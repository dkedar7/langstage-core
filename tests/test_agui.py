"""Tests for the AG-UI bridge (langstage_core.agui).

These run end to end against the keyless demo stub: build an AG-UI ASGI app,
POST a RunAgentInput, and assert the AG-UI event stream comes back with a clean
run lifecycle (and the echoed content when token streaming is available).

Skipped automatically if the optional ``agui`` extra (ag-ui-langgraph) isn't
installed.
"""
import json

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from langstage_core.agui import build_agent, build_app  # noqa: E402
from langstage_core.demo import create_stub_agent  # noqa: E402


def _run_input(text: str) -> dict:
    """A minimal valid AG-UI RunAgentInput body."""
    return {
        "threadId": "t-test",
        "runId": "r-test",
        "messages": [{"id": "m1", "role": "user", "content": text}],
        "tools": [],
        "context": [],
        "state": {},
        "forwardedProps": {},
    }


def _event_types(sse_text: str) -> list[str]:
    types = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "type" in payload:
            types.append(payload["type"])
    return types


def _text_deltas(sse_text: str) -> str:
    out = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "TEXT_MESSAGE_CONTENT":
            out.append(payload.get("delta", ""))
    return "".join(out)


def test_build_agent_wraps_graph():
    graph = create_stub_agent()
    agent = build_agent(graph, name="Test Agent")
    assert agent.name == "Test Agent"
    assert hasattr(agent, "clone") and hasattr(agent, "run")


def test_endpoint_is_mounted():
    """The AG-UI route exists at the configured path (POST). We don't assert
    the adapter's health-route placement — just that '/' is wired, not 404."""
    app = build_app(create_stub_agent(), name="Demo Agent")
    client = TestClient(app)
    assert client.get("/").status_code != 404


def test_agui_run_lifecycle_and_echo():
    app = build_app(create_stub_agent(reply_prefix="(demo agent) You said: "))
    client = TestClient(app)
    resp = client.post("/", json=_run_input("hi there"))
    assert resp.status_code == 200, resp.text

    types = _event_types(resp.text)
    # Clean lifecycle: started, finished, no error.
    assert "RUN_STARTED" in types, types
    assert "RUN_FINISHED" in types, types
    assert "RUN_ERROR" not in types, resp.text

    # The echo content travels as AG-UI text deltas — proving rich content
    # (not just lifecycle) survives the bridge.
    deltas = _text_deltas(resp.text)
    assert "hi there" in deltas, f"expected echoed text in AG-UI deltas, got: {deltas!r}"


def test_build_agent_rejects_wrong_type():
    # gh langstage-jupyter #92: a non-graph input fails fast with an actionable
    # TypeError, not the leaked `'dict' object has no attribute 'nodes'` from deep in
    # the adapter (the anti-pattern gh #112 / #100 fixed elsewhere).
    for obj in ({"hello": "world"}, None, 42):
        with pytest.raises(TypeError, match="compiled LangGraph graph"):
            build_agent(obj)


def test_build_agent_rejects_uncompiled_stategraph():
    # A StateGraph the caller forgot to .compile() gets pointed at the fix, not a
    # cryptic 'StateGraph' object has no attribute 'aget_state' at first turn.
    from langgraph.graph import END, MessagesState, START, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("n", lambda s: s)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    with pytest.raises(TypeError, match=r"\.compile\(\)"):
        build_agent(b)  # forgot .compile()
