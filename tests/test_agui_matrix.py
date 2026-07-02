"""AG-UI edge-case matrix — exercises the bridge against purpose-built agents
that each stress one event class, asserting the *observed* AG-UI output.

This is the audit that converts the langstage-core -> AG-UI mapping
from "claimed" to "proven", and it pins the three robustness fixes the audit
surfaced:
  1. graphs compiled WITHOUT a checkpointer are auto-handled (AG-UI's
     aget_state would otherwise hard-crash with "No checkpointer set");
  2. an agent exception mid-run becomes a terminal RUN_ERROR event instead of
     a silently-dying stream / unhandled 500;
  3. interrupts surface as a CUSTOM `on_interrupt` event and resume via
     forwardedProps.command.resume.

Every agent here is compiled WITHOUT a checkpointer on purpose, so the suite
also proves the auto-attach fix end to end.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
pytest.importorskip("langgraph", reason="needs langgraph")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langstage_core.agui import build_app  # noqa: E402
from langstage_core.demo import create_stub_agent  # noqa: E402


# ── driver ───────────────────────────────────────────────────────────

def _run_input(text, *, thread="t1", run="r1", resume=None):
    # NB: each turn needs a UNIQUE message id — the AG-UI/LangGraph adapter
    # dedupes messages by id, so reusing an id silently drops later turns.
    body = {
        "threadId": thread, "runId": run,
        "messages": [{"id": f"msg-{thread}-{run}", "role": "user", "content": text}],
        "tools": [], "context": [], "state": {}, "forwardedProps": {},
    }
    if resume is not None:
        body["forwardedProps"] = {"command": {"resume": resume}}
    return body


def _drive(app, body):
    """POST a run, return (status_code, [event dicts])."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/", json=body)
    events = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return resp.status_code, events


def _types(events):
    return [e.get("type") for e in events]


def _text(events):
    return "".join(e.get("delta", "") for e in events if e.get("type") == "TEXT_MESSAGE_CONTENT")


# ── purpose-built agents (all compiled WITHOUT a checkpointer) ─────────

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class _ToolThenAnswer(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "tool-matrix-stub"

    def bind_tools(self, *a, **k):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _reply(self, messages: List[BaseMessage]) -> AIMessage:
        if messages and getattr(messages[-1], "type", None) == "tool":
            return AIMessage(content="the sum is 5")
        return AIMessage(content="", tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 2, "b": 3}}])

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        if messages and getattr(messages[-1], "type", None) == "tool":
            for tok in ["the ", "sum ", "is ", "5"]:
                ch = ChatGenerationChunk(message=AIMessageChunk(content=tok))
                if run_manager:
                    run_manager.on_llm_new_token(tok, chunk=ch)
                yield ch
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": "add", "args": '{"a": 2, "b": 3}', "id": "call_1", "index": 0}],
            ))


def _tool_agent():
    model = _ToolThenAnswer()

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    b = StateGraph(MessagesState)
    b.add_node("model", call_model)
    b.add_node("tools", ToolNode([add]))
    b.add_edge(START, "model")
    b.add_conditional_edges("model", route, {"tools": "tools", END: END})
    b.add_edge("tools", "model")
    return b.compile()  # NO checkpointer — exercises the auto-attach fix


def _interrupt_agent():
    def ask(state):
        decision = interrupt({"question": "approve?", "tool": "deploy"})
        return {"messages": [AIMessage(content=f"resumed: {decision}")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile()  # NO checkpointer


def _error_agent():
    def boom(state):
        raise ValueError("intentional boom")

    b = StateGraph(MessagesState)
    b.add_node("boom", boom)
    b.add_edge(START, "boom")
    b.add_edge("boom", END)
    return b.compile()  # NO checkpointer


def _plain_echo_no_checkpointer():
    """An echo graph with NO checkpointer (create_stub_agent ships one)."""
    from langchain_core.messages import AIMessage as _AI

    def respond(state):
        last = next((m.content for m in reversed(state["messages"])
                     if getattr(m, "type", None) == "human"), "")
        return {"messages": [_AI(content=f"echo: {last}")]}

    b = StateGraph(MessagesState)
    b.add_node("respond", respond)
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    return b.compile()


# ── tests ────────────────────────────────────────────────────────────

def test_tool_calls_map_to_agui():
    status, events = _drive(build_app(_tool_agent()), _run_input("add 2 and 3"))
    assert status == 200
    t = _types(events)
    assert "TOOL_CALL_START" in t
    assert "TOOL_CALL_RESULT" in t
    start = next(e for e in events if e["type"] == "TOOL_CALL_START")
    result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
    assert start.get("toolCallName") == "add"
    assert result.get("content") == "5"
    assert "the sum is 5" in _text(events)
    assert "RUN_FINISHED" in t and "RUN_ERROR" not in t


def test_interrupt_is_signaled_and_resumes():
    app = build_app(_interrupt_agent())

    status, first = _drive(app, _run_input("go", thread="ti", run="r1"))
    assert status == 200
    customs = [e for e in first if e.get("type") == "CUSTOM" and e.get("name") == "on_interrupt"]
    assert customs, f"interrupt not signaled as CUSTOM/on_interrupt: {_types(first)}"
    # the interrupt payload travels to the client
    assert "approve?" in json.dumps(customs[0])
    assert "RUN_ERROR" not in _types(first)

    # resume the SAME thread with a decision via forwardedProps.command.resume
    status, second = _drive(app, _run_input("", thread="ti", run="r2", resume="approved"))
    assert status == 200
    assert "RUN_FINISHED" in _types(second)
    assert "RUN_ERROR" not in _types(second)
    # the resumed value reached the agent (it echoed "resumed: approved" into state)
    assert "resumed: approved" in json.dumps(second)


def test_agent_error_becomes_run_error():
    """The resilient endpoint emits a terminal RUN_ERROR instead of a
    silently-dying stream / unhandled 500."""
    status, events = _drive(build_app(_error_agent()), _run_input("crash"))
    assert status == 200
    errs = [e for e in events if e.get("type") == "RUN_ERROR"]
    assert errs, f"expected a RUN_ERROR event, got: {_types(events)}"
    assert "intentional boom" in errs[0].get("message", "")


def test_graph_without_checkpointer_is_handled():
    """AG-UI's aget_state needs a checkpointer; the bridge auto-attaches one
    so a plain compiled graph doesn't hard-crash."""
    status, events = _drive(build_app(_plain_echo_no_checkpointer()), _run_input("ping"))
    assert status == 200
    # The fix's job: no hard-crash, run completes. (This node appends an
    # AIMessage directly rather than streaming a model, so its content lands in
    # the state/messages snapshot, not as TEXT_MESSAGE_CONTENT deltas.)
    assert "RUN_ERROR" not in _types(events)
    assert "RUN_FINISHED" in _types(events)
    assert "echo: ping" in json.dumps(events)


def test_multi_turn_thread_persists():
    app = build_app(create_stub_agent(reply_prefix="echo: "))
    s1, e1 = _drive(app, _run_input("first", thread="conv", run="r1"))
    s2, e2 = _drive(app, _run_input("second", thread="conv", run="r2"))
    assert s1 == 200 and s2 == 200
    assert "first" in _text(e1)
    assert "second" in _text(e2)
    # state persisted across turns: the second run carries BOTH turns
    blob = json.dumps(e2)
    assert "first" in blob and "second" in blob


def test_concurrent_requests_are_isolated():
    """The adapter clones per request; distinct threads must not cross-talk."""
    app = build_app(create_stub_agent(reply_prefix="echo: "))
    inputs = [f"msg-{i}" for i in range(8)]

    def one(i):
        _, events = _drive(app, _run_input(inputs[i], thread=f"th-{i}", run=f"r-{i}"))
        return inputs[i], _text(events)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, range(8)))

    for sent, got in results:
        assert sent in got, f"{sent!r} not echoed in {got!r}"
        # no other request's payload leaked into this response
        for other in inputs:
            if other != sent:
                assert other not in got, f"cross-talk: {other!r} leaked into {got!r}"
