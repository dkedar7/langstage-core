"""A streamed message whose chunks don't share one id is still sent once (gh #242).

On the OpenAI Responses API only a model call's first, empty chunk carries the provider's
id (``resp_…``). The text chunks carry LangChain's run id (``lc_run--…``), which is the id
the AG-UI adapter opens the text message under, while the finished message in the
checkpoint takes the provider's id. ``_TurnStream`` matched streamed messages to checkpoint
messages by id, never found this one, and sent it a second time in full, after its own
tool call.

``fixtures/responses_api_turn.json`` holds the chunks of a real two-call turn (narrate and
call a tool, then answer), captured from langchain-openai on that API with ids renamed.
``_ReplayModel`` streams them back, so every test here is keyless.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from langstage_core.agui import (  # noqa: E402
    build_app,
    collect_chunk_frames,
    collect_event_frames,
    iter_chunk_frames,
    iter_event_frames,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "responses_api_turn.json").read_text(encoding="utf-8"))
VARIANTS = ["default", "v1"]  # ChatOpenAI output_version: its default, and "v1"
FIRST = "I’ll add the numbers."
FINAL = "The result is 42.75."


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


class _ReplayModel(BaseChatModel):
    """Streams the captured chunks of the n-th model call, n = AI messages so far."""

    calls: list

    @property
    def _llm_type(self) -> str:
        return "replay"

    def bind_tools(self, tools, **kwargs):
        return self

    def _chunks(self, messages):
        n = sum(1 for m in messages if getattr(m, "type", None) == "ai")
        for raw in self.calls[n]:
            # id None: LangChain fills in ``lc_run--<run id>``, as it did in the capture.
            yield ChatGenerationChunk(message=AIMessageChunk(**raw))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield from self._chunks(messages)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._chunks(messages):
            yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        merged = None
        for chunk in self._chunks(messages):
            merged = chunk if merged is None else merged + chunk
        m = merged.message
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=m.content, tool_calls=m.tool_calls, id=m.id,
            response_metadata=m.response_metadata))])


def _agent_graph(variant, *, checkpointer=True):
    """A model node and a ToolNode in a loop: the shape ``create_agent`` builds."""
    model = _ReplayModel(calls=_FIXTURE[variant])

    async def call_model(state):
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    g.add_edge("tools", "model")
    return g.compile(checkpointer=InMemorySaver() if checkpointer else None)


def _subgraph_graph(variant):
    """The same agent as a node of an outer graph (the supervisor shape, gh #240)."""
    outer = StateGraph(MessagesState)
    outer.add_node("worker", _agent_graph(variant, checkpointer=False))
    outer.add_edge(START, "worker")
    outer.add_edge("worker", END)
    return outer.compile(checkpointer=InMemorySaver())


GRAPHS = {"flat": _agent_graph, "subgraph": _subgraph_graph}


async def _collect(aiter):
    return [f async for f in aiter]


def _served_events(graph):
    client = TestClient(build_app(graph))
    body = {
        "threadId": "t-served", "runId": "r-served",
        "messages": [{"id": "u1", "role": "user", "content": "add them"}],
        "tools": [], "context": [], "state": {}, "forwardedProps": {},
    }
    resp = client.post("/", json=body, headers={"Accept": "text/event-stream"})
    assert resp.status_code == 200
    return [json.loads(line.strip()[len("data:"):].strip())
            for line in resp.text.splitlines() if line.strip().startswith("data:")]


def _runs(frames, kind_of, text_of):
    """Collapse consecutive text frames into one ``("text", joined)`` entry."""
    out = []
    for f in frames:
        kind = kind_of(f)
        if kind is None:
            continue
        if kind == "text":
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1] + text_of(f))
            else:
                out.append(("text", text_of(f)))
        else:
            out.append((kind, None))
    return out


EXPECTED = [("text", FIRST), ("tool_start", None), ("tool_end", None), ("text", FINAL)]


def test_the_fixture_has_the_id_split_the_bug_needs():
    # Guards the fixture itself: a provider id on the first, empty chunk only.
    for variant in VARIANTS:
        for call in _FIXTURE[variant]:
            assert call[0]["id"].startswith("resp_") and not call[0]["content"]
            assert all(c["id"] is None for c in call[1:])


@pytest.mark.parametrize("shape", list(GRAPHS))
@pytest.mark.parametrize("variant", VARIANTS)
async def test_event_wire_sends_each_message_once_and_in_order(variant, shape):
    frames = await _collect(iter_event_frames(GRAPHS[shape](variant), "add them", f"e-{variant}"))
    got = _runs(
        frames,
        lambda f: {"content": "text", "tool_start": "tool_start", "tool_end": "tool_end"}.get(f["type"]),
        lambda f: f["content"])
    assert got == EXPECTED
    assert frames[-1]["type"] == "complete"
    # It still streams token by token: the fix must not turn it into one finished frame.
    assert len([f for f in frames if f["type"] == "content"]) > 4
    # One message id per message, so the boundary between the two is still marked.
    ids = [f["message_id"] for f in frames if f["type"] == "content"]
    assert len(set(ids)) == 2


@pytest.mark.parametrize("shape", list(GRAPHS))
@pytest.mark.parametrize("variant", VARIANTS)
async def test_chunk_wire_sends_each_message_once_and_in_order(variant, shape):
    frames = await _collect(iter_chunk_frames(GRAPHS[shape](variant), "add them", f"c-{variant}"))

    def kind(f):
        if f.get("status") != "streaming":
            return None
        return ("text" if "chunk" in f else "tool_start" if "tool_calls" in f
                else "tool_end" if "tool_result" in f else None)

    assert _runs(frames, kind, lambda f: f["chunk"]) == EXPECTED
    assert frames[-1]["status"] == "complete"


@pytest.mark.parametrize("shape", list(GRAPHS))
@pytest.mark.parametrize("variant", VARIANTS)
def test_served_wire_sends_each_message_once_and_in_order(variant, shape):
    events = _served_events(GRAPHS[shape](variant))
    got = _runs(
        events,
        lambda e: {"TEXT_MESSAGE_CONTENT": "text", "TOOL_CALL_START": "tool_start",
                   "TOOL_CALL_RESULT": "tool_end"}.get(e["type"]),
        lambda e: e["delta"])
    assert got == EXPECTED
    types = [e["type"] for e in events]
    assert types.count("TEXT_MESSAGE_START") == 2 == types.count("TEXT_MESSAGE_END")
    assert types.count("TOOL_CALL_START") == 1 == types.count("TOOL_CALL_RESULT")


@pytest.mark.parametrize("variant", VARIANTS)
async def test_collectors_hold_the_text_once(variant):
    ev = await collect_event_frames(_agent_graph(variant), "add them", f"ce-{variant}")
    ch = await collect_chunk_frames(_agent_graph(variant), "add them", f"cc-{variant}")
    for result in (ev, ch):
        assert result.text.count(FIRST) == 1 and result.text.count(FINAL) == 1
        assert [c["name"] for c in result.tool_calls] == ["add"]
    assert ev.outcome == "complete"


async def test_a_finished_message_after_a_streamed_one_is_still_sent():
    # The alias only covers ids from one model call. A later node's finished message
    # (not streamed, its own id) must still reach the wire, once.
    model = _ReplayModel(calls=[_FIXTURE["default"][1]])

    async def call_model(state):
        return {"messages": [await model.ainvoke(state["messages"][:1])]}

    def sign_off(state):
        return {"messages": [AIMessage(content="Anything else?", id="closing-1")]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("sign_off", sign_off)
    g.add_edge(START, "model")
    g.add_edge("model", "sign_off")
    g.add_edge("sign_off", END)
    for wire, text_key in ((iter_event_frames, "content"), (iter_chunk_frames, "chunk")):
        frames = await _collect(wire(g.compile(checkpointer=InMemorySaver()), "hi", f"mix-{text_key}"))
        text = "".join(f[text_key] for f in frames if text_key in f)
        assert text == FINAL + "Anything else?"
