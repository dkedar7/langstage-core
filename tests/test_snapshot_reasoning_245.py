"""A finished (non-streamed) assistant message keeps its reasoning (gh #245).

A message that reaches the stream without token streaming (a model with
``disable_streaming=True``, a non-streaming provider, a node that returns an
``AIMessage``) used to lose its ``reasoning`` / ``thinking`` blocks: only the streaming
``ReasoningMessageContentEvent`` ever became a ``reasoning`` frame, and the checkpoint
walk that sends finished messages yielded text, tool calls and tool results only.

These tests cover both in-process wires and the served endpoint, the block shapes
providers use, the order (reasoning, then text, then tool calls), and the cases where
the reasoning must NOT be sent again: a token-streamed message, a resumed turn, and a
message whose stream events arrive after it was sent from the checkpoint.
"""
import json

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langstage_core.agui import (  # noqa: E402
    _message_reasoning,
    _snapshot_items,
    _TurnStream,
    build_agent,
    build_app,
    collect_chunk_frames,
    collect_event_frames,
    iter_chunk_frames,
    iter_event_frames,
)

THINK = "Let me think: 2+2."
ANSWER = "It is 4."

# One entry per shape a provider uses for the reasoning of a finished message.
SHAPES = {
    "reasoning-block": dict(content=[{"type": "reasoning", "reasoning": THINK},
                                     {"type": "text", "text": ANSWER}]),
    "thinking-block": dict(content=[{"type": "thinking", "thinking": THINK, "signature": "s"},
                                    {"type": "text", "text": ANSWER}]),
    "responses-summary": dict(content=[
        {"type": "reasoning", "id": "rs_1",
         "summary": [{"type": "summary_text", "text": THINK}]},
        {"type": "text", "text": ANSWER}]),
    "bedrock": dict(content=[
        {"type": "reasoning_content", "reasoning_content": {"text": THINK, "signature": "s"}},
        {"type": "text", "text": ANSWER}]),
    "reasoning_content-kwarg": dict(content=ANSWER,
                                    additional_kwargs={"reasoning_content": THINK}),
    "reasoning-kwarg": dict(content=ANSWER, additional_kwargs={
        "reasoning": {"id": "rs_2", "summary": [{"type": "summary_text", "text": THINK}]}}),
}


async def _collect(aiter):
    return [f async for f in aiter]


def _node_graph(**message):
    """A plain node that returns a finished ``AIMessage``."""
    def respond(state):
        return {"messages": [AIMessage(**message)]}

    g = StateGraph(MessagesState)
    g.add_node("model", respond)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


def _non_streaming_model_graph():
    """The issue's repro: a chat model with ``disable_streaming=True``."""
    msg = AIMessage(**SHAPES["reasoning-block"])
    model = GenericFakeChatModel(messages=iter([msg] * 10), disable_streaming=True)

    def node(state):
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", node)
    g.add_edge(START, "model")
    g.add_edge("model", END)
    return g.compile()


def _event_kinds(frames):
    return [(f["type"], f.get("content")) for f in frames
            if f["type"] in ("reasoning", "content", "tool_start", "tool_end")]


def _chunk_kinds(frames):
    out = []
    for f in frames:
        if "reasoning" in f:
            out.append(("reasoning", f["reasoning"]))
        elif "chunk" in f:
            out.append(("content", f["chunk"]))
        elif "tool_calls" in f:
            out.append(("tool_start", None))
        elif "tool_result" in f:
            out.append(("tool_end", None))
    return out


def _served_events(graph, thread="t-served"):
    client = TestClient(build_app(graph))
    body = {
        "threadId": thread, "runId": "r-served",
        "messages": [{"id": "u1", "role": "user", "content": "q"}],
        "tools": [], "context": [], "state": {}, "forwardedProps": {},
    }
    resp = client.post("/", json=body, headers={"Accept": "text/event-stream"})
    assert resp.status_code == 200
    return [json.loads(line.strip()[len("data:"):].strip())
            for line in resp.text.splitlines() if line.strip().startswith("data:")]


def _served_kinds(events):
    names = {"REASONING_MESSAGE_CONTENT": "reasoning", "TEXT_MESSAGE_CONTENT": "content"}
    out = []
    for e in events:
        if e["type"] in names:
            out.append((names[e["type"]], e["delta"]))
        elif e["type"] == "TOOL_CALL_START":
            out.append(("tool_start", None))
        elif e["type"] == "TOOL_CALL_RESULT":
            out.append(("tool_end", None))
    return out


EXPECTED = [("reasoning", THINK), ("content", ANSWER)]


# ── the reasoning of a finished message reaches every wire ───────────


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_event_wire_emits_a_finished_messages_reasoning(shape):
    frames = await _collect(iter_event_frames(_node_graph(**SHAPES[shape]), "q", "e-" + shape))
    assert _event_kinds(frames) == EXPECTED
    assert [f["node"] for f in frames if f["type"] == "reasoning"] == ["model"]


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_chunk_wire_emits_a_finished_messages_reasoning(shape):
    frames = await _collect(iter_chunk_frames(_node_graph(**SHAPES[shape]), "q", "c-" + shape))
    assert _chunk_kinds(frames) == EXPECTED
    assert [f["node"] for f in frames if "reasoning" in f] == ["model"]


@pytest.mark.parametrize("shape", list(SHAPES))
def test_served_wire_emits_a_finished_messages_reasoning(shape):
    events = _served_events(_node_graph(**SHAPES[shape]), "s-" + shape)
    assert _served_kinds(events) == EXPECTED
    types = [e["type"] for e in events if e["type"].startswith("REASONING")]
    assert types == ["REASONING_START", "REASONING_MESSAGE_START", "REASONING_MESSAGE_CONTENT",
                     "REASONING_MESSAGE_END", "REASONING_END"]


async def test_non_streaming_model_keeps_its_reasoning_on_every_wire():
    # The issue's repro: disable_streaming=True.
    ev = await _collect(iter_event_frames(_non_streaming_model_graph(), "q", "n1"))
    ch = await _collect(iter_chunk_frames(_non_streaming_model_graph(), "q", "n2"))
    assert _event_kinds(ev) == EXPECTED
    assert _chunk_kinds(ch) == EXPECTED
    assert _served_kinds(_served_events(_non_streaming_model_graph())) == EXPECTED


async def test_collectors_carry_the_reasoning():
    ev = await collect_event_frames(_node_graph(**SHAPES["reasoning-block"]), "q", thread_id="k1")
    ch = await collect_chunk_frames(_node_graph(**SHAPES["thinking-block"]), "q", thread_id="k2")
    assert ev.reasoning == THINK and ev.text == ANSWER
    assert ch.reasoning == THINK and ch.text == ANSWER


def test_served_reasoning_id_matches_the_snapshots_reasoning_message():
    # ag-ui-langgraph puts a ``reasoning`` block in MESSAGES_SNAPSHOT as its own message.
    # The events sent for it carry the same id, so a client can match the two.
    for shape in ("reasoning-block", "responses-summary"):
        events = _served_events(_node_graph(**SHAPES[shape]), "id-" + shape)
        sent = {e["messageId"] for e in events if e["type"] == "REASONING_MESSAGE_CONTENT"}
        snapshot = [m for e in events if e["type"] == "MESSAGES_SNAPSHOT" for m in e["messages"]]
        in_snapshot = {m["id"] for m in snapshot if m["role"] == "reasoning"}
        assert sent and sent == in_snapshot, (shape, sent, in_snapshot)


async def test_a_message_without_reasoning_is_unchanged():
    frames = await _collect(iter_event_frames(_node_graph(content=ANSWER), "q", "plain"))
    assert _event_kinds(frames) == [("content", ANSWER)]
    assert not any(e["type"].startswith("REASONING")
                   for e in _served_events(_node_graph(content=ANSWER), "plain-s"))


async def test_redacted_and_empty_reasoning_emit_nothing():
    frames = await _collect(iter_event_frames(_node_graph(content=[
        {"type": "redacted_thinking", "data": "xyz"},
        {"type": "reasoning", "id": "rs_9", "summary": []},
        {"type": "reasoning", "reasoning": ""},
        {"type": "text", "text": ANSWER}]), "q", "redacted"))
    assert _event_kinds(frames) == [("content", ANSWER)]


async def test_several_reasoning_blocks_each_arrive_in_order():
    frames = await _collect(iter_event_frames(_node_graph(content=[
        {"type": "thinking", "thinking": "First."},
        {"type": "thinking", "thinking": "Second."},
        {"type": "text", "text": ANSWER}]), "q", "two"))
    assert _event_kinds(frames) == [
        ("reasoning", "First."), ("reasoning", "Second."), ("content", ANSWER)]


# ── order: reasoning, then text, then tool calls ─────────────────────


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _tool_graph(first_text=""):
    """A finished message that reasons and calls a tool (with or without narration),
    then a finished answer that reasons again."""
    def agent(state):
        if isinstance(state["messages"][-1], ToolMessage):
            return {"messages": [AIMessage(content=[
                {"type": "thinking", "thinking": "Sum is 42."},
                {"type": "text", "text": "The answer is 42."}])]}
        content = [{"type": "thinking", "thinking": "I should add."}]
        if first_text:
            content.append({"type": "text", "text": first_text})
        return {"messages": [AIMessage(
            content=content,
            tool_calls=[{"name": "add", "args": {"a": 17, "b": 25}, "id": "c1"}])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    g.add_edge("tools", "agent")
    return g


def _flat(first_text=""):
    return _tool_graph(first_text).compile()


def _subgraph(first_text=""):
    outer = StateGraph(MessagesState)
    outer.add_node("worker", _tool_graph(first_text).compile())
    outer.add_edge(START, "worker")
    outer.add_edge("worker", END)
    return outer.compile(checkpointer=InMemorySaver())


def _expected_tool_turn(first_text):
    out = [("reasoning", "I should add.")]
    if first_text:
        out.append(("content", first_text))
    return out + [("tool_start", None), ("tool_end", None),
                  ("reasoning", "Sum is 42."), ("content", "The answer is 42.")]


@pytest.mark.parametrize("first_text", ["", "Let me add those."], ids=["tool-only", "narrated"])
@pytest.mark.parametrize("factory", [_flat, _subgraph], ids=["flat", "subgraph"])
async def test_reasoning_precedes_the_messages_text_and_tool_call(factory, first_text):
    ev = await _collect(iter_event_frames(factory(first_text), "17+25?", "o1"))
    ch = await _collect(iter_chunk_frames(factory(first_text), "17+25?", "o2"))
    expected = _expected_tool_turn(first_text)
    assert [(k, t if k != "tool_end" else None) for k, t in _event_kinds(ev)] == expected
    assert _chunk_kinds(ch) == expected
    assert _served_kinds(_served_events(factory(first_text))) == expected


# ── a token-streamed message's reasoning is not sent a second time ───


class _StreamingReasoner(BaseChatModel):
    """Streams reasoning deltas, then (optionally) text or a tool call, as content
    blocks, the way a reasoning model does under token streaming."""

    text: str = ANSWER
    call_tool: bool = False

    @property
    def _llm_type(self) -> str:
        return "streaming-reasoner"

    def bind_tools(self, tools, **kwargs):
        return self

    def _chunks(self, messages):
        if self.call_tool and isinstance(messages[-1], ToolMessage):
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[{"type": "text", "text": "Done.", "index": 0}]))
            return
        for part in ("Let me think: ", "2+2."):
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[{"type": "reasoning", "reasoning": part, "index": 0}]))
        if self.call_tool:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": "add", "args": '{"a": 2, "b": 2}', "id": "s1", "index": 0}]))
        elif self.text:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[{"type": "text", "text": self.text, "index": 1}]))

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
            content=m.content, tool_calls=m.tool_calls, id=m.id))])


def _streamed_graph(**model_kwargs):
    model = _StreamingReasoner(**model_kwargs)

    async def call_model(state):
        return {"messages": [await model.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode([add]))
    g.add_edge(START, "model")
    g.add_conditional_edges(
        "model", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    g.add_edge("tools", "model")
    return g.compile(checkpointer=InMemorySaver())


def _joined(kinds):
    """Join consecutive reasoning (and content) entries, as a renderer would."""
    out = []
    for kind, text in kinds:
        if kind in ("reasoning", "content") and out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + text)
        else:
            out.append((kind, text))
    return out


STREAMED = {
    "text": (dict(), [("reasoning", THINK), ("content", ANSWER)]),
    "tool-call-only": (dict(call_tool=True), [
        ("reasoning", THINK), ("tool_start", None), ("tool_end", None), ("content", "Done.")]),
    "reasoning-only": (dict(text=""), [("reasoning", THINK)]),
}


@pytest.mark.parametrize("case", list(STREAMED))
async def test_streamed_reasoning_is_sent_once(case):
    kwargs, expected = STREAMED[case]
    ev = await _collect(iter_event_frames(_streamed_graph(**kwargs), "q", "st1"))
    ch = await _collect(iter_chunk_frames(_streamed_graph(**kwargs), "q", "st2"))
    assert _joined((k, t if k != "tool_end" else None) for k, t in _event_kinds(ev)) == expected
    assert _joined(_chunk_kinds(ch)) == expected
    assert _joined(_served_kinds(_served_events(_streamed_graph(**kwargs)))) == expected


# ── a resumed turn does not repeat the reasoning from before the pause ─


def _pausing_graph():
    def think(state):
        return {"messages": [AIMessage(**SHAPES["reasoning-block"])]}

    def confirm(state):
        answer = interrupt({"question": "go on?"})
        return {"messages": [AIMessage(content=[
            {"type": "reasoning", "reasoning": "They said " + str(answer) + "."},
            {"type": "text", "text": "Continuing."}])]}

    g = StateGraph(MessagesState)
    g.add_node("think", think)
    g.add_node("confirm", confirm)
    g.add_edge(START, "think")
    g.add_edge("think", "confirm")
    g.add_edge("confirm", END)
    return g.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize("wire", ["event", "chunk"])
async def test_resume_does_not_replay_pre_interrupt_reasoning(wire):
    agent = build_agent(_pausing_graph())
    it, kinds = ((iter_event_frames, _event_kinds) if wire == "event"
                 else (iter_chunk_frames, _chunk_kinds))
    first = kinds(await _collect(it(agent, "start", "p-" + wire)))
    assert first == EXPECTED
    resumed = kinds(await _collect(it(agent, "", "p-" + wire, resume="yes")))
    assert resumed == [("reasoning", "They said yes."), ("content", "Continuing.")]


# ── the graph runs ahead of the reader ───────────────────────────────


class _State:
    def __init__(self, messages):
        self.values = {"messages": list(messages)}
        self.tasks = ()
        self.config = {}


class _ScriptedAgent:
    """``run()`` replays scripted AG-UI events; the checkpoint already holds
    ``committed`` (the model call finished before its events were read)."""

    def __init__(self, events, committed):
        self._events, self._committed, self._running = events, committed, False
        self.graph = self

    async def aget_state(self, config, **kwargs):
        return _State(self._committed if self._running else [])

    async def run(self, run_input):
        self._running = True
        for ev in self._events:
            yield ev


async def test_reasoning_streamed_after_a_checkpoint_send_is_dropped():
    from ag_ui.core import (
        EventType,
        RawEvent,
        ReasoningEndEvent,
        ReasoningMessageContentEvent,
        ReasoningMessageEndEvent,
        ReasoningMessageStartEvent,
        ReasoningStartEvent,
        RunAgentInput,
        StepFinishedEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
    )

    def raw(content):
        return RawEvent(type=EventType.RAW, event={
            "event": "on_chat_model_stream", "run_id": "r1", "metadata": {},
            "data": {"chunk": AIMessageChunk(content=content, id="lc_run--r1")}})

    events = [
        # An earlier step ends. The model call after it has already committed its message.
        StepFinishedEvent(type=EventType.STEP_FINISHED, step_name="tools"),
        # Only now are that call's stream events read: its reasoning, then its text.
        raw([{"type": "reasoning", "reasoning": THINK, "index": 0}]),
        ReasoningStartEvent(type=EventType.REASONING_START, message_id="uuid-1"),
        ReasoningMessageStartEvent(
            type=EventType.REASONING_MESSAGE_START, message_id="uuid-1", role="reasoning"),
        ReasoningMessageContentEvent(
            type=EventType.REASONING_MESSAGE_CONTENT, message_id="uuid-1", delta=THINK),
        raw([{"type": "text", "text": ANSWER, "index": 1}]),
        ReasoningMessageEndEvent(type=EventType.REASONING_MESSAGE_END, message_id="uuid-1"),
        ReasoningEndEvent(type=EventType.REASONING_END, message_id="uuid-1"),
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START, message_id="lc_run--r1", role="assistant"),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id="lc_run--r1", delta=ANSWER),
        TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="lc_run--r1"),
        StepFinishedEvent(type=EventType.STEP_FINISHED, step_name="model"),
    ]
    committed = [AIMessage(id="lc_run--r1", **SHAPES["reasoning-block"])]
    run_input = RunAgentInput(thread_id="t", run_id="r", state={}, messages=[], tools=[],
                              context=[], forwarded_props={})
    out = [ev async for ev in _TurnStream(_ScriptedAgent(events, committed), run_input).events()]
    reasoning = [(type(ev).__name__, ev.message_id) for ev in out
                 if type(ev).__name__.startswith("Reasoning")]
    rid = "lc_run--r1-reasoning-0"
    assert reasoning == [
        ("ReasoningStartEvent", rid), ("ReasoningMessageStartEvent", rid),
        ("ReasoningMessageContentEvent", rid), ("ReasoningMessageEndEvent", rid),
        ("ReasoningEndEvent", rid)]
    assert [ev.delta for ev in out if type(ev).__name__ == "ReasoningMessageContentEvent"] == [THINK]
    assert [ev.delta for ev in out if type(ev).__name__ == "TextMessageContentEvent"] == [ANSWER]


# ── the pieces ───────────────────────────────────────────────────────


def test_message_reasoning_reads_every_shape():
    for shape, kwargs in SHAPES.items():
        got = _message_reasoning(AIMessage(id="m1", **kwargs))
        assert [text for _, text in got] == [THINK], shape
    assert _message_reasoning(AIMessage(id="m1", **SHAPES["reasoning-block"]))[0][0] == "m1-reasoning-0"
    assert _message_reasoning(AIMessage(id="m1", **SHAPES["responses-summary"]))[0][0] == "rs_1"
    assert _message_reasoning(AIMessage(content=ANSWER)) == []
    assert _message_reasoning(None) == []
    # Content that isn't the expected shape is ignored, never an error.
    assert _message_reasoning(AIMessage(content=[
        {"type": "reasoning", "summary": [{"text": 7}, "x", {"text": ""}]},
        {"type": "reasoning_content", "reasoning_content": 12},
        {"type": "thinking", "thinking": None}, "plain"])) == []


class _Msg:
    def __init__(self, role, id, content="", tool_calls=None):
        self.role, self.id, self.content, self.tool_calls = role, id, content, tool_calls or []


def _walk(messages, **kw):
    args = dict(streamed_ids=set(), tool_names={}, streamed_result_ids=set(),
                step_nodes=[], current_node="n")
    args.update(kw)
    return [(i["kind"], i["text"]) for i in _snapshot_items(messages, **args)
            if i["kind"] in ("reasoning", "content")]


def test_snapshot_walk_reads_ag_ui_reasoning_messages():
    # An agent with no readable checkpoint: the final snapshot is the only source, and
    # ag-ui carries reasoning there as ``role == "reasoning"`` messages before the owner.
    snapshot = [_Msg("user", "u1", "q"), _Msg("reasoning", "a1-reasoning-0", THINK),
                _Msg("assistant", "a1", ANSWER)]
    assert _walk(snapshot) == [("reasoning", THINK), ("content", ANSWER)]
    # Already out: the owner's text was streamed, or its reasoning was.
    assert _walk(snapshot, streamed_ids={"a1"}) == []
    assert _walk(snapshot, reasoned_ids={"a1"}) == [("content", ANSWER)]
    # ...or that very reasoning message was (the adapter lists a subagent's streamed
    # reasoning in the snapshot too, ahead of whichever message comes next).
    assert _walk(snapshot, sent_reasoning_ids={"a1-reasoning-0"}) == [("content", ANSWER)]


def test_this_turn_drops_reasoning_with_its_pre_run_owner():
    from ag_ui.core import RunAgentInput

    stream = _TurnStream(object(), RunAgentInput(
        thread_id="t", run_id="r", state={}, messages=[], tools=[], context=[],
        forwarded_props={}))
    stream._pre_ids = {"u1", "a1"}
    messages = [_Msg("user", "u1", "q"), _Msg("reasoning", "a1-reasoning-0", "old"),
                _Msg("assistant", "a1", "before the pause"),
                _Msg("reasoning", "a2-reasoning-0", "new"), _Msg("assistant", "a2", "after")]
    assert [m.id for m in stream.this_turn(messages)] == ["a2-reasoning-0", "a2"]
