"""Wire consistency: served vs in-process, event vs chunk wire, message boundaries,
tool status/duration, terminal frames, and isolated one-shots.

Every graph here is keyless and deterministic (finished ``AIMessage``s, real
``ToolNode`` tools), so each test pins a wire-shape guarantee rather than a model's
behavior. Grouped by the issue each guards:

- served wire emits TEXT_MESSAGE_* for a finished message, at parity with the
  in-process wire (gh #140; the served-vs-in-process check #146 asked for);
- chunk wire: ``error`` is terminal (gh #161), tool-error status (gh #168), tool-call
  ids (gh #149); no ``extraction`` for a failed tool on either wire (gh #177);
- ``TurnResult`` parity across collectors (gh #149, #154);
- the terminal ``complete`` frame names the outcome (gh #152);
- message boundaries (langstage-vscode #108), partial content before an error
  (langstage-vscode #105), in-message order (langstage-cli #119);
- ``tool_end.duration_ms`` (langstage #160);
- isolated one-shots (gh #163);
- a stdlib ``typing.TypedDict`` state (langstage #166).
"""
import json
import typing

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from langstage_core import GenericToolExtractor  # noqa: E402
from langstage_core.agui import (  # noqa: E402
    build_agent,
    build_app,
    collect_chunk_frames,
    collect_event_frames,
    iter_chunk_frames,
    iter_event_frames,
    run_turn,
)
from langstage_core.demo.tools import create_tool_demo_agent  # noqa: E402


async def _collect(aiter):
    return [f async for f in aiter]


# ── graphs ───────────────────────────────────────────────────────────


def _finished_graph(text="Hello from a finished AIMessage."):
    def respond(state):
        return {"messages": [AIMessage(content=text)]}

    b = StateGraph(MessagesState)
    b.add_node("respond", respond)
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    return b.compile()


def _two_node_graph(fail_second=False):
    def plan(state):
        return {"messages": [AIMessage(content="PLAN: two parts.")]}

    def answer(state):
        if fail_second:
            raise RuntimeError("tool call failed")
        return {"messages": [AIMessage(content="ANSWER: the sky is blue.")]}

    b = StateGraph(MessagesState)
    b.add_node("plan", plan)
    b.add_node("answer", answer)
    b.add_edge(START, "plan")
    b.add_edge("plan", "answer")
    b.add_edge("answer", END)
    return b.compile()


def _custom_tool_graph():
    """The langstage-cli #119 shape: a finished AIMessage with text AND a tool call,
    a hand-written tool node, then a final reply — nothing token-streamed."""
    def call_tool(state):
        return {"messages": [AIMessage(content="Let me check the weather.", tool_calls=[
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}])]}

    def run_tool(state):
        return {"messages": [ToolMessage(content="Sunny, 24C", tool_call_id="call_1")]}

    def final(state):
        return {"messages": [AIMessage(content="The weather in Paris is sunny, 24C.")]}

    b = StateGraph(MessagesState)
    for n, f in [("call_tool", call_tool), ("run_tool", run_tool), ("final", final)]:
        b.add_node(n, f)
    b.add_edge(START, "call_tool")
    b.add_edge("call_tool", "run_tool")
    b.add_edge("run_tool", "final")
    b.add_edge("final", END)
    return b.compile()


@tool
def lookup(city: str) -> str:
    """Look up the weather for a city (fails for 'fail')."""
    if city == "fail":
        raise ValueError("kaboom")
    return json.dumps({"ok": True, "city": city})


def _toolnode_graph():
    """A real ToolNode agent whose planner returns finished AIMessages; the tool call's
    city is the user's message, so one graph covers the success and failure paths."""
    def planner(state):
        msgs = state["messages"]
        if getattr(msgs[-1], "type", None) == "tool":
            return {"messages": [AIMessage(content="Done.")]}
        return {"messages": [AIMessage(content="Let me check.", tool_calls=[
            {"name": "lookup", "args": {"city": msgs[-1].content}, "id": "call_1"}])]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    b = StateGraph(MessagesState)
    b.add_node("planner", planner)
    b.add_node("tools", ToolNode([lookup], handle_tool_errors=True))
    b.add_edge(START, "planner")
    b.add_conditional_edges("planner", route, {"tools": "tools", END: END})
    b.add_edge("tools", "planner")
    return b.compile()


# ── served vs in-process (gh #140) ───────────────────────────────────


def _served_events(graph, text="hi"):
    client = TestClient(build_app(graph))
    body = {
        "threadId": "t-served", "runId": "r-served",
        "messages": [{"id": "u1", "role": "user", "content": text}],
        "tools": [], "context": [], "state": {}, "forwardedProps": {},
    }
    resp = client.post("/", json=body, headers={"Accept": "text/event-stream"})
    assert resp.status_code == 200
    events = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_served_wire_streams_a_finished_message():
    events = _served_events(_finished_graph())
    types = [e["type"] for e in events]
    # The finished reply now streams as TEXT_MESSAGE_* BEFORE the closing snapshot,
    # instead of appearing only inside MESSAGES_SNAPSHOT (gh #140).
    assert "TEXT_MESSAGE_CONTENT" in types
    assert types.index("TEXT_MESSAGE_START") < types.index("TEXT_MESSAGE_CONTENT") \
        < types.index("TEXT_MESSAGE_END") < types.index("MESSAGES_SNAPSHOT")
    served_text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert served_text == "Hello from a finished AIMessage."
    # The synthesized message id is the snapshot's own id, so a client that rebuilds
    # messages from events and then applies the snapshot sees ONE message, not two.
    start_id = next(e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_START")
    snapshot = next(e for e in events if e["type"] == "MESSAGES_SNAPSHOT")
    assert start_id in [m["id"] for m in snapshot["messages"] if m["role"] == "assistant"]


@pytest.mark.parametrize("graph_factory", [_finished_graph, _two_node_graph, _custom_tool_graph])
async def test_served_and_in_process_wires_agree(graph_factory):
    """ASGI-level parity: the same graph's served TEXT_MESSAGE_* text and tool calls
    match what the in-process event wire streams."""
    events = _served_events(graph_factory())
    frames = await _collect(iter_event_frames(graph_factory(), "hi", "t-inproc"))

    served_text = [e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
    inproc_text = [f["content"] for f in frames if f["type"] == "content"]
    assert served_text == inproc_text and served_text

    served_calls = [(e["toolCallId"], e["toolCallName"]) for e in events
                    if e["type"] == "TOOL_CALL_START"]
    inproc_calls = [(f["id"], f["name"]) for f in frames if f["type"] == "tool_start"]
    assert served_calls == inproc_calls
    served_results = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_RESULT"]
    inproc_results = [f["id"] for f in frames if f["type"] == "tool_end"]
    assert served_results == inproc_results


def test_served_toolnode_turn_announces_each_tool_call_once():
    # The call is synthesized when the planner step finishes; the ToolNode's own
    # TOOL_CALL_START for the same id must be suppressed, not repeated.
    events = _served_events(_toolnode_graph(), "Paris")
    starts = [e["toolCallId"] for e in events if e["type"] == "TOOL_CALL_START"]
    assert starts == ["call_1"]
    assert [e["type"] for e in events].count("TOOL_CALL_RESULT") == 1


# ── chunk wire: terminal error, tool status, ids (gh #161, #168, #149, #177) ─────


class RunErrorEvent:  # what the ag-ui adapter emits (then returns) on an upstream error
    def __init__(self, message):
        self.message = message


class _RunErrorAgent:
    async def run(self, run_input):
        yield RunErrorEvent(message="upstream boom")


async def test_chunk_wire_error_is_terminal():
    event = await _collect(iter_event_frames(_RunErrorAgent(), "hi", "e"))
    chunk = await _collect(iter_chunk_frames(_RunErrorAgent(), "hi", "c"))
    assert [f["type"] for f in event] == ["error"]
    assert [f["status"] for f in chunk] == ["error"]  # no trailing `complete` (gh #161)


async def test_failed_tool_status_on_both_wires_and_no_extraction():
    ex = [GenericToolExtractor()]
    ev = await _collect(iter_event_frames(_toolnode_graph(), "fail", "e1", extractors=ex))
    ch = await _collect(iter_chunk_frames(_toolnode_graph(), "fail", "c1", extractors=ex))

    tool_end = next(f for f in ev if f["type"] == "tool_end")
    assert tool_end["status"] == "error" and tool_end["id"] == "call_1"
    tool_result = next(f for f in ch if "tool_result" in f)
    assert tool_result["tool_status"] == "error"  # gh #168
    assert tool_result["id"] == "call_1" and tool_result["name"] == "lookup"
    assert isinstance(tool_result["tool_result"], str)  # the payload key is unchanged
    # No success-shaped extraction card built from the error string (gh #177).
    assert not any(f["type"] == "extraction" for f in ev)
    assert not any("extraction" in f for f in ch)


async def test_successful_tool_still_extracts_on_both_wires():
    ex = [GenericToolExtractor()]
    ev = await _collect(iter_event_frames(_toolnode_graph(), "Paris", "e2", extractors=ex))
    ch = await _collect(iter_chunk_frames(_toolnode_graph(), "Paris", "c2", extractors=ex))
    assert next(f for f in ev if f["type"] == "tool_end")["status"] == "success"
    assert next(f for f in ch if "tool_result" in f)["tool_status"] == "success"
    assert any(f["type"] == "extraction" for f in ev)
    assert any("extraction" in f for f in ch)


async def test_chunk_tool_calls_carry_the_id():
    ch = await _collect(iter_chunk_frames(_custom_tool_graph(), "hi", "c3"))
    calls = [tc for f in ch for tc in f.get("tool_calls", [])]
    assert calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}]


# ── TurnResult parity (gh #149, #154) ────────────────────────────────


async def test_collectors_agree_on_tool_calls():
    ev = await collect_event_frames(_custom_tool_graph(), "hi", "e")
    ch = await collect_chunk_frames(_custom_tool_graph(), "hi", "c")
    assert ev.tool_calls == ch.tool_calls
    assert ev.tool_calls[0]["id"] == "call_1"
    assert ev.text == ch.text


async def test_collectors_agree_on_interrupt():
    ev = await collect_event_frames(build_agent(create_tool_demo_agent()), "ask me", "e")
    ch = await collect_chunk_frames(build_agent(create_tool_demo_agent()), "ask me", "c")
    assert ev.outcome == ch.outcome == "interrupted"
    assert ev.interrupt == ch.interrupt
    assert "type" not in ev.interrupt and ev.interrupt["action_requests"]


# ── terminal frame names the outcome (gh #152) ───────────────────────


async def test_complete_frame_distinguishes_an_interrupt():
    agent = build_agent(create_tool_demo_agent())
    ev = await _collect(iter_event_frames(agent, "ask me", "s1"))
    ch = await _collect(iter_chunk_frames(agent, "ask me", "s2"))
    assert ev[-1] == {"type": "complete", "outcome": "interrupted"}
    assert ch[-1] == {"status": "complete", "outcome": "interrupted"}

    ev = await _collect(iter_event_frames(agent, "hello", "s3"))
    ch = await _collect(iter_chunk_frames(agent, "hello", "s4"))
    assert ev[-1] == {"type": "complete", "outcome": "complete"}
    assert ch[-1] == {"status": "complete", "outcome": "complete"}


# ── message boundaries / partial content / in-message order ─────────


async def test_two_nodes_messages_carry_distinct_message_ids():
    ev = await _collect(iter_event_frames(_two_node_graph(), "hi", "b1"))
    ch = await _collect(iter_chunk_frames(_two_node_graph(), "hi", "b2"))
    content = [f for f in ev if f["type"] == "content"]
    chunks = [f for f in ch if "chunk" in f]
    assert [f["node"] for f in content] == ["plan", "answer"]
    # A message_id change is the boundary a renderer joins with a paragraph break
    # (langstage-vscode #108).
    assert len({f["message_id"] for f in content}) == 2
    assert len({f["message_id"] for f in chunks}) == 2


async def test_earlier_node_content_survives_a_later_node_error():
    ev = await _collect(iter_event_frames(_two_node_graph(fail_second=True), "hi", "p1"))
    ch = await _collect(iter_chunk_frames(_two_node_graph(fail_second=True), "hi", "p2"))
    assert [f["type"] for f in ev] == ["content", "error"]  # langstage-vscode #105
    assert ev[0]["content"] == "PLAN: two parts."
    assert [f["status"] for f in ch] == ["streaming", "error"]
    assert ch[0]["chunk"] == "PLAN: two parts."


async def test_message_text_precedes_its_own_tool_call():
    ev = await _collect(iter_event_frames(_custom_tool_graph(), "hi", "o1"))
    ch = await _collect(iter_chunk_frames(_custom_tool_graph(), "hi", "o2"))
    assert [f["type"] for f in ev] == ["content", "tool_start", "tool_end", "content", "complete"]
    assert ev[0]["content"] == "Let me check the weather."  # langstage-cli #119
    kinds = [next(k for k in ("chunk", "tool_calls", "tool_result") if k in f)
             for f in ch if f["status"] == "streaming"]
    assert kinds == ["chunk", "tool_calls", "tool_result", "chunk"]


async def test_toolnode_turn_text_before_tool_and_before_result():
    ev = await _collect(iter_event_frames(_toolnode_graph(), "Paris", "o3"))
    assert [f["type"] for f in ev] == ["content", "tool_start", "tool_end", "content", "complete"]


async def test_resume_does_not_replay_pre_interrupt_messages():
    agent = build_agent(create_tool_demo_agent())
    first = await _collect(iter_event_frames(agent, "ask me", "r1"))
    first_ids = {f["message_id"] for f in first if f["type"] == "content"}
    resumed = await _collect(iter_event_frames(
        agent, "", "r1", resume={"decisions": [{"type": "approve"}]}))
    resumed_ids = [f["message_id"] for f in resumed if f["type"] == "content"]
    assert resumed_ids and not (set(resumed_ids) & first_ids)
    assert len(resumed_ids) == len(set(resumed_ids))  # nothing emitted twice


# ── tool duration (langstage #160) ───────────────────────────────────


async def test_tool_end_carries_duration_for_a_real_tool():
    ev = await _collect(iter_event_frames(_toolnode_graph(), "Paris", "d1"))
    ch = await _collect(iter_chunk_frames(_toolnode_graph(), "Paris", "d2"))
    tool_end = next(f for f in ev if f["type"] == "tool_end")
    assert isinstance(tool_end["duration_ms"], int) and tool_end["duration_ms"] >= 0
    assert isinstance(next(f for f in ch if "tool_result" in f)["duration_ms"], int)


async def test_hand_written_tool_node_reports_no_duration():
    # No tool runtime -> no timing events -> an honest None, not a made-up number.
    ev = await _collect(iter_event_frames(_custom_tool_graph(), "hi", "d3"))
    assert next(f for f in ev if f["type"] == "tool_end")["duration_ms"] is None


# ── isolated one-shots (gh #163) ─────────────────────────────────────


def _history_graph():
    def node(state):
        return {"messages": [AIMessage(content=f"history_len={len(state['messages'])}")]}

    b = StateGraph(MessagesState)
    b.add_node("node", node)
    b.add_edge(START, "node")
    b.add_edge("node", END)
    return b.compile()


def test_run_turn_on_one_graph_is_isolated_and_does_not_mutate_it():
    g = _history_graph()
    texts = [run_turn(g, p).text for p in ("p1", "p2", "p3")]
    assert texts == ["history_len=1"] * 3
    assert g.checkpointer is None  # the caller's graph is left untouched


async def test_collectors_on_a_bare_graph_do_not_share_state():
    g = _history_graph()
    a = await collect_event_frames(g, "p1", "same-thread")
    b = await collect_event_frames(g, "p2", "same-thread")
    assert a.text == b.text == "history_len=1"
    assert g.checkpointer is None


def test_prebuilt_agent_with_shared_thread_keeps_state():
    agent = build_agent(_history_graph())
    assert run_turn(agent, "p1", thread_id="t").text == "history_len=1"
    assert run_turn(agent, "p2", thread_id="t").text == "history_len=3"


def test_graph_with_its_own_checkpointer_is_used_as_is():
    b = StateGraph(MessagesState)
    b.add_node("n", lambda s: {"messages": [AIMessage(content=f"n={len(s['messages'])}")]})
    b.add_edge(START, "n")
    b.add_edge("n", END)
    saver = InMemorySaver()
    g = b.compile(checkpointer=saver)
    assert build_agent(g).graph is g
    assert run_turn(g, "p1", thread_id="k").text == "n=1"
    assert run_turn(g, "p2", thread_id="k").text == "n=3"


# ── stdlib TypedDict state (langstage #166) ──────────────────────────


class _StdlibState(typing.TypedDict):
    messages: typing.Annotated[list, add_messages]


def _stdlib_typeddict_graph():
    def respond(state):
        return {"messages": [AIMessage(content="echo: " + state["messages"][-1].content)]}

    b = StateGraph(_StdlibState)
    b.add_node("respond", respond)
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    return b.compile()


async def test_stdlib_typeddict_state_runs_on_both_wires():
    ev = await collect_event_frames(_stdlib_typeddict_graph(), "hi", "td1")
    ch = await collect_chunk_frames(_stdlib_typeddict_graph(), "hi", "td2")
    assert ev.outcome == ch.outcome == "complete", (ev.error, ch.error)
    assert ev.text == ch.text == "echo: hi"


def test_stdlib_typeddict_state_serves():
    events = _served_events(_stdlib_typeddict_graph())
    assert not any(e["type"] == "RUN_ERROR" for e in events)
    assert "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT") == "echo: hi"
