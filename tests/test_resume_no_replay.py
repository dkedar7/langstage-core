"""A resumed turn doesn't re-send what the turn said before it paused.

A turn that pauses on an interrupt has already streamed everything before the pause
(here, "Checking the order." and the refund tool call). When it resumes, the end-of-turn
snapshot walk in both ``iter_*`` mappings used to emit those messages again: it walked
every message after the last user message, and on a resume that is still the original
prompt. So every surface showed the pre-pause text and tool calls a second time, after
the resumed turn's own reply (the web Board's task detail, the CLI, the chat).
"""

import json

import pytest

pytest.importorskip("ag_ui_langgraph")

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames

APPROVE = {"decisions": [{"type": "approve"}]}


def _refund_graph():
    """Narrate + call a tool, pause in the tool for approval, then answer."""

    def agent(state):
        if isinstance(state["messages"][-1], ToolMessage):
            return {"messages": [AIMessage(content="Refund issued.")]}
        return {"messages": [AIMessage(
            content="Checking the order.",
            tool_calls=[{"name": "refund", "args": {"amount": 189}, "id": "call_1"}],
        )]}

    def tools(state):
        answer = interrupt({"action_requests": [{"name": "refund", "args": {"amount": 189}}]})
        return {"messages": [ToolMessage(content=f"refunded: {answer}", tool_call_id="call_1", name="refund")]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_node("tools", tools)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    b.add_edge("tools", "agent")
    return b.compile(checkpointer=InMemorySaver())


async def _frames(it):
    return [f async for f in it]


async def test_event_wire_resume_sends_only_the_resumed_part():
    agent = build_agent(_refund_graph())
    first = await _frames(iter_event_frames(agent, "refund NW-1041", "t-event"))
    assert any(f.get("type") == "interrupt" for f in first)
    assert "Checking the order." in json.dumps(first)

    resumed = await _frames(iter_event_frames(agent, "", "t-event", resume=APPROVE))
    text = "".join(f.get("content", "") for f in resumed if f.get("type") == "content")
    assert text == "Refund issued."
    # The paused call's result arrives; the call itself was announced before the pause.
    assert [f["id"] for f in resumed if f.get("type") == "tool_end"] == ["call_1"]
    assert [f for f in resumed if f.get("type") == "tool_start"] == []


async def test_chunk_wire_resume_sends_only_the_resumed_part():
    agent = build_agent(_refund_graph())
    first = await _frames(iter_chunk_frames(agent, "refund NW-1041", "t-chunk"))
    assert "Checking the order." in json.dumps(first)

    resumed = await _frames(iter_chunk_frames(agent, "", "t-chunk", resume=APPROVE))
    text = "".join(f.get("chunk", "") for f in resumed if f.get("status") == "streaming")
    assert text == "Refund issued."
    assert "Checking the order." not in json.dumps(resumed)


async def test_next_turn_on_the_thread_still_sends_its_reply():
    # The filter keys on what was in the checkpoint before the run, so a new user
    # turn on the same thread (history before it) still sends its own messages.
    agent = build_agent(_refund_graph())
    await _frames(iter_event_frames(agent, "refund NW-1041", "t-next"))
    await _frames(iter_event_frames(agent, "", "t-next", resume=APPROVE))
    again = await _frames(iter_event_frames(agent, "and NW-1047?", "t-next"))
    # The graph starts another refund: its own narration once, nothing from turn one.
    text = "".join(f.get("content", "") for f in again if f.get("type") == "content")
    assert text == "Checking the order."
    assert "Refund issued." not in json.dumps(again)
