"""Node attribution does not depend on checkpointer timing (gh #188, langstage #173).

``_TurnStream`` surfaces a finished (non-token-streamed) message by reading the thread's
checkpoint after each step. With an async/durable checkpointer (the web app's
``AsyncSqliteSaver``) and LangGraph's default ``durability="async"``, that read can run
before the step's write lands, so the message surfaces at a *later* flush. The frame's
``node`` must still name the node that produced the message, not whichever step is
current when the late read finally sees it.

``_SlowSaver`` makes the race deterministic: every checkpoint write lands 50 ms late.
The ``AsyncSqliteSaver`` test drives the real thing the web app uses.
"""
import asyncio

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames  # noqa: E402


class _SlowSaver(InMemorySaver):
    """An async checkpointer whose writes land late, like a durable (SQLite) saver."""

    async def aput(self, config, checkpoint, metadata, new_versions):
        await asyncio.sleep(0.05)
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        await asyncio.sleep(0.05)
        return self.put_writes(config, writes, task_id, task_path)


def _react3_builder():
    """langstage #173's graph: agent (narration + tool call) -> tools -> final."""

    def agent(state):
        return {"messages": [AIMessage(content="Let me check.", tool_calls=[
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}])]}

    def tools(state):
        return {"messages": [ToolMessage(content="Sunny, 21C", tool_call_id="call_1",
                                         name="get_weather")]}

    def final(state):
        return {"messages": [AIMessage(content="The weather in Paris is sunny, 21C.")]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent)
    b.add_node("tools", tools)
    b.add_node("final", final)
    b.add_edge(START, "agent")
    b.add_edge("agent", "tools")
    b.add_edge("tools", "final")
    b.add_edge("final", END)
    return b


def _assert_event_nodes(frames):
    starts = [f for f in frames if f["type"] == "tool_start"]
    assert [f["node"] for f in starts] == ["agent"], frames
    content = {f["content"]: f["node"] for f in frames if f["type"] == "content"}
    assert content == {"Let me check.": "agent",
                       "The weather in Paris is sunny, 21C.": "final"}, frames


async def test_event_wire_nodes_survive_a_late_checkpoint_write():
    agent = build_agent(_react3_builder().compile(checkpointer=_SlowSaver()))
    frames = [f async for f in iter_event_frames(agent, "weather?", "t1")]
    _assert_event_nodes(frames)


async def test_chunk_wire_nodes_survive_a_late_checkpoint_write():
    agent = build_agent(_react3_builder().compile(checkpointer=_SlowSaver()))
    frames = [f async for f in iter_chunk_frames(agent, "weather?", "t1")]
    text = {f["chunk"]: f["node"] for f in frames if "chunk" in f}
    assert text == {"Let me check.": "agent",
                    "The weather in Paris is sunny, 21C.": "final"}, frames


async def test_async_sqlite_saver_attributes_the_tool_call_to_its_node(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio",
                        reason="needs langgraph-checkpoint-sqlite")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        agent = build_agent(_react3_builder().compile(checkpointer=saver))
        # The race is timing-dependent against a real saver; several turns make a
        # regression all but certain to show up.
        for i in range(5):
            frames = [f async for f in iter_event_frames(agent, "weather?", f"t{i}")]
            _assert_event_nodes(frames)


async def test_in_memory_saver_attribution_is_unchanged():
    agent = build_agent(_react3_builder().compile())
    frames = [f async for f in iter_event_frames(agent, "weather?", "t1")]
    _assert_event_nodes(frames)
