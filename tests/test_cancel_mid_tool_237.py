"""A turn stopped mid-tool no longer leaves an unanswered tool call (gh #237).

Stopping a turn while a tool runs left the model's AIMessage (with tool_calls) in the
checkpoint with no ToolMessage, so the next turn on that thread sent the model a tool
call with no result, which providers reject. Before a new turn the shared run path now
answers each such call with a "cancelled" ToolMessage.
"""
from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("ag_ui_langgraph")

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langstage_core.agui import (  # noqa: E402
    CANCELLED_TOOL_RESULT,
    _answer_cancelled_tool_calls,
    _is_resume_run,
    build_agent,
    iter_chunk_frames,
    iter_event_frames,
)


def _strict_model(state):
    """A node that fails like a provider does on a tool call without a result."""
    msgs = state["messages"]
    answered = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] not in answered:
                    raise RuntimeError(f"provider 400: tool call {tc['id']} has no result")
    last = msgs[-1]
    if isinstance(last, HumanMessage) and "slow" in last.content:
        calls = [{"name": "slow", "args": {}, "id": f"call_{len(msgs)}_a"}]
        return {"messages": [AIMessage(content="", tool_calls=calls)]}
    return {"messages": [AIMessage(content=f"ok, saw {len(msgs)} messages")]}


async def _tools(state):
    ai = state["messages"][-1]
    await asyncio.sleep(30)
    return {"messages": [ToolMessage(content="slow done", tool_call_id=ai.tool_calls[0]["id"])]}


async def _slow_tools(state):
    await asyncio.sleep(30)
    return {}


def _route(state):
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def _graph(tools=_tools):
    g = StateGraph(MessagesState)
    g.add_node("model", _strict_model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", _route, ["tools", END])
    g.add_edge("tools", "model")
    return g.compile(checkpointer=InMemorySaver())


async def _cancel_mid_tool(agent, thread_id, prompt="do slow thing"):
    async def turn():
        async for _ in iter_chunk_frames(agent, prompt, thread_id):
            pass

    task = asyncio.create_task(turn())
    await asyncio.sleep(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _messages(graph, thread_id):
    st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return st.values["messages"]


def test_next_turn_after_a_cancel_mid_tool_succeeds():
    async def go():
        graph = _graph()
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        before = await _messages(graph, "t")
        assert isinstance(before[-1], AIMessage) and before[-1].tool_calls  # the dangling call

        frames = [c async for c in iter_chunk_frames(agent, "hello again", "t")]
        assert frames[-1]["status"] == "complete", frames
        assert not any(f["status"] == "error" for f in frames)
        # The synthetic result belongs to history, not to this turn's output.
        assert not any(f.get("tool_result") or f.get("status") == "tool_result" for f in frames)

        msgs = await _messages(graph, "t")
        assert [type(m).__name__ for m in msgs] == [
            "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage", "AIMessage"]
        tm = msgs[2]
        assert tm.content == CANCELLED_TOOL_RESULT
        assert tm.status == "error" and tm.tool_call_id == before[-1].tool_calls[0]["id"]

    asyncio.run(go())


def test_event_wire_gets_the_same_repair():
    """vscode's sidecar and the web SessionAdapter run on iter_event_frames."""
    async def go():
        graph = _graph()
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        frames = [f async for f in iter_event_frames(agent, "hello again", "t")]
        assert not any(f.get("type") == "error" for f in frames), frames
        assert frames[-1].get("type") == "complete"

    asyncio.run(go())


def test_only_the_unanswered_call_of_a_parallel_pair_is_answered():
    """create_agent runs parallel calls as separate tasks, so one result can be committed
    while the other tool is still running when the turn is stopped."""
    async def go():
        graph = _graph()
        agent = build_agent(graph)
        config = {"configurable": {"thread_id": "t"}}
        ai = AIMessage(content="", id="ai-1", tool_calls=[
            {"name": "slow", "args": {}, "id": "call_slow"},
            {"name": "fast", "args": {}, "id": "call_fast"},
        ])
        await graph.aupdate_state(config, {"messages": [
            HumanMessage(content="slow two tools", id="h-1"), ai,
            ToolMessage(content="fast done", tool_call_id="call_fast", id="tm-1"),
        ]}, as_node="model")
        assert await _answer_cancelled_tool_calls(agent, "t") == ["call_slow"]
        msgs = await _messages(graph, "t")
        results = {m.tool_call_id: m.content for m in msgs if isinstance(m, ToolMessage)}
        assert results == {"call_slow": CANCELLED_TOOL_RESULT, "call_fast": "fast done"}
        frames = [c async for c in iter_chunk_frames(agent, "hello again", "t")]
        assert frames[-1]["status"] == "complete", frames

    asyncio.run(go())


def test_repair_is_idempotent():
    async def go():
        graph = _graph()
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        assert len(await _answer_cancelled_tool_calls(agent, "t")) == 1
        assert await _answer_cancelled_tool_calls(agent, "t") == []
        msgs = await _messages(graph, "t")
        assert sum(isinstance(m, ToolMessage) for m in msgs) == 1

    asyncio.run(go())


def test_a_thread_paused_on_an_interrupt_is_left_alone():
    """Unanswered calls awaiting a human decision are not "cancelled"."""
    def ask(state):
        interrupt({"question": "run it?"})
        return {}

    async def go():
        graph = _graph(tools=ask)
        agent = build_agent(graph)
        frames = [c async for c in iter_chunk_frames(agent, "do slow thing", "t")]
        assert any(f["status"] == "interrupt" for f in frames)
        before = await _messages(graph, "t")
        assert await _answer_cancelled_tool_calls(agent, "t") == []
        assert await _messages(graph, "t") == before

    asyncio.run(go())


def test_a_messages_channel_without_add_messages_is_not_written():
    class PlainState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]

    async def go():
        g = StateGraph(PlainState)
        g.add_node("model", _strict_model)
        g.add_node("tools", _slow_tools)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", _route, ["tools", END])
        g.add_edge("tools", "model")
        graph = g.compile(checkpointer=InMemorySaver())
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        before = await _messages(graph, "t")
        assert await _answer_cancelled_tool_calls(agent, "t") == []
        assert await _messages(graph, "t") == before

    asyncio.run(go())


def test_no_checkpointer_or_no_graph_is_a_no_op():
    async def go():
        class Double:
            pass

        assert await _answer_cancelled_tool_calls(Double(), "t") == []
        assert await _answer_cancelled_tool_calls(build_agent(_graph()), None) == []

    asyncio.run(go())


def test_a_resume_is_never_repaired():
    from types import SimpleNamespace

    assert _is_resume_run(SimpleNamespace(resume=[object()], forwarded_props={}))
    assert _is_resume_run(SimpleNamespace(resume=None, forwarded_props={"command": {"resume": "y"}}))
    assert not _is_resume_run(SimpleNamespace(resume=None, forwarded_props={}))
    assert not _is_resume_run(SimpleNamespace(resume=[], forwarded_props={"command": {}}))


def test_patch_tool_calls_middleware_does_not_answer_twice():
    """deepagents' PatchToolCallsMiddleware finds nothing left to patch."""
    pytest.importorskip("deepagents")
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

    async def go():
        graph = _graph()
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        await _answer_cancelled_tool_calls(agent, "t")
        msgs = await _messages(graph, "t")
        update = PatchToolCallsMiddleware().before_agent({"messages": msgs}, None)
        patched = (update or {}).get("messages")
        # Either no update, or the same history with no extra ToolMessage.
        if patched is not None:
            items = getattr(patched, "value", patched)
            tool_msgs = [m for m in items if isinstance(m, ToolMessage)]
            assert len(tool_msgs) == 1

    asyncio.run(go())


def test_create_agent_cancelled_mid_tool_then_next_turn_succeeds():
    """The real-world shape: langchain's create_agent, a tool cancelled while running."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool

    class StrictFake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
            for m in messages:
                if isinstance(m, AIMessage):
                    for tc in m.tool_calls:
                        assert tc["id"] in answered, f"provider 400: {tc['id']} has no result"
            return super()._generate(messages, *args, **kwargs)

    @tool
    async def slow() -> str:
        """Takes a while."""
        await asyncio.sleep(30)
        return "slow done"

    model = StrictFake(responses=[
        AIMessage(content="", tool_calls=[{"name": "slow", "args": {}, "id": "call_slow"}]),
        AIMessage(content="ok, after the cancel"),
    ])

    async def go():
        graph = create_agent(model, tools=[slow], checkpointer=InMemorySaver())
        agent = build_agent(graph)
        await _cancel_mid_tool(agent, "t")
        frames = [c async for c in iter_chunk_frames(agent, "hello again", "t")]
        assert frames[-1]["status"] == "complete", frames
        assert any(f.get("chunk") == "ok, after the cancel" for f in frames), frames
        msgs = await _messages(graph, "t")
        tm = next(m for m in msgs if isinstance(m, ToolMessage))
        assert tm.tool_call_id == "call_slow" and tm.content == CANCELLED_TOOL_RESULT

    asyncio.run(go())
