"""Answers to a legacy or custom interrupt aren't refused by the frame checks (gh #257).

Every ``interrupt`` frame carries ``action_requests``, because the wire normalizes
every interrupt into that shape. ``decision_problems(resume, frame)`` therefore ran
the HumanInTheLoopMiddleware checks on a legacy ``HumanInterrupt`` or a custom
``interrupt(...)`` frame, and ``TaskRunner.resume`` (which checks the stored frame)
refused answers the graph accepts. Frames now say ``hitl: True/False``.
"""
import asyncio

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langstage_core import InvalidDecisionError, SessionAdapter, decision_problems  # noqa: E402
from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames  # noqa: E402
from langstage_core.tasks import (  # noqa: E402
    DONE,
    REVIEW_NEEDED,
    TERMINAL_STATES,
    InMemoryTaskStore,
    TaskRunner,
)

LEGACY_ANSWER = [{"type": "response", "args": "Bob"}]


def _legacy_graph():
    def ask(state):
        answer = interrupt(
            [
                {
                    "action_request": {"action": "ask_name", "args": {}},
                    "config": {"allow_respond": True},
                    "description": "What should I call you?",
                }
            ]
        )
        return {"messages": [AIMessage(content=f"Nice to meet you, {answer[0]['args']}.")]}

    g = StateGraph(MessagesState)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile()


def _custom_graph():
    def ask(state):
        answer = interrupt({"question": "Proceed?"})
        return {"messages": [AIMessage(content=f"got {answer!r}")]}

    g = StateGraph(MessagesState)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile()


def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


class _Model(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="Done.")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "delete_file", "args": {"path": "a.txt"}, "id": "c1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _hitl_graph():
    return create_agent(
        _Model(),
        tools=[delete_file],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"delete_file": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
    )


async def _interrupt_frame(graph, thread="s1"):
    agent = build_agent(graph)
    frame = None
    async for f in iter_event_frames(agent, "hi", thread):
        if f["type"] == "interrupt":
            frame = f
    return agent, frame


async def _wait(runner, tid, states):
    task = None
    for _ in range(300):
        task = await runner.store.get(tid)
        if task["state"] in states:
            return task
        await asyncio.sleep(0.02)
    return task


def test_frames_flag_their_source():
    async def main():
        _, legacy = await _interrupt_frame(_legacy_graph())
        _, custom = await _interrupt_frame(_custom_graph())
        _, hitl = await _interrupt_frame(_hitl_graph())
        return legacy, custom, hitl

    legacy, custom, hitl = asyncio.run(main())
    assert legacy["hitl"] is False
    assert custom["hitl"] is False
    assert hitl["hitl"] is True


def test_chunk_wire_carries_the_flag():
    async def main():
        agent = build_agent(_custom_graph())
        out = None
        async for c in iter_chunk_frames(agent, "hi", "c1"):
            if c.get("status") == "interrupt":
                out = c["interrupt"]
        return out

    data = asyncio.run(main())
    assert data["hitl"] is False


def test_decision_problems_skips_legacy_and_custom_frames():
    async def main():
        _, legacy = await _interrupt_frame(_legacy_graph())
        _, custom = await _interrupt_frame(_custom_graph())
        return legacy, custom

    legacy, custom = asyncio.run(main())
    assert decision_problems({"decisions": LEGACY_ANSWER}, legacy) == []
    assert decision_problems({"decisions": [{"type": "edit"}]}, custom) == []
    assert decision_problems({"decisions": [{"a": 1}, {"b": 2}]}, custom) == []


def test_decision_problems_still_checks_middleware_frames():
    async def main():
        return (await _interrupt_frame(_hitl_graph()))[1]

    frame = asyncio.run(main())
    assert decision_problems({"decisions": [{"type": "edit"}]}, frame)
    assert decision_problems({"decisions": [{"type": "approve"}]}, frame) == []


def test_frame_without_flag_keeps_shape_test():
    # A frame from an older core (no "hitl" key) is still treated as a request.
    old = {"action_requests": [{"name": "x", "args": {}}], "review_configs": [],
           "allowed_decisions": ["approve"]}
    assert decision_problems({"decisions": [{"type": "edit"}]}, old)


def test_task_runner_accepts_an_answer_to_a_legacy_interrupt():
    # The verdict now matches the wire: no InvalidDecisionError for the graph's own
    # vocabulary. (TaskRunner sends the {"decisions": [...]} envelope, as documented.)
    async def main():
        runner = TaskRunner(SessionAdapter(graph=_legacy_graph()), InMemoryTaskStore())
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="hi")
            task = await _wait(runner, tid, {REVIEW_NEEDED} | TERMINAL_STATES)
            assert task["state"] == REVIEW_NEEDED
            assert task["interrupt"]["hitl"] is False
            return await runner.resume(tid, LEGACY_ANSWER)
        finally:
            await runner.shutdown()

    assert asyncio.run(main()) is True


def test_task_runner_resumes_a_custom_interrupt():
    async def main():
        runner = TaskRunner(SessionAdapter(graph=_custom_graph()), InMemoryTaskStore())
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="hi")
            task = await _wait(runner, tid, {REVIEW_NEEDED} | TERMINAL_STATES)
            assert task["state"] == REVIEW_NEEDED
            assert await runner.resume(tid, [{"type": "edit"}])
            return await _wait(runner, tid, TERMINAL_STATES)
        finally:
            await runner.shutdown()

    task = asyncio.run(main())
    assert task["state"] == DONE
    assert "edit" in (task.get("result") or "")


def test_task_runner_still_refuses_bad_middleware_answer():
    async def main():
        runner = TaskRunner(SessionAdapter(graph=_hitl_graph()), InMemoryTaskStore())
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="delete a.txt")
            task = await _wait(runner, tid, {REVIEW_NEEDED} | TERMINAL_STATES)
            assert task["state"] == REVIEW_NEEDED
            with pytest.raises(InvalidDecisionError):
                await runner.resume(tid, [{"type": "edit"}])
        finally:
            await runner.shutdown()

    asyncio.run(main())
