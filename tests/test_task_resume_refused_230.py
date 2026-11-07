"""A refused HITL decision on the task engine keeps the approval answerable (gh #230).

Since #227 a refused answer never reaches the graph, and the error says to answer the
approval again. ``TaskRunner.resume`` still flipped the task to ``ongoing`` with no
interrupt first, so the refused run recorded it as ``failed``: the corrected
``resume()`` then returned False and only ``retry()`` (a fresh turn) was left.
"""
import asyncio

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from langstage_core import InvalidDecisionError, SessionAdapter  # noqa: E402
from langstage_core.tasks import (  # noqa: E402
    DONE,
    REVIEW_NEEDED,
    TERMINAL_STATES,
    InMemoryTaskStore,
    TaskRunner,
)


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


def _graph():
    return create_agent(
        _Model(),
        tools=[delete_file],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"delete_file": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
    )


async def _wait(runner, tid, states):
    task = None
    for _ in range(200):
        task = await runner.store.get(tid)
        if task["state"] in states:
            return task
        await asyncio.sleep(0.02)
    return task


async def _paused_task():
    runner = TaskRunner(SessionAdapter(graph=_graph()), InMemoryTaskStore())
    await runner.start()
    tid = await runner.enqueue(title="cleanup", prompt="delete a.txt")
    task = await _wait(runner, tid, {REVIEW_NEEDED} | TERMINAL_STATES)
    assert task["state"] == REVIEW_NEEDED
    return runner, tid, task


@pytest.mark.parametrize(
    "decisions",
    [
        [{"type": "edit"}],  # not allowed for this tool
        [{"type": "approve"}, {"type": "approve"}],  # two answers for one action
        [{"type": "aprove"}],  # typo
    ],
)
def test_refused_decision_keeps_task_in_review(decisions):
    async def go():
        runner, tid, before = await _paused_task()
        try:
            with pytest.raises(InvalidDecisionError, match="answer the approval again"):
                await runner.resume(tid, decisions)
            after = await runner.store.get(tid)
            assert after["state"] == REVIEW_NEEDED
            assert after["interrupt"] == before["interrupt"]
            # The corrected answer the error asks for now works and finishes the task.
            assert await runner.resume(tid, [{"type": "approve"}]) is True
            done = await _wait(runner, tid, TERMINAL_STATES)
            assert done["state"] == DONE
            assert done["result"] == "Done."
        finally:
            await runner.shutdown()

    asyncio.run(go())


def test_envelope_is_checked_too():
    async def go():
        runner, tid, _ = await _paused_task()
        try:
            with pytest.raises(InvalidDecisionError):
                await runner.resume(tid, {"decisions": [{"type": "edit"}]})
            assert (await runner.store.get(tid))["state"] == REVIEW_NEEDED
        finally:
            await runner.shutdown()

    asyncio.run(go())


def test_refusal_by_the_graph_restores_the_approval():
    """When the stored interrupt can't be checked (not a HITL request), the graph's
    own refusal still puts the task back in review instead of failing it."""

    async def go():
        runner, tid, before = await _paused_task()
        try:
            opaque = {"type": "interrupt", "value": "legacy"}
            await runner.store.update(tid, interrupt=opaque)
            assert await runner.resume(tid, [{"type": "edit"}]) is True
            task = await _wait(runner, tid, {REVIEW_NEEDED} | TERMINAL_STATES)
            for _ in range(100):  # the resume flips to ongoing first
                if task["state"] == REVIEW_NEEDED and task.get("error"):
                    break
                await asyncio.sleep(0.02)
                task = await runner.store.get(tid)
            assert task["state"] == REVIEW_NEEDED
            assert task["interrupt"] == opaque
            assert task["error"].startswith("InvalidDecisionError:")
            assert await runner.resume(tid, [{"type": "approve"}]) is True
            done = await _wait(runner, tid, TERMINAL_STATES)
            assert done["state"] == DONE
            assert done.get("error") is None
        finally:
            await runner.shutdown()

    asyncio.run(go())
