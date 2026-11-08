"""A task's event log records the human's side of each later turn (gh #233).

``followup()`` and ``resume()`` used to start a run on the task's thread without
recording what the human said, so a replay of ``get_events`` (the web Board's task
view) showed agent replies with no visible prompt. Each accepted human turn is now
appended ahead of the run's frames: a follow-up as ``{"type": "user", ...}`` and a
HITL answer as ``{"type": "resume", "decisions": [...]}``.
"""
import pytest

from langstage_core import InvalidDecisionError
from langstage_core.adapters.session import SessionAdapter
from langstage_core.tasks import DONE, REVIEW_NEEDED, InMemoryTaskStore, TaskRunner

from tests.test_tasks import _wait_state, echo_graph, interrupt_graph


def _runner(graph):
    store = InMemoryTaskStore()
    adapter = SessionAdapter(graph=graph, stream_mode="updates")
    return TaskRunner(adapter, store, concurrency=1, poll_interval=0.05), store


async def test_followup_is_recorded_before_its_reply():
    runner, store = _runner(echo_graph())
    await runner.start()
    try:
        tid = await runner.enqueue(title="t", prompt="first question")
        await _wait_state(store, tid, DONE)
        first_run = len(await store.get_events(tid))
        assert await runner.followup(tid, "  second question  ") is True
        await _wait_state(store, tid, DONE)
        events = await store.get_events(tid)
        # The first turn is untouched: no user frame for task["prompt"].
        assert all(e["type"] != "user" for e in events[:first_run])
        assert events[first_run] == {"type": "user", "role": "user", "content": "second question"}
        rest = [e["type"] for e in events[first_run + 1:]]
        assert "content" in rest and rest[-1] == "complete"
        assert (await store.get(tid))["prompt"] == "first question"
    finally:
        await runner.shutdown()


async def test_each_followup_gets_its_own_frame_in_order():
    runner, store = _runner(echo_graph())
    await runner.start()
    try:
        tid = await runner.enqueue(title="t", prompt="q1")
        await _wait_state(store, tid, DONE)
        for msg in ("q2", "q3"):
            assert await runner.followup(tid, msg) is True
            await _wait_state(store, tid, DONE)
        events = await store.get_events(tid)
        assert [e["content"] for e in events if e["type"] == "user"] == ["q2", "q3"]
        # Each user frame is followed by that turn's reply, ending in a complete.
        idx = [i for i, e in enumerate(events) if e["type"] == "user"]
        assert events[idx[1] - 1]["type"] == "complete"
    finally:
        await runner.shutdown()


async def test_rejected_followup_records_nothing():
    runner, store = _runner(echo_graph())
    await runner.start()
    try:
        tid = await runner.enqueue(title="t", prompt="hi")
        await _wait_state(store, tid, DONE)
        before = await store.get_events(tid)
        assert await runner.followup(tid, "   ") is False
        assert await runner.followup("missing", "hello") is False
        assert await store.get_events(tid) == before
    finally:
        await runner.shutdown()


async def test_resume_answer_is_recorded_before_the_resumed_run():
    runner, store = _runner(interrupt_graph())
    await runner.start()
    try:
        tid = await runner.enqueue(title="hitl", prompt="run it")
        await _wait_state(store, tid, REVIEW_NEEDED)
        paused = len(await store.get_events(tid))
        decisions = [{"type": "approve"}]
        assert await runner.resume(tid, decisions) is True
        await _wait_state(store, tid, DONE)
        events = await store.get_events(tid)
        assert events[paused] == {"type": "resume", "decisions": decisions}
        assert events[-1]["type"] == "complete"
    finally:
        await runner.shutdown()


async def test_envelope_resume_is_recorded_unwrapped():
    runner, store = _runner(interrupt_graph())
    await runner.start()
    try:
        tid = await runner.enqueue(title="hitl", prompt="run it")
        await _wait_state(store, tid, REVIEW_NEEDED)
        assert await runner.resume(tid, {"decisions": [{"type": "approve"}]}) is True
        await _wait_state(store, tid, DONE)
        frames = [e for e in await store.get_events(tid) if e["type"] == "resume"]
        assert frames == [{"type": "resume", "decisions": [{"type": "approve"}]}]
    finally:
        await runner.shutdown()


async def test_refused_resume_records_nothing():
    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    from tests.test_task_resume_refused_230 import _paused_task

    runner, tid, _ = await _paused_task()
    try:
        before = await runner.store.get_events(tid)
        with pytest.raises(InvalidDecisionError):
            await runner.resume(tid, [{"type": "aprove"}])
        assert await runner.store.get_events(tid) == before
        assert await runner.resume(tid, [{"type": "approve"}]) is True
        await _wait_state(runner.store, tid, DONE)
        frames = [e for e in await runner.store.get_events(tid) if e["type"] == "resume"]
        assert frames == [{"type": "resume", "decisions": [{"type": "approve"}]}]
    finally:
        await runner.shutdown()
