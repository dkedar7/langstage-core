"""Tests for the async task-delegation engine (tasks/).

Covers:
- the ``Session.outcome`` terminal-outcome primitive on the adapter,
- ``InMemoryTaskStore`` (atomic claim, requeue, filters),
- ``TaskRunner`` end-to-end (enqueue→done/failed/review_needed) and controls
  (cancel, retry).
"""
import asyncio
from typing import Any, AsyncIterator

import pytest

from langgraph_stream_parser.adapters.session import SessionAdapter
from langgraph_stream_parser.tasks import (
    CANCELLED,
    DONE,
    FAILED,
    ONGOING,
    QUEUED,
    REVIEW_NEEDED,
    InMemoryTaskStore,
    TaskRunner,
    get_runner,
    set_runner,
)
from langgraph_stream_parser.tasks.store import now_iso

from .fixtures.mocks import (
    AI_MESSAGE_WITH_TOOL_CALLS,
    INTERRUPT_WITH_ACTIONS,
    SIMPLE_AI_MESSAGE,
    TOOL_MESSAGE_SUCCESS,
)


# ── Mock graphs (mirror test_session_adapter) ────────────────────────


class MockGraph:
    def __init__(self, chunks_per_call: list[list[Any]]):
        self._chunks_per_call = chunks_per_call
        self._call_idx = 0

    def astream(self, input_data, config=None, stream_mode="updates") -> AsyncIterator[Any]:
        chunks = self._chunks_per_call[min(self._call_idx, len(self._chunks_per_call) - 1)]
        self._call_idx += 1

        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()


class SlowGraph:
    def astream(self, input_data, config=None, stream_mode="updates"):
        async def gen():
            await asyncio.sleep(5)
            yield SIMPLE_AI_MESSAGE

        return gen()


class BoomGraph:
    def astream(self, input_data, config=None, stream_mode="updates"):
        async def gen():
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        return gen()


async def _wait_state(store, task_id, *targets, timeout=3.0):
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    last = None
    while loop.time() < end:
        last = await store.get(task_id)
        if last and last["state"] in targets:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {task_id} never reached {targets}; last={last}")


# ── 1. Session.outcome primitive ─────────────────────────────────────


class TestOutcomePrimitive:
    async def test_complete(self):
        adapter = SessionAdapter(graph=MockGraph([[SIMPLE_AI_MESSAGE]]), stream_mode="updates")
        session = adapter.submit_message("s", "hi")
        await session.current_task
        assert session.outcome == "complete"
        assert session.interrupt is None
        assert session.error is None

    async def test_interrupted(self):
        # A trailing CompleteEvent still follows the interrupt; outcome must
        # be 'interrupted', not 'complete'.
        graph = MockGraph([[AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS]])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        session = adapter.submit_message("s", "run it")
        await session.current_task
        assert session.outcome == "interrupted"
        assert session.interrupt is not None
        assert session.interrupt["type"] == "interrupt"
        assert "action_requests" in session.interrupt

    async def test_error(self):
        adapter = SessionAdapter(graph=BoomGraph())
        session = adapter.submit_message("s", "go")
        await session.current_task
        assert session.outcome == "error"
        assert session.error and "kaboom" in session.error

    async def test_cancelled(self):
        adapter = SessionAdapter(graph=SlowGraph())
        session = adapter.submit_message("s", "hi")
        await asyncio.sleep(0.01)
        adapter.cancel("s")
        await asyncio.gather(session.current_task, return_exceptions=True)
        assert session.outcome == "cancelled"

    async def test_outcome_resets_between_turns(self):
        # An interrupt turn then a clean turn → outcome flips back to complete.
        graph = MockGraph([
            [AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS],
            [SIMPLE_AI_MESSAGE],
        ])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        session = adapter.submit_message("s", "a")
        await session.current_task
        assert session.outcome == "interrupted"
        session = adapter.submit_decisions("s", [{"type": "approve"}])
        await session.current_task
        assert session.outcome == "complete"
        assert session.interrupt is None


# ── 2. InMemoryTaskStore ─────────────────────────────────────────────


def _row(task_id: str, state=QUEUED, created_at=None) -> dict:
    return {
        "task_id": task_id,
        "parent_id": None,
        "title": task_id,
        "prompt": "do",
        "agent_spec": None,
        "state": state,
        "thread_id": f"task-{task_id}",
        "created_at": created_at or now_iso(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "artifacts": None,
        "error": None,
        "interrupt": None,
    }


class TestInMemoryStore:
    async def test_claim_is_atomic_and_fifo(self):
        store = InMemoryTaskStore()
        await store.create(_row("a", created_at="2026-01-01T00:00:00+00:00"))
        await store.create(_row("b", created_at="2026-01-01T00:00:01+00:00"))
        # Two concurrent claims must hand back two distinct tasks (no dup).
        c1, c2 = await asyncio.gather(store.claim_next(), store.claim_next())
        ids = {c1["task_id"], c2["task_id"]}
        assert ids == {"a", "b"}
        assert c1["task_id"] == "a"  # FIFO: oldest first
        assert all(c["state"] == ONGOING for c in (c1, c2))
        assert await store.claim_next() is None

    async def test_requeue_orphans(self):
        store = InMemoryTaskStore()
        await store.create(_row("x", state=ONGOING))
        await store.create(_row("y", state=DONE))
        n = await store.requeue_orphans()
        assert n == 1
        assert (await store.get("x"))["state"] == QUEUED
        assert (await store.get("y"))["state"] == DONE

    async def test_list_filters(self):
        store = InMemoryTaskStore()
        await store.create(_row("p", state=DONE))
        child = _row("c"); child["parent_id"] = "p"
        await store.create(child)
        assert {t["task_id"] for t in await store.list(state=DONE)} == {"p"}
        assert {t["task_id"] for t in await store.list(parent_id="p")} == {"c"}


# ── 3. TaskRunner end-to-end ─────────────────────────────────────────


class TestRunnerEndToEnd:
    async def test_enqueue_to_done_with_result(self):
        adapter = SessionAdapter(graph=MockGraph([[SIMPLE_AI_MESSAGE]]), stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=2, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="hello", prompt="hi there")
            row = await _wait_state(store, tid, DONE)
            assert row["result"]  # captured assistant text
            assert row["finished_at"]
            assert row["thread_id"] == f"task-{tid}"
        finally:
            await runner.shutdown()

    async def test_failing_task_goes_failed(self):
        adapter = SessionAdapter(graph=BoomGraph())
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="boom", prompt="explode")
            row = await _wait_state(store, tid, FAILED)
            assert row["error"] and "kaboom" in row["error"]
        finally:
            await runner.shutdown()

    async def test_interrupt_goes_review_needed(self):
        graph = MockGraph([[AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS]])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="approve me", prompt="run it")
            row = await _wait_state(store, tid, REVIEW_NEEDED)
            assert row["interrupt"] and row["interrupt"]["type"] == "interrupt"
            assert row["finished_at"] is None  # paused, not finished
        finally:
            await runner.shutdown()

    async def test_resume_review_to_done(self):
        graph = MockGraph([
            [AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS],
            [TOOL_MESSAGE_SUCCESS, SIMPLE_AI_MESSAGE],
        ])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="hitl", prompt="run it")
            await _wait_state(store, tid, REVIEW_NEEDED)
            assert await runner.resume(tid, [{"type": "approve"}]) is True
            row = await _wait_state(store, tid, DONE)
            assert row["interrupt"] is None
        finally:
            await runner.shutdown()


# ── 4. Runner controls (no workers needed for pure transitions) ──────


class TestRunnerControls:
    async def test_cancel_queued_task(self):
        adapter = SessionAdapter(graph=MockGraph([[SIMPLE_AI_MESSAGE]]))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store)  # not started → no workers
        tid = await runner.enqueue(title="t", prompt="p")
        assert await runner.cancel(tid) is True
        assert (await store.get(tid))["state"] == CANCELLED
        # cancelling a terminal task is a no-op
        assert await runner.cancel(tid) is False

    async def test_cancel_ongoing_task(self):
        adapter = SessionAdapter(graph=SlowGraph())
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="slow", prompt="wait")
            await _wait_state(store, tid, ONGOING)
            assert await runner.cancel(tid) is True
            row = await _wait_state(store, tid, CANCELLED)
            assert row["state"] == CANCELLED
        finally:
            await runner.shutdown()

    async def test_retry_failed_task(self):
        adapter = SessionAdapter(graph=MockGraph([[SIMPLE_AI_MESSAGE]]))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store)  # not started
        tid = await runner.enqueue(title="t", prompt="p")
        await store.update(tid, state=FAILED, error="boom", finished_at=now_iso())
        assert await runner.retry(tid) is True
        row = await store.get(tid)
        assert row["state"] == QUEUED and row["error"] is None
        # retry only applies to failed/cancelled
        await store.update(tid, state=DONE)
        assert await runner.retry(tid) is False

    async def test_runner_singleton(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(SessionAdapter(graph=MockGraph([[]])), store)
        set_runner(runner)
        assert get_runner() is runner
        set_runner(None)
        assert get_runner() is None


# ── 5. Hardening: concurrency, recovery, resilience, edge cases ──────


class RecordingGraph:
    """Records the thread_id of every astream call; yields one content chunk
    after a brief delay (so concurrent workers actually overlap)."""

    def __init__(self) -> None:
        self.threads: list[str] = []

    def astream(self, input_data, config=None, stream_mode="updates"):
        tid = (config or {}).get("configurable", {}).get("thread_id")
        self.threads.append(tid)

        async def gen():
            await asyncio.sleep(0.02)
            yield SIMPLE_AI_MESSAGE

        return gen()


class FlakyClaimStore(InMemoryTaskStore):
    """Raises on the first claim_next to simulate a transient store error."""

    def __init__(self) -> None:
        super().__init__()
        self._raised = False

    async def claim_next(self):
        if not self._raised:
            self._raised = True
            raise RuntimeError("transient store error")
        return await super().claim_next()


class TestRunnerHardening:
    async def test_shutdown_during_active_run_does_not_hang(self):
        # Regression for the worker/shutdown cancel deadlock: shutting down
        # while a task is mid-run must not block.
        adapter = SessionAdapter(graph=SlowGraph())
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        tid = await runner.enqueue(title="slow", prompt="wait")
        await _wait_state(store, tid, ONGOING)
        await asyncio.wait_for(runner.shutdown(), timeout=5)  # must not hang
        assert (await store.get(tid))["state"] == ONGOING  # left for recovery

    async def test_many_tasks_run_once_each_across_workers(self):
        # No double-execution and full drain with N workers / M>N tasks.
        graph = RecordingGraph()
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=3, poll_interval=0.02)
        await runner.start()
        try:
            ids = [await runner.enqueue(title=f"t{i}", prompt=f"p{i}") for i in range(8)]
            for tid in ids:
                await _wait_state(store, tid, DONE, timeout=6)
            threads = graph.threads
            expected = {f"task-{tid}" for tid in ids}
            assert set(threads) == expected            # every task ran
            assert len(threads) == len(expected)       # exactly once — no dup
        finally:
            await runner.shutdown()

    async def test_retry_actually_reruns_to_done(self):
        # stream_mode must match the raw chunks the healed MockGraph yields.
        adapter = SessionAdapter(graph=BoomGraph(), stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            await _wait_state(store, tid, FAILED)
            adapter._graph = MockGraph([[SIMPLE_AI_MESSAGE]])  # heal the graph
            assert await runner.retry(tid) is True
            row = await _wait_state(store, tid, DONE, timeout=4)
            assert row["error"] is None and row["result"]
        finally:
            await runner.shutdown()

    async def test_worker_survives_claim_error(self):
        adapter = SessionAdapter(graph=MockGraph([[SIMPLE_AI_MESSAGE]]), stream_mode="updates")
        store = FlakyClaimStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            row = await _wait_state(store, tid, DONE, timeout=4)
            assert row["state"] == DONE  # worker recovered from the claim error
        finally:
            await runner.shutdown()

    async def test_enqueue_requires_prompt(self):
        runner = TaskRunner(SessionAdapter(graph=MockGraph([[]])), InMemoryTaskStore())
        with pytest.raises(ValueError):
            await runner.enqueue(title="x", prompt="   ")

    async def test_result_text_dual_mode(self):
        # The real default stream mode is dual ("updates","messages"); make sure
        # result reconstruction works there too, not just in updates mode.
        from .fixtures.mocks import DUAL_MESSAGES_TOKEN_1, DUAL_MESSAGES_TOKEN_2

        graph = MockGraph([[DUAL_MESSAGES_TOKEN_1, DUAL_MESSAGES_TOKEN_2]])
        adapter = SessionAdapter(graph=graph)  # default dual mode
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="d", prompt="hi")
            row = await _wait_state(store, tid, DONE, timeout=4)
            assert row["result"]  # tokens concatenated into the final text
        finally:
            await runner.shutdown()
