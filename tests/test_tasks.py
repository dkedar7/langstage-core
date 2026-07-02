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
    current_task_id,
    get_runner,
    set_runner,
)
from langgraph_stream_parser.tasks.store import now_iso
from langgraph_stream_parser.tasks.tools import (
    cancel_async_task,
    check_async_task,
    list_async_tasks,
    start_async_task,
    update_async_task,
)

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt


# ── Mock graphs (mirror test_session_adapter) ────────────────────────


def echo_graph(text: str = "task result text"):
    def n(state):
        return {"messages": [AIMessage(content=text)]}
    b = StateGraph(MessagesState)
    b.add_node("n", n); b.add_edge(START, "n"); b.add_edge("n", END)
    return b.compile(checkpointer=InMemorySaver())


def slow_graph():
    async def n(state):
        await asyncio.sleep(5)
        return {"messages": [AIMessage(content="done")]}
    b = StateGraph(MessagesState)
    b.add_node("n", n); b.add_edge(START, "n"); b.add_edge("n", END)
    return b.compile(checkpointer=InMemorySaver())


def boom_graph():
    def n(state):
        raise RuntimeError("kaboom")
    b = StateGraph(MessagesState)
    b.add_node("n", n); b.add_edge(START, "n"); b.add_edge("n", END)
    return b.compile(checkpointer=InMemorySaver())


def interrupt_graph():
    def n(state):
        decision = interrupt({"action_requests": [{"tool": "approve", "args": {}}]})
        return {"messages": [AIMessage(content=f"approved: {decision}")]}
    b = StateGraph(MessagesState)
    b.add_node("n", n); b.add_edge(START, "n"); b.add_edge("n", END)
    return b.compile(checkpointer=InMemorySaver())


def recording_graph(runs: list):
    def n(state):
        runs.append(1)  # one entry per node execution — count == task count means no dup
        return {"messages": [AIMessage(content="ok")]}
    b = StateGraph(MessagesState)
    b.add_node("n", n); b.add_edge(START, "n"); b.add_edge("n", END)
    return b.compile(checkpointer=InMemorySaver())


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
        adapter = SessionAdapter(graph=echo_graph(), stream_mode="updates")
        session = adapter.submit_message("s", "hi")
        await session.current_task
        assert session.outcome == "complete"
        assert session.interrupt is None
        assert session.error is None

    async def test_interrupted(self):
        # A trailing CompleteEvent still follows the interrupt; outcome must
        # be 'interrupted', not 'complete'.
        graph = interrupt_graph()
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        session = adapter.submit_message("s", "run it")
        await session.current_task
        assert session.outcome == "interrupted"
        assert session.interrupt is not None
        assert session.interrupt["type"] == "interrupt"
        assert "action_requests" in session.interrupt

    async def test_error(self):
        adapter = SessionAdapter(graph=boom_graph())
        session = adapter.submit_message("s", "go")
        await session.current_task
        assert session.outcome == "error"
        assert session.error and "kaboom" in session.error

    async def test_cancelled(self):
        adapter = SessionAdapter(graph=slow_graph())
        session = adapter.submit_message("s", "hi")
        await asyncio.sleep(0.01)
        adapter.cancel("s")
        await asyncio.gather(session.current_task, return_exceptions=True)
        assert session.outcome == "cancelled"

    async def test_outcome_resets_between_turns(self):
        # An interrupt turn then a clean turn → outcome flips back to complete.
        graph = interrupt_graph()
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
        adapter = SessionAdapter(graph=echo_graph(), stream_mode="updates")
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
        adapter = SessionAdapter(graph=boom_graph())
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
        graph = interrupt_graph()
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
        graph = interrupt_graph()
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
        adapter = SessionAdapter(graph=echo_graph())
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store)  # not started → no workers
        tid = await runner.enqueue(title="t", prompt="p")
        assert await runner.cancel(tid) is True
        assert (await store.get(tid))["state"] == CANCELLED
        # cancelling a terminal task is a no-op
        assert await runner.cancel(tid) is False

    async def test_cancel_ongoing_task(self):
        adapter = SessionAdapter(graph=slow_graph())
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
        adapter = SessionAdapter(graph=echo_graph())
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
        runner = TaskRunner(SessionAdapter(graph=echo_graph()), store)
        set_runner(runner)
        assert get_runner() is runner
        set_runner(None)
        assert get_runner() is None


# ── 5. Hardening: concurrency, recovery, resilience, edge cases ──────


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
        adapter = SessionAdapter(graph=slow_graph())
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        tid = await runner.enqueue(title="slow", prompt="wait")
        await _wait_state(store, tid, ONGOING)
        await asyncio.wait_for(runner.shutdown(), timeout=5)  # must not hang
        assert (await store.get(tid))["state"] == ONGOING  # left for recovery

    async def test_many_tasks_run_once_each_across_workers(self):
        # No double-execution and full drain with N workers / M>N tasks.
        runs: list = []
        graph = recording_graph(runs)
        adapter = SessionAdapter(graph=graph)
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=3, poll_interval=0.02)
        await runner.start()
        try:
            ids = [await runner.enqueue(title=f"t{i}", prompt=f"p{i}") for i in range(8)]
            for tid in ids:
                await _wait_state(store, tid, DONE, timeout=6)
            # exactly one node execution per task → no task ran twice (atomic claim);
            # distinct per-task threads are guaranteed by the store (task-<id>).
            assert len(runs) == len(ids)
        finally:
            await runner.shutdown()

    async def test_retry_actually_reruns_to_done(self):
        # stream_mode must match the raw chunks the healed MockGraph yields.
        adapter = SessionAdapter(graph=boom_graph(), stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            await _wait_state(store, tid, FAILED)
            adapter._graph = echo_graph()  # heal the graph
            adapter._agui_agent = None  # rebuild the wrapped agent around it
            assert await runner.retry(tid) is True
            row = await _wait_state(store, tid, DONE, timeout=4)
            assert row["error"] is None and row["result"]
        finally:
            await runner.shutdown()

    async def test_worker_survives_claim_error(self):
        adapter = SessionAdapter(graph=echo_graph(), stream_mode="updates")
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
        runner = TaskRunner(SessionAdapter(graph=echo_graph()), InMemoryTaskStore())
        with pytest.raises(ValueError):
            await runner.enqueue(title="x", prompt="   ")


class TestEventTranscript:
    async def test_events_streamed_to_store(self):
        adapter = SessionAdapter(graph=echo_graph(), stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="hi")
            await _wait_state(store, tid, DONE)
            events = await store.get_events(tid)
            types = [e.get("type") for e in events]
            assert "content" in types
            assert types[-1] == "complete"  # terminal event recorded last
        finally:
            await runner.shutdown()

    async def test_followup_reruns_thread(self):
        graph = echo_graph()
        adapter = SessionAdapter(graph=graph, stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="hi")
            await _wait_state(store, tid, DONE)
            assert await runner.followup(tid, "now do more") is True
            await _wait_state(store, tid, DONE)
            events = await store.get_events(tid)
            # the thread ran twice → two terminal completes in the transcript
            assert sum(1 for e in events if e.get("type") == "complete") == 2
        finally:
            await runner.shutdown()

    async def test_followup_rejected_on_unknown(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(SessionAdapter(graph=echo_graph()), store)
        assert await runner.followup("nope", "hi") is False


def _task_id_from(out: str) -> str:
    return out.split("task_id:")[1].split()[0].strip().rstrip(".")


class TestDelegationTools:
    async def test_start_check_list(self):
        adapter = SessionAdapter(graph=echo_graph(), stream_mode="updates")
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        set_runner(runner)
        try:
            out = await start_async_task.ainvoke({"title": "research", "prompt": "go research"})
            assert "task_id:" in out
            tid = _task_id_from(out)
            await _wait_state(store, tid, DONE)
            assert "done" in (await check_async_task.ainvoke({"task_id": tid})).lower()
            assert tid in await list_async_tasks.ainvoke({})
        finally:
            set_runner(None)
            await runner.shutdown()

    async def test_parent_id_from_context(self):
        # The runner sets current_task_id while an agent runs; a tool the agent
        # calls must stamp the spawned task's parent_id from it.
        store = InMemoryTaskStore()
        runner = TaskRunner(SessionAdapter(graph=echo_graph()), store)  # not started
        set_runner(runner)
        try:
            token = current_task_id.set("parent-123")
            try:
                out = await start_async_task.ainvoke({"title": "child", "prompt": "p"})
            finally:
                current_task_id.reset(token)
            child = await store.get(_task_id_from(out))
            assert child["parent_id"] == "parent-123"
        finally:
            set_runner(None)

    async def test_cancel_tool(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(SessionAdapter(graph=echo_graph()), store)  # not started
        set_runner(runner)
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            assert "Cancelled" in await cancel_async_task.ainvoke({"task_id": tid})
            assert (await store.get(tid))["state"] == CANCELLED
        finally:
            set_runner(None)

    async def test_tools_unavailable_without_runner(self):
        set_runner(None)
        assert "unavailable" in (
            await start_async_task.ainvoke({"title": "x", "prompt": "p"})
        ).lower()
        assert "unavailable" in (await list_async_tasks.ainvoke({})).lower()
