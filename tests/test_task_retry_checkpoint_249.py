"""Regression tests for gh #249: ``TaskRunner.retry()`` must resume the thread
from its last checkpoint instead of re-sending the prompt.

``retry()`` used to just re-queue the task with the same thread_id; the worker
then called ``submit_message(thread_id, prompt)`` again — a NEW user message
that re-runs the whole graph from the start. On a graph with side effects that
meant every completed step ran twice, and the prompt showed up twice in the
thread history. ``retry()`` now marks the record ``resume_from_checkpoint`` and
the worker goes through ``SessionAdapter.submit_resume`` —
``graph.astream(None, config)`` — which LangGraph resolves against the thread's
checkpoint: completed nodes stay complete, only the pending node re-runs, and
no new message is appended. Threads with nothing to resume (no durable saver /
no pending checkpoint) fall back to the established fresh submit, which keeps
retry working for graphs without a checkpointer.
"""
import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langstage_core.adapters.session import SessionAdapter
from langstage_core.tasks import (
    DONE,
    FAILED,
    QUEUED,
    REVIEW_NEEDED,
    InMemoryTaskStore,
    TaskRunner,
)


async def _wait_state(store, task_id, *targets, timeout=4.0):
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    last = None
    while loop.time() < end:
        last = await store.get(task_id)
        if last and last["state"] in targets:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {task_id} never reached {targets}; last={last}")


def fail_once_two_step_graph(steps: list):
    """``n1`` always succeeds; ``n2`` fails its first run and succeeds after.

    Both nodes append to ``steps`` — one entry per invocation — so a re-run of
    completed work is visible as a repeated letter.
    """

    def n1(state):
        steps.append("n1")
        return {"messages": [AIMessage(content="step one")]}

    def n2(state):
        steps.append("n2")
        if steps.count("n2") < 2:
            raise RuntimeError("n2 blew up")
        return {"messages": [AIMessage(content="step two")]}

    b = StateGraph(MessagesState)
    b.add_node("n1", n1)
    b.add_node("n2", n2)
    b.add_edge(START, "n1")
    b.add_edge("n1", "n2")
    b.add_edge("n2", END)
    return b.compile(checkpointer=InMemorySaver())


def fail_once_then_gate_graph(n2_runs: list):
    """``n2`` fails its first run; on the resume it pauses on a HITL gate."""

    def n1(state):
        return {"messages": [AIMessage(content="one")]}

    def n2(state):
        n2_runs.append(1)
        if len(n2_runs) == 1:
            raise RuntimeError("n2 blew up")
        decision = interrupt({"action_requests": [{"name": "ship", "args": {}}]})
        return {"messages": [AIMessage(content=f"shipped: {decision}")]}

    b = StateGraph(MessagesState)
    b.add_node("n1", n1)
    b.add_node("n2", n2)
    b.add_edge(START, "n1")
    b.add_edge("n1", "n2")
    b.add_edge("n2", END)
    return b.compile(checkpointer=InMemorySaver())


class TestRetryResumesFromCheckpoint:
    async def test_retry_reruns_only_the_pending_node(self):
        steps: list = []
        adapter = SessionAdapter(graph=fail_once_two_step_graph(steps))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="only-once-prompt")
            row = await _wait_state(store, tid, FAILED)
            assert "n2 blew up" in row["error"]
            assert steps == ["n1", "n2"]

            assert await runner.retry(tid) is True
            row = await _wait_state(store, tid, DONE)

            # n1 did NOT re-run: the old behavior re-ran the whole graph, which
            # shows up here as ['n1', 'n2', 'n1', 'n2'].
            assert steps == ["n1", "n2", "n2"]
            assert row["result"] == "step two"
            # the prompt is in the thread exactly once — no duplicate user message
            state = await adapter._graph.aget_state(
                {"configurable": {"thread_id": row["thread_id"]}}
            )
            user_texts = [
                m.content
                for m in state.values["messages"]
                if getattr(m, "type", "") == "human"
            ]
            # exactly one human turn, the labeled prompt (no duplicate on retry)
            assert len(user_texts) == 1
            assert user_texts[0].endswith("only-once-prompt")
        finally:
            await runner.shutdown()

    async def test_retry_marks_and_consumes_resume_from_checkpoint(self):
        steps: list = []
        adapter = SessionAdapter(graph=fail_once_two_step_graph(steps))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store)  # not started: drive the handoff only
        tid = await runner.enqueue(title="t", prompt="p")
        await store.update(tid, state=FAILED, error="boom")
        assert await runner.retry(tid) is True
        row = await store.get(tid)
        assert row["state"] == QUEUED
        assert row["resume_from_checkpoint"] is True

        # A task retried without ever having the flag has no field to consume.
        tid2 = await runner.enqueue(title="t2", prompt="p2")
        assert (await store.get(tid2)).get("resume_from_checkpoint") is None

    async def test_no_resumable_checkpoint_falls_back_to_fresh_submit(self):
        # A fresh graph has a brand-new (empty) checkpointer, so the retried
        # thread has no checkpoint to resume — retry must still work, via the
        # fallback: a plain submit_message on the new graph.
        steps: list = []
        adapter = SessionAdapter(graph=fail_once_two_step_graph(steps))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            await _wait_state(store, tid, FAILED)
            adapter._graph = fail_once_two_step_graph(steps)
            adapter._agui_agent = None
            assert await runner.retry(tid) is True
            row = await _wait_state(store, tid, DONE)
            assert row["result"]
        finally:
            await runner.shutdown()

    async def test_submit_resume_returns_none_without_checkpoint(self):
        adapter = SessionAdapter(graph=fail_once_two_step_graph([]))
        # A thread that never ran has no pending checkpoint to resume.
        assert await adapter.submit_resume("never-run") is None

    async def test_retry_into_a_hitl_gate_goes_review_needed_and_answers(self):
        n2_runs: list = []
        adapter = SessionAdapter(graph=fail_once_then_gate_graph(n2_runs))
        store = InMemoryTaskStore()
        runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
        await runner.start()
        try:
            tid = await runner.enqueue(title="t", prompt="p")
            await _wait_state(store, tid, FAILED)
            assert await runner.retry(tid) is True
            # the resumed run pauses on the gate → the standard review flow
            row = await _wait_state(store, tid, REVIEW_NEEDED)
            assert row["interrupt"] is not None
            assert row["interrupt"]["type"] == "interrupt"
            assert row["interrupt"]["action_requests"]
            assert await runner.resume(tid, [{"type": "approve"}]) is True
            row = await _wait_state(store, tid, DONE)
            assert "shipped" in (row["result"] or "")
        finally:
            await runner.shutdown()

    async def test_resume_stream_frames_carry_the_turn(self):
        # The direct (non-AG-UI) resume path emits the standard vocabulary:
        # content frames for finished messages, a terminal complete.
        steps: list = []
        graph = fail_once_two_step_graph(steps)
        adapter = SessionAdapter(graph=graph)
        session = adapter.submit_message("s1", "go")
        await session.current_task
        assert session.outcome == "error"
        assert steps == ["n1", "n2"]

        session = await adapter.submit_resume("s1")
        assert session is not None
        await session.current_task
        assert session.outcome == "complete"
        frames = []
        while not session.event_queue.empty():
            frames.append(session.event_queue.get_nowait())
        kinds = [f["type"] for f in frames]
        assert "content" in kinds
        assert kinds[-1] == "complete"
        content = " ".join(f["content"] for f in frames if f["type"] == "content")
        assert "step two" in content
        # no duplicate user message: the resume appended nothing human
        assert not any(f.get("role") == "user" for f in frames if f["type"] == "content")
