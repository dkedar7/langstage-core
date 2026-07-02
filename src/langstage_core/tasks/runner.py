"""TaskRunner: a single-process async worker pool for delegated agent tasks.

Generalizes the cron-scheduler pattern (an asyncio task driving the shared
``SessionAdapter``) into a durable, board-backed task queue:

- ``enqueue`` writes a ``queued`` row and returns a ``task_id`` immediately —
  the caller (an HTTP handler or an agent tool) never blocks on the run.
- ``concurrency`` worker coroutines each claim the oldest ``queued`` task
  (atomically, via the store), run it as its own ``SessionAdapter`` session,
  and transition it to ``done`` / ``failed`` / ``review_needed`` based on the
  session's typed ``outcome``.
- on ``start`` any ``ongoing`` rows left by a crash are requeued.

The runner depends only on the :class:`TaskStore` protocol and the public
``SessionAdapter`` surface, so a surface can back it with any store (in-memory,
SQLite, …) without touching this code.

Single-process by design: the worker count *is* the concurrency cap, and the
atomic claim is only atomic within one process — run one server worker.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from typing import Any, Optional

from .state import (
    CANCELLED,
    DONE,
    FAILED,
    ONGOING,
    QUEUED,
    REVIEW_NEEDED,
    TERMINAL_STATES,
    outcome_to_state,
)
from .store import Task, TaskStore, now_iso

logger = logging.getLogger(__name__)

#: Set to the running task's id while its agent executes, so delegation tools
#: called by that agent can record the spawned sub-task's ``parent_id``.
#: ``asyncio.create_task`` copies the context, so the value propagates into the
#: agent's run task and the tools it invokes.
current_task_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "langstage_current_task_id", default=None
)

#: Event ``type`` values that end a run's stream (the parser emits exactly one
#: per turn; an interrupt is followed by a trailing ``complete``).
_TERMINAL_EVENT_TYPES = frozenset({"complete", "error", "cancelled"})


class TaskRunner:
    """Runs delegated tasks on the app's asyncio loop. See module docstring."""

    def __init__(
        self,
        adapter: Any,
        store: TaskStore,
        *,
        concurrency: int = 3,
        thread_prefix: str = "task-",
        context_label: str = "Async task",
        poll_interval: float = 2.0,
    ) -> None:
        self._adapter = adapter
        self._store = store
        self._concurrency = max(1, concurrency)
        self._thread_prefix = thread_prefix
        self._context_label = context_label
        self._poll_interval = poll_interval
        self._workers: list[asyncio.Task] = []
        self._side_tasks: set[asyncio.Task] = set()  # resume runs, etc.
        self._wake = asyncio.Event()
        self._started = False

    @property
    def store(self) -> TaskStore:
        """The task store backing this runner (for read access from tools)."""
        return self._store

    # ── lifecycle ────────────────────────────────────────────────────
    async def start(self) -> None:
        """Set up the store, recover orphans, and spawn worker coroutines."""
        if self._started:
            return
        await self._store.setup()
        recovered = await self._store.requeue_orphans()
        if recovered:
            logger.info("TaskRunner requeued %d orphaned task(s) on startup", recovered)
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self._concurrency)
        ]
        # Kick the workers in case there's already queued work.
        self._wake.set()

    async def shutdown(self) -> None:
        tasks = [*self._workers, *self._side_tasks]
        for t in tasks:
            t.cancel()
        # Await the cancellations so no tasks linger past shutdown (otherwise
        # the event loop can't close cleanly on the way down).
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._side_tasks.clear()
        self._started = False

    # ── public API ───────────────────────────────────────────────────
    async def enqueue(
        self,
        *,
        title: str,
        prompt: str,
        agent_spec: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Create a queued task and return its id immediately (non-blocking)."""
        if not prompt or not prompt.strip():
            raise ValueError("Task prompt is required.")
        task_id = uuid.uuid4().hex
        task: Task = {
            "task_id": task_id,
            "parent_id": parent_id,
            "title": (title or prompt).strip()[:200],
            "prompt": prompt.strip(),
            "agent_spec": agent_spec,
            "state": QUEUED,
            "thread_id": f"{self._thread_prefix}{task_id}",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "artifacts": None,
            "error": None,
            "interrupt": None,
        }
        await self._store.create(task)
        self._wake.set()
        return task_id

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task. Stops the in-flight run if it's ``ongoing``."""
        task = await self._store.get(task_id)
        if task is None or task.get("state") in TERMINAL_STATES:
            return False
        # Mark cancelled first, then interrupt the in-flight run. The worker
        # awaiting that run sees the run task's CancelledError (its own
        # ``cancelling()`` is 0) and leaves the already-set state alone.
        await self._store.update(
            task_id, state=CANCELLED, finished_at=now_iso()
        )
        if task.get("state") == ONGOING:
            self._adapter.cancel(task["thread_id"])
        return True

    async def resume(self, task_id: str, decisions: list[dict[str, Any]]) -> bool:
        """Resume a ``review_needed`` task with HITL decisions (non-blocking).

        Flips the task back to ``ongoing`` and runs the resume in the
        background so the caller (an approve/reject HTTP handler) returns at
        once.
        """
        task = await self._store.get(task_id)
        if task is None or task.get("state") != REVIEW_NEEDED:
            return False
        await self._store.update(
            task_id, state=ONGOING, interrupt=None, started_at=now_iso()
        )
        t = asyncio.create_task(self._resume_run(task, decisions))
        self._side_tasks.add(t)
        t.add_done_callback(self._side_tasks.discard)
        return True

    async def followup(self, task_id: str, message: str) -> bool:
        """Send a follow-up message to a finished task's thread (talk-back).

        Continues the conversation on the same thread (the checkpointer keeps
        history), runs in the background, and flips the task to ``ongoing`` →
        ``done`` with the new reply. For a ``review_needed`` task use
        :meth:`resume` instead. Non-blocking.
        """
        if not message or not message.strip():
            return False
        task = await self._store.get(task_id)
        if task is None or task.get("state") not in {DONE, FAILED, CANCELLED}:
            return False
        await self._store.update(
            task_id, state=ONGOING, finished_at=None, error=None, started_at=now_iso()
        )
        t = asyncio.create_task(self._followup_run(task, message.strip()))
        self._side_tasks.add(t)
        t.add_done_callback(self._side_tasks.discard)
        return True

    async def _followup_run(self, task: Task, message: str) -> None:
        task_id = task["task_id"]
        token = current_task_id.set(task_id)
        try:
            session = self._adapter.submit_message(task["thread_id"], message)
        finally:
            current_task_id.reset(token)
        await self._await_run(task_id, session)

    async def retry(self, task_id: str) -> bool:
        """Re-queue a ``failed`` / ``cancelled`` task (same thread → resumes
        from its last checkpoint where a durable saver is configured)."""
        task = await self._store.get(task_id)
        if task is None or task.get("state") not in {FAILED, CANCELLED}:
            return False
        await self._store.update(
            task_id,
            state=QUEUED,
            error=None,
            finished_at=None,
            started_at=None,
        )
        self._wake.set()
        return True

    # ── internals ────────────────────────────────────────────────────
    async def _worker(self, idx: int) -> None:
        while True:
            try:
                task = await self._store.claim_next()
            except Exception:  # pragma: no cover - defensive
                logger.exception("worker %d: claim_next failed", idx)
                task = None
            if task is None:
                # Nothing to do: sleep until woken or the poll timeout fires
                # (the timeout self-heals any missed wake signal).
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                continue
            await self._run_one(task)

    async def _run_one(self, task: Task) -> None:
        task_id = task["task_id"]
        # Tag the context so any sub-task the agent spawns records this as parent.
        token = current_task_id.set(task_id)
        try:
            session = self._adapter.submit_message(
                task["thread_id"],
                task["prompt"],
                context_parts=[f"[{self._context_label}: {task.get('title', task_id)}]"],
            )
        finally:
            current_task_id.reset(token)
        await self._await_run(task_id, session)

    async def _resume_run(
        self, task: Task, decisions: list[dict[str, Any]]
    ) -> None:
        task_id = task["task_id"]
        token = current_task_id.set(task_id)
        try:
            session = self._adapter.submit_decisions(task["thread_id"], decisions)
        finally:
            current_task_id.reset(token)
        await self._await_run(task_id, session)

    async def _await_run(self, task_id: str, session: Any) -> None:
        """Stream the run's events to the store as they arrive, then record the
        terminal outcome. Streaming (vs draining at the end) is what makes the
        per-task detail/replay view possible — including live-tailing."""
        content_parts: list[str] = []
        q = getattr(session, "event_queue", None)
        run_task = getattr(session, "current_task", None)
        try:
            if q is not None:
                while True:
                    event = await q.get()
                    await self._store.append_events(task_id, [event])
                    if event.get("type") == "content" and event.get("role", "assistant") == "assistant":
                        content_parts.append(event.get("content", ""))
                    if event.get("type") in _TERMINAL_EVENT_TYPES:
                        break
            if run_task is not None and not run_task.done():
                await run_task
        except asyncio.CancelledError:
            cur = asyncio.current_task()
            if cur is not None and cur.cancelling() > 0:
                # This worker is itself being cancelled (shutdown) → propagate
                # so it can exit; leave the task ``ongoing`` for restart recovery.
                raise
            # Otherwise the *run* task was cancelled via ``cancel()`` — its state
            # is already set; persist any stragglers and stop.
            await self._flush_remaining(task_id, q)
            return
        except Exception:  # pragma: no cover - _produce captures its own errors
            logger.exception("task %s run raised", task_id)

        result_text = "".join(content_parts).strip() or None
        await self._record_outcome(task_id, session, result_text)

    async def _flush_remaining(self, task_id: str, q: Any) -> None:
        if q is None:
            return
        leftover: list[dict[str, Any]] = []
        while not q.empty():
            try:
                leftover.append(q.get_nowait())
            except asyncio.QueueEmpty:  # pragma: no cover
                break
        if leftover:
            await self._store.append_events(task_id, leftover)

    async def _record_outcome(
        self, task_id: str, session: Any, result_text: Optional[str]
    ) -> None:
        outcome = getattr(session, "outcome", None)
        state = outcome_to_state(outcome)
        fields: dict[str, Any] = {"state": state}
        if state in TERMINAL_STATES:
            fields["finished_at"] = now_iso()
        if state == DONE:
            fields["result"] = result_text
        elif state == REVIEW_NEEDED:
            fields["interrupt"] = getattr(session, "interrupt", None)
        elif state == FAILED:
            fields["error"] = getattr(session, "error", None) or "Task run failed."
        await self._store.update(task_id, **fields)


# ── process-global singleton (so agent tools can reach the runner) ──
_runner: Optional[TaskRunner] = None


def set_runner(runner: Optional[TaskRunner]) -> None:
    global _runner
    _runner = runner


def get_runner() -> Optional[TaskRunner]:
    return _runner
