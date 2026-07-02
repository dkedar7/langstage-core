"""Task store: the persistence contract for the task-delegation engine.

The engine (:class:`~langstage_core.tasks.runner.TaskRunner`) depends
only on the :class:`TaskStore` *protocol* — it never imports a database driver.
Surfaces provide a concrete store:

- the **web** stage ships a SQLite-backed store (durable across restarts);
- :class:`InMemoryTaskStore` here is a dependency-free reference implementation,
  used by the test-suite and fine for ephemeral, single-process surfaces.

The shape of a task record mirrors deepagents' ``AsyncTask`` plus the board
fields, so a future graduation to an Agent Protocol server (where runs/threads
live remotely) is a store swap, not a redesign.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TypedDict, runtime_checkable

from .state import ONGOING, QUEUED, TaskState


def now_iso() -> str:
    """ISO-8601 UTC timestamp, second precision (``YYYY-MM-DDTHH:MM:SS+00:00``)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Task(TypedDict, total=False):
    """A delegated task record. ``total=False`` so partial updates type-check."""

    task_id: str          # uuid hex (primary key)
    parent_id: Optional[str]   # delegating task, for sub-agent trees
    title: str
    prompt: str
    agent_spec: Optional[str]  # which agent/spec to run (None = host default)
    state: TaskState
    thread_id: str        # langgraph thread / SessionAdapter session id
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    result: Optional[str]      # final assistant text
    artifacts: Optional[list[Any]]
    error: Optional[str]
    interrupt: Optional[dict[str, Any]]  # serialized InterruptEvent when review_needed


@runtime_checkable
class TaskStore(Protocol):
    """Async CRUD + atomic claim for tasks. Implementations must be safe to
    call concurrently from multiple worker coroutines on one event loop."""

    async def setup(self) -> None:
        """Create tables / indices if needed. Idempotent."""
        ...

    async def create(self, task: Task) -> Task:
        """Insert a new task (already populated with id/state/timestamps)."""
        ...

    async def get(self, task_id: str) -> Optional[Task]:
        ...

    async def list(
        self,
        *,
        state: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> list[Task]:
        """All tasks, newest first, optionally filtered by state and/or parent."""
        ...

    async def claim_next(self) -> Optional[Task]:
        """Atomically take the oldest ``queued`` task → ``ongoing`` and return
        it, or ``None`` if the queue is empty. The atomicity is what prevents
        two workers from running the same task."""
        ...

    async def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        """Patch the given fields on a task; returns the updated record."""
        ...

    async def requeue_orphans(self) -> int:
        """Reset any ``ongoing`` tasks back to ``queued`` (call on startup to
        recover from a crash). Returns the number requeued."""
        ...

    async def append_events(self, task_id: str, events: list[dict[str, Any]]) -> None:
        """Append serialized stream events to a task's transcript (the live
        stream the runner produces). Enables a detail/replay view per task."""
        ...

    async def get_events(self, task_id: str) -> list[dict[str, Any]]:
        """Return a task's full event transcript in order (empty if none)."""
        ...


class InMemoryTaskStore:
    """Dependency-free reference store. Not durable across process restarts.

    Implements :class:`TaskStore`. An :class:`asyncio.Lock` serializes the
    claim so two workers never grab the same row.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        return None

    async def create(self, task: Task) -> Task:
        self._tasks[task["task_id"]] = dict(task)  # type: ignore[assignment]
        return self._tasks[task["task_id"]]

    async def get(self, task_id: str) -> Optional[Task]:
        t = self._tasks.get(task_id)
        return dict(t) if t is not None else None  # type: ignore[return-value]

    async def list(
        self,
        *,
        state: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> list[Task]:
        items = list(self._tasks.values())
        if state is not None:
            items = [t for t in items if t.get("state") == state]
        if parent_id is not None:
            items = [t for t in items if t.get("parent_id") == parent_id]
        # newest first by created_at
        items.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return [dict(t) for t in items]  # type: ignore[misc]

    async def claim_next(self) -> Optional[Task]:
        async with self._lock:
            queued = [t for t in self._tasks.values() if t.get("state") == QUEUED]
            if not queued:
                return None
            queued.sort(key=lambda t: t.get("created_at", ""))
            task = queued[0]
            task["state"] = ONGOING
            task["started_at"] = now_iso()
            return dict(task)  # type: ignore[return-value]

    async def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.update(fields)  # type: ignore[typeddict-item]
        return dict(task)  # type: ignore[return-value]

    async def requeue_orphans(self) -> int:
        n = 0
        for task in self._tasks.values():
            if task.get("state") == ONGOING:
                task["state"] = QUEUED
                task["started_at"] = None
                n += 1
        return n

    async def append_events(self, task_id: str, events: list[dict[str, Any]]) -> None:
        self._events.setdefault(task_id, []).extend(events)

    async def get_events(self, task_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(task_id, []))
