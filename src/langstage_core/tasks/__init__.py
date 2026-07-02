"""Async task-delegation engine for LangGraph agents.

A small, single-process control plane that lets a host accept tasks, run them
as background agent sessions, and track them through a four-column board
(queued → ongoing → review_needed → done/failed/cancelled).

- :class:`TaskRunner` — the worker pool that drives a ``SessionAdapter``.
- :class:`TaskStore` — the persistence protocol (surfaces provide a concrete
  store; :class:`InMemoryTaskStore` is a dependency-free reference impl).
- :data:`TASK_TOOLS` — agent tools so an agent can delegate to copies of
  itself (added in a later release).

Example:
    from langstage_core import SessionAdapter
    from langstage_core.tasks import TaskRunner, InMemoryTaskStore

    runner = TaskRunner(adapter, InMemoryTaskStore(), concurrency=3)
    await runner.start()
    task_id = await runner.enqueue(title="research", prompt="...")
"""
from __future__ import annotations

from .runner import TaskRunner, current_task_id, get_runner, set_runner
from .tools import TASK_TOOLS
from .state import (
    CANCELLED,
    DONE,
    FAILED,
    ONGOING,
    QUEUED,
    REVIEW_NEEDED,
    TERMINAL_STATES,
    TaskState,
    outcome_to_state,
)
from .store import InMemoryTaskStore, Task, TaskStore, now_iso

__all__ = [
    "TaskRunner",
    "set_runner",
    "get_runner",
    "current_task_id",
    "TASK_TOOLS",
    "TaskStore",
    "InMemoryTaskStore",
    "Task",
    "now_iso",
    "TaskState",
    "outcome_to_state",
    "TERMINAL_STATES",
    "QUEUED",
    "ONGOING",
    "REVIEW_NEEDED",
    "DONE",
    "FAILED",
    "CANCELLED",
]
