"""Async task-delegation engine for LangGraph agents.

A small, single-process control plane that lets a host accept tasks, run them
as background agent sessions, and track them through a four-column board
(queued → ongoing → review_needed → done/failed/cancelled).

- :class:`TaskRunner` — the worker pool that drives a ``SessionAdapter``.
- :class:`TaskStore` — the persistence protocol (surfaces provide a concrete
  store; :class:`InMemoryTaskStore` is a dependency-free reference impl).
- :data:`TASK_TOOLS` — agent tools so an agent can delegate to copies of
  itself (added in a later release).

Example (delegate-and-walk-away — enqueue, then read the result off the board)::

    import asyncio
    from langstage_core import SessionAdapter, load_agent_spec
    from langstage_core.tasks import TaskRunner, InMemoryTaskStore, TERMINAL_STATES

    async def main():
        # the runner needs a SessionAdapter over any CompiledGraph
        adapter = SessionAdapter(graph=load_agent_spec("langstage_core.demo.stub:graph"))
        runner = TaskRunner(adapter, InMemoryTaskStore(), concurrency=3)
        await runner.start()

        task_id = await runner.enqueue(title="research", prompt="Summarize the plan.")

        # poll the board until the task reaches a terminal state; a HITL agent parks
        # at review_needed (NOT terminal) until runner.resume() answers it (gh #166)
        while (task := await runner.store.get(task_id))["state"] not in TERMINAL_STATES:
            if task["state"] == "review_needed":
                await runner.resume(task_id, [{"type": "approve"}])
            await asyncio.sleep(0.1)

        print(task["state"])    # 'done'
        print(task["result"])   # the agent's answer
        await runner.shutdown()

    asyncio.run(main())

A ``Task`` is a ``TypedDict`` — read it with ``task["state"]`` / ``task["result"]``
(not attribute access). States flow
``queued -> ongoing -> review_needed -> done | failed | cancelled``;
``TERMINAL_STATES`` is the set to stop polling on. ``review_needed`` is not in it: a
task paused on an interrupt waits for :meth:`TaskRunner.resume`.
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
