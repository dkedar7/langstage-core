"""Task states for the async task-delegation engine.

A delegated task moves through a small state machine that maps directly onto a
four-column task board:

    queued ──▶ ongoing ──▶ done
                   │
                   ├──▶ review_needed  (paused on a HITL interrupt; awaits a human)
                   ├──▶ failed         (the run errored)
                   └──▶ cancelled      (stopped by a human)

The board groups ``failed``/``cancelled`` under the "Done" column (terminal,
non-success) or shows them distinctly — that's a presentation choice for the
surface, not part of the engine.
"""
from __future__ import annotations

from typing import Literal

TaskState = Literal[
    "queued", "ongoing", "review_needed", "done", "failed", "cancelled"
]

QUEUED: TaskState = "queued"
ONGOING: TaskState = "ongoing"
REVIEW_NEEDED: TaskState = "review_needed"
DONE: TaskState = "done"
FAILED: TaskState = "failed"
CANCELLED: TaskState = "cancelled"

#: States a task will never leave on its own (no further work without a human).
TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})

#: ``Session.outcome`` value → board state after a run finishes.
_OUTCOME_TO_STATE: dict[str, TaskState] = {
    "complete": DONE,
    "interrupted": REVIEW_NEEDED,
    "error": FAILED,
    "cancelled": CANCELLED,
}


def outcome_to_state(outcome: str | None) -> TaskState:
    """Map a :attr:`Session.outcome` onto a board state.

    An unknown / missing outcome is treated as ``failed`` rather than ``done``
    — we never silently mark an indeterminate run successful (it can be retried).
    """
    return _OUTCOME_TO_STATE.get(outcome or "", FAILED)
