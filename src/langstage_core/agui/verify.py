"""Live preflight — run ONE real turn and report whether the agent actually works.

Every surface grew its own health check (vscode ``--selfcheck``, hermes ``verify``,
web ``langstage check``, cli ``--show-config``, jupyter ``/health``) and they
disagree on the one thing that matters: whether a configured agent can complete a
turn. The two that drive a real turn (vscode, hermes) can't give a false green;
the ones that assert static facts — "imports fine", "loads", "key is set" — do,
and that gap is the single largest class in the family's issue backlog (a clean
``doctor`` for a missing extra, ``[ok] loads`` for a non-graph, a green selfcheck
for a path that errors).

This is the shared primitive those checks were each reinventing: build the agent,
stream one turn through :func:`iter_event_frames`, and return a structured verdict.
A missing API key, a broken tool, or a bad state schema fails HERE — because it
runs the real path — instead of passing preflight and failing at first chat. See
ADR 0004.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from . import _is_langgraph_agent, build_agent, iter_event_frames

# A tiny, model-agnostic probe. It only needs the turn to *complete*; the content
# is incidental (the keyless demo stub echoes it, a real model answers it).
DEFAULT_VERIFY_MESSAGE = "Reply with the single word: OK."


@dataclass
class VerifyResult:
    """The verdict of one preflight turn.

    ``ok`` is True iff the turn reached a ``complete`` frame with no ``error``
    frame and no raised exception. Truthy in a boolean context, so
    ``if verify(graph): ...`` reads naturally.
    """

    ok: bool
    reason: str
    saw_complete: bool = False
    saw_error: bool = False
    error_message: str | None = None
    content_chars: int = 0
    frames: int = 0

    def __bool__(self) -> bool:
        return self.ok


async def averify(
    agent_or_graph: Any,
    *,
    message: str = DEFAULT_VERIFY_MESSAGE,
    thread_id: str = "verify",
    timeout: float = 60.0,
    state: Any = None,
) -> VerifyResult:
    """Drive one real turn through the AG-UI adapter and return a structured verdict.

    Args:
        agent_or_graph: A compiled LangGraph graph OR an already-built
            ``LangGraphAgent`` (see :func:`build_agent`) — both accepted so a
            surface can verify whatever it already holds.
        message: The probe message. Defaults to a trivial model-agnostic prompt.
        thread_id: Checkpoint thread for the probe turn (kept off real threads).
        timeout: Seconds to wait for the turn to finish before failing.
        state: Optional extra graph input (e.g. an agent with a richer contract
            than ``messages``), forwarded to :func:`iter_event_frames`.

    Returns:
        A :class:`VerifyResult`. ``ok`` is True only if the turn *completed with
        no error* — a missing key / broken tool / bad schema yields ``ok=False``
        with a human-readable ``reason``, never an exception to the caller.
    """
    agent = (
        agent_or_graph
        if _is_langgraph_agent(agent_or_graph)
        else build_agent(agent_or_graph)
    )

    result = VerifyResult(ok=False, reason="turn produced no completion frame")

    async def _run() -> None:
        async for frame in iter_event_frames(agent, message, thread_id, state=state):
            result.frames += 1
            kind = frame.get("type")
            if kind == "content":
                result.content_chars += len(frame.get("content") or "")
            elif kind == "error":
                result.saw_error = True
                result.error_message = frame.get("error")
            elif kind == "complete":
                result.saw_complete = True

    try:
        await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        result.reason = f"timed out after {timeout:g}s"
        return result
    except Exception as exc:  # noqa: BLE001 - any failure IS a failed preflight
        result.reason = f"{type(exc).__name__}: {exc}"
        return result

    result.ok = result.saw_complete and not result.saw_error
    if result.ok:
        result.reason = "one turn completed cleanly"
    elif result.saw_error:
        result.reason = f"agent errored: {result.error_message}"
    return result


def verify(agent_or_graph: Any, **kwargs: Any) -> VerifyResult:
    """Synchronous wrapper around :func:`averify`.

    For a sync caller (a CLI ``doctor``/``check``/``selfcheck``). Async callers
    already inside an event loop should await :func:`averify` instead.
    """
    return asyncio.run(averify(agent_or_graph, **kwargs))
