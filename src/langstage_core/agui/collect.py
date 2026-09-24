"""One-shot turn collectors — run a turn, get a typed result (gh #110).

The library exposes exactly two consumers, :func:`iter_event_frames` and
:func:`iter_chunk_frames`, both **streaming** async generators. That is the right
primitive for a live UI, but a large share of real usage is *not* a live UI: a
test, an eval/grading harness, a batch job, a ``@tool`` that delegates to a
sub-agent, a "run my agent once, give me the answer" script. Every one of those
wants a **single call that runs one turn and returns the result** — the final
text, the tool calls it made, any extractions and reasoning, and whether it
completed / interrupted / errored.

Before this module each such consumer hand-rolled the same accumulator loop over
:func:`iter_event_frames` (collect ``content`` deltas, watch for ``interrupt``,
map ``complete`` → outcome, catch ``error``) — and, crucially, re-implemented the
terminal-outcome state machine that already lives in ``SessionAdapter._produce``.
This centralizes both: a small accumulator plus the shared
:func:`~langstage_core.agui._terminal_outcome` rule, returning a typed
:class:`TurnResult`.

Three entry points, mirroring the two ``iter_*`` mappings and the
``averify`` / ``verify`` async+sync pairing:

- :func:`collect_event_frames` — async, over :func:`iter_event_frames` (the
  ``{"type": ...}`` event wire); the primary one.
- :func:`collect_chunk_frames` — async, over :func:`iter_chunk_frames` (the
  ``{"status": ...}`` chunk wire), for CLI/Jupyter parity.
- :func:`run_turn` — the sync "one call, one answer" convenience: it accepts a
  compiled graph **or** a prebuilt ``LangGraphAgent`` (like ``verify``), builds
  the agent if needed, and drives :func:`collect_event_frames` under
  ``asyncio.run``.

Relationship to the sibling ``langstage`` package's ``langstage/oneturn.py``
(``complete_turn`` / ``run_turn_sync``): that one buffers a ``SessionAdapter`` for
the web server's one-turn HTTP endpoint — it needs the session/queue plumbing.
These helpers are the lower-level, **session-free** collectors the issue asked
for: they just iterate the ``iter_*`` mappings, no ``SessionAdapter`` involved.
Reach for these in tests/evals/scripts; the web endpoint keeps using its own.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import (
    _is_langgraph_agent,
    _terminal_outcome,
    build_agent,
    iter_chunk_frames,
    iter_event_frames,
)


@dataclass
class TurnResult:
    """The typed result of running one turn to termination.

    Produced by :func:`collect_event_frames` / :func:`collect_chunk_frames` /
    :func:`run_turn`. Every field is populated from the frames a single turn
    emitted — nothing is thrown away (contrast ``VerifyResult``, which reports
    ``content_chars`` but discards the content itself).

    Attributes:
        text: The concatenated ``content`` deltas — the agent's answer.
        outcome: ``"complete"`` | ``"interrupted"`` | ``"error"`` — the shared
            :func:`~langstage_core.agui._terminal_outcome` verdict.
        tool_calls: One ``{"name", "args", "id"}`` per tool the turn called — the
            same ids on both collectors (the chunk wire carries the call ``id`` since
            gh #149).
        extractions: One ``{"tool_name", "extracted_type", "data"}`` per
            ``extraction`` frame (empty unless ``extractors=`` was passed).
        reasoning: The concatenated ``reasoning`` deltas (reasoning-model
            chain-of-thought), kept separate from ``text``.
        interrupt: The interrupt payload when ``outcome == "interrupted"`` —
            ``{"action_requests", "review_configs", "allowed_decisions"}``, identical
            across both collectors (no frame ``type`` key, gh #154) — else ``None``.
        error: The error message when ``outcome == "error"``, else ``None``.
        traceback: The crash traceback when ``outcome == "error"`` **and**
            ``LANGSTAGE_DEBUG`` is set — showing *where* the agent crashed, not just
            the one-line message. ``None`` otherwise (no error, or debug off). Sourced
            from the error frame's ``traceback`` (gh #132; the frame carries it under
            debug since 1.0.32), so the one-shot path reaches the same "where" the
            streaming wires do.
        frames: Total frames seen (an ``int`` count — NOT the list of frame dicts) —
            handy for smoke checks.
    """

    text: str
    outcome: str
    tool_calls: list[dict] = field(default_factory=list)
    extractions: list[dict] = field(default_factory=list)
    reasoning: str = ""
    interrupt: dict | None = None
    error: str | None = None
    traceback: str | None = None
    frames: int = 0


async def collect_event_frames(
    agent: Any,
    message: str,
    thread_id: str,
    *,
    resume: Any = None,
    max_result_len: int = 500,
    extractors: Any = (),
    state: Any = None,
) -> TurnResult:
    """Run one turn through :func:`iter_event_frames` and return a :class:`TurnResult`.

    A drop-in "I don't need the stream, just the result" over the event wire:
    same signature and kwargs as :func:`iter_event_frames`, all forwarded
    unchanged. ``agent`` may be a prebuilt ``LangGraphAgent``, a compiled graph, or
    a ``module:attr`` / ``path/to/file.py:attr`` **spec string** (resolved through
    :func:`build_agent`, like the CLI) (gh #112). Accumulates ``content`` / ``reasoning`` deltas, ``tool_start`` and
    ``extraction`` frames, captures any ``interrupt`` / ``error``, and derives the
    typed ``outcome`` from the shared
    :func:`~langstage_core.agui._terminal_outcome` rule — so it can't drift from
    ``SessionAdapter._produce``.
    """
    agent = agent if _is_langgraph_agent(agent) else build_agent(agent)

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    extractions: list[dict] = []
    interrupt: dict | None = None
    error: str | None = None
    tb: str | None = None
    frames = 0

    async for frame in iter_event_frames(
        agent,
        message,
        thread_id,
        resume=resume,
        max_result_len=max_result_len,
        extractors=extractors,
        state=state,
    ):
        frames += 1
        kind = frame.get("type")
        if kind == "content":
            text_parts.append(frame.get("content") or "")
        elif kind == "reasoning":
            reasoning_parts.append(frame.get("content") or "")
        elif kind == "tool_start":
            tool_calls.append(
                {"name": frame.get("name"), "args": frame.get("args"), "id": frame.get("id")}
            )
        elif kind == "extraction":
            extractions.append(
                {
                    "tool_name": frame.get("tool_name"),
                    "extracted_type": frame.get("extracted_type"),
                    "data": frame.get("data"),
                }
            )
        elif kind == "interrupt":
            # The interrupt DATA, without the event wire's frame discriminator: the
            # chunk collector stores the inner dict (no `type` key), so storing the
            # whole frame here made `TurnResult.interrupt` differ across the two
            # collectors for the same turn (gh #154). `Session.interrupt` is a
            # separate, documented contract and keeps the full frame.
            interrupt = {k: v for k, v in frame.items() if k != "type"}
        elif kind == "error":
            error = frame.get("error")
            tb = frame.get("traceback")  # present under LANGSTAGE_DEBUG (gh #132)

    outcome = _terminal_outcome(saw_interrupt=interrupt is not None, saw_error=error is not None)
    return TurnResult(
        text="".join(text_parts),
        outcome=outcome,
        tool_calls=tool_calls,
        extractions=extractions,
        reasoning="".join(reasoning_parts),
        # Match _produce: surface the interrupt only when that's the outcome
        # (an interrupt followed by an error is an errored turn, not a paused one).
        interrupt=interrupt if outcome == "interrupted" else None,
        error=error,
        traceback=tb if outcome == "error" else None,
        frames=frames,
    )


async def collect_chunk_frames(
    agent: Any,
    message: str,
    thread_id: str,
    *,
    resume: Any = None,
    max_result_len: int = 500,
    extractors: Any = (),
    state: Any = None,
) -> TurnResult:
    """Run one turn through :func:`iter_chunk_frames` and return a :class:`TurnResult`.

    The chunk-wire counterpart of :func:`collect_event_frames`, for CLI/Jupyter
    parity. Same kwargs, same :class:`TurnResult` shape, same shared
    :func:`~langstage_core.agui._terminal_outcome` verdict — only the frame
    vocabulary differs (``{"status": "streaming", "chunk"/"reasoning"/"tool_calls"/
    "extraction": ...}``, ``{"status": "interrupt"/"error"/"complete"}``). For the
    same turn both collectors return equal ``tool_calls`` (ids included, gh #149)
    and an equal ``interrupt`` dict (gh #154).
    """
    agent = agent if _is_langgraph_agent(agent) else build_agent(agent)

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    extractions: list[dict] = []
    interrupt: dict | None = None
    error: str | None = None
    tb: str | None = None
    frames = 0

    async for chunk in iter_chunk_frames(
        agent,
        message,
        thread_id,
        resume=resume,
        max_result_len=max_result_len,
        extractors=extractors,
        state=state,
    ):
        frames += 1
        status = chunk.get("status")
        if status == "streaming":
            if "chunk" in chunk:
                text_parts.append(chunk.get("chunk") or "")
            if "reasoning" in chunk:
                reasoning_parts.append(chunk.get("reasoning") or "")
            for tc in chunk.get("tool_calls") or []:
                tool_calls.append(
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                )
            if "extraction" in chunk:
                ex = chunk["extraction"] or {}
                extractions.append(
                    {
                        "tool_name": ex.get("tool_name"),
                        "extracted_type": ex.get("extracted_type"),
                        "data": ex.get("data"),
                    }
                )
        elif status == "interrupt":
            interrupt = chunk.get("interrupt", chunk)
        elif status == "error":
            error = chunk.get("error")
            tb = chunk.get("traceback")  # present under LANGSTAGE_DEBUG (gh #132)

    outcome = _terminal_outcome(saw_interrupt=interrupt is not None, saw_error=error is not None)
    return TurnResult(
        text="".join(text_parts),
        outcome=outcome,
        tool_calls=tool_calls,
        extractions=extractions,
        reasoning="".join(reasoning_parts),
        interrupt=interrupt if outcome == "interrupted" else None,
        error=error,
        traceback=tb if outcome == "error" else None,
        frames=frames,
    )


def run_turn(
    graph_or_agent: Any,
    message: str,
    *,
    thread_id: str | None = None,
    **kwargs: Any,
) -> TurnResult:
    """Synchronous "one call, one answer" convenience over :func:`collect_event_frames`.

    Accepts a compiled LangGraph graph **or** an already-built ``LangGraphAgent``
    (like :func:`~langstage_core.agui.verify`) — builds the agent if needed, then
    runs one turn under ``asyncio.run`` and returns the :class:`TurnResult`::

        from langstage_core.agui import run_turn
        from langstage_core.demo.tools import create_tool_demo_agent, demo_extractors

        r = run_turn(create_tool_demo_agent(), "use a tool", extractors=demo_extractors())
        assert r.outcome == "complete"
        assert r.tool_calls[0]["name"] == "demo_lookup"

    Each call is an **isolated** one-shot by default: ``thread_id`` defaults to a fresh
    id per call, and a bare graph is wrapped without mutating it (gh #163) — so
    ``for prompt in dataset: run_turn(graph, prompt)`` never leaks one turn's
    conversation into the next. Extra kwargs (``extractors``, ``max_result_len``,
    ``resume``, ``state``) are forwarded to :func:`collect_event_frames`. For a caller
    already inside an event loop, await :func:`collect_event_frames` directly.

    To carry state across calls (multi-turn, or resuming an interrupt), pass the
    *same* prebuilt ``build_agent(...)`` **and** an explicit, shared ``thread_id`` on
    each call (or a graph compiled with your own checkpointer)::

        agent = build_agent(graph)
        r = run_turn(agent, "ask me", thread_id="t1")          # outcome == "interrupted"
        run_turn(agent, "", thread_id="t1", resume={"decisions": [...]})

    Raises a clear ``RuntimeError`` if called from inside a running event loop (e.g. a
    Jupyter cell), instead of ``asyncio.run``'s opaque "cannot be called from a running
    event loop" + a leaked un-awaited coroutine — pointing at the async alternative
    (gh #135).
    """
    # Check for a running loop BEFORE building the coroutine, so the actionable error
    # replaces asyncio.run's generic one AND nothing is left un-awaited (gh #135).
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — the normal sync case
    else:
        raise RuntimeError(
            "run_turn() cannot be called from inside a running event loop (e.g. a "
            "Jupyter cell or other async code). Use "
            "`await collect_event_frames(agent, message, thread_id, ...)` "
            "(or collect_chunk_frames) instead."
        )
    agent = (
        graph_or_agent if _is_langgraph_agent(graph_or_agent) else build_agent(graph_or_agent)
    )
    if thread_id is None:
        thread_id = f"oneshot-{uuid.uuid4().hex}"
    return asyncio.run(collect_event_frames(agent, message, thread_id, **kwargs))
