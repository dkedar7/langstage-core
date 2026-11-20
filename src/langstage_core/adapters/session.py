"""Session-scoped streaming adapter for web hosts.

Where a request-scoped adapter runs one turn per call, ``SessionAdapter`` keeps a
long-lived session that:

- survives client disconnects (a page refresh resumes the same session),
- multiplexes a per-session event queue so a producer turn and out-of-band
  events (file-change notifications, etc.) interleave on one SSE stream,
- supports cancelling an in-flight turn,
- delegates conversation state to LangGraph's checkpointer (keyed by the
  session id used as ``thread_id``).

This absorbs the ``session_manager`` + ``sse_adapter`` + chat-route plumbing
that ``cowork-dash`` used to carry in-tree.

Requires: ``pip install langstage-core[fastapi]`` for the SSE helper
(only the JSON framing needs nothing; ``asyncio`` is stdlib).
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

from ..resume import prepare_agent_input


class Session:
    """State for a single chat session. Outlives any one transport connection."""

    def __init__(self, session_id: str | None = None):
        self.id: str = session_id or str(uuid.uuid4())
        self.config: dict[str, Any] = {"configurable": {"thread_id": self.id}}
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.current_task: asyncio.Task | None = None
        self.sse_connected: bool = False
        self.created_at: datetime = datetime.now()
        # Typed terminal outcome of the most recent turn, set by ``_produce``.
        # One of ``"complete" | "interrupted" | "error" | "cancelled"``, or
        # ``None`` before the first turn finishes. Headless consumers (e.g. a
        # task runner) read this instead of re-inspecting the event stream to
        # tell whether a turn finished, paused on a HITL interrupt, or failed.
        self.outcome: str | None = None
        # When ``outcome == "interrupted"``, the serialized InterruptEvent
        # (``{"type": "interrupt", "action_requests": [...], ...}``) that paused
        # the turn — everything a consumer needs to render a review gate and
        # build resume decisions. ``None`` otherwise.
        self.interrupt: dict[str, Any] | None = None
        # When ``outcome in {"error"}``, a human-readable error string.
        self.error: str | None = None

    def cancel_current(self) -> bool:
        """Cancel the in-flight turn if any. Returns True if one was cancelled."""
        if self.current_task is not None and not self.current_task.done():
            self.current_task.cancel()
            return True
        return False

    def push(self, event: dict[str, Any]) -> None:
        """Enqueue a serialized event for SSE consumers (non-blocking)."""
        # The queue is unbounded, so put_nowait never raises QueueFull and is
        # safe to call from within an except/cancellation block.
        self.event_queue.put_nowait(event)


class SessionAdapter:
    """Session-scoped streaming for LangGraph agents.

    Turns stream through the in-process AG-UI adapter (:func:`agui.iter_event_frames`)
    since langstage-core 1.0 (ADR 0003).

    Attributes:
        graph: A compiled LangGraph graph (with a checkpointer for resumption).
        max_result_len: Max length for serialized tool results.
        extractors: Optional iterable of
            :class:`~langstage_core.extractors.base.ToolExtractor` forwarded to
            :func:`agui.iter_event_frames`. After a matching tool result the
            extractor runs and its non-None return is emitted as an ``extraction``
            frame on the SSE stream — the same skill/memory/todo/``display_inline``
            callouts the ``iter_*`` mappings emit, for parity with the CLI/Jupyter
            (``iter_chunk_frames``) and VS Code (``iter_event_frames``) surfaces.
            Defaults to none (no ``extraction`` frames). An extractor whose
            ``tool_name`` is the ``"*"`` sentinel (e.g.
            :class:`~langstage_core.GenericToolExtractor`) is the fallback for any
            tool without a dedicated extractor.

    Example:
        adapter = SessionAdapter(graph=agent)

        # POST /api/chat  →  start a turn
        adapter.submit_message(session_id, content, context_parts=[...])

        # GET /api/stream  →  consume events (persistent EventSource)
        return StreamingResponse(adapter.sse(session_id),
                                 media_type="text/event-stream")

        # POST /api/chat/interrupt  →  resume from a HITL interrupt
        adapter.submit_decisions(session_id, decisions)

        # POST /api/chat/cancel
        adapter.cancel(session_id)
    """

    def __init__(
        self,
        *,
        graph: Any,
        max_result_len: int = 500,
        extractors: Any = (),
        **_legacy: Any,  # accepts + ignores the removed stream_mode/agui/parser kwargs
    ):
        self._graph = graph
        self._max_result_len = max_result_len
        from ..agui import _normalize_extractors

        # Forwarded to iter_event_frames so the web/task-board surface can emit
        # `extraction` frames too — parity with the iter_* surfaces (gh #96). None / a
        # single extractor are accepted; a non-iterable fails here, not mid-turn (#178).
        self._extractors = _normalize_extractors(extractors)
        self._sessions: dict[str, Session] = {}
        # AG-UI-only since langstage-core 1.0 (ADR 0003): turns stream through the
        # in-process AG-UI adapter (``agui.iter_event_frames``) — the wrapped agent
        # is built lazily and reused (its checkpointer keys per-session by thread_id).
        self._agui_agent: Any = None

    # ── Session lifecycle ────────────────────────────────────────────

    def get_or_create(self, session_id: str | None = None) -> Session:
        """Resume an existing session by id, or create a fresh one.

        A provided-but-unknown ``session_id`` is honored as the new session's
        id, so a reconnecting client keeps its ``thread_id`` (and thus its
        checkpointed history).
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = Session(session_id)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Look up a session by id, or None."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Cancel and remove a session. Returns True if it existed."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.cancel_current()
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """Summaries of all live sessions."""
        return [
            {
                "session_id": s.id,
                "created_at": s.created_at.isoformat(),
                "connected": s.sse_connected,
            }
            for s in self._sessions.values()
        ]

    @property
    def active_count(self) -> int:
        """Number of sessions with a connected SSE consumer."""
        return sum(1 for s in self._sessions.values() if s.sse_connected)

    # ── Producing turns ──────────────────────────────────────────────

    def submit_message(
        self,
        session_id: str | None,
        content: str,
        *,
        context_parts: list[str] | None = None,
    ) -> Session:
        """Start a turn for a user message. Cancels any in-flight turn first.

        The turn runs as a background task that pushes serialized events onto
        the session's queue; consume them via :meth:`sse`.
        """
        session = self.get_or_create(session_id)
        session.cancel_current()
        # Reuse prepare_agent_input's context-combining, then hand the raw text to
        # the AG-UI producer (which builds its own RunAgentInput).
        combined = prepare_agent_input(message=content, context_parts=context_parts)
        text = combined["messages"][0]["content"]
        session.current_task = asyncio.create_task(self._produce(session, message=text))
        return session

    async def submit_resume(self, session_id: str | None) -> Session | None:
        """Resume a thread from its last checkpoint (the ``retry`` path).

        Drives ``graph.astream(None, config)`` — LangGraph's checkpoint-resume
        form: completed nodes stay complete (their side effects do not repeat),
        only the node the thread is waiting on re-runs, and no new user message
        is appended. Re-submitting the prompt instead re-runs the whole graph
        from the start and duplicates the message in the thread (gh #249).

        The AG-UI layer cannot express this: a ``RunAgentInput`` always carries
        a user message, and its continue path runs ``aupdate_state``, which
        marks the failed node complete — the wrong semantics. So this drives
        the graph directly (:meth:`_iter_resume_frames`) and maps the update
        stream onto the same frame vocabulary ``iter_event_frames`` emits,
        through the same ``_produce`` tail — ``session.outcome`` /
        ``interrupt`` / ``error`` behave exactly as a submitted message's.

        Returns the Session to consume, or ``None`` when the thread has
        nothing to resume: a graph without ``astream``/``aget_state``, no
        checkpointer (``aget_state`` needs one), or no pending work (``next``
        empty — e.g. the failure predates any checkpoint, or the saver is
        per-process and this thread state is gone). The caller falls back to
        :meth:`submit_message` on ``None``, which keeps retry working for
        graphs without a durable saver.
        """
        session = self.get_or_create(session_id)
        session.cancel_current()
        graph = self._graph
        aget_state = getattr(graph, "aget_state", None)
        if getattr(graph, "astream", None) is None or aget_state is None:
            return None
        try:
            snapshot = await aget_state(dict(session.config))
        except Exception:
            # A graph without a checkpointer raises here — nothing to resume.
            return None
        if snapshot is None or not getattr(snapshot, "next", ()):
            return None
        session.current_task = asyncio.create_task(
            self._produce(session, frames=self._iter_resume_frames(session))
        )
        return session

    async def _iter_resume_frames(self, session: Session) -> AsyncIterator[dict[str, Any]]:
        """Yield the standard frame vocabulary from a checkpoint-resume stream.

        ``stream_mode="updates"`` delivers one ``{node: update}`` chunk per
        completed super-step. An ``AIMessage`` in an update becomes ``content``
        (its text) and ``tool_start`` (its tool calls) frames; a ``ToolMessage``
        becomes the matching ``tool_end``. A HITL interrupt arrives as an
        ``__interrupt__`` chunk of raw ``Interrupt`` objects — decoded with the
        shared :func:`~langstage_core.agui._normalize_interrupt` into the same
        ``interrupt`` frame the AG-UI wire emits (so a resumed run that pauses
        on a gate flows into ``review_needed`` and answers via
        :meth:`submit_decisions` like any other). Errors surface as the
        terminal ``error`` frame instead of propagating out of the iterator
        (gh #93's contract); a clean end yields ``complete``, marked
        ``interrupted`` when an interrupt frame went out (gh #152's contract).

        Frames here carry finished messages, not token deltas — updates mode
        sees each message once, when its node completes. The transcript of a
        resumed run is therefore chunkier than a streamed one, but the
        vocabulary and the terminal contract are identical.
        """
        from ..agui import _message_text, _normalize_interrupt, _truncate_result

        graph = self._graph
        saw_interrupt = False
        try:
            async for chunk in graph.astream(None, dict(session.config), stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupts = update if isinstance(update, (tuple, list)) else (update,)
                        for ev in interrupts:
                            saw_interrupt = True
                            action_requests, review_configs, decisions = _normalize_interrupt(
                                getattr(ev, "value", None)
                            )
                            yield {
                                "type": "interrupt",
                                "action_requests": action_requests,
                                "review_configs": review_configs,
                                "allowed_decisions": decisions,
                            }
                        continue
                    if not isinstance(update, dict):
                        continue
                    for m in update.get("messages") or ():
                        if getattr(m, "type", "") == "ai":
                            text = _message_text(m)
                            if text:
                                yield {"type": "content", "content": text, "role": "assistant",
                                       "node": node, "message_id": getattr(m, "id", None)}
                            for call in getattr(m, "tool_calls", None) or ():
                                yield {"type": "tool_start",
                                       "id": call.get("id") or "",
                                       "name": call.get("name") or "tool",
                                       "args": call.get("args") or {},
                                       "node": node}
                        elif getattr(m, "type", "") == "tool":
                            result = _truncate_result(
                                str(getattr(m, "content", "")), self._max_result_len
                            )
                            is_error = getattr(m, "status", "success") == "error"
                            yield {"type": "tool_end",
                                   "id": getattr(m, "tool_call_id", "") or "",
                                   "name": getattr(m, "name", None) or "tool",
                                   "result": result,
                                   "status": "error" if is_error else "success",
                                   "error_message": result if is_error else None,
                                   "duration_ms": None}
        except Exception as exc:  # noqa: BLE001 — a node exception mid-resume surfaces
            # as the terminal `error` frame, matching iter_event_frames (gh #93).
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            return
        yield {"type": "complete", "outcome": "interrupted" if saw_interrupt else "complete"}

    def submit_decisions(
        self,
        session_id: str | None,
        decisions: list[dict[str, Any]],
    ) -> Session:
        """Resume a session from an interrupt with HITL decisions."""
        session = self.get_or_create(session_id)
        session.cancel_current()
        session.current_task = asyncio.create_task(
            self._produce(session, resume={"decisions": decisions})
        )
        return session

    def cancel(self, session_id: str) -> bool:
        """Cancel the in-flight turn for a session. Returns True if one ran."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        return session.cancel_current()

    def push_event(self, session_id: str | None, event: dict[str, Any]) -> Session:
        """Push an out-of-band event onto a session's stream (side channel).

        Use for events that don't originate from the agent — e.g. a file
        watcher pushing ``{"type": "file_changed", ...}`` into the same SSE
        stream the agent events flow through.
        """
        session = self.get_or_create(session_id)
        session.push(event)
        return session

    async def _produce(
        self,
        session: Session,
        *,
        message: str = "",
        resume: Any = None,
        frames: Any = None,
    ) -> None:
        """Run one turn through the in-process AG-UI adapter, pushing serialized
        ``event_to_dict`` frames onto the session queue and recording the terminal
        outcome (``session.outcome`` + ``session.interrupt`` / ``session.error``) so
        headless consumers (the task runner) don't re-inspect the event stream.

        ``frames`` swaps the source: an already-built async iterator of frames
        (the checkpoint-resume stream, :meth:`_iter_resume_frames`) replaces the
        AG-UI turn entirely, skipping the build/clone below. The tail — push,
        terminal-outcome bookkeeping, cancellation/error mapping — is shared, so
        a resumed run is indistinguishable from a submitted one to consumers.
        """
        from ..agui import _terminal_outcome, build_agent, iter_event_frames

        session.outcome = None
        session.interrupt = None
        session.error = None
        pending_interrupt: dict[str, Any] | None = None
        try:
            if frames is None:
                # build_agent()/clone() run INSIDE the try: a graph build_agent can't wrap,
                # a version-incompatible ag-ui-langgraph, or the documented bare install
                # without the [agui] extra all raise HERE, before the stream. Left outside
                # the try, that exception escaped _produce uncaught — the session pushed no
                # terminal frame, session.outcome/error stayed None, the task wedged in
                # `ongoing` forever (error=None) and the SSE client got keepalives with no
                # `error` frame. Now a build/clone failure degrades exactly like a stream
                # error below: an `error` frame + session.outcome="error". (gh #115)
                if self._agui_agent is None:
                    self._agui_agent = build_agent(self._graph)
                # Clone per turn: the ag-ui-langgraph agent carries per-run state, so
                # concurrent sessions (the task runner runs many at once) must not share
                # one instance. clone() keeps the graph + checkpointer (thread state) but
                # isolates the run — the same pattern build_app uses per request.
                run_agent = self._agui_agent.clone()
                thread_id = session.config.get("configurable", {}).get("thread_id", session.id)
                frames = iter_event_frames(
                    run_agent,
                    message,
                    thread_id,
                    resume=resume,
                    max_result_len=self._max_result_len,
                    extractors=self._extractors,
                )
            async for data in frames:
                session.push(data)
                kind = data.get("type")
                # The complete/interrupted/error decision is the shared
                # _terminal_outcome rule (gh #110) — the same one collect_event_frames
                # / collect_chunk_frames use — so this session-scoped path and the
                # one-shot collectors can't drift. (`cancelled`, below, is orthogonal:
                # a transport concern, not part of the frame-driven rule.)
                if kind == "interrupt":
                    pending_interrupt = data
                elif kind == "error":
                    session.outcome = _terminal_outcome(saw_interrupt=False, saw_error=True)
                    session.error = data.get("error")
                    return
                elif kind == "complete":
                    session.outcome = _terminal_outcome(
                        saw_interrupt=pending_interrupt is not None, saw_error=False
                    )
                    if session.outcome == "interrupted":
                        session.interrupt = pending_interrupt
                    return
        except asyncio.CancelledError:
            session.outcome = "cancelled"
            session.push({"type": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced to the client, not swallowed
            session.outcome = "error"
            session.error = f"{type(exc).__name__}: {exc}"
            session.push({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

    # ── Consuming (SSE) ──────────────────────────────────────────────

    async def sse(
        self,
        session_id: str | None = None,
        *,
        keepalive: float = 30.0,
        send_init: bool = True,
    ) -> AsyncIterator[str]:
        """Yield Server-Sent Events for a session, persistently.

        Drains the session queue, emitting one ``data: {json}\\n\\n`` frame per
        event. On idle, emits a ``: keepalive`` comment every ``keepalive``
        seconds to keep proxies from closing the connection. The generator runs
        until the consumer stops iterating (e.g. the ASGI server detects the
        client disconnected), so a single EventSource spans many turns.

        Args:
            session_id: Resume this session, or create one if None/unknown.
            keepalive: Seconds between keepalive comments while idle.
            send_init: Emit a ``session_init`` frame first so the client learns
                its (possibly newly minted) session id for reconnects.
        """
        session = self.get_or_create(session_id)
        session.sse_connected = True
        try:
            if send_init:
                yield _sse_frame({"type": "session_init", "session_id": session.id})
            while True:
                # asyncio.timeout, not wait_for: on Python 3.11 wait_for drops a cancel
                # (client gone) that lands as an event arrives, and the loop runs on
                # (CPython gh-86296). Yield outside the timeout block.
                try:
                    async with asyncio.timeout(keepalive):
                        event = await session.event_queue.get()
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse_frame(event)
        finally:
            session.sse_connected = False


def _sse_frame(event: dict[str, Any]) -> str:
    """Format an event dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(event)}\n\n"
