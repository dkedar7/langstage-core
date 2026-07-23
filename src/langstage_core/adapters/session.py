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
        # Forwarded to iter_event_frames so the web/task-board surface can emit
        # `extraction` frames too — parity with the iter_* surfaces (gh #96).
        self._extractors = extractors
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
    ) -> None:
        """Run one turn through the in-process AG-UI adapter, pushing serialized
        ``event_to_dict`` frames onto the session queue and recording the terminal
        outcome (``session.outcome`` + ``session.interrupt`` / ``session.error``) so
        headless consumers (the task runner) don't re-inspect the event stream.
        """
        from ..agui import _terminal_outcome, build_agent, iter_event_frames

        session.outcome = None
        session.interrupt = None
        session.error = None
        pending_interrupt: dict[str, Any] | None = None
        if self._agui_agent is None:
            self._agui_agent = build_agent(self._graph)
        # Clone per turn: the ag-ui-langgraph agent carries per-run state, so
        # concurrent sessions (the task runner runs many at once) must not share
        # one instance. clone() keeps the graph + checkpointer (thread state) but
        # isolates the run — the same pattern build_app uses per request.
        run_agent = self._agui_agent.clone()
        thread_id = session.config.get("configurable", {}).get("thread_id", session.id)
        try:
            async for data in iter_event_frames(
                run_agent,
                message,
                thread_id,
                resume=resume,
                max_result_len=self._max_result_len,
                extractors=self._extractors,
            ):
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
                try:
                    event = await asyncio.wait_for(
                        session.event_queue.get(), timeout=keepalive
                    )
                    yield _sse_frame(event)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            session.sse_connected = False


def _sse_frame(event: dict[str, Any]) -> str:
    """Format an event dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(event)}\n\n"
