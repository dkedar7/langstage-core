"""Session-scoped streaming adapter for web hosts.

Where :class:`~langgraph_stream_parser.adapters.fastapi.FastAPIAdapter` is
request-scoped (one turn per call), ``SessionAdapter`` keeps a long-lived
session that:

- survives client disconnects (a page refresh resumes the same session),
- multiplexes a per-session event queue so a producer turn and out-of-band
  events (file-change notifications, etc.) interleave on one SSE stream,
- supports cancelling an in-flight turn,
- delegates conversation state to LangGraph's checkpointer (keyed by the
  session id used as ``thread_id``).

This absorbs the ``session_manager`` + ``sse_adapter`` + chat-route plumbing
that ``cowork-dash`` used to carry in-tree.

Requires: ``pip install langgraph-stream-parser[fastapi]`` for the SSE helper
(only the JSON framing needs nothing; ``asyncio`` is stdlib).
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

from ..events import CompleteEvent, ErrorEvent, InterruptEvent, event_to_dict
from ..parser import StreamParser
from ..resume import create_resume_input, prepare_agent_input


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

    Attributes:
        graph: A compiled LangGraph graph (with a checkpointer for resumption).
        stream_mode: Stream mode for ``astream`` and the parser. Defaults to
            dual mode so content streams token-by-token while tool/interrupt
            events arrive complete.
        max_result_len: Max length for serialized tool results (see
            :func:`event_to_dict`).
        parser_kwargs: Extra kwargs forwarded to each ``StreamParser``
            (e.g. ``skip_tools``, custom ``extractors`` via registration).

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
        stream_mode: str | list[str] = ("updates", "messages"),
        max_result_len: int = 500,
        **parser_kwargs: Any,
    ):
        self._graph = graph
        # Normalize tuple default to a list (StreamParser expects str | list).
        self._stream_mode = list(stream_mode) if isinstance(stream_mode, tuple) else stream_mode
        self._max_result_len = max_result_len
        self._parser_kwargs = parser_kwargs
        self._sessions: dict[str, Session] = {}

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
        input_data = prepare_agent_input(message=content, context_parts=context_parts)
        session.current_task = asyncio.create_task(self._produce(session, input_data))
        return session

    def submit_decisions(
        self,
        session_id: str | None,
        decisions: list[dict[str, Any]],
    ) -> Session:
        """Resume a session from an interrupt with HITL decisions."""
        session = self.get_or_create(session_id)
        session.cancel_current()
        input_data = create_resume_input(decisions=decisions)
        session.current_task = asyncio.create_task(self._produce(session, input_data))
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

    async def _produce(self, session: Session, input_data: Any) -> None:
        """Run one turn, pushing serialized events onto the session queue.

        Also records the turn's terminal outcome on the session
        (``session.outcome`` + ``session.interrupt`` / ``session.error``) so
        headless consumers don't have to re-inspect the event stream.
        """
        parser = StreamParser(stream_mode=self._stream_mode, **self._parser_kwargs)
        # Fresh turn → clear any prior outcome.
        session.outcome = None
        session.interrupt = None
        session.error = None
        pending_interrupt: dict[str, Any] | None = None
        try:
            stream = self._graph.astream(
                input_data,
                config=session.config,
                stream_mode=self._stream_mode,
            )
            async for event in parser.aparse(stream):
                data = event_to_dict(event, max_result_len=self._max_result_len)
                session.push(data)
                if isinstance(event, InterruptEvent):
                    # The graph paused for HITL. The parser still emits a
                    # trailing CompleteEvent when the stream ends, so remember
                    # the interrupt and reinterpret that Complete below.
                    pending_interrupt = data
                elif isinstance(event, ErrorEvent):
                    session.outcome = "error"
                    session.error = event.error
                    return
                elif isinstance(event, CompleteEvent):
                    # A Complete that follows an interrupt means "paused",
                    # not "finished" — distinguish the two for consumers.
                    if pending_interrupt is not None:
                        session.outcome = "interrupted"
                        session.interrupt = pending_interrupt
                    else:
                        session.outcome = "complete"
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
