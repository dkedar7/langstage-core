"""Tests for SessionAdapter — session-scoped queue streaming."""
import asyncio
import json
from typing import Any, AsyncIterator

import pytest

from langgraph_stream_parser.adapters.session import SessionAdapter, Session

from .fixtures.mocks import (
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    INTERRUPT_WITH_ACTIONS,
)


# ── Mock graphs ──────────────────────────────────────────────────────


class MockGraph:
    """Async graph yielding canned chunks per call; records calls."""

    def __init__(self, chunks_per_call: list[list[Any]]):
        self._chunks_per_call = chunks_per_call
        self._call_idx = 0
        self.calls: list[dict[str, Any]] = []

    def astream(self, input_data, config=None, stream_mode="updates") -> AsyncIterator[Any]:
        self.calls.append({"input": input_data, "config": config, "stream_mode": stream_mode})
        chunks = self._chunks_per_call[self._call_idx]
        self._call_idx += 1

        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()


class SlowGraph:
    """Async graph that sleeps before yielding, to test cancellation."""

    def astream(self, input_data, config=None, stream_mode="updates"):
        async def gen():
            await asyncio.sleep(5)
            yield SIMPLE_AI_MESSAGE

        return gen()


class BoomGraph:
    """Async graph whose stream raises."""

    def astream(self, input_data, config=None, stream_mode="updates"):
        async def gen():
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        return gen()


def _drain(queue: asyncio.Queue) -> list[dict]:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


# ── Producing turns ──────────────────────────────────────────────────


class TestSubmitMessage:
    async def test_streams_content_and_complete(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")

        session = adapter.submit_message("s1", "Hello")
        await session.current_task
        events = _drain(session.event_queue)

        types = [e["type"] for e in events]
        assert "content" in types
        assert types[-1] == "complete"
        assert graph.calls[0]["config"] == {"configurable": {"thread_id": "s1"}}

    async def test_tool_lifecycle(self):
        graph = MockGraph([[AI_MESSAGE_WITH_TOOL_CALLS, TOOL_MESSAGE_SUCCESS]])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")

        session = adapter.submit_message("s-tools", "search")
        await session.current_task
        types = [e["type"] for e in _drain(session.event_queue)]

        assert "tool_start" in types
        assert "tool_end" in types
        assert "complete" in types

    async def test_context_parts_prepended(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")

        session = adapter.submit_message(
            "s-ctx", "What time is it?", context_parts=["[Current time: noon]"]
        )
        await session.current_task

        sent = graph.calls[0]["input"]["messages"][0]["content"]
        assert sent.startswith("[Current time: noon]")
        assert "What time is it?" in sent

    async def test_default_dual_mode_end_to_end(self):
        # Exercises the real default stream_mode=("updates","messages"):
        # content arrives from the messages channel, tool lifecycle from updates.
        from .fixtures.mocks import (
            DUAL_MESSAGES_TOKEN_1,
            DUAL_UPDATES_TOOL_CALL,
            DUAL_UPDATES_TOOL_RESULT,
        )

        graph = MockGraph([[
            DUAL_MESSAGES_TOKEN_1,
            DUAL_UPDATES_TOOL_CALL,
            DUAL_UPDATES_TOOL_RESULT,
        ]])
        adapter = SessionAdapter(graph=graph)  # default dual mode

        session = adapter.submit_message("s-dual", "hi")
        await session.current_task
        types = [e["type"] for e in _drain(session.event_queue)]

        assert "content" in types
        assert "tool_start" in types
        assert "tool_end" in types
        assert types[-1] == "complete"

    async def test_new_message_cancels_previous(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE], [SIMPLE_AI_MESSAGE]])
        adapter = SessionAdapter(graph=SlowGraph())
        # First turn is slow; submitting again should cancel it.
        session = adapter.submit_message("s-x", "first")
        first_task = session.current_task
        await asyncio.sleep(0.01)

        adapter._graph = graph  # swap to fast graph for the second turn
        adapter.submit_message("s-x", "second")
        await asyncio.gather(first_task, return_exceptions=True)
        assert first_task.cancelled()


class TestResume:
    async def test_interrupt_then_decision(self):
        graph = MockGraph([
            [AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS],
            [TOOL_MESSAGE_SUCCESS],
        ])
        adapter = SessionAdapter(graph=graph, stream_mode="updates")

        session = adapter.submit_message("s-int", "run it")
        await session.current_task
        turn1 = _drain(session.event_queue)
        assert any(e["type"] == "interrupt" for e in turn1)

        session = adapter.submit_decisions("s-int", [{"type": "approve"}])
        await session.current_task
        turn2 = _drain(session.event_queue)
        assert any(e["type"] == "tool_end" for e in turn2)

        from langgraph.types import Command
        assert isinstance(graph.calls[1]["input"], Command)


class TestCancel:
    async def test_cancel_pushes_cancelled_event(self):
        adapter = SessionAdapter(graph=SlowGraph())
        session = adapter.submit_message("s-cancel", "hi")
        await asyncio.sleep(0.01)

        assert adapter.cancel("s-cancel") is True
        await asyncio.gather(session.current_task, return_exceptions=True)

        events = _drain(session.event_queue)
        assert {"type": "cancelled"} in events

    async def test_cancel_unknown_session(self):
        adapter = SessionAdapter(graph=SlowGraph())
        assert adapter.cancel("nope") is False


class TestErrorPath:
    async def test_stream_error_pushed(self):
        adapter = SessionAdapter(graph=BoomGraph())
        session = adapter.submit_message("s-err", "go")
        await session.current_task

        events = _drain(session.event_queue)
        err = [e for e in events if e["type"] == "error"]
        assert err
        assert "kaboom" in err[0]["error"]


class TestSideChannel:
    async def test_push_event_interleaves(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = SessionAdapter(graph=graph)
        adapter.get_or_create("s-side")

        adapter.push_event("s-side", {"type": "file_changed", "path": "a.txt"})
        session = adapter.submit_message("s-side", "hi")
        await session.current_task

        events = _drain(session.event_queue)
        assert events[0] == {"type": "file_changed", "path": "a.txt"}
        assert any(e["type"] == "complete" for e in events)


# ── Session management ───────────────────────────────────────────────


class TestSessionManagement:
    def test_get_or_create_reuses_known_id(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        a = adapter.get_or_create("fixed")
        b = adapter.get_or_create("fixed")
        assert a is b
        assert a.id == "fixed"

    def test_get_or_create_generates_id(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        s = adapter.get_or_create()
        assert s.id
        assert adapter.get(s.id) is s

    def test_delete_session(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        adapter.get_or_create("gone")
        assert adapter.delete_session("gone") is True
        assert adapter.get("gone") is None
        assert adapter.delete_session("gone") is False

    def test_list_and_active_count(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        s = adapter.get_or_create("one")
        s.sse_connected = True
        adapter.get_or_create("two")
        listed = {row["session_id"] for row in adapter.list_sessions()}
        assert listed == {"one", "two"}
        assert adapter.active_count == 1


# ── SSE consumer ─────────────────────────────────────────────────────


class TestSSE:
    async def test_sse_emits_init_then_events(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        session = adapter.get_or_create("s-sse")
        session.push({"type": "content", "content": "hi"})
        session.push({"type": "complete"})

        gen = adapter.sse("s-sse", keepalive=10)
        frames = [await asyncio.wait_for(gen.__anext__(), timeout=1) for _ in range(3)]
        await gen.aclose()

        assert all(f.startswith("data: ") and f.endswith("\n\n") for f in frames)
        payloads = [json.loads(f[len("data: "):].rstrip()) for f in frames]
        assert payloads[0]["type"] == "session_init"
        assert payloads[0]["session_id"] == "s-sse"
        assert payloads[1] == {"type": "content", "content": "hi"}
        assert payloads[2] == {"type": "complete"}

    async def test_sse_keepalive_on_idle(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        adapter.get_or_create("s-idle")

        gen = adapter.sse("s-idle", keepalive=0.02, send_init=False)
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1)
        await gen.aclose()
        assert frame == ": keepalive\n\n"

    async def test_sse_marks_connected(self):
        adapter = SessionAdapter(graph=MockGraph([]))
        gen = adapter.sse("s-conn", keepalive=0.02, send_init=True)
        await asyncio.wait_for(gen.__anext__(), timeout=1)  # init frame
        assert adapter.get("s-conn").sse_connected is True
        await gen.aclose()
        assert adapter.get("s-conn").sse_connected is False
