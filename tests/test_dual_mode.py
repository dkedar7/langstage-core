"""Tests for dual stream mode support (stream_mode=["updates", "messages"])."""
import pytest
from typing import AsyncIterator, Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
)

from .fixtures.mocks import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    HumanMessage,
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    INTERRUPT_WITH_ACTIONS,
    MESSAGES_METADATA,
    MESSAGES_CHUNK_TOKEN_1,
    MESSAGES_CHUNK_TOKEN_2,
    MESSAGES_CHUNK_EMPTY,
    MESSAGES_CHUNK_WITH_TOOL_CALL_CHUNKS,
    MESSAGES_CHUNK_TOOL_WITH_CONTENT,
    MESSAGES_CHUNK_WITH_TOOL_CALLS,
    DUAL_MESSAGES_TOKEN_1,
    DUAL_MESSAGES_TOKEN_2,
    DUAL_MESSAGES_EMPTY,
    DUAL_MESSAGES_TOOL_CHUNK,
    DUAL_MESSAGES_TOOL_WITH_CONTENT,
    DUAL_MESSAGES_TOOL_CALLS,
    DUAL_UPDATES_SIMPLE,
    DUAL_UPDATES_TOOL_CALL,
    DUAL_UPDATES_TOOL_RESULT,
    DUAL_UPDATES_INTERRUPT,
    NAMESPACE_PARENT,
    NAMESPACE_CHILD,
    SUBGRAPH_SINGLE_PARENT,
    SUBGRAPH_SINGLE_CHILD,
    SUBGRAPH_SINGLE_CHILD_TOOL,
    SUBGRAPH_MULTI_PARENT_MSG,
    SUBGRAPH_MULTI_CHILD_MSG,
    SUBGRAPH_MULTI_PARENT_UPD,
    SUBGRAPH_MULTI_CHILD_UPD,
    SUBGRAPH_MULTI_CHILD_TOOL_RESULT,
    SUBGRAPH_MULTI_CHILD_INTERRUPT,
)


def make_stream(chunks: list) -> Iterator:
    return iter(chunks)


async def make_async_stream(chunks: list) -> AsyncIterator:
    for chunk in chunks:
        yield chunk


# ── Constructor validation ──────────────────────────────────────────


class TestStreamModeValidation:
    def test_default_is_updates(self):
        parser = StreamParser()
        assert parser._stream_mode == "updates"

    def test_string_updates(self):
        parser = StreamParser(stream_mode="updates")
        assert parser._stream_mode == "updates"

    def test_string_messages(self):
        parser = StreamParser(stream_mode="messages")
        assert parser._stream_mode == "messages"

    def test_string_auto(self):
        parser = StreamParser(stream_mode="auto")
        assert parser._stream_mode == "auto"

    def test_list_mode(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        assert parser._stream_mode == ["updates", "messages"]

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported stream_mode"):
            StreamParser(stream_mode="values")

    def test_invalid_list_element_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode in stream_mode list"):
            StreamParser(stream_mode=["updates", "values"])

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be a string or list"):
            StreamParser(stream_mode=123)


# ── Messages handler (single messages mode) ─────────────────────────


class TestMessagesMode:
    def test_ai_chunk_yields_content(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_TOKEN_1])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello"
        assert content_events[0].node == "agent"

    def test_multiple_tokens(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([
            MESSAGES_CHUNK_TOKEN_1,
            MESSAGES_CHUNK_TOKEN_2,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"

    def test_empty_content_skipped(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_EMPTY])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 0

    def test_tool_call_chunks_ignored(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([
            MESSAGES_CHUNK_WITH_TOOL_CALL_CHUNKS,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(content_events) == 0
        assert len(tool_events) == 0

    def test_tool_call_chunks_with_content_ignored(self):
        """AIMessageChunk with tool_call_chunks AND content (stringified tool dict) should emit nothing."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([
            MESSAGES_CHUNK_TOOL_WITH_CONTENT,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 0

    def test_tool_calls_list_ignored(self):
        """AIMessageChunk with tool_calls list should emit nothing."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([
            MESSAGES_CHUNK_WITH_TOOL_CALLS,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 0

    def test_non_ai_chunk_ignored(self):
        """HumanMessage and ToolMessage chunks produce no events in messages mode."""
        parser = StreamParser(stream_mode="messages")
        human_chunk = (HumanMessage(content="hi"), MESSAGES_METADATA)
        tool_chunk = (
            ToolMessage(content="result", name="search", tool_call_id="c1"),
            MESSAGES_METADATA,
        )
        events = list(parser.parse(make_stream([human_chunk, tool_chunk])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 0

    def test_completes_with_complete_event(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_TOKEN_1])))
        assert isinstance(events[-1], CompleteEvent)

    def test_metadata_node_name(self):
        parser = StreamParser(stream_mode="messages")
        chunk = (AIMessageChunk(content="test"), {"langgraph_node": "custom_node"})
        events = list(parser.parse(make_stream([chunk])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].node == "custom_node"

    def test_missing_metadata(self):
        parser = StreamParser(stream_mode="messages")
        chunk = (AIMessageChunk(content="test"), {})
        events = list(parser.parse(make_stream([chunk])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].node is None


# ── Dual mode deduplication ──────────────────────────────────────────


class TestDualModeDeduplication:
    def test_content_from_messages_only(self):
        """In dual mode, ContentEvent comes from messages, not updates."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_MESSAGES_TOKEN_1,
            DUAL_MESSAGES_TOKEN_2,
            DUAL_UPDATES_SIMPLE,  # has "Hello, how can I help?" — should be suppressed
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"

    def test_tool_calls_from_updates_only(self):
        """ToolCallStartEvent comes from updates, not messages."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_MESSAGES_TOOL_CHUNK,  # tool_call_chunks — should be ignored
            DUAL_UPDATES_TOOL_CALL,  # complete tool call — should emit event
        ]
        events = list(parser.parse(make_stream(chunks)))

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "search"

    def test_tool_end_from_updates_only(self):
        """ToolCallEndEvent comes from updates mode."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_UPDATES_TOOL_CALL,
            DUAL_UPDATES_TOOL_RESULT,
        ]
        events = list(parser.parse(make_stream(chunks)))

        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_ends) == 1
        assert tool_ends[0].name == "search"
        assert tool_ends[0].status == "success"

    def test_interrupt_from_updates_only(self):
        """InterruptEvent comes from updates mode."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [DUAL_UPDATES_INTERRUPT]
        events = list(parser.parse(make_stream(chunks)))

        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].needs_approval is True

    def test_unstreamed_updates_message_emits_fallback_content(self):
        """A finished AIMessage that never token-streamed emits fallback content.

        This is the dual-mode content fallback: when no messages/token stream
        delivered the content (e.g. a node returning a prebuilt AIMessage), the
        updates handler renders it instead of dropping it. Streamed content is
        still deduped — see test_content_from_messages_only.
        """
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [DUAL_UPDATES_SIMPLE]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"


# ── Full interleaved conversation ────────────────────────────────────


class TestDualModeFullConversation:
    def test_interleaved_stream(self):
        """Full dual-mode conversation with interleaved updates and messages."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            # Token-level content streaming
            ("messages", (AIMessageChunk(content="I'll"), MESSAGES_METADATA)),
            ("messages", (AIMessageChunk(content=" search"), MESSAGES_METADATA)),
            ("messages", (AIMessageChunk(content=" for that."), MESSAGES_METADATA)),
            # Tool call from updates (complete)
            ("updates", {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="I'll search for that.",
                            tool_calls=[{
                                "id": "call_1",
                                "name": "search",
                                "args": {"query": "weather"},
                            }],
                        )
                    ]
                }
            }),
            # Tool result from updates
            ("updates", {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="The weather is sunny and 72F",
                            name="search",
                            tool_call_id="call_1",
                        )
                    ]
                }
            }),
            # Final response tokens
            ("messages", (AIMessageChunk(content="It's"), MESSAGES_METADATA)),
            ("messages", (AIMessageChunk(content=" sunny!"), MESSAGES_METADATA)),
            # Final update (content suppressed)
            ("updates", {
                "agent": {
                    "messages": [
                        AIMessage(content="It's sunny!")
                    ]
                }
            }),
        ]

        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]

        # 5 content tokens from messages mode
        assert len(content_events) == 5
        assert content_events[0].content == "I'll"
        assert content_events[4].content == " sunny!"

        # 1 tool call from updates
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "search"

        # 1 tool end from updates
        assert len(tool_ends) == 1
        assert tool_ends[0].status == "success"

        # Ends with CompleteEvent
        assert isinstance(events[-1], CompleteEvent)


# ── Auto-detection ───────────────────────────────────────────────────


class TestAutoDetect:
    def test_auto_detect_single_mode(self):
        """Auto mode detects plain dict chunks as single (updates) mode."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"

    def test_auto_detect_multi_mode(self):
        """Auto mode detects tuple chunks as multi mode."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([
            DUAL_MESSAGES_TOKEN_1,
            DUAL_MESSAGES_TOKEN_2,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"

    def test_auto_detect_preserves_first_chunk(self):
        """Auto mode doesn't lose the first chunk during detection."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_auto_detect_empty_stream(self):
        """Auto mode handles empty stream gracefully."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([])))

        assert len(events) == 1
        assert isinstance(events[0], CompleteEvent)

    def test_auto_detect_multi_preserves_first_chunk(self):
        """Auto mode doesn't lose the first chunk in multi-mode detection."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([DUAL_MESSAGES_TOKEN_1])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello"


# ── Async variants ───────────────────────────────────────────────────


class TestAsyncDualMode:
    @pytest.mark.asyncio
    async def test_aparse_dual_mode(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_MESSAGES_TOKEN_1,
            DUAL_MESSAGES_TOKEN_2,
            DUAL_UPDATES_SIMPLE,
        ]

        events = []
        async for event in parser.aparse(make_async_stream(chunks)):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert isinstance(events[-1], CompleteEvent)

    @pytest.mark.asyncio
    async def test_aparse_auto_detect_single(self):
        parser = StreamParser(stream_mode="auto")

        events = []
        async for event in parser.aparse(make_async_stream([SIMPLE_AI_MESSAGE])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    @pytest.mark.asyncio
    async def test_aparse_auto_detect_multi(self):
        parser = StreamParser(stream_mode="auto")

        events = []
        async for event in parser.aparse(make_async_stream([
            DUAL_MESSAGES_TOKEN_1,
            DUAL_MESSAGES_TOKEN_2,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2

    @pytest.mark.asyncio
    async def test_aparse_auto_detect_empty(self):
        parser = StreamParser(stream_mode="auto")

        events = []
        async for event in parser.aparse(make_async_stream([])):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], CompleteEvent)


# ── parse_chunk ──────────────────────────────────────────────────────


class TestParseChunkDualMode:
    def test_parse_chunk_dual_mode_updates(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = parser.parse_chunk(DUAL_UPDATES_TOOL_CALL)

        # Updates handler with suppress_content=True — no content, but tool events
        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "search"

    def test_parse_chunk_dual_mode_messages(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = parser.parse_chunk(DUAL_MESSAGES_TOKEN_1)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello"

    def test_parse_chunk_dual_mode_malformed(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = parser.parse_chunk({"not": "a tuple"})
        assert events == []

    def test_parse_chunk_auto_raises(self):
        parser = StreamParser(stream_mode="auto")
        with pytest.raises(ValueError, match="parse_chunk.*does not support.*auto"):
            parser.parse_chunk(SIMPLE_AI_MESSAGE)


# ── UpdatesHandler suppress_content ─────────────────────────────────


class TestSuppressContent:
    def test_unstreamed_content_falls_back_to_updates(self):
        """Dual mode renders a finished AIMessage's content when nothing
        token-streamed it (the content fallback), rather than suppressing it."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [DUAL_UPDATES_SIMPLE]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"

    def test_suppress_content_tool_events_unaffected(self):
        """With suppress_content, tool events still come through."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [DUAL_UPDATES_TOOL_CALL, DUAL_UPDATES_TOOL_RESULT]
        events = list(parser.parse(make_stream(chunks)))

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_starts) == 1
        assert len(tool_ends) == 1

    def test_suppress_content_human_message(self):
        """Human message content is also suppressed in dual mode."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        human_update = ("updates", {
            "user_input": {
                "messages": [HumanMessage(content="Hello agent")]
            }
        })
        events = list(parser.parse(make_stream([human_update])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 0


# ── Edge cases ───────────────────────────────────────────────────────


class TestDualModeEdgeCases:
    def test_unknown_mode_in_stream_silently_ignored(self):
        """Unknown mode names in multi-mode stream are silently skipped."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            ("debug", {"some": "debug_data"}),
            DUAL_MESSAGES_TOKEN_1,
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_malformed_chunk_in_multi_mode_skipped(self):
        """Non-tuple chunks in multi-mode stream are silently skipped."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            {"raw": "dict"},  # not a tuple
            DUAL_MESSAGES_TOKEN_1,
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_empty_dual_mode_stream(self):
        """Empty stream in dual mode yields only CompleteEvent."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([])))

        assert len(events) == 1
        assert isinstance(events[0], CompleteEvent)

    def test_tool_call_content_leak_filtered_in_dual_mode(self):
        """In dual mode, tool call content leaking from messages mode should not produce ContentEvents."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_MESSAGES_TOKEN_1,          # "Hello" — real content
            DUAL_MESSAGES_TOOL_WITH_CONTENT, # tool dict in content — should be filtered
            DUAL_UPDATES_TOOL_CALL,          # tool call from updates — should produce ToolCallStartEvent
            DUAL_MESSAGES_TOKEN_2,           # " world" — real content
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_events = [e for e in events if isinstance(e, ToolCallStartEvent)]

        # Only the two real content tokens, not the tool dict leak
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"
        # Tool call comes from updates handler
        assert len(tool_events) == 1
        assert tool_events[0].name == "search"

    def test_single_updates_mode_unchanged(self):
        """Default updates mode behavior is unchanged."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"
        assert isinstance(events[-1], CompleteEvent)


# ── Subgraph (subgraphs=True) support ────────────────────────────────


class TestSubgraphSingleMode:
    """Single stream mode with subgraphs=True: chunks are (namespace, data)."""

    def test_parent_chunk_processed(self):
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_PARENT])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"

    def test_child_chunk_processed(self):
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_CHILD])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert "Subgraph response" in content_events[0].content

    def test_mixed_parent_and_child(self):
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([
            SUBGRAPH_SINGLE_PARENT,
            SUBGRAPH_SINGLE_CHILD,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert isinstance(events[-1], CompleteEvent)

    def test_child_tool_calls(self):
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_CHILD_TOOL])))

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "search"

    def test_parse_chunk_single_subgraph(self):
        parser = StreamParser(stream_mode="updates")
        events = parser.parse_chunk(SUBGRAPH_SINGLE_CHILD)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_regular_dict_still_works(self):
        """Plain dict chunks (no subgraphs) still work in single mode."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1


class TestSubgraphMultiMode:
    """Multi stream mode with subgraphs=True: chunks are (namespace, mode, data)."""

    def test_parent_messages_processed(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_PARENT_MSG])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello"

    def test_child_messages_processed(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_CHILD_MSG])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Sub token"

    def test_child_unstreamed_updates_emits_fallback_content(self):
        """A subgraph updates message that never token-streamed still renders
        (content fallback), with the namespace preserved."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_PARENT_UPD])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"

    def test_child_tool_lifecycle(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([
            SUBGRAPH_MULTI_CHILD_UPD,
            SUBGRAPH_MULTI_CHILD_TOOL_RESULT,
        ])))

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "search"
        assert len(tool_ends) == 1
        assert tool_ends[0].status == "success"

    def test_child_interrupt(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_CHILD_INTERRUPT])))

        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1

    def test_mixed_parent_child_interleaved(self):
        """Full conversation with parent and child subgraph chunks interleaved."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            SUBGRAPH_MULTI_PARENT_MSG,   # parent token
            SUBGRAPH_MULTI_CHILD_MSG,    # child token
            SUBGRAPH_MULTI_CHILD_UPD,    # child tool call (updates)
            SUBGRAPH_MULTI_CHILD_TOOL_RESULT,  # child tool result
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]

        assert len(content_events) == 2  # parent + child tokens
        assert len(tool_starts) == 1
        assert len(tool_ends) == 1
        assert isinstance(events[-1], CompleteEvent)

    def test_parse_chunk_multi_subgraph(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = parser.parse_chunk(SUBGRAPH_MULTI_CHILD_MSG)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Sub token"

    def test_regular_dual_chunks_still_work(self):
        """Regular 2-tuple chunks still work alongside subgraph 3-tuples."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        chunks = [
            DUAL_MESSAGES_TOKEN_1,       # regular 2-tuple
            SUBGRAPH_MULTI_CHILD_MSG,    # subgraph 3-tuple
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2


class TestSubgraphAutoDetect:
    """Auto-detection with subgraph formats."""

    def test_auto_detect_subgraph_multi(self):
        """Auto mode detects subgraph 3-tuple as multi mode."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([
            SUBGRAPH_MULTI_PARENT_MSG,
            SUBGRAPH_MULTI_CHILD_MSG,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2

    def test_auto_detect_subgraph_single(self):
        """Auto mode detects subgraph single (namespace, dict) as updates."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_PARENT])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_auto_detect_subgraph_preserves_first_chunk(self):
        """Auto mode doesn't lose the first subgraph chunk."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_CHILD_MSG])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Sub token"


class TestSubgraphAsync:
    @pytest.mark.asyncio
    async def test_aparse_subgraph_multi(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = []
        async for event in parser.aparse(make_async_stream([
            SUBGRAPH_MULTI_PARENT_MSG,
            SUBGRAPH_MULTI_CHILD_MSG,
            SUBGRAPH_MULTI_CHILD_UPD,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(content_events) == 2
        assert len(tool_starts) == 1

    @pytest.mark.asyncio
    async def test_aparse_subgraph_single(self):
        parser = StreamParser(stream_mode="updates")
        events = []
        async for event in parser.aparse(make_async_stream([
            SUBGRAPH_SINGLE_PARENT,
            SUBGRAPH_SINGLE_CHILD,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2

    @pytest.mark.asyncio
    async def test_aparse_auto_detect_subgraph_multi(self):
        parser = StreamParser(stream_mode="auto")
        events = []
        async for event in parser.aparse(make_async_stream([
            SUBGRAPH_MULTI_PARENT_MSG,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
