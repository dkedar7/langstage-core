"""Dual-mode content fallback: a finished AIMessage from a non-token-streaming
node must still produce a ContentEvent — without double-emitting content that
the messages/token stream already delivered.

Regression for the family-wide "custom CompiledGraph replies with nothing in
chat" bug (langstage + langstage-vscode): the parser ran
``stream_mode=["updates","messages"]`` with the updates handler hard-suppressing
content, so a node returning a prebuilt AIMessage (rule-based / router /
retrieval agents, or any LLM call made outside a token-streaming node) emitted
no content at all.
"""
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langgraph_stream_parser import ContentEvent, StreamParser


def _contents(events):
    return [e.content for e in events if isinstance(e, ContentEvent)]


def _parse(stream):
    return list(StreamParser(stream_mode=["updates", "messages"]).parse(iter(stream)))


def test_finished_aimessage_with_no_token_stream_emits_content():
    """The bug: a node that returns a prebuilt AIMessage (no AIMessageChunk)."""
    stream = [
        ("updates", {"respond": {"messages": [AIMessage(content="hello world", id="m1")]}}),
    ]
    assert _contents(_parse(stream)) == ["hello world"]


def test_token_streamed_message_is_not_duplicated_by_updates():
    """Streamed content comes from `messages`; the final updates AIMessage with
    the same id must NOT re-emit it."""
    stream = [
        ("messages", (AIMessageChunk(content="he", id="m1"), {"langgraph_node": "agent"})),
        ("messages", (AIMessageChunk(content="llo", id="m1"), {"langgraph_node": "agent"})),
        ("updates", {"agent": {"messages": [AIMessage(content="hello", id="m1")]}}),
    ]
    # Only the two streamed fragments — no third "hello" from updates.
    assert _contents(_parse(stream)) == ["he", "llo"]


def test_no_id_streaming_dedupes_by_node():
    """If chunks carry no id, the final no-id AIMessage from the same node is
    still treated as already-streamed (conservative: avoid a duplicate)."""
    stream = [
        ("messages", (AIMessageChunk(content="hi", id=None), {"langgraph_node": "agent"})),
        ("updates", {"agent": {"messages": [AIMessage(content="hi", id=None)]}}),
    ]
    assert _contents(_parse(stream)) == ["hi"]


def test_distinct_unstreamed_message_still_emits_alongside_streamed_one():
    """A streamed message in node A and a separate prebuilt message in node B
    (never streamed) both render."""
    stream = [
        ("messages", (AIMessageChunk(content="streamed", id="a1"), {"langgraph_node": "A"})),
        ("updates", {"A": {"messages": [AIMessage(content="streamed", id="a1")]}}),
        ("updates", {"B": {"messages": [AIMessage(content="prebuilt", id="b1")]}}),
    ]
    assert _contents(_parse(stream)) == ["streamed", "prebuilt"]


def test_human_message_is_not_echoed_in_dual_mode():
    """The updates stream's HumanMessage echo stays suppressed in dual mode."""
    stream = [
        ("updates", {"__start__": {"messages": [HumanMessage(content="user input")]}}),
        ("updates", {"respond": {"messages": [AIMessage(content="reply", id="m1")]}}),
    ]
    assert _contents(_parse(stream)) == ["reply"]
