"""
Handler for stream_mode="messages".

This mode produces token-level streaming of LLM outputs. Each chunk is a
(message_chunk, metadata) tuple where message_chunk is typically an
AIMessageChunk and metadata contains langgraph_node, langgraph_step, etc.
"""
from typing import Any, Iterator

from ..events import ContentEvent, ReasoningEvent, StreamEvent
from ..extractors.messages import (
    extract_message_content,
    extract_reasoning_content,
    clean_tool_dict_from_content,
    get_message_type_name,
)


class MessagesHandler:
    """Handler for stream_mode='messages' chunks.

    Processes (message_chunk, metadata) tuples and produces typed
    StreamEvent objects. Only yields ContentEvent from AIMessageChunk
    text content. Tool call chunks are ignored (the updates handler
    provides complete tool calls in dual mode).
    """

    def __init__(self) -> None:
        # Track which messages actually token-streamed text content, so the
        # dual-mode UpdatesHandler can emit a *fallback* ContentEvent for a
        # finished AIMessage that never streamed (non-LLM / non-token-streaming
        # nodes) without double-emitting one that did. Shared by reference with
        # the updates handler in _parse_multi_mode.
        self.streamed_content_ids: set[str] = set()
        self.streamed_content_nodes: set[str] = set()

    def process_chunk(self, chunk: Any) -> Iterator[StreamEvent]:
        """Process a single messages-mode chunk.

        Args:
            chunk: A (message_chunk, metadata) tuple from
                graph.stream(stream_mode="messages"), or the inner
                data when unwrapped from a multi-mode tuple.

        Yields:
            StreamEvent objects (ContentEvent only).
        """
        if isinstance(chunk, tuple) and len(chunk) == 2:
            message, metadata = chunk
        else:
            message = chunk
            metadata = {}

        message_type = get_message_type_name(message)

        if message_type == "AIMessageChunk":
            yield from self._process_ai_chunk(message, metadata)

    def _process_ai_chunk(
        self, chunk: Any, metadata: dict
    ) -> Iterator[StreamEvent]:
        """Process an AIMessageChunk for reasoning and text content.

        Skips chunks that are tool-call-only. Emits ReasoningEvent for
        reasoning/thinking content blocks (langchain-core standardized
        format), then ContentEvent for any remaining text.

        Args:
            chunk: An AIMessageChunk object.
            metadata: Metadata dict with langgraph_node etc.

        Yields:
            ReasoningEvent for reasoning blocks, ContentEvent for text.
        """
        # Skip chunks that carry tool call data — tool lifecycle is
        # handled by the updates handler in dual mode.
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_call_chunks or tool_calls:
            return

        node_name = None
        agent_name = None
        is_subagent = False
        if isinstance(metadata, dict):
            node_name = metadata.get("langgraph_node")
            agent_name = metadata.get("lc_agent_name")
            # deepagents >= 0.6 tags subagent runs with ls_agent_type
            is_subagent = metadata.get("ls_agent_type") == "subagent"

        # Reasoning / thinking blocks — emitted before text so UI
        # consumers can render the "thinking" indicator first.
        reasoning = extract_reasoning_content(chunk)
        if reasoning:
            yield ReasoningEvent(
                content=reasoning,
                source="content_block",
                node=node_name,
                agent_name=agent_name,
                is_subagent=is_subagent,
            )

        content = extract_message_content(chunk)
        if not content:
            return

        # Clean any stringified tool-call dicts that leak into content
        content = clean_tool_dict_from_content(content)
        if not content:
            return

        # Record that this message (by id) and node token-streamed content, so
        # the dual-mode updates handler knows not to re-emit it as a fallback.
        msg_id = getattr(chunk, "id", None)
        if msg_id:
            self.streamed_content_ids.add(msg_id)
        if node_name:
            self.streamed_content_nodes.add(node_name)

        yield ContentEvent(
            content=content,
            role="assistant",
            node=node_name,
            agent_name=agent_name,
            is_subagent=is_subagent,
        )
