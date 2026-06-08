"""
Handler for stream_mode="updates".

This is the most common stream mode and produces updates in the format:
{node_name: {key: value, ...}}
"""
from typing import Any, Iterator

from ..events import (
    ContentEvent,
    DisplayEvent,
    InterruptEvent,
    ReasoningEvent,
    StateUpdateEvent,
    StreamEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolExtractedEvent,
    UsageEvent,
)
from ..extractors.base import ToolExtractor
from ..extractors.interrupts import process_interrupt
from ..extractors.messages import (
    clean_tool_dict_from_content,
    detect_tool_error,
    extract_message_content,
    extract_tool_calls,
    get_message_type_name,
)


class UpdatesHandler:
    """Handler for stream_mode='updates' chunks.

    Processes update dicts in the format {node_name: state_dict}
    and produces typed StreamEvent objects.
    """

    def __init__(
        self,
        extractors: dict[str, ToolExtractor],
        skip_tools: set[str],
        track_tool_lifecycle: bool,
        include_state_updates: bool,
        pending_tool_calls: dict[str, ToolCallStartEvent],
        suppress_content: bool = False,
        default_extractor: ToolExtractor | None = None,
    ):
        """Initialize the handler.

        Args:
            extractors: Registered tool extractors by tool name.
            skip_tools: Set of tool names to skip entirely.
            track_tool_lifecycle: Whether to emit tool lifecycle events.
            include_state_updates: Whether to emit StateUpdateEvents.
            pending_tool_calls: Shared dict tracking pending tool calls.
            suppress_content: If True, skip ContentEvent generation for
                AI and human messages. Used in dual mode where the
                messages handler provides token-level content instead.
            default_extractor: Fallback extractor invoked when no entry
                in ``extractors`` matches a tool's name. Lets hosts emit
                a generic ``ToolExtractedEvent`` for custom tools the
                parser doesn't know about.
        """
        self._extractors = extractors
        self._default_extractor = default_extractor
        self._skip_tools = skip_tools
        self._track_tool_lifecycle = track_tool_lifecycle
        self._include_state_updates = include_state_updates
        self._pending_tool_calls = pending_tool_calls
        self._suppress_content = suppress_content

    def process_chunk(self, chunk: Any) -> Iterator[StreamEvent]:
        """Process a single update chunk.

        Args:
            chunk: A chunk from graph.stream(stream_mode="updates").

        Yields:
            StreamEvent objects.
        """
        if not isinstance(chunk, dict):
            return

        # Check for interrupts
        if "__interrupt__" in chunk:
            interrupt_data = process_interrupt(chunk["__interrupt__"])
            yield InterruptEvent(
                action_requests=interrupt_data["action_requests"],
                review_configs=interrupt_data["review_configs"],
                raw_value=chunk["__interrupt__"],
            )
            return

        # Process regular node updates
        for node_name, state_data in chunk.items():
            yield from self._process_node_update(node_name, state_data)

    def _process_node_update(
        self, node_name: str, state_data: Any
    ) -> Iterator[StreamEvent]:
        """Process an update from a specific node.

        Args:
            node_name: The name of the node.
            state_data: The state data from the node.

        Yields:
            StreamEvent objects.
        """
        if not isinstance(state_data, dict):
            return

        # Handle messages key
        if "messages" in state_data:
            messages = state_data["messages"]
            if messages:
                yield from self._process_messages(node_name, messages)

        # Handle other state keys if requested
        if self._include_state_updates:
            for key, value in state_data.items():
                if key != "messages":
                    yield StateUpdateEvent(
                        node=node_name,
                        key=key,
                        value=value,
                    )

    def _process_messages(
        self, node_name: str, messages: Any
    ) -> Iterator[StreamEvent]:
        """Process messages from a node update.

        Args:
            node_name: The name of the node.
            messages: List of messages or single message.

        Yields:
            StreamEvent objects.
        """
        if not messages:
            return

        # Normalize to list
        if not isinstance(messages, list):
            messages = [messages]

        # Process each message in sequence
        for message in messages:
            message_type = get_message_type_name(message)

            if message_type == "ToolMessage":
                yield from self._process_tool_message(message)
            elif message_type in ("AIMessage", "AIMessageChunk"):
                yield from self._process_ai_message(node_name, message)
            elif message_type == "HumanMessage":
                yield from self._process_human_message(node_name, message)

    def _process_ai_message(
        self, node_name: str, message: Any
    ) -> Iterator[StreamEvent]:
        """Process an AI message.

        Args:
            node_name: The name of the node.
            message: The AIMessage or AIMessageChunk.

        Yields:
            ContentEvent and/or ToolCallStartEvent objects.
        """
        # Extract tool calls
        tool_calls = extract_tool_calls(message)

        # Filter out skipped tools and emit ToolCallStartEvent for others
        if self._track_tool_lifecycle:
            for tc in tool_calls:
                tool_name = tc.get("name")
                if tool_name in self._skip_tools:
                    continue

                event = ToolCallStartEvent(
                    id=tc["id"],
                    name=tool_name,
                    args=tc["args"],
                    node=node_name,
                )
                self._pending_tool_calls[tc["id"]] = event
                yield event

        # Extract and yield content (unless suppressed for dual mode)
        if not self._suppress_content:
            content = extract_message_content(message)
            content = content.strip() if content else ""

            # Clean tool dict representations from content
            if content and tool_calls:
                content = clean_tool_dict_from_content(content)

            # Yield content if non-empty
            if content:
                yield ContentEvent(
                    content=content,
                    role="assistant",
                    node=node_name,
                )

        # Extract token usage if present
        usage = getattr(message, 'usage_metadata', None)
        if usage and isinstance(usage, dict):
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
            input_details = usage.get('input_token_details') or {}
            cache_read = input_details.get('cache_read', 0) if isinstance(input_details, dict) else 0
            cache_creation = input_details.get('cache_creation', 0) if isinstance(input_details, dict) else 0
            if total_tokens > 0:
                yield UsageEvent(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                    node=node_name,
                )

    def _process_human_message(
        self, node_name: str, message: Any
    ) -> Iterator[StreamEvent]:
        """Process a Human message.

        Args:
            node_name: The name of the node.
            message: The HumanMessage.

        Yields:
            ContentEvent with role="human" (unless content is suppressed).
        """
        if self._suppress_content:
            return

        content = extract_message_content(message)
        content = content.strip() if content else ""

        if content:
            yield ContentEvent(
                content=content,
                role="human",
                node=node_name,
            )

    def _event_from_extraction(
        self,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        extracted_type: str,
        data: Any,
    ) -> Iterator[StreamEvent]:
        """Route extractor output to the appropriate typed event.

        Known extracted_types map to first-class events (ReasoningEvent,
        DisplayEvent). Everything else falls through to the generic
        ToolExtractedEvent so user-registered extractors still work.
        """
        if extracted_type == "reflection":
            # ThinkToolExtractor output. ``data`` is the reflection text.
            yield ReasoningEvent(
                content=str(data) if data is not None else "",
                source="think_tool",
            )
        elif extracted_type == "display_inline":
            # DisplayInlineExtractor output. ``data`` is a dict with
            # display_type / data / title / status / error.
            if isinstance(data, dict):
                yield DisplayEvent(
                    display_type=data.get("display_type", "unknown"),
                    data=data.get("data"),
                    title=data.get("title"),
                    status=data.get("status", "success"),
                    error=data.get("error"),
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            else:
                # Malformed extraction — fall back to generic event.
                yield ToolExtractedEvent(
                    tool_name=tool_name,
                    extracted_type=extracted_type,
                    data=data,
                )
        else:
            # Unknown extracted_type — emit generic event for user extractors.
            yield ToolExtractedEvent(
                tool_name=tool_name,
                extracted_type=extracted_type,
                data=data,
            )

    def _process_tool_message(self, message: Any) -> Iterator[StreamEvent]:
        """Process a ToolMessage.

        Args:
            message: The ToolMessage.

        Yields:
            ToolCallEndEvent and/or ToolExtractedEvent objects.
        """
        tool_name = getattr(message, 'name', None)
        tool_call_id = getattr(message, 'tool_call_id', None)
        content = getattr(message, 'content', None)
        artifact = getattr(message, 'artifact', None)

        # Check for errors
        is_error, error_message = detect_tool_error(message)

        # Try to extract special content FIRST — extractors run even for
        # skipped tools. Skipping a tool hides its lifecycle (start/end)
        # events, but the whole point of an extractor is to surface that
        # tool's content in another form (e.g. write_todos -> todos,
        # think_tool -> reflection). Gating the extractor on skip_tools would
        # silently drop those, which is what hosts use skip_tools to keep.
        # Prefer artifact over content (artifact carries full data from
        # tools using response_format="content_and_artifact")
        extractor = self._extractors.get(tool_name) if tool_name else None
        if extractor is None and tool_name and self._default_extractor is not None:
            # No per-tool extractor matched. Fall through to the host's
            # default extractor (e.g. GenericToolExtractor) so unknown
            # tools still produce a ToolExtractedEvent. Opt-in via the
            # `default_extractor` kwarg on StreamParser — None preserves
            # the historical "only named extractors fire" behaviour.
            extractor = self._default_extractor
        if extractor:
            try:
                extracted = extractor.extract(artifact if artifact is not None else content)
                if extracted is not None:
                    yield from self._event_from_extraction(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        extracted_type=extractor.extracted_type,
                        data=extracted,
                    )
            except Exception:
                # Graceful degradation - continue without extraction
                pass

        # Skipped tools emit no lifecycle end event (their start was already
        # suppressed in _process_ai_message).
        if tool_name in self._skip_tools:
            return

        # Emit ToolCallEndEvent if tracking lifecycle
        if self._track_tool_lifecycle and tool_call_id:
            # Calculate duration if we have the start event
            start_event = self._pending_tool_calls.pop(tool_call_id, None)
            duration_ms = None
            if start_event:
                from datetime import datetime
                duration_ms = (
                    datetime.now() - start_event.timestamp
                ).total_seconds() * 1000

            yield ToolCallEndEvent(
                id=tool_call_id,
                name=tool_name or "unknown",
                result=content,
                status="error" if is_error else "success",
                error_message=error_message,
                duration_ms=duration_ms,
            )
