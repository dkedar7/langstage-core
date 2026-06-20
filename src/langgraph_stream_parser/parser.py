"""
Main StreamParser class for parsing LangGraph streaming outputs.

This is the primary interface for the langgraph-stream-parser package.
"""
from itertools import chain
from typing import Any, AsyncIterator, Iterator

from .events import (
    CompleteEvent,
    CustomEvent,
    DebugEvent,
    ErrorEvent,
    StreamEvent,
    ToolCallStartEvent,
    ValuesEvent,
)
from .extractors.base import ToolExtractor
from .extractors.builtins import ThinkToolExtractor, TodoExtractor, DisplayInlineExtractor
from .handlers.messages import MessagesHandler
from .handlers.updates import UpdatesHandler

_VALID_MODES = {"updates", "messages", "custom"}
_V2_TYPES = {"updates", "messages", "custom", "values", "debug", "checkpoints", "tasks"}


def _is_v2_stream_part(chunk: Any) -> bool:
    """Check if a chunk is a v2 StreamPart dict.

    v2 StreamParts have the shape ``{"type": str, "ns": tuple, "data": ...}``.
    """
    return (
        isinstance(chunk, dict)
        and "type" in chunk
        and "ns" in chunk
        and "data" in chunk
        and isinstance(chunk["type"], str)
        and isinstance(chunk["ns"], tuple)
    )


def _unwrap_v2_chunk(chunk: dict) -> tuple[str, Any, tuple[str, ...] | None]:
    """Unwrap a v2 StreamPart dict.

    Args:
        chunk: A v2 StreamPart dict with "type", "ns", "data" keys.

    Returns:
        (stream_type, data, namespace) where namespace is None for root graph
        or a non-empty tuple for subgraphs.
    """
    namespace = chunk["ns"] if chunk["ns"] else None  # () → None
    return chunk["type"], chunk["data"], namespace


def _is_multi_mode(chunk: Any) -> bool:
    """Check if a chunk is from multi-mode streaming.

    Handles both regular and subgraph formats:
    - Regular: (mode_name: str, data)
    - Subgraph: (namespace: tuple, mode_name: str, data)
    """
    if not isinstance(chunk, tuple):
        return False
    if len(chunk) == 2 and isinstance(chunk[0], str) and chunk[0] in _VALID_MODES:
        return True
    if len(chunk) == 3 and isinstance(chunk[0], tuple) and isinstance(chunk[1], str) and chunk[1] in _VALID_MODES:
        return True
    return False


def _is_subgraph_single_mode(chunk: Any) -> bool:
    """Check if a chunk is from single-mode streaming with subgraphs=True.

    Format: (namespace: tuple, data: dict)
    """
    return (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[0], tuple)
        and isinstance(chunk[1], dict)
    )


def _unwrap_multi_chunk(chunk: tuple) -> tuple[str, Any, tuple[str, ...] | None]:
    """Unwrap a multi-mode chunk, extracting namespace if present.

    Args:
        chunk: Either (mode_name, data) or (namespace, mode_name, data).

    Returns:
        (mode_name, data, namespace) where namespace is None for parent graph
        or a non-empty tuple for subgraphs.
    """
    if len(chunk) == 3 and isinstance(chunk[0], tuple):
        # Subgraph format: (namespace, mode_name, data)
        namespace = chunk[0] if chunk[0] else None  # () → None
        return chunk[1], chunk[2], namespace
    # Regular format: (mode_name, data)
    return chunk[0], chunk[1], None


def _unwrap_single_chunk(chunk: Any) -> tuple[Any, tuple[str, ...] | None]:
    """Unwrap a single-mode chunk, extracting namespace if present.

    Args:
        chunk: Either a plain dict or (namespace, data).

    Returns:
        (data, namespace) where namespace is None for parent graph
        or a non-empty tuple for subgraphs.
    """
    if _is_subgraph_single_mode(chunk):
        namespace = chunk[0] if chunk[0] else None  # () → None
        return chunk[1], namespace
    return chunk, None


def _stamp_namespace(
    events: Iterator[StreamEvent],
    namespace: tuple[str, ...] | None,
) -> Iterator[StreamEvent]:
    """Stamp namespace onto events that support it.

    Args:
        events: Iterator of events from a handler.
        namespace: Namespace tuple, or None for parent graph.

    Yields:
        Events with namespace set (if applicable).
    """
    if namespace is None:
        yield from events
        return
    for event in events:
        if hasattr(event, "namespace"):
            event.namespace = namespace
        yield event


class StreamParser:
    """Universal parser for LangGraph streaming outputs.

    Normalizes various output formats into typed StreamEvent objects
    that are easy to consume in application code.

    Example:
        parser = StreamParser(stream_mode=["updates", "messages"])

        for event in parser.parse(graph.stream(input, stream_mode=["updates", "messages"])):
            match event:
                case ContentEvent(content=text):
                    print(text, end="")
                case ToolCallStartEvent(name=name):
                    print(f"Calling {name}...")
                case InterruptEvent(action_requests=actions):
                    # Handle HITL
                    ...
    """

    def __init__(
        self,
        *,
        stream_mode: str | list[str] = "updates",
        track_tool_lifecycle: bool = True,
        skip_tools: list[str] | None = None,
        include_state_updates: bool = False,
        default_extractor: ToolExtractor | None = None,
    ):
        """Initialize the parser.

        Args:
            stream_mode: Tells the parser what stream format to expect.
                - "updates" (default): chunks are plain dicts
                - "messages": chunks are (AIMessageChunk, metadata) tuples
                - ["updates", "messages"]: chunks are (mode_name, data) tuples
                - "auto": auto-detect from the first chunk
            track_tool_lifecycle: If True, emit ToolCallStartEvent when tools
                are called and ToolCallEndEvent when results arrive.
                If False, only emit ToolExtractedEvent for registered extractors.
            skip_tools: Tool names to skip entirely (no events emitted).
                Useful for internal tools you don't want to expose in UI.
            include_state_updates: If True, emit StateUpdateEvent for non-message
                state keys in updates mode.
            default_extractor: Fallback extractor invoked for any tool name
                that has no per-tool extractor registered. Without this,
                custom tools only emit ``ToolCallStartEvent`` and
                ``ToolCallEndEvent`` — never ``ToolExtractedEvent`` — so
                hosts that switch on extracted events to render rich
                cards have no signal to act on for unknown tools. Pass
                :class:`~langgraph_stream_parser.GenericToolExtractor`
                to surface every otherwise-unhandled tool as a generic
                ``extracted_type="tool_call"`` event. Default ``None``
                preserves the historical behaviour (only named
                extractors fire).

        Raises:
            ValueError: If stream_mode is invalid.
        """
        self._stream_mode = stream_mode
        self._validate_stream_mode(stream_mode)

        self._track_tool_lifecycle = track_tool_lifecycle
        self._skip_tools = set(skip_tools or [])
        self._include_state_updates = include_state_updates
        self._extractors: dict[str, ToolExtractor] = {}
        self._default_extractor = default_extractor
        self._pending_tool_calls: dict[str, ToolCallStartEvent] = {}

        # Register built-in extractors
        self._register_builtin_extractors()

    @staticmethod
    def _validate_stream_mode(stream_mode: str | list[str]) -> None:
        """Validate the stream_mode parameter."""
        if isinstance(stream_mode, str):
            valid = _VALID_MODES | {"auto", "v2"}
            if stream_mode not in valid:
                raise ValueError(
                    f"Unsupported stream_mode: {stream_mode!r}. "
                    f"Must be one of {sorted(valid)} or a list of modes."
                )
        elif isinstance(stream_mode, list):
            for mode in stream_mode:
                if mode not in _VALID_MODES:
                    raise ValueError(
                        f"Unsupported mode in stream_mode list: {mode!r}. "
                        f"Each element must be one of {sorted(_VALID_MODES)}."
                    )
        else:
            raise ValueError(
                f"stream_mode must be a string or list of strings, "
                f"got {type(stream_mode).__name__}."
            )

    def _register_builtin_extractors(self) -> None:
        """Register the built-in tool extractors."""
        self.register_extractor(ThinkToolExtractor())
        self.register_extractor(TodoExtractor())
        self.register_extractor(DisplayInlineExtractor())

    def register_extractor(self, extractor: ToolExtractor) -> None:
        """Register a custom tool extractor.

        Extractors process ToolMessage content and emit ToolExtractedEvent
        with the extracted data.

        Args:
            extractor: An object implementing the ToolExtractor protocol.
        """
        self._extractors[extractor.tool_name] = extractor

    def unregister_extractor(self, tool_name: str) -> None:
        """Remove a registered extractor.

        Args:
            tool_name: The tool name to unregister.
        """
        self._extractors.pop(tool_name, None)

    def parse(self, stream: Iterator[Any]) -> Iterator[StreamEvent]:
        """Parse a LangGraph stream into typed events.

        This is the main entry point for parsing. It iterates over the
        stream, processes each chunk, and yields typed events.

        Supports subgraph streaming (``subgraphs=True``): namespace tuples
        are stripped automatically and all chunks are processed uniformly.

        Args:
            stream: Iterator from graph.stream().

        Yields:
            StreamEvent objects.

        Example:
            for event in parser.parse(graph.stream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            effective_mode = self._stream_mode

            if effective_mode == "auto":
                stream, effective_mode = self._peek_and_detect(stream)

            if effective_mode == "v2":
                yield from self._parse_v2(stream)
            elif isinstance(effective_mode, list):
                yield from self._parse_multi_mode(stream)
            elif effective_mode == "custom":
                for chunk in stream:
                    data, namespace = _unwrap_single_chunk(chunk)
                    yield CustomEvent(data=data, namespace=namespace)
            else:
                handler = self._create_handler_for_mode(effective_mode)
                for chunk in stream:
                    data, namespace = _unwrap_single_chunk(chunk)
                    yield from _stamp_namespace(
                        handler.process_chunk(data), namespace
                    )

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    async def aparse(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        """Async version of parse().

        Args:
            stream: AsyncIterator from graph.astream().

        Yields:
            StreamEvent objects.

        Example:
            async for event in parser.aparse(graph.astream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            effective_mode = self._stream_mode

            if effective_mode == "auto":
                stream, effective_mode = await self._apeek_and_detect(stream)

            if effective_mode == "v2":
                async for event in self._aparse_v2(stream):
                    yield event
            elif isinstance(effective_mode, list):
                async for event in self._aparse_multi_mode(stream):
                    yield event
            elif effective_mode == "custom":
                async for chunk in stream:
                    data, namespace = _unwrap_single_chunk(chunk)
                    yield CustomEvent(data=data, namespace=namespace)
            else:
                handler = self._create_handler_for_mode(effective_mode)
                async for chunk in stream:
                    data, namespace = _unwrap_single_chunk(chunk)
                    for event in _stamp_namespace(
                        handler.process_chunk(data), namespace
                    ):
                        yield event

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    def parse_chunk(self, chunk: Any) -> list[StreamEvent]:
        """Parse a single chunk into events.

        Useful for manual iteration or when you need to process
        chunks individually.

        Args:
            chunk: A single chunk from graph.stream().

        Returns:
            List of events (may be empty, one, or multiple).

        Raises:
            ValueError: If stream_mode is "auto" (requires stream context).
        """
        if self._stream_mode == "auto":
            raise ValueError(
                "parse_chunk() does not support stream_mode='auto'. "
                "Use parse() or aparse() instead."
            )

        if self._stream_mode == "v2" or _is_v2_stream_part(chunk):
            if not _is_v2_stream_part(chunk):
                return []
            return self._parse_v2_chunk(chunk)

        if isinstance(self._stream_mode, list):
            # Multi-mode: expect (mode_name, data) or (namespace, mode_name, data)
            if not (_is_multi_mode(chunk)):
                return []
            mode_name, data, namespace = _unwrap_multi_chunk(chunk)
            if mode_name == "custom":
                return [CustomEvent(data=data, namespace=namespace)]
            handler = self._create_handler_for_mode(
                mode_name,
                suppress_content=(mode_name == "updates"),
            )
            return list(_stamp_namespace(
                handler.process_chunk(data), namespace
            ))

        if self._stream_mode == "custom":
            data, namespace = _unwrap_single_chunk(chunk)
            return [CustomEvent(data=data, namespace=namespace)]

        handler = self._create_handler_for_mode(self._stream_mode)
        data, namespace = _unwrap_single_chunk(chunk)
        return list(_stamp_namespace(
            handler.process_chunk(data), namespace
        ))

    def _create_updates_handler(
        self,
        suppress_content: bool = False,
        *,
        content_fallback: bool = False,
        streamed_content_ids: set[str] | None = None,
        streamed_content_nodes: set[str] | None = None,
    ) -> UpdatesHandler:
        """Create an UpdatesHandler configured for this parser."""
        return UpdatesHandler(
            extractors=self._extractors,
            default_extractor=self._default_extractor,
            skip_tools=self._skip_tools,
            track_tool_lifecycle=self._track_tool_lifecycle,
            include_state_updates=self._include_state_updates,
            pending_tool_calls=self._pending_tool_calls,
            suppress_content=suppress_content,
            content_fallback=content_fallback,
            streamed_content_ids=streamed_content_ids,
            streamed_content_nodes=streamed_content_nodes,
        )

    def _create_messages_handler(self) -> MessagesHandler:
        """Create a MessagesHandler."""
        return MessagesHandler()

    def _create_handler_for_mode(
        self, mode: str, suppress_content: bool = False
    ) -> UpdatesHandler | MessagesHandler:
        """Create the appropriate handler for a given mode string.

        Note: "custom" mode does not use a handler — it is handled
        directly in the parse/routing methods. Do not call this for
        custom mode.
        """
        if mode == "updates":
            return self._create_updates_handler(
                suppress_content=suppress_content
            )
        elif mode == "messages":
            return self._create_messages_handler()
        else:
            raise ValueError(f"Unsupported stream_mode: {mode!r}.")

    def _parse_multi_mode(self, stream: Iterator[Any]) -> Iterator[StreamEvent]:
        """Parse a multi-mode stream with deduplication.

        In dual mode, ContentEvent comes from "messages" (token-level)
        and tool/interrupt/state events come from "updates".

        Handles both regular ``(mode, data)`` and subgraph
        ``(namespace, mode, data)`` chunk formats.
        """
        messages_handler = self._create_messages_handler()
        updates_handler = self._create_updates_handler(
            content_fallback=True,
            streamed_content_ids=messages_handler.streamed_content_ids,
            streamed_content_nodes=messages_handler.streamed_content_nodes,
        )

        for chunk in stream:
            if not _is_multi_mode(chunk):
                continue

            mode_name, data, namespace = _unwrap_multi_chunk(chunk)

            if mode_name == "updates":
                yield from _stamp_namespace(
                    updates_handler.process_chunk(data), namespace
                )
            elif mode_name == "messages":
                yield from _stamp_namespace(
                    messages_handler.process_chunk(data), namespace
                )
            elif mode_name == "custom":
                yield CustomEvent(data=data, namespace=namespace)

    async def _aparse_multi_mode(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        """Async version of _parse_multi_mode."""
        messages_handler = self._create_messages_handler()
        updates_handler = self._create_updates_handler(
            content_fallback=True,
            streamed_content_ids=messages_handler.streamed_content_ids,
            streamed_content_nodes=messages_handler.streamed_content_nodes,
        )

        async for chunk in stream:
            if not _is_multi_mode(chunk):
                continue

            mode_name, data, namespace = _unwrap_multi_chunk(chunk)

            if mode_name == "updates":
                for event in _stamp_namespace(
                    updates_handler.process_chunk(data), namespace
                ):
                    yield event
            elif mode_name == "messages":
                for event in _stamp_namespace(
                    messages_handler.process_chunk(data), namespace
                ):
                    yield event
            elif mode_name == "custom":
                yield CustomEvent(data=data, namespace=namespace)

    def _route_v2_type(
        self,
        stream_type: str,
        data: Any,
        namespace: tuple[str, ...] | None,
        updates_handler: UpdatesHandler,
        messages_handler: MessagesHandler,
    ) -> Iterator[StreamEvent]:
        """Route a single v2 stream type to the appropriate handler/event."""
        match stream_type:
            case "updates":
                yield from _stamp_namespace(
                    updates_handler.process_chunk(data), namespace
                )
            case "messages":
                yield from _stamp_namespace(
                    messages_handler.process_chunk(data), namespace
                )
            case "custom":
                yield CustomEvent(data=data, namespace=namespace)
            case "values":
                yield ValuesEvent(data=data, namespace=namespace)
            case "debug":
                yield DebugEvent(data=data, debug_type="debug", namespace=namespace)
            case "checkpoints":
                yield DebugEvent(data=data, debug_type="checkpoint", namespace=namespace)
            case "tasks":
                yield DebugEvent(data=data, debug_type="task", namespace=namespace)

    def _parse_v2(self, stream: Iterator[Any]) -> Iterator[StreamEvent]:
        """Parse a v2 StreamPart stream.

        v2 chunks are dicts with ``{"type": str, "ns": tuple, "data": ...}``.
        Routes each type to the appropriate handler or event.
        """
        updates_handler = self._create_updates_handler()
        messages_handler = self._create_messages_handler()

        for chunk in stream:
            if not _is_v2_stream_part(chunk):
                continue
            stream_type, data, namespace = _unwrap_v2_chunk(chunk)
            yield from self._route_v2_type(
                stream_type, data, namespace,
                updates_handler, messages_handler,
            )

    async def _aparse_v2(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        """Async version of _parse_v2."""
        updates_handler = self._create_updates_handler()
        messages_handler = self._create_messages_handler()

        async for chunk in stream:
            if not _is_v2_stream_part(chunk):
                continue
            stream_type, data, namespace = _unwrap_v2_chunk(chunk)
            for event in self._route_v2_type(
                stream_type, data, namespace,
                updates_handler, messages_handler,
            ):
                yield event

    def _parse_v2_chunk(self, chunk: dict) -> list[StreamEvent]:
        """Parse a single v2 StreamPart chunk into events."""
        stream_type, data, namespace = _unwrap_v2_chunk(chunk)
        updates_handler = self._create_updates_handler()
        messages_handler = self._create_messages_handler()
        return list(self._route_v2_type(
            stream_type, data, namespace,
            updates_handler, messages_handler,
        ))

    def _peek_and_detect(
        self, stream: Iterator[Any]
    ) -> tuple[Iterator[Any], str | list[str]]:
        """Peek at the first chunk to auto-detect stream format.

        Detects regular and subgraph formats:
        - Multi-mode: ``(mode, data)`` or ``(namespace, mode, data)``
        - Single-mode: plain dict or ``(namespace, data)``

        Returns:
            (chained_stream, detected_mode) where detected_mode is
            either "updates" or ["updates", "messages"].
        """
        try:
            first_chunk = next(stream)
        except StopIteration:
            return iter([]), "updates"

        if _is_multi_mode(first_chunk):
            return chain([first_chunk], stream), ["updates", "messages"]

        if _is_v2_stream_part(first_chunk):
            return chain([first_chunk], stream), "v2"

        return chain([first_chunk], stream), "updates"

    async def _apeek_and_detect(
        self, stream: AsyncIterator[Any]
    ) -> tuple[AsyncIterator[Any], str | list[str]]:
        """Async version of _peek_and_detect."""
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration:
            return _empty_async_iter(), "updates"

        if _is_multi_mode(first_chunk):
            return _async_chain(first_chunk, stream), ["updates", "messages"]

        if _is_v2_stream_part(first_chunk):
            return _async_chain(first_chunk, stream), "v2"

        return _async_chain(first_chunk, stream), "updates"

    def reset(self) -> None:
        """Reset parser state.

        Clears pending tool calls. Call this when starting a new
        conversation or stream.
        """
        self._pending_tool_calls.clear()


async def _empty_async_iter() -> AsyncIterator[Any]:
    """Empty async iterator."""
    return
    yield  # noqa: unreachable — makes this an async generator


async def _async_chain(
    first: Any, rest: AsyncIterator[Any]
) -> AsyncIterator[Any]:
    """Async equivalent of itertools.chain([first], rest)."""
    yield first
    async for item in rest:
        yield item
