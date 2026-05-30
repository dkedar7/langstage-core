"""
Event dataclasses for LangGraph stream parsing.

These typed event objects provide a consistent interface for consuming
LangGraph streaming outputs, regardless of the underlying stream mode
or message types.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union


@dataclass
class ContentEvent:
    """Text content from a message.

    Attributes:
        content: The text content from the message.
        role: The role of the message sender ("assistant" or "human").
        node: The name of the graph node that produced this content.
        agent_name: The deep agent name from ``lc_agent_name`` metadata.
        is_subagent: True when ``ls_agent_type == "subagent"`` was set on
            the run metadata (deepagents >= 0.6). Useful when the chunk
            comes from a subagent but ``agent_name`` is not set.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    content: str
    role: Literal["assistant", "human"] = "assistant"
    node: str | None = None
    agent_name: str | None = None
    is_subagent: bool = False
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "content",
            "content": self.content,
            "role": self.role,
            "node": self.node,
        }
        if self.agent_name is not None:
            d["agent_name"] = self.agent_name
        if self.is_subagent:
            d["is_subagent"] = True
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class ToolCallStartEvent:
    """Tool call initiated by AI.

    Emitted when an AI message contains tool calls. This indicates
    the tool is about to be executed.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool being called.
        args: Arguments passed to the tool.
        node: The name of the graph node that initiated the call.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    id: str
    name: str
    args: dict[str, Any]
    node: str | None = None
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "tool_start",
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "node": self.node,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class ToolCallEndEvent:
    """Tool call completed with result.

    Emitted when a ToolMessage is received, indicating the tool
    has finished executing.

    Attributes:
        id: Unique identifier matching the ToolCallStartEvent.
        name: Name of the tool that was called.
        result: The result returned by the tool.
        status: Whether the tool succeeded or errored.
        error_message: Error details if status is "error".
        duration_ms: Execution time in milliseconds if available.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    id: str
    name: str
    result: Any
    status: Literal["success", "error"]
    error_message: str | None = None
    duration_ms: float | None = None
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self, max_result_len: int = 500) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs.

        Args:
            max_result_len: Maximum length for result string (truncated if longer).
        """
        result_str = str(self.result)
        if len(result_str) > max_result_len:
            result_str = result_str[:max_result_len] + "..."
        d: dict[str, Any] = {
            "type": "tool_end",
            "id": self.id,
            "name": self.name,
            "result": result_str,
            "status": self.status,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class ToolExtractedEvent:
    """Special content extracted from a tool result.

    Emitted when a registered ToolExtractor successfully extracts
    meaningful data from a tool's output. For example, extracting
    a reflection from think_tool or a todo list from write_todos.

    Attributes:
        tool_name: Name of the tool the content was extracted from.
        extracted_type: Type identifier for the extracted content
            (e.g., "reflection", "todos", "canvas_item").
        data: The extracted data.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    tool_name: str
    extracted_type: str
    data: Any
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "extraction",
            "tool_name": self.tool_name,
            "extracted_type": self.extracted_type,
            "data": self.data,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class InterruptEvent:
    """Human-in-the-loop interrupt requiring user decision.

    Emitted when the graph hits an interrupt point and requires
    user input to continue. Use create_resume_input() to create
    the appropriate input to resume execution.

    Attributes:
        action_requests: List of actions requiring user approval.
            Each item contains 'tool', 'tool_call_id', 'args', etc.
        review_configs: Configuration for how actions should be reviewed.
            Each item may contain 'allowed_decisions' list.
        raw_value: The original interrupt value for custom handling.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    action_requests: list[dict[str, Any]]
    review_configs: list[dict[str, Any]]
    raw_value: Any = None
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def needs_approval(self) -> bool:
        """Check if this interrupt has action requests needing approval."""
        return len(self.action_requests) > 0

    @property
    def allowed_decisions(self) -> set[str]:
        """Get the set of allowed decision types from review configs.

        Defaults to ``{"approve", "reject", "edit", "respond"}`` when no
        review configs are present — this matches the deepagents 0.6+
        decision verb set.
        """
        allowed = set()
        for config in self.review_configs:
            allowed.update(config.get("allowed_decisions", []))
        if not allowed:
            allowed = {"approve", "reject", "edit", "respond"}
        return allowed

    def build_decisions(
        self,
        decision_type: str,
        args_modifier: Any = None,
        *,
        response: str | None = None,
        use_edited_action: bool = True,
    ) -> list[dict[str, Any]]:
        """Build decision list for resuming from this interrupt.

        Args:
            decision_type: The decision type — one of ``"approve"``,
                ``"reject"``, ``"edit"``, or ``"respond"``.
            args_modifier: Optional function to modify args for "edit"
                decisions. Takes original args dict and returns modified
                args dict.
            response: Text reply for ``"respond"`` decisions. Sent back
                to the agent as the user's response in place of the
                tool call.
            use_edited_action: When True (default, LangGraph >= 1.1 /
                deepagents >= 0.5 shape), edit decisions emit
                ``{"type": "edit", "edited_action": {"name", "args"}}``.
                When False, the legacy ``{"type": "edit", "args": ...}``
                shape is emitted for older runtimes.

        Returns:
            List of decision dicts ready for create_resume_input().
            Order matches ``action_requests`` — required by the LangGraph
            interrupt API.

        Example:
            # Approve all actions
            decisions = interrupt.build_decisions("approve")

            # Edit args before approval (modern shape)
            decisions = interrupt.build_decisions(
                "edit", args_modifier=lambda a: {**a, "safe": True}
            )

            # Reply with text instead of running the tool
            decisions = interrupt.build_decisions(
                "respond", response="Please rephrase that."
            )
        """
        decisions = []
        for action in self.action_requests:
            decision: dict[str, Any] = {"type": decision_type}

            if decision_type == "edit" and args_modifier is not None:
                original_args = action.get("args", {})
                modified_args = args_modifier(original_args)
                if use_edited_action:
                    tool_name = action.get("tool") or action.get("name")
                    decision["edited_action"] = {
                        "name": tool_name,
                        "args": modified_args,
                    }
                else:
                    decision["args"] = modified_args
            elif decision_type == "respond" and response is not None:
                decision["args"] = {"response": response}

            decisions.append(decision)

        return decisions

    def create_resume(
        self,
        decision_type: str,
        args_modifier: Any = None,
        *,
        response: str | None = None,
        use_edited_action: bool = True,
    ) -> Any:
        """Create resume input to continue from this interrupt.

        Convenience wrapper around ``build_decisions()`` +
        ``create_resume_input()``. See ``build_decisions()`` for the
        full parameter set.

        Example:
            # Approve and resume in one call
            resume_input = interrupt.create_resume("approve")
            for event in parser.parse(graph.stream(resume_input, config=config)):
                handle_event(event)
        """
        # Import here to avoid circular dependency
        from .resume import create_resume_input

        decisions = self.build_decisions(
            decision_type,
            args_modifier,
            response=response,
            use_edited_action=use_edited_action,
        )
        return create_resume_input(decisions=decisions)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "interrupt",
            "action_requests": self.action_requests,
            "review_configs": self.review_configs,
            "allowed_decisions": list(self.allowed_decisions),
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class StateUpdateEvent:
    """Raw state update for non-message state keys.

    Emitted when include_state_updates=True and the update contains
    state keys other than "messages".

    Attributes:
        node: The name of the graph node that produced this update.
        key: The state key that was updated.
        value: The new value for the state key.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    node: str
    key: str
    value: Any
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "state_update",
            "node": self.node,
            "key": self.key,
            "value": self.value,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class UsageEvent:
    """Token usage metadata from a model invocation.

    Emitted when an AIMessage contains usage_metadata with token counts.
    These are per-invocation counts (not cumulative); consumers should
    accumulate them if a running total is desired.

    Attributes:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        total_tokens: Sum of input and output tokens.
        cache_read_tokens: Cached prompt tokens read (from
            ``input_token_details.cache_read``). 0 when unavailable.
        cache_creation_tokens: Tokens used to create a cache entry
            (from ``input_token_details.cache_creation``). 0 when
            unavailable.
        node: The name of the graph node that produced this usage.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    node: str | None = None
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "usage",
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "node": self.node,
        }
        if self.cache_read_tokens:
            d["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_creation_tokens:
            d["cache_creation_tokens"] = self.cache_creation_tokens
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class CustomEvent:
    """Custom data emitted via ``get_stream_writer()``.

    Emitted when the graph uses ``stream_mode="custom"`` or when
    custom chunks appear in multi-mode streaming. The data field
    contains whatever the agent wrote with ``get_stream_writer()``.

    Attributes:
        data: The custom data. Can be any type.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    data: Any
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "custom",
            "data": self.data,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class ValuesEvent:
    """Full state snapshot from ``stream_mode="values"`` (v2 streaming).

    Emitted when using LangGraph v2 streaming with the "values" stream type.
    Contains the complete graph state after each node execution.

    Attributes:
        data: The full state snapshot dict.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    data: dict[str, Any]
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "values",
            "data": self.data,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class DebugEvent:
    """Debug, checkpoint, or task trace from v2 streaming.

    Emitted when using LangGraph v2 streaming with "debug", "checkpoints",
    or "tasks" stream types. The ``debug_type`` field discriminates between them.

    Attributes:
        data: The raw trace data.
        debug_type: Discriminator — one of "debug", "checkpoint", "task".
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    data: Any
    debug_type: str = "debug"
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "debug",
            "debug_type": self.debug_type,
            "data": self.data,
        }
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class ReasoningEvent:
    """Reasoning / thinking content from an AI message.

    Emitted for the langchain-core standardized ``reasoning`` content
    block (Anthropic thinking, OpenAI reasoning summaries) and for
    ``think_tool`` reflections. Consumers can render reasoning
    differently from final answer text (e.g., greyed out, collapsible).

    Attributes:
        content: The reasoning text.
        source: Where the reasoning came from — "content_block" for
            langchain-core reasoning blocks, "think_tool" for
            ThinkToolExtractor output.
        node: The graph node that produced the reasoning.
        agent_name: The deep agent name, if from a subagent.
        is_subagent: True when ``ls_agent_type == "subagent"`` was set
            on the run metadata (deepagents >= 0.6).
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    content: str
    source: Literal["content_block", "think_tool"] = "content_block"
    node: str | None = None
    agent_name: str | None = None
    is_subagent: bool = False
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "reasoning",
            "content": self.content,
            "source": self.source,
            "node": self.node,
        }
        if self.agent_name is not None:
            d["agent_name"] = self.agent_name
        if self.is_subagent:
            d["is_subagent"] = True
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class DisplayEvent:
    """Rich inline content from a ``display_inline``-style tool.

    Emitted when a tool returns structured display metadata — typically
    a serialized dataframe, matplotlib/plotly figure, image, or HTML
    blob that the frontend should render inline rather than treat as
    plain text.

    Tool authors typically produce this by returning a JSON string with
    ``{"display_type": str, "data": str, "title": str, "status": str}``
    from a tool. The parser's ``DisplayInlineExtractor`` handles the
    parse and yields this event.

    Attributes:
        display_type: The display kind — e.g., "dataframe", "image",
            "plotly", "html", "json". Consumer-defined.
        data: The serialized payload (HTML string, base64 image, plotly
            JSON, etc.). Format depends on ``display_type``.
        title: Optional display title.
        status: "success" or "error".
        error: Optional error message when ``status == "error"``.
        tool_name: The tool that produced the display.
        tool_call_id: The originating tool call ID.
        node: The graph node that produced the event.
        namespace: The subgraph namespace path, if from a subgraph.
        timestamp: When the event was created.
    """
    display_type: str
    data: Any
    title: str | None = None
    status: str = "success"
    error: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    node: str | None = None
    namespace: tuple[str, ...] | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        d: dict[str, Any] = {
            "type": "display",
            "display_type": self.display_type,
            "data": self.data,
            "status": self.status,
        }
        if self.title is not None:
            d["title"] = self.title
        if self.error is not None:
            d["error"] = self.error
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.node is not None:
            d["node"] = self.node
        if self.namespace is not None:
            d["namespace"] = list(self.namespace)
        return d


@dataclass
class CompleteEvent:
    """Stream completed successfully.

    Emitted when the graph stream finishes without error.

    Attributes:
        timestamp: When the stream completed.
    """
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {"type": "complete"}


@dataclass
class ErrorEvent:
    """Error occurred during streaming.

    Emitted when an error occurs during stream processing.
    The parser catches exceptions and yields ErrorEvent instead
    of raising, allowing graceful error handling.

    Attributes:
        error: Human-readable error message.
        exception: The original exception if available.
        timestamp: When the error occurred.
    """
    error: str
    exception: Exception | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "error",
            "error": self.error,
        }


def event_to_dict(
    event: "StreamEvent", *, max_result_len: int = 500
) -> dict[str, Any]:
    """Convert any StreamEvent to a JSON-serializable dict.

    This is a convenience function for web APIs that need to serialize
    events to JSON for transmission over WebSockets, HTTP responses, etc.

    Args:
        event: Any StreamEvent instance.
        max_result_len: Maximum length for the ``result`` string of a
            ``ToolCallEndEvent`` (truncated with an ellipsis if longer).
            Ignored for every other event type. Raise this when a UI
            needs to show full tool output instead of a preview.

    Returns:
        A dict with a "type" key and event-specific fields.

    Example:
        for event in parser.parse(stream):
            await websocket.send_json(event_to_dict(event))

        # Show full tool results in a rich UI:
        event_to_dict(tool_end_event, max_result_len=50_000)
    """
    to_dict = getattr(event, "to_dict", None)
    if to_dict is None:
        return {"type": "unknown", "event": str(event)}
    # Only ToolCallEndEvent.to_dict accepts max_result_len.
    if isinstance(event, ToolCallEndEvent):
        return to_dict(max_result_len=max_result_len)
    return to_dict()


# Union type for all events - useful for type hints
StreamEvent = Union[
    ContentEvent,
    ReasoningEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    DisplayEvent,
    InterruptEvent,
    StateUpdateEvent,
    UsageEvent,
    CustomEvent,
    ValuesEvent,
    DebugEvent,
    CompleteEvent,
    ErrorEvent,
]
