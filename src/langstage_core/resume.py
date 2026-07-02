"""
Utilities for resuming from interrupts.

These functions help create the appropriate input to resume
a LangGraph agent after an interrupt.
"""
from typing import Any


def create_resume_input(
    decisions: list[dict[str, Any]] | None = None,
    value: Any = None,
) -> Any:
    """Create input to resume a LangGraph agent from an interrupt.

    Use this function to create the appropriate Command object
    to resume execution after receiving an InterruptEvent.

    Args:
        decisions: List of decision dicts for action requests.
            Each decision should have a 'type' key (e.g., "approve",
            "reject", "edit") and may have additional fields.
        value: Simple value to resume with (for simple interrupts).
            Use this for interrupts that just need a boolean or
            simple confirmation.

    Returns:
        A LangGraph Command object ready to pass to graph.stream().

    Raises:
        ValueError: If neither decisions nor value is provided,
            or if both are provided.

    Example:
        # Resume with approval decisions
        resume_input = create_resume_input(
            decisions=[{"type": "approve"}]
        )
        for event in parser.parse(graph.stream(resume_input, config=config)):
            handle_event(event)

        # Resume with simple value
        resume_input = create_resume_input(value=True)
        for event in parser.parse(graph.stream(resume_input, config=config)):
            handle_event(event)
    """
    if decisions is None and value is None:
        raise ValueError("Must provide either 'decisions' or 'value'")
    if decisions is not None and value is not None:
        raise ValueError("Cannot provide both 'decisions' and 'value'")

    # Import Command lazily to avoid hard dependency
    from langgraph.types import Command

    if decisions is not None:
        return Command(resume={"decisions": decisions})
    else:
        return Command(resume=value)


def prepare_agent_input(
    message: str | None = None,
    decisions: list[dict[str, Any]] | None = None,
    raw_input: Any = None,
    context_parts: list[str] | None = None,
) -> Any:
    """Prepare input for a LangGraph agent.

    This function handles different input types and converts them
    to the appropriate format for LangGraph.

    Args:
        message: Regular user message string. Will be converted to
            {"messages": [{"role": "user", "content": message}]}.
        decisions: Resume decisions for interrupts. Will be converted
            to a Command object.
        raw_input: Raw input passed directly (for custom formats).
            Bypasses all processing.
        context_parts: Optional list of context lines to prepend to the
            message (e.g., timestamp, working directory). Only used with
            ``message``. Each part becomes a line before the message content.

    Returns:
        Prepared input for the agent.

    Raises:
        ValueError: If no input is provided or multiple inputs are provided.

    Example:
        # Regular message
        input_data = prepare_agent_input(message="Hello!")

        # Message with context
        input_data = prepare_agent_input(
            message="What time is it?",
            context_parts=["[Current time: 2025-01-01 12:00:00 UTC]"],
        )

        # Resume from interrupt
        input_data = prepare_agent_input(decisions=[{"type": "approve"}])

        # Custom format
        input_data = prepare_agent_input(raw_input={"custom": "data"})
    """
    # Count how many inputs are provided
    inputs_provided = sum([
        message is not None,
        decisions is not None,
        raw_input is not None,
    ])

    if inputs_provided == 0:
        raise ValueError("Must provide one of: message, decisions, or raw_input")
    if inputs_provided > 1:
        raise ValueError("Can only provide one of: message, decisions, or raw_input")

    # Handle raw input (pass through)
    if raw_input is not None:
        return raw_input

    # Handle regular message
    if message is not None:
        content = message
        if context_parts:
            content = "\n".join(context_parts) + "\n\n" + content
        return {"messages": [{"role": "user", "content": content}]}

    # Handle resume from interrupt
    if decisions is not None:
        return create_resume_input(decisions=decisions)

    # Should never reach here
    raise ValueError("Invalid input")
