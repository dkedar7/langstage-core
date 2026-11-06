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
        A LangGraph ``Command``. Pass it to the 1.0 streaming path via
        ``iter_event_frames``/``iter_chunk_frames``'s ``resume=`` (they accept
        either this ``Command`` or the raw resume payload — gh #82), or to
        ``graph.stream(...)`` directly.

    Raises:
        ValueError: If neither decisions nor value is provided,
            or if both are provided.

    Example:
        # Resume an interrupt over the 1.0 AG-UI streaming path:
        from langstage_core.agui import iter_event_frames
        resume = create_resume_input(decisions=[{"type": "approve"}])
        async for frame in iter_event_frames(agent, "", thread_id, resume=resume):
            handle(frame)

        # A raw payload works too — resume={"decisions": [{"type": "approve"}]}.

        # Or resume a bare graph.stream() directly (a Command is what it expects):
        for event in graph.stream(resume, config=config):
            ...
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


# ── Decision verbs (HITL) ──────────────────────────────────────────────
#
# ONE canonical vocabulary (gh langstage-vscode #114 / #117):
#
# - Core ADVERTISES LangChain's HumanInTheLoopMiddleware verbs -- ``approve``, ``edit``,
#   ``reject``, ``respond`` -- in every ``interrupt`` frame's ``allowed_decisions``. A
#   legacy LangGraph ``HumanInterrupt`` (deprecated since LangGraph 1.0) is mapped onto
#   them from its ``config``: ``allow_accept`` -> approve, ``allow_edit`` -> edit,
#   ``allow_respond`` -> respond, ``allow_ignore`` -> reject.
# - Core ACCEPTS the legacy ``HumanResponse`` verbs as aliases on resume and in these
#   helpers: ``accept`` -> approve, ``ignore`` -> reject, ``response`` -> respond
#   (``edit`` is the same word in both).
# - Core TRANSLATES to the graph only where the graph needs it: when the pending
#   interrupt is a HumanInTheLoopMiddleware request (an ``action_requests`` payload),
#   each alias in the ``{"decisions": [...]}`` envelope is rewritten to its canonical
#   verb (the middleware raises on ``accept``). Any other interrupt -- a legacy
#   ``HumanInterrupt`` list, a custom ``interrupt(...)`` -- receives the payload
#   verbatim, because that graph reads its own vocabulary. The envelope shape is never
#   changed.

DECISION_VERBS: tuple[str, ...] = ("approve", "edit", "reject", "respond")
"""The canonical decision verbs core advertises in ``allowed_decisions``."""

DECISION_ALIASES: dict[str, str] = {"accept": "approve", "ignore": "reject", "response": "respond"}
"""Legacy LangGraph ``HumanResponse`` verbs -> the canonical verb they mean."""


def _canonical(verb: Any) -> str | None:
    if not isinstance(verb, str):
        return None
    v = verb.strip().lower()
    if not v:
        return None
    return DECISION_ALIASES.get(v, v)


def normalize_decision(verb: Any, allowed: Any = None) -> str | None:
    """The canonical decision verb for ``verb``, or ``None`` if it isn't allowed.

    ``verb`` may be canonical (``approve`` / ``edit`` / ``reject`` / ``respond``) or a
    legacy alias (``accept`` / ``ignore`` / ``response``), in any case. ``allowed`` is an
    interrupt frame's ``allowed_decisions`` (either vocabulary); pass ``None`` to check
    only that ``verb`` is a known decision. A verb outside the canonical four is accepted
    only when ``allowed`` lists it (a custom verb a graph advertises itself).

    Surfaces use this to validate a user's decision against the pending interrupt
    before resuming::

        verb = normalize_decision("accept", frame["allowed_decisions"])  # -> "approve"
        if verb is None:
            ...  # refuse: not a decision this interrupt allows
    """
    canon = _canonical(verb)
    if canon is None:
        return None
    if allowed is None:
        return canon if canon in DECISION_VERBS else None
    if isinstance(allowed, str):
        allowed = [allowed]
    permitted = {_canonical(a) for a in allowed}
    return canon if canon in permitted else None


def is_allowed_decision(verb: Any, allowed: Any) -> bool:
    """``True`` when ``verb`` (canonical or alias) is in ``allowed`` (either vocabulary)."""
    return normalize_decision(verb, allowed) is not None


def canonicalize_resume(resume: Any) -> Any:
    """Rewrite legacy-alias decision ``type``s in a ``{"decisions": [...]}`` resume
    payload to the canonical verbs (a copy; the input is not mutated).

    Anything else -- a bare value, a non-envelope dict, a decision with an unknown
    ``type`` -- is returned unchanged. ``iter_event_frames`` / ``iter_chunk_frames``
    apply this to a ``resume=`` answering a HumanInTheLoopMiddleware request, so an
    ``accept`` reaches the middleware as the ``approve`` it understands.
    """
    if not isinstance(resume, dict) or not isinstance(resume.get("decisions"), list):
        return resume
    out = []
    changed = False
    for d in resume["decisions"]:
        if isinstance(d, dict) and isinstance(d.get("type"), str):
            canon = _canonical(d["type"])
            if canon in DECISION_VERBS and canon != d["type"]:
                d = {**d, "type": canon}
                changed = True
        out.append(d)
    return {**resume, "decisions": out} if changed else resume


class InvalidDecisionError(ValueError):
    """A resume the pending HumanInTheLoopMiddleware request would refuse (gh #227).

    ``iter_event_frames`` / ``iter_chunk_frames`` and the served endpoint raise it, as a
    terminal ``error`` frame / ``RUN_ERROR``, *before* anything reaches the graph: a
    refused answer that did reach it stayed on the thread and was replayed on every
    later resume. The thread is left as it was, so the same interrupt can be answered
    again.
    """


def _is_hitl_request(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("action_requests"), list)


def _allowed_for(i: int, action: Any, interrupt: dict) -> list:
    """The verbs allowed for action ``i``: its own review_config, else the interrupt's."""
    configs = interrupt.get("review_configs") or []
    if not isinstance(configs, list):
        configs = []
    actions = interrupt.get("action_requests") or []
    name = action.get("name") if isinstance(action, dict) else None
    # HumanInTheLoopMiddleware builds review_configs index-aligned with action_requests.
    rc = configs[i] if len(configs) == len(actions) and i < len(configs) else next(
        (c for c in configs if isinstance(c, dict) and c.get("action_name") == name), None
    )
    allowed = rc.get("allowed_decisions") if isinstance(rc, dict) else None
    if allowed is None:
        allowed = interrupt.get("allowed_decisions")
    return list(allowed) if isinstance(allowed, (list, tuple)) and allowed else list(DECISION_VERBS)


def decision_problems(resume: Any, interrupt: Any) -> list[str]:
    """What is wrong with ``resume`` as the answer to ``interrupt``; ``[]`` when it's fine.

    ``interrupt`` is a LangChain HumanInTheLoopMiddleware request (``action_requests``
    plus ``review_configs``), either the graph's own value or an ``interrupt`` frame.
    The checks are the middleware's own: a ``{"decisions": [...]}`` envelope with one
    decision per action, each a type that action allows (aliases such as ``accept``
    count), an ``edit`` carrying ``edited_action`` ``{"name": str, "args": dict}``, a
    ``respond`` carrying a text ``message``. Any other interrupt (a legacy
    ``HumanInterrupt`` list, a custom ``interrupt(...)``) reads its own payload, so it
    is never checked here and always returns ``[]``.

    Surfaces can call it to re-prompt before resuming::

        problems = decision_problems({"decisions": decisions}, frame)
        if problems:
            ...  # show them and ask again
    """
    if not _is_hitl_request(interrupt):
        return []
    actions = interrupt["action_requests"]
    if not isinstance(resume, dict) or not isinstance(resume.get("decisions"), list):
        return [
            (
                'the answer must be {"decisions": [...]}, one decision per action; '
                f"got {type(resume).__name__}"
            )
        ]
    decisions = resume["decisions"]
    if len(decisions) != len(actions):
        return [f"{len(actions)} action(s) are waiting for a decision but {len(decisions)} were sent"]
    problems: list[str] = []
    for i, (decision, action) in enumerate(zip(decisions, actions)):
        name = action.get("name") if isinstance(action, dict) else None
        where = f"decision {i + 1}" + (f" ({name})" if name else "")
        if not isinstance(decision, dict):
            problems.append(f'{where}: must be an object like {{"type": "approve"}}')
            continue
        allowed = _allowed_for(i, action, interrupt)
        verb = normalize_decision(decision.get("type"), allowed)
        if verb is None:
            problems.append(
                f"{where}: type {decision.get('type')!r} is not allowed; "
                f"allowed: {', '.join(map(str, allowed))}"
            )
            continue
        if verb == "edit":
            ea = decision.get("edited_action")
            if not (
                isinstance(ea, dict) and isinstance(ea.get("name"), str) and ea["name"]
                and isinstance(ea.get("args"), dict)
            ):
                problems.append(
                    f'{where}: edit needs "edited_action": {{"name": "<tool>", "args": {{...}}}}'
                )
        elif verb == "respond" and not isinstance(decision.get("message"), str):
            problems.append(f'{where}: respond needs a text "message"')
        elif (
            verb == "reject"
            and decision.get("message") is not None
            and not isinstance(decision["message"], str)
        ):
            problems.append(f'{where}: a reject "message" must be text')
    return problems
