"""The shared default agent factory.

``deepagents`` and ``langgraph`` are imported lazily inside the factory so the
base ``langgraph-stream-parser`` install stays dependency-light — importing
this module does not require the ``[demo]`` extra; only *calling* the factory
does.
"""
import re
import warnings
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

# OpenAI (and OpenAI-compatible providers, e.g. via OpenRouter) require the
# message `name` field to match ^[^\s<|\\/>]+$ — no spaces or <|\/>. An agent's
# display name flows into that field, so a name with a space 400s on the second
# turn. Slugify unsafe names so any human-readable name "just works".
_UNSAFE_AGENT_NAME = re.compile(r"[\s<|\\/>]+")


def _safe_agent_name(name: str) -> str:
    """Return a name safe for the LLM message `name` field (slugify spaces /
    <|\\/>), warning if it had to change."""
    safe = _UNSAFE_AGENT_NAME.sub("-", name).strip("-") or "agent"
    if safe != name:
        warnings.warn(
            f"Agent name {name!r} contains characters not allowed in the LLM "
            f"message 'name' field (spaces or <|\\/>); using {safe!r}. "
            "OpenAI-compatible providers reject the original on multi-turn calls.",
            stacklevel=3,
        )
    return safe

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a \
filesystem workspace.

- Explain what you are about to do before using a tool, and summarize results \
after.
- Use the todo tool to plan multi-step work and keep the user informed of \
progress.
- Be proactive in exploring the workspace when it helps, but never run risky \
or destructive actions without explicit user consent.

The workspace is your sandbox — read, create, and organize files to help the \
user with their tasks."""


def create_default_agent(
    workspace: str | Path = ".",
    *,
    model: str | None = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: list[Any] | None = None,
    name: str = "deep-agent",
    virtual_mode: bool = True,
    checkpointer: Any = None,
    **extra: Any,
):
    """Create a filesystem-backed deep agent for hosts to use as a default.

    This owns the boilerplate (filesystem backend, checkpointer, model wiring)
    that every host's default agent repeated. Hosts supply their own prompt,
    tools, and any deepagents extras (``middleware``, ``interrupt_on``, ...) via
    ``**extra``.

    Args:
        workspace: Root directory the agent's filesystem backend operates in.
        model: Model identifier passed to deepagents (e.g.
            ``"anthropic:claude-sonnet-4-6"``). Requires the matching provider
            API key in the environment. Pass ``None`` to let deepagents pick its
            own default model.
        system_prompt: System prompt for the agent.
        tools: Extra tools to register. Hosts inject their domain tools here
            (notebook cells, canvas, ...). Defaults to none — filesystem +
            built-in deepagents tools only.
        name: Display name surfaced in host UIs.
        virtual_mode: Passed to deepagents' ``FilesystemBackend``.
        checkpointer: Checkpointer for the agent. Defaults to an
            ``InMemorySaver`` when not provided.
        **extra: Forwarded verbatim to ``deepagents.create_deep_agent`` —
            e.g. ``middleware=[...]`` or ``interrupt_on={"bash": True}``.

    Returns:
        A compiled deep agent (LangGraph ``CompiledGraph``).

    Raises:
        RuntimeError: If the ``deepagents`` extra is not installed.

    Example:
        from langgraph_stream_parser.demo import create_default_agent
        agent = create_default_agent("./workspace")

        # A host layering its own tools + middleware on the shared boilerplate:
        agent = create_default_agent(
            "./ws", name="Cowork Dash", system_prompt=PROMPT,
            tools=MY_TOOLS, middleware=[CanvasMiddleware()],
            interrupt_on={"bash": True},
        )
    """
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise RuntimeError(
            "create_default_agent requires the 'deepagents' extra. "
            "Install it with: pip install langgraph-stream-parser[demo]"
        ) from exc

    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=virtual_mode)

    kwargs: dict[str, Any] = dict(
        name=_safe_agent_name(name),
        system_prompt=system_prompt,
        backend=backend,
        tools=tools or [],
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        **extra,
    )
    if model is not None:
        kwargs["model"] = model

    return create_deep_agent(**kwargs)
