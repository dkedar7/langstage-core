"""A keyless, network-free demo agent that exercises **every** rich frame type.

The echo stub (:mod:`langstage_core.demo.stub`) is deliberately minimal — it only
ever emits ``content`` frames. But ``content`` is one of *eight* documented frame
types, and the rich ones (``tool_start`` / ``tool_end`` / ``extraction`` /
``reasoning`` / ``interrupt``) are the library's whole selling point. Before this
module the only keyless, documented path exercised exactly one of them, so an
adopter had no copy-pasteable example of a tool call flowing through to a
``tool_end`` / ``extraction`` frame, or of an ``interrupt`` → ``resume`` (gh #99).

This agent closes that gap. It is a *real* compiled LangGraph graph — a router
node plus a real :class:`langgraph.prebuilt.ToolNode` — streaming through the exact
same parser path a production agent uses, but the "model" is a local, deterministic
fake, so it needs no network and no API key. Its behaviour is keyed off a trigger
in the user's message:

===================  ==================================================  ====================================
Message contains     What the agent does                                 Frame types produced
===================  ==================================================  ====================================
``"use a tool"``     calls the built-in :data:`demo_lookup` tool via a   ``tool_start`` → ``tool_end`` →
                     real ``ToolNode``, then summarizes the result       ``extraction`` (with the paired
                                                                         :class:`DemoLookupExtractor`) →
                                                                         ``content``
``"think"``          streams a chain-of-thought then the answer          ``reasoning`` → ``content``
``"ask me"``         raises ``interrupt(...)`` (HITL); resume to finish  ``interrupt`` → (resume) ``content``
anything else        echoes the message token-by-token                   ``content``
===================  ==================================================  ====================================

Every path ends with the terminal ``complete`` frame (or ``error`` on failure).

Why a real ``ToolNode`` and not a hand-rolled ``AIMessage(tool_calls=...)`` +
manually-appended ``ToolMessage``? Because the AG-UI adapter emits streaming
``ToolCall*`` events **only** when a tool runs through a real ``ToolNode`` — a
manually-appended ``ToolMessage`` arrives via the snapshot path instead (the very
distinction gh #91 / core 1.0.24 hinges on). A new adopter has no way to know that
tribal fact; this demo bakes the correct wiring in.

Point any surface at it with the standard spec string::

    LANGSTAGE_AGENT_SPEC=langstage_core.demo.tools:graph

serve it keyless from the CLI::

    langstage-agui --demo=tools

or drive it in code (this is the runnable snippet the README's frame-type list
points at)::

    import asyncio
    from langstage_core.agui import build_agent, iter_event_frames
    from langstage_core.demo.tools import create_tool_demo_agent, demo_extractors

    agent = build_agent(create_tool_demo_agent())

    async def main():
        for turn in ("hello", "think about it", "use a tool"):
            async for frame in iter_event_frames(agent, turn, "s1",
                                                 extractors=demo_extractors()):
                print(frame["type"])

    asyncio.run(main())

``langgraph`` and ``langchain-core`` are imported lazily — importing this module
stays dependency-free; only building the agent requires them (the lightweight
``[stub]`` extra, or any host surface, already provides both).
"""
from __future__ import annotations

import json
from typing import Any

DEFAULT_NAME = "Tool Demo Agent"
DEFAULT_REPLY_PREFIX = "(demo agent) You said: "

# The documented trigger phrases (case-insensitive substrings). Exported so tests
# and docs reference one source of truth instead of hard-coding the strings.
TOOL_TRIGGER = "use a tool"
REASONING_TRIGGER = "think"
INTERRUPT_TRIGGER = "ask me"

# The keyless "fact" the built-in tool returns — deterministic, so the demo doubles
# as a regression fixture (the extraction frame's data is stable across runs).
_DEMO_ANSWER = "42"

# A HumanInterrupt-shaped payload for the "ask me" path. `_normalize_interrupt`
# (agui) turns a dict already keyed `action_requests` into a populated interrupt
# frame, so a caller sees a real action request + allowed decisions to resume with.
_INTERRUPT_REQUEST: dict[str, Any] = {
    "action_requests": [
        {"action": "ask_user", "args": {"question": "What should I call you?"}}
    ],
    "allowed_decisions": ["respond", "approve"],
}

_GRAPH_CACHE: Any = None


class DemoLookupExtractor:
    """Extractor paired with the built-in :data:`demo_lookup` tool.

    Implements the :class:`~langstage_core.extractors.base.ToolExtractor` protocol
    (``tool_name`` / ``extracted_type`` / ``extract``) — the same machinery hosts
    use for their own tools — so a ``demo_lookup`` result surfaces as an
    ``extraction`` frame of ``extracted_type="demo_fact"``. This is what makes the
    ``extraction`` frame reachable keyless; pass it (via :func:`demo_extractors`) to
    ``iter_event_frames`` / ``iter_chunk_frames``' ``extractors=`` argument.
    """

    tool_name = "demo_lookup"
    extracted_type = "demo_fact"

    def extract(self, content: Any) -> dict[str, Any] | None:
        """Pull ``{query, answer}`` out of a ``demo_lookup`` result.

        Accepts the tool's JSON string (or an already-parsed dict); returns
        ``None`` for anything that doesn't look like a ``demo_lookup`` payload, so
        no spurious ``extraction`` frame fires.
        """
        data: Any = content
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(data, dict) and "answer" in data:
            return {"query": data.get("query"), "answer": data["answer"]}
        return None


def demo_extractors() -> list[Any]:
    """The built-in extractors that pair with this demo's tools.

    Pass to ``iter_event_frames`` / ``iter_chunk_frames``' ``extractors=`` so the
    ``demo_lookup`` result renders as an ``extraction`` frame::

        async for frame in iter_event_frames(agent, "use a tool", "s1",
                                             extractors=demo_extractors()):
            ...
    """
    return [DemoLookupExtractor()]


def _last_human_text(messages: list[Any]) -> str:
    """The most recent human message's text (''  if none)."""
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _last_tool_result(messages: list[Any]) -> str | None:
    """The tool result produced *this turn* (after the last human message), or None.

    Scans back only to the last human message so a previous turn's tool result
    doesn't leak into the current turn's routing.
    """
    for message in reversed(messages):
        if getattr(message, "type", None) == "tool":
            content = message.content
            return content if isinstance(content, str) else str(content)
        if getattr(message, "type", None) == "human":
            return None
    return None


def create_tool_demo_agent(
    *,
    name: str = DEFAULT_NAME,
    checkpointer: Any = None,
):
    """Build the keyless rich-frame demo agent.

    Args:
        name: Display name surfaced in host UIs.
        checkpointer: Optional LangGraph checkpointer. Defaults to ``None`` — like
            the echo stub, the graph compiles **without** one so the config-free
            keyless paths (content / reasoning / tool) just work. The ``interrupt``
            path needs threaded state to resume, but every documented resume route
            supplies it: ``build_agent`` attaches an ``InMemorySaver`` when the
            graph has none, and the AG-UI wire always passes a ``thread_id``. Pass
            one explicitly to persist across process restarts.

    Returns:
        A compiled LangGraph graph (``messages``-state) that routes on the trigger
        phrases documented at the module level and emits every rich frame type.

    Raises:
        RuntimeError: If ``langgraph`` / ``langchain-core`` are not installed.
    """
    try:
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
        from langchain_core.outputs import (
            ChatGeneration,
            ChatGenerationChunk,
            ChatResult,
        )
        from langchain_core.tools import tool
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode
        from langgraph.types import interrupt
    except ImportError as e:  # pragma: no cover — exercised only without the deps
        raise RuntimeError(
            "The tool demo agent needs langgraph + langchain-core, which aren't "
            "installed. Install the lightweight stub extra "
            '(`pip install "langstage-core[stub]"`) — or your stage\'s own demo '
            f"extra, or plain `pip install langgraph`: {e}"
        ) from e

    from typing import Iterator, List, Optional

    @tool
    def demo_lookup(query: str) -> str:
        """Look up a fact in the demo knowledge base (keyless, deterministic)."""
        return json.dumps({"query": query, "answer": _DEMO_ANSWER, "source": "demo-kb"})

    def _reasoning_and_reply(messages: List[BaseMessage]) -> tuple[list[str], str]:
        """Decide the (reasoning deltas, answer text) the fake model should stream.

        Pure function of the conversation so ``_generate`` and ``_stream`` can't
        drift. The node handles the tool-call and interrupt control flow; the model
        only ever streams ``content`` (and ``reasoning`` on the ``think`` trigger).
        """
        tool_result = _last_tool_result(messages)
        if tool_result is not None:
            return [], (
                f"The demo tool returned {tool_result}. That completes the "
                "tool_start -> tool_end -> extraction flow."
            )
        trigger = _last_human_text(messages).lower()
        if REASONING_TRIGGER in trigger:
            return (
                [
                    "Let me reason about this step by step. ",
                    "The demo streams this as a `reasoning` frame, kept separate "
                    "from the answer. ",
                ],
                "Done reasoning — that was the `reasoning` frame demo.",
            )
        return [], DEFAULT_REPLY_PREFIX + _last_human_text(messages)

    class DemoChatModel(BaseChatModel):
        """A no-API chat model that streams a deterministic content (and, on the
        ``think`` trigger, ``reasoning``) reply — the keyless stand-in for a real
        LLM. Implements ``_stream`` so tokens surface one-by-one through the parser,
        exactly as a streaming provider would."""

        @property
        def _llm_type(self) -> str:
            return "tool-demo-stub"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            _, text = _reasoning_and_reply(messages)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            reasoning, text = _reasoning_and_reply(messages)
            # Reasoning deltas carry the chain-of-thought in the `reasoning_content`
            # additional_kwarg (the DeepSeek / Qwen / xAI shape the AG-UI adapter's
            # resolve_reasoning_content recognizes), with empty text content so they
            # surface as `reasoning` frames, never as the answer. (gh #71 / #99)
            for delta in reasoning:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="", additional_kwargs={"reasoning_content": delta}
                    )
                )
                if run_manager is not None:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            tokens = text.split(" ")
            for i, token in enumerate(tokens):
                piece = token if i == len(tokens) - 1 else token + " "
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                if run_manager is not None:
                    run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk

    model = DemoChatModel()

    def _agent(state: MessagesState) -> dict:
        messages = state["messages"]
        trigger = _last_human_text(messages).lower()
        tool_ran = _last_tool_result(messages) is not None

        # Interrupt path (HITL). `interrupt()` raises a GraphInterrupt on the first
        # pass (yielding the `interrupt` frame) and returns the resume decision on
        # the second — at which point we finish the turn with a `content` reply.
        # Only before any tool has run, so the tool path's second pass can't re-fire.
        if INTERRUPT_TRIGGER in trigger and not tool_ran:
            decision = interrupt(_INTERRUPT_REQUEST)
            return {
                "messages": [
                    AIMessage(content=f"Resumed. Your decision was: {decision}.")
                ]
            }

        # Tool path (first pass): return a tool-call AIMessage and let the real
        # ToolNode execute it — the only wiring that makes the adapter emit
        # streaming ToolCall* events (core 1.0.24 / gh #91). The conditional edge
        # routes to "tools"; the second pass sees the ToolMessage and summarizes it.
        if TOOL_TRIGGER in trigger and not tool_ran:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "demo_lookup",
                                "args": {"query": _last_human_text(messages)},
                                "id": "demo_lookup_1",
                            }
                        ],
                    )
                ]
            }

        # Content / reasoning / tool-summary: the streaming model handles the reply.
        return {"messages": [model.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", _agent)
    builder.add_node("tools", ToolNode([demo_lookup]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=checkpointer)
    graph.name = name
    return graph


def __getattr__(attr: str) -> Any:
    # ``graph`` is built lazily on first access so plain imports of this module
    # never require langgraph — but ``load_agent_spec("...demo.tools:graph")`` gets
    # a ready compiled agent (mirrors demo.stub).
    if attr == "graph":
        global _GRAPH_CACHE
        if _GRAPH_CACHE is None:
            _GRAPH_CACHE = create_tool_demo_agent()
        return _GRAPH_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
