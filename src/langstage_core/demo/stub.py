"""A keyless, deterministic stub agent — the engine behind every ``--demo`` mode.

Every LangStage surface (langstage, langstage-cli, langstage-jupyter,
langstage-vscode) offers a demo mode so a new user can see the surface working
before configuring a real agent or any API key. This module is the single
shared implementation: a real compiled LangGraph graph with a checkpointer,
streaming token-by-token through the exact same parser path as a production
agent — but the "model" is a local echo, so it needs no network and no keys.

Point any surface at it with the standard spec string::

    LANGSTAGE_AGENT_SPEC=langstage_core.demo.stub:graph

or build a customized one in code::

    from langstage_core.demo import create_stub_agent
    agent = create_stub_agent(name="My Demo")

``langgraph`` and ``langchain-core`` are imported lazily — importing this
module stays dependency-free; only building the agent requires them (every
host surface already depends on both).
"""
from __future__ import annotations

from typing import Any

DEFAULT_REPLY_PREFIX = "(demo agent) You said: "
DEFAULT_NAME = "Demo Agent"

_GRAPH_CACHE: Any = None


def create_stub_agent(
    *,
    name: str = DEFAULT_NAME,
    reply_prefix: str = DEFAULT_REPLY_PREFIX,
    checkpointer: Any = None,
):
    """Build the echo stub agent.

    Args:
        name: Display name surfaced in host UIs.
        reply_prefix: Prepended to the echoed user message in every reply.
        checkpointer: Optional LangGraph checkpointer. Defaults to ``None`` —
            the graph is compiled **without** a checkpointer so it streams with
            zero ``config``/``thread_id`` bookkeeping (the documented keyless
            Quick Start just works). Pass one if you want multi-turn
            persistence; hosts that need threaded state (e.g. AG-UI) attach one
            automatically.

    Returns:
        A compiled LangGraph graph that replies to each user message with
        ``reply_prefix + <last human message>``, streamed token-by-token.

    Raises:
        RuntimeError: If ``langgraph`` / ``langchain-core`` are not installed.
    """
    try:
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
        from langgraph.graph import END, START, MessagesState, StateGraph
    except ImportError as e:  # pragma: no cover — exercised only without the deps
        raise RuntimeError(
            "The stub agent needs langgraph + langchain-core, which aren't "
            "installed. Install the lightweight stub extra "
            '(`pip install "langstage-core[stub]"`) — or your stage\'s '
            "own demo extra, or plain `pip install langgraph`: "
            f"{e}"
        ) from e

    from typing import Iterator, List, Optional

    def _last_human(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    class EchoChatModel(BaseChatModel):
        """A no-API chat model that echoes the user's last message.

        Implements ``_stream`` so that under LangGraph's ``messages`` stream
        mode the reply is emitted token-by-token, matching how a real chat
        model behaves through the parser.
        """

        @property
        def _llm_type(self) -> str:
            return "echo-stub"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            text = reply_prefix + _last_human(messages)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            text = reply_prefix + _last_human(messages)
            tokens = text.split(" ")
            for i, token in enumerate(tokens):
                piece = token if i == len(tokens) - 1 else token + " "
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
                if run_manager is not None:
                    run_manager.on_llm_new_token(piece, chunk=chunk)
                yield chunk

    model = EchoChatModel()

    def _respond(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", _respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)

    # No default checkpointer: a checkpointer requires a ``thread_id`` in the
    # stream config, and without one LangGraph raises — which the parser turns
    # into a lone ErrorEvent and a silent, blank reply for anyone running the
    # documented config-free Quick Start. Hosts that need threaded state (AG-UI)
    # attach a checkpointer themselves.
    graph = builder.compile(checkpointer=checkpointer)
    graph.name = name
    return graph


def __getattr__(attr: str) -> Any:
    # ``graph`` is built lazily on first access so plain imports of this module
    # never require langgraph — but ``load_agent_spec("...demo.stub:graph")``
    # gets a ready compiled agent.
    if attr == "graph":
        global _GRAPH_CACHE
        if _GRAPH_CACHE is None:
            _GRAPH_CACHE = create_stub_agent()
        return _GRAPH_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
