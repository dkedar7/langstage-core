"""A minimal, self-contained agent for the FastAPI streaming example.

Keyless and dependency-light, so the WebSocket example runs with just the
documented install — `pip install "langstage-core[agui]" uvicorn` — with no API
key and no extra packages. It echoes the user's message back through a one-node
LangGraph graph compiled with a checkpointer (iter_event_frames needs one for
threaded per-session state).

Swap in your own compiled LangGraph agent for real use — e.g. a `deepagents`
agent with real tools and a model::

    pip install deepagents langchain-anthropic   # + export ANTHROPIC_API_KEY
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    agent = create_deep_agent(name="Example", checkpointer=InMemorySaver())

build_agent() / iter_event_frames() take any ``CompiledGraph``.
"""

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


def respond(state: MessagesState) -> dict:
    user = state["messages"][-1].content if state["messages"] else ""
    return {"messages": [AIMessage(content=f"You said: {user}")]}


_builder = StateGraph(MessagesState)
_builder.add_node("respond", respond)
_builder.add_edge(START, "respond")
_builder.add_edge("respond", END)

# iter_event_frames needs a checkpointer for threaded state per session.
agent = _builder.compile(checkpointer=InMemorySaver())
