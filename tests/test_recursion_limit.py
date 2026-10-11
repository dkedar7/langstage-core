"""A graph's own recursion limit survives the AG-UI adapter.

``ag_ui_langgraph`` builds each run's config with ``ensure_config()``, which stamps
LangGraph's default ``recursion_limit=25``; that passed config overrides the graph's
own default. So a limit set with ``graph.with_config(recursion_limit=...)`` was
ignored, and so was deepagents' built-in 1000: any agent run through LangStage
stopped with ``GraphRecursionError`` after 25 steps. ``build_agent`` now carries the
graph's limit into the adapter's config.
"""

import pytest

pytest.importorskip("ag_ui_langgraph")

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langstage_core.agui import run_turn

STEPS = 30  # more than LangGraph's default limit of 25


def _looping_graph():
    """One node that loops STEPS times, then answers."""

    def step(state):
        n = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
        return {"messages": [AIMessage(content="done" if n + 1 >= STEPS else "", id=f"m{n}")]}

    b = StateGraph(MessagesState)
    b.add_node("step", step)
    b.add_edge(START, "step")
    b.add_conditional_edges("step", lambda s: END if s["messages"][-1].content == "done" else "step")
    return b.compile()


def test_default_limit_still_applies_without_one():
    r = run_turn(_looping_graph(), "go")
    assert r.outcome == "error"
    assert "Recursion limit of 25" in (r.error or "")


def test_graph_with_config_limit_is_honored():
    r = run_turn(_looping_graph().with_config(recursion_limit=100), "go")
    assert r.outcome == "complete", r.error
    assert r.text.endswith("done")


def test_explicit_config_wins_over_the_graphs_limit():
    from langstage_core.agui import build_agent

    agent = build_agent(_looping_graph().with_config(recursion_limit=100), config={"recursion_limit": 10})
    r = run_turn(agent, "go")
    assert r.outcome == "error"
    assert "Recursion limit of 10" in (r.error or "")
