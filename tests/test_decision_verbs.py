"""One canonical HITL decision vocabulary (langstage-vscode #114 / #117).

Core advertises LangChain HumanInTheLoopMiddleware's verbs (``approve`` / ``edit`` /
``reject`` / ``respond``), accepts LangGraph's legacy ``HumanResponse`` verbs
(``accept`` / ``ignore`` / ``response``) as aliases, and forwards the canonical verb to
the graph inside the ``{"decisions": [...]}`` envelope.
"""
from typing import List

import pytest

from langstage_core import (
    DECISION_ALIASES,
    DECISION_VERBS,
    create_resume_input,
    is_allowed_decision,
    normalize_decision,
)
from langstage_core.resume import canonicalize_resume

# ── the helper ───────────────────────────────────────────────────────────


def test_canonical_set_and_aliases():
    assert DECISION_VERBS == ("approve", "edit", "reject", "respond")
    assert DECISION_ALIASES == {"accept": "approve", "ignore": "reject", "response": "respond"}


@pytest.mark.parametrize("verb, canon", [
    ("approve", "approve"), ("accept", "approve"), ("ACCEPT ", "approve"),
    ("reject", "reject"), ("ignore", "reject"),
    ("respond", "respond"), ("response", "respond"),
    ("edit", "edit"),
])
def test_both_vocabularies_normalize_to_the_canonical_verb(verb, canon):
    assert normalize_decision(verb) == canon
    assert normalize_decision(verb, list(DECISION_VERBS)) == canon


def test_unknown_or_empty_verbs_are_rejected():
    for bad in ("yes", "", "   ", None, 3):
        assert normalize_decision(bad) is None
        assert not is_allowed_decision(bad, list(DECISION_VERBS))


def test_allowed_restricts_and_accepts_either_vocabulary():
    # an approve-only interrupt (vscode #114's shape)
    assert is_allowed_decision("approve", ["approve"])
    assert is_allowed_decision("accept", ["approve"])
    assert not is_allowed_decision("edit", ["approve"])
    assert not is_allowed_decision("reject", ["approve"])
    # the demo's [respond, approve] interrupt refuses reject (vscode #117)
    assert not is_allowed_decision("reject", ["respond", "approve"])
    assert normalize_decision("response", ["respond", "approve"]) == "respond"
    # an allowed list written in the LEGACY vocabulary works too
    assert normalize_decision("approve", ["accept", "ignore"]) == "approve"
    assert normalize_decision("reject", ["accept", "ignore"]) == "reject"
    assert normalize_decision("edit", ["accept", "ignore"]) is None


def test_custom_verb_only_when_the_interrupt_advertises_it():
    assert normalize_decision("defer") is None
    assert normalize_decision("defer", ["approve", "defer"]) == "defer"


def test_canonicalize_resume_rewrites_aliases_without_mutating():
    payload = {"decisions": [{"type": "accept"}, {"type": "ignore", "message": "no"},
                             {"type": "edit", "edited_action": {"name": "x", "args": {}}}]}
    out = canonicalize_resume(payload)
    assert [d["type"] for d in out["decisions"]] == ["approve", "reject", "edit"]
    assert out["decisions"][1]["message"] == "no"
    assert payload["decisions"][0]["type"] == "accept"  # input untouched
    canonical = {"decisions": [{"type": "approve"}]}
    assert canonicalize_resume(canonical) is canonical
    for other in ("yes", ["accept"], {"answer": "accept"}, None):
        assert canonicalize_resume(other) == other


# ── end to end: the graph receives the canonical verb ────────────────────


def _echo_decision_graph():
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    def ask(state):
        got = interrupt({
            "action_requests": [{"name": "delete", "args": {}}],
            "review_configs": [{"action_name": "delete", "allowed_decisions": ["approve", "reject"]}],
        })
        return {"messages": [AIMessage(content=f"decision={got['decisions'][0]['type']}")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile()


@pytest.mark.parametrize("resume", [
    {"decisions": [{"type": "accept"}]},
    create_resume_input(decisions=[{"type": "accept"}]),
    {"decisions": [{"type": "approve"}]},
])
async def test_alias_reaches_the_graph_as_the_canonical_verb(resume):
    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    from langstage_core.agui import build_agent, collect_event_frames

    agent = build_agent(_echo_decision_graph())
    first = await collect_event_frames(agent, "go", "t1")
    assert first.outcome == "interrupted"
    assert first.interrupt["allowed_decisions"] == ["reject", "approve"]
    done = await collect_event_frames(agent, "", "t1", resume=resume)
    assert done.text == "decision=approve"


async def test_real_hitl_middleware_accepts_the_legacy_alias():
    pytest.importorskip("langchain.agents.middleware", reason="needs langchain>=1")
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    from langstage_core.agui import build_agent, collect_event_frames

    @tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"deleted {path}"

    class _Model(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None,
                      **kwargs) -> ChatResult:
            if any(isinstance(m, ToolMessage) for m in messages):
                msg = AIMessage(content="Done.")
            else:
                msg = AIMessage(content="", tool_calls=[
                    {"name": "delete_file", "args": {"path": "a.txt"}, "id": "c1"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    graph = create_agent(_Model(), tools=[delete_file], middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"delete_file": {"allowed_decisions": ["approve", "reject"]}}),
    ])
    agent = build_agent(graph)
    first = await collect_event_frames(agent, "delete a.txt", "t1")
    assert first.outcome == "interrupted"
    allowed = first.interrupt["allowed_decisions"]
    assert allowed == ["reject", "approve"]
    assert is_allowed_decision("accept", allowed) and not is_allowed_decision("edit", allowed)
    # "accept" would make the middleware raise; core forwards it as "approve".
    done = await collect_event_frames(agent, "", "t1", resume={"decisions": [{"type": "accept"}]})
    assert done.outcome == "complete", done.error
    assert any(tc["name"] == "delete_file" for tc in first.tool_calls + done.tool_calls)
    assert "Done." in done.text


async def test_legacy_human_interrupt_gets_its_own_verb_verbatim():
    # A legacy HumanInterrupt graph reads its own vocabulary: core must not rewrite it.
    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    from langstage_core.agui import build_agent, collect_event_frames

    def act(state):
        got = interrupt([{"action_request": {"action": "delete", "args": {}},
                          "config": {"allow_accept": True, "allow_ignore": True}}])
        return {"messages": [AIMessage(content=f"chose={got['decisions'][0]['type']}")]}

    b = StateGraph(MessagesState)
    b.add_node("act", act)
    b.add_edge(START, "act")
    b.add_edge("act", END)
    agent = build_agent(b.compile())
    first = await collect_event_frames(agent, "go", "t1")
    assert first.interrupt["allowed_decisions"] == ["reject", "approve"]
    done = await collect_event_frames(agent, "", "t1", resume={"decisions": [{"type": "accept"}]})
    assert done.text == "chose=accept"
