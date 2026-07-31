"""The shared live-preflight primitive (ADR 0004).

``verify`` / ``averify`` run ONE real turn through the AG-UI adapter and report
whether the agent actually completed — the thing each surface's health check was
reinventing. A turn that errors (or a non-runnable object) must fail here, not
pass a static check and blow up at first chat.
Skipped unless the agui extra is installed (dev pulls it, so CI runs it).
"""

import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langstage_core import load_agent_spec
from langstage_core.agui import VerifyResult, averify, build_agent, verify

# asyncio_mode = "auto" (pyproject) runs the async tests as coroutines; the sync
# wrapper test stays sync, so no module-level asyncio mark here.


def _erroring_graph():
    """A compiled graph whose only node raises — the AG-UI adapter surfaces this
    as a RunError frame, so a real turn fails while a static load would pass."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    def boom(state):
        raise RuntimeError("tool exploded")

    b = StateGraph(MessagesState)
    b.add_node("boom", boom)
    b.add_edge(START, "boom")
    b.add_edge("boom", END)
    return b.compile()


async def test_averify_demo_stub_passes():
    # The keyless echo stub completes a turn and emits content -> ok.
    r = await averify(load_agent_spec("langstage_core.demo.stub:graph"))
    assert isinstance(r, VerifyResult)
    assert r.ok and bool(r) is True
    assert r.saw_complete and not r.saw_error
    assert r.content_chars > 0
    assert r.reason == "one turn completed cleanly"


async def test_averify_accepts_a_prebuilt_agent():
    # Passing an already-built LangGraphAgent must work too (not just a graph).
    agent = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    r = await averify(agent)
    assert r.ok


async def test_averify_erroring_agent_fails_not_raises():
    # A turn that errors is a FAILED preflight returned as data, never an
    # exception to the caller — and never a false green.
    r = await averify(_erroring_graph())
    assert r.ok is False and bool(r) is False
    # Either a RunError frame or a surfaced exception — both are non-ok with a reason.
    assert r.reason and (r.saw_error or "Error" in r.reason)


def test_verify_sync_wrapper_runs_a_turn():
    # The sync convenience a CLI doctor/check/selfcheck would call.
    r = verify(load_agent_spec("langstage_core.demo.stub:graph"))
    assert r.ok and r.saw_complete


def _uncompiled_stategraph():
    """A common mistake: exporting the StateGraph builder, forgetting .compile()."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("n", lambda s: s)
    b.add_edge(START, "n")
    b.add_edge("n", END)
    return b  # NOT compiled


async def test_averify_wrong_type_export_fails_not_raises():
    # gh langstage-jupyter #92: a wrong-type export (dict/None/function) used to make
    # build_agent raise a raw AttributeError that escaped averify()/verify() and
    # crashed the caller with a full traceback. It must now be a clean ok=False verdict
    # with an actionable reason, never an exception.
    r = await averify({"hello": "world"})
    assert r.ok is False and bool(r) is False
    assert "compiled LangGraph graph" in r.reason
    assert "has no attribute" not in r.reason, r.reason


async def test_averify_uncompiled_stategraph_gives_actionable_reason():
    # gh langstage-jupyter #92 (case 2): an uncompiled StateGraph must fail with the
    # actionable ".compile()" guidance, not a leaked internal AttributeError
    # ('StateGraph' object has no attribute 'aget_state').
    r = await averify(_uncompiled_stategraph())
    assert r.ok is False
    assert ".compile()" in r.reason, r.reason
    assert "aget_state" not in r.reason, r.reason


def _empty_turn_graph():
    """Loads and completes fine, but appends NO assistant message — the empty turn."""
    from langgraph.graph import END, MessagesState, START, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("respond", lambda s: {})  # no message appended
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    return b.compile()


async def test_averify_empty_turn_fails():
    # gh #119: a graph that loads and completes but produces ZERO content is the
    # "false green" --verify is documented to catch — ok=False, not a clean pass.
    r = await averify(_empty_turn_graph())
    assert r.ok is False and bool(r) is False
    assert r.saw_complete and not r.saw_error
    assert r.content_chars == 0
    assert "no content" in r.reason or "0 chars" in r.reason, r.reason


async def test_averify_blank_aimessage_turn_fails():
    # The realistic variant: a node returns AIMessage(content="") (a model that
    # returned nothing). Still zero content -> not a clean pass (gh #119).
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, MessagesState, START, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("respond", lambda s: {"messages": [AIMessage(content="")]})
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    r = await averify(b.compile())
    assert r.ok is False
    assert r.content_chars == 0


def _hitl_graph():
    """A keyless HITL agent: pauses once on a well-formed interrupt."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, MessagesState, START, StateGraph
    from langgraph.types import interrupt

    def node(state):
        d = interrupt({"question": "Approve?"})
        return {"messages": [AIMessage(content=f"done {d}")]}

    b = StateGraph(MessagesState)
    b.add_node("node", node)
    b.add_edge(START, "node")
    b.add_edge("node", END)
    return b.compile(checkpointer=InMemorySaver())


async def test_averify_hitl_interrupt_is_a_healthy_preflight():
    # gh langstage-jupyter #95: a HITL agent that reaches a well-formed interrupt is a
    # HEALTHY preflight outcome (the human-in-the-loop feature working), not a failure.
    # This intentionally supersedes the earlier "interrupt = not a clean pass" (gh #110).
    r = await averify(_hitl_graph())
    assert r.ok is True and bool(r) is True
    assert "interrupt" in r.reason.lower(), r.reason


async def test_averify_tool_only_turn_is_not_empty():
    # A turn that calls a tool but emits no assistant text is NOT the empty-turn false
    # green (gh #119) — it did real work. content_chars can be 0 yet ok=True.
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import END, MessagesState, START, StateGraph

    def respond(state):
        # A finished AIMessage carrying a tool call + its result, no assistant text.
        return {"messages": [
            AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": "c1"}]),
            ToolMessage(content="pong", tool_call_id="c1"),
        ]}

    b = StateGraph(MessagesState)
    b.add_node("respond", respond)
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    r = await averify(b.compile())
    assert r.ok is True, r.reason
