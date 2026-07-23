"""One-shot turn collectors — run a turn, get a typed result (gh #110).

``collect_event_frames`` / ``collect_chunk_frames`` / ``run_turn`` are the
non-streaming counterpart to the two ``iter_*`` mappings: one call runs a turn and
returns a typed :class:`TurnResult` (text, tool_calls, extractions, reasoning,
outcome, interrupt, error). These tests assert the typed result for each terminal
outcome — a plain text turn, a tool turn, an interrupt turn, and an error turn —
and that the ``complete``/``interrupted``/``error`` verdict comes from the one
shared ``_terminal_outcome`` rule (so it can't drift from ``SessionAdapter._produce``).

Skipped unless the agui extra is installed (the dev extra pulls it, so CI runs it).
"""

import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langstage_core import load_agent_spec
from langstage_core.agui import (
    TurnResult,
    build_agent,
    collect_chunk_frames,
    collect_event_frames,
    run_turn,
)
from langstage_core.agui import _terminal_outcome
from langstage_core.demo.tools import create_tool_demo_agent, demo_extractors

# asyncio_mode = "auto" (pyproject) runs the async tests as coroutines; the sync
# run_turn tests stay sync, so no module-level asyncio mark here.


def _erroring_graph():
    """A compiled graph whose only node raises — the AG-UI adapter surfaces this
    as an ``error`` frame, so the turn's outcome is ``error`` (not an exception)."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    def boom(state):
        raise RuntimeError("tool exploded")

    b = StateGraph(MessagesState)
    b.add_node("boom", boom)
    b.add_edge(START, "boom")
    b.add_edge("boom", END)
    return b.compile()


# ── The shared terminal-outcome rule (single source of truth) ────────────────


def test_terminal_outcome_rule_precedence():
    # error wins over interrupt; interrupt over complete; else complete.
    assert _terminal_outcome(saw_interrupt=False, saw_error=False) == "complete"
    assert _terminal_outcome(saw_interrupt=True, saw_error=False) == "interrupted"
    assert _terminal_outcome(saw_interrupt=False, saw_error=True) == "error"
    assert _terminal_outcome(saw_interrupt=True, saw_error=True) == "error"


# ── (a) a plain text turn ────────────────────────────────────────────────────


async def test_collect_event_frames_plain_text_turn():
    agent = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    r = await collect_event_frames(agent, "hello there", "s1")
    assert isinstance(r, TurnResult)
    assert r.outcome == "complete"
    assert "hello there" in r.text
    assert r.tool_calls == [] and r.extractions == []
    assert r.reasoning == ""
    assert r.interrupt is None and r.error is None
    assert r.frames > 0


def test_run_turn_sync_plain_text_turn():
    # The sync "one call, one answer" convenience, accepting a bare graph.
    r = run_turn(load_agent_spec("langstage_core.demo.stub:graph"), "hi from a script")
    assert r.outcome == "complete"
    assert "hi from a script" in r.text


def test_run_turn_accepts_a_prebuilt_agent():
    # Like verify/averify, run_turn takes a graph OR an already-built agent.
    agent = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    r = run_turn(agent, "prebuilt")
    assert r.outcome == "complete" and "prebuilt" in r.text


# ── (b) a tool turn (text + tool_calls + extraction) ─────────────────────────


async def test_collect_event_frames_tool_turn():
    agent = build_agent(create_tool_demo_agent())
    r = await collect_event_frames(agent, "use a tool", "s1", extractors=demo_extractors())
    assert r.outcome == "complete"
    # text: the agent summarizes the tool result
    assert r.text and "demo tool returned" in r.text
    # tool_calls: the real ToolNode call, carrying name + args + id
    assert len(r.tool_calls) == 1
    call = r.tool_calls[0]
    assert call["name"] == "demo_lookup"
    assert call["args"] == {"query": "use a tool"}
    assert call["id"]  # a non-empty tool_call id
    # extraction: the paired DemoLookupExtractor fired
    assert len(r.extractions) == 1
    ex = r.extractions[0]
    assert ex["tool_name"] == "demo_lookup"
    assert ex["extracted_type"] == "demo_fact"
    assert ex["data"] == {"query": "use a tool", "answer": "42"}
    assert r.interrupt is None and r.error is None


def test_run_turn_tool_turn_sync():
    r = run_turn(create_tool_demo_agent(), "use a tool", extractors=demo_extractors())
    assert r.outcome == "complete"
    assert r.tool_calls[0]["name"] == "demo_lookup"
    assert r.extractions[0]["data"]["answer"] == "42"


async def test_collect_chunk_frames_tool_turn_parity():
    # The chunk-wire collector accumulates the same typed result (its tool_calls
    # carry no id — that's the chunk vocabulary — but name/args/extraction match).
    agent = build_agent(create_tool_demo_agent())
    r = await collect_chunk_frames(agent, "use a tool", "s1", extractors=demo_extractors())
    assert r.outcome == "complete"
    assert r.tool_calls[0]["name"] == "demo_lookup"
    assert r.tool_calls[0]["args"] == {"query": "use a tool"}
    assert r.extractions[0]["extracted_type"] == "demo_fact"
    assert r.extractions[0]["data"] == {"query": "use a tool", "answer": "42"}


# ── (c) an interrupt turn (outcome interrupted, interrupt captured) ───────────


async def test_collect_event_frames_interrupt_turn():
    agent = build_agent(create_tool_demo_agent())
    r = await collect_event_frames(agent, "ask me", "s-int")
    assert r.outcome == "interrupted"
    assert r.interrupt is not None
    assert r.interrupt["type"] == "interrupt"
    assert r.interrupt["action_requests"]  # a populated action request to resume with
    assert r.error is None


def test_run_turn_interrupt_turn_sync():
    r = run_turn(create_tool_demo_agent(), "ask me")
    assert r.outcome == "interrupted"
    assert r.interrupt and r.interrupt["action_requests"]


# ── (d) an error turn (outcome error) ────────────────────────────────────────


async def test_collect_event_frames_error_turn():
    agent = build_agent(_erroring_graph())
    r = await collect_event_frames(agent, "go", "s-err")
    assert r.outcome == "error"
    assert r.error and "tool exploded" in r.error
    assert r.interrupt is None


def test_run_turn_error_turn_sync():
    r = run_turn(_erroring_graph(), "go")
    assert r.outcome == "error"
    assert r.error


# ── Export surface ───────────────────────────────────────────────────────────


def test_collectors_exported_from_agui():
    import langstage_core.agui as agui

    for name in ("collect_event_frames", "collect_chunk_frames", "run_turn", "TurnResult"):
        assert name in agui.__all__, f"{name} missing from langstage_core.agui.__all__"
        assert hasattr(agui, name)
