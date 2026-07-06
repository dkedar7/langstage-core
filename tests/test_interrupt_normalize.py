"""The interrupt payload normalizer (gh langstage-vscode #40).

The on_interrupt handler used to do ``payload.get("action_requests", ...)``, which
crashed on the standard **HumanInterrupt list** that deepagents / langchain HITL emit
(``'list' object has no attribute 'get'``) and returned an empty ``action_requests``
for any other dict. ``_normalize_interrupt`` handles all three shapes; these tests pin
the unit behavior and drive both frame iterators end-to-end with a list-shape agent.
"""

import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langstage_core.agui import (
    _normalize_interrupt,
    build_agent,
    iter_chunk_frames,
    iter_event_frames,
)

# The standard HumanInterrupt list shape deepagents / langchain HITL produce.
HUMAN_INTERRUPT = [
    {
        "action_request": {"action": "delete_file", "args": {"path": "/tmp/x"}},
        "config": {"allow_accept": True, "allow_respond": True},
        "description": "Approve deleting the file?",
    }
]


# ── unit: _normalize_interrupt across shapes ─────────────────────────


def test_human_interrupt_list_unwraps_action_request_and_derives_decisions():
    action_requests, review_configs, decisions = _normalize_interrupt(HUMAN_INTERRUPT)
    assert action_requests == [{"action": "delete_file", "args": {"path": "/tmp/x"}}]
    assert review_configs == []
    # config allow_accept + allow_respond -> approve + respond (in default order)
    assert decisions == ["respond", "approve"]


def test_list_without_config_falls_back_to_all_decisions():
    action_requests, _, decisions = _normalize_interrupt([{"action_request": {"a": 1}}])
    assert action_requests == [{"a": 1}]
    assert decisions == ["reject", "edit", "respond", "approve"]


def test_plain_dict_becomes_a_single_action_request():
    # Previously returned action_requests=[] (the advertised field never populated).
    action_requests, _, decisions = _normalize_interrupt({"action": "x", "path": "/y"})
    assert action_requests == [{"action": "x", "path": "/y"}]
    assert decisions == ["reject", "edit", "respond", "approve"]


def test_our_keyed_dict_is_used_as_is():
    action_requests, _, decisions = _normalize_interrupt(
        {"action_requests": [{"tool": "t"}], "allowed_decisions": ["approve"]}
    )
    assert action_requests == [{"tool": "t"}]
    assert decisions == ["approve"]


def test_empty_or_none_is_safe():
    assert _normalize_interrupt(None) == ([], [], ["reject", "edit", "respond", "approve"])
    assert _normalize_interrupt({}) == ([], [], ["reject", "edit", "respond", "approve"])


# ── integration: both iterators survive a list-shape interrupt ───────


def _human_interrupt_agent():
    def ask(state):
        decision = interrupt(HUMAN_INTERRUPT)
        return {"messages": [AIMessage(content=f"resumed: {decision}")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile(checkpointer=InMemorySaver())


async def _collect(aiter):
    return [f async for f in aiter]


def test_iter_event_frames_surfaces_human_interrupt_without_crashing():
    agent = build_agent(_human_interrupt_agent(), name="HITL")
    frames = asyncio.run(_collect(iter_event_frames(agent, "delete it", thread_id="e1")))
    interrupts = [f for f in frames if f.get("type") == "interrupt"]
    assert interrupts, f"no interrupt frame; got {[f.get('type') for f in frames]}"
    assert not any(f.get("type") == "error" for f in frames)
    # the advertised action_requests is populated from the HumanInterrupt list
    assert interrupts[0]["action_requests"] == [
        {"action": "delete_file", "args": {"path": "/tmp/x"}}
    ]


def test_iter_chunk_frames_surfaces_human_interrupt_without_crashing():
    agent = build_agent(_human_interrupt_agent(), name="HITL")
    frames = asyncio.run(_collect(iter_chunk_frames(agent, "delete it", thread_id="c1")))
    interrupts = [f for f in frames if f.get("status") == "interrupt"]
    assert interrupts, f"no interrupt frame; got {[f.get('status') for f in frames]}"
    assert not any(f.get("status") == "error" for f in frames)
    # a chunk-wire consumer (cli) reads frame["interrupt"]["action_requests"]
    assert interrupts[0]["interrupt"]["action_requests"] == [
        {"action": "delete_file", "args": {"path": "/tmp/x"}}
    ]
