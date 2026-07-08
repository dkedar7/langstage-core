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

from langstage_core import create_resume_input
from langstage_core.agui import (
    _normalize_interrupt,
    _unwrap_resume,
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


# ── resume: create_resume_input()'s Command must not double-wrap (gh #82) ──


def test_unwrap_resume_accepts_command_and_raw_payload():
    cmd = create_resume_input(decisions=[{"type": "approve"}])
    # a Command is unwrapped to its .resume payload...
    assert _unwrap_resume(cmd) == {"decisions": [{"type": "approve"}]}
    # ...and a raw payload passes through untouched; None stays None.
    raw = {"decisions": [{"type": "approve"}]}
    assert _unwrap_resume(raw) is raw
    assert _unwrap_resume(None) is None


def _decision_reading_agent():
    """A realistic HITL node that reads the resume decision as a mapping — it crashes
    (`'Command' object is not subscriptable`) if resume was double-wrapped (gh #82)."""
    def act(state):
        decision = interrupt([{"action_request": {"action": "delete", "args": {}},
                               "config": {"allow_accept": True}}])
        choice = decision["decisions"][0]["type"]
        return {"messages": [AIMessage(content=f"chose: {choice}")]}

    b = StateGraph(MessagesState)
    b.add_node("act", act)
    b.add_edge(START, "act")
    b.add_edge("act", END)
    return b.compile(checkpointer=InMemorySaver())


def _resume_roundtrip(thread_id, resume):
    agent = build_agent(_decision_reading_agent(), name="HITL")
    asyncio.run(_collect(iter_event_frames(agent, "go", thread_id)))  # hit the interrupt
    return asyncio.run(_collect(iter_event_frames(agent, "", thread_id, resume=resume)))


def test_resume_with_create_resume_input_command_does_not_double_wrap():
    # The exact #82 repro: resume= create_resume_input(...) (a Command) used to
    # double-wrap and crash the HITL node.
    frames = _resume_roundtrip("r-cmd", create_resume_input(decisions=[{"type": "accept"}]))
    assert not any(f.get("type") == "error" for f in frames), frames
    content = "".join(f.get("content", "") for f in frames if f.get("type") == "content")
    assert "chose: accept" in content


def test_resume_with_raw_payload_still_works():
    frames = _resume_roundtrip("r-raw", {"decisions": [{"type": "accept"}]})
    assert not any(f.get("type") == "error" for f in frames), frames
    content = "".join(f.get("content", "") for f in frames if f.get("type") == "content")
    assert "chose: accept" in content
