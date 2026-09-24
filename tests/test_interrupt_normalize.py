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


# ── gh langstage-cli #95: a bare-string / scalar interrupt must surface ──────────
def test_plain_string_interrupt_becomes_a_single_action_request():
    """The canonical `interrupt("Approve deleting X?")` HITL form used to normalize to
    an EMPTY action_requests list -> renderers showed "(no action details provided)"
    and asked the human to approve blind. It now surfaces as a single request the
    renderer can display (cli's format_interrupt_request returns str(action))."""
    ars, _reviews, decisions = _normalize_interrupt("Approve deleting ALL files in /home?")
    assert ars == ["Approve deleting ALL files in /home?"]
    assert decisions  # still offers the default decisions


def test_scalar_interrupt_surfaces_but_empty_still_vanishes():
    assert _normalize_interrupt(42)[0] == [42]
    assert _normalize_interrupt("")[0] == []      # a degenerate empty payload
    assert _normalize_interrupt(None)[0] == []


import pytest


@pytest.mark.parametrize("wire", ["event", "chunk"])
def test_string_interrupt_renders_end_to_end(wire):
    """Drive a real graph that raises interrupt("...") and confirm the interrupt frame
    carries the string on BOTH wires — the cli #95 symptom lived in the on_interrupt
    handler, which exists once per wire. (cli uses the chunk wire; the web/vscode
    surfaces use the event wire — both must surface the string.)"""
    from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames

    def ask(state):
        interrupt("Approve deleting ALL files? (y/n)")
        return {"messages": [AIMessage(content="done")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    agent = build_agent(b.compile(checkpointer=InMemorySaver()))

    it = iter_event_frames if wire == "event" else iter_chunk_frames

    async def go():
        return [f async for f in it(agent, "go", f"s95-{wire}")]

    frames = asyncio.run(go())
    # event wire: {"type":"interrupt","action_requests":[...]}
    # chunk wire: {"status":"interrupt","interrupt":{"action_requests":[...]}}
    if wire == "event":
        hits = [f["action_requests"] for f in frames if f.get("type") == "interrupt"]
    else:
        hits = [f["interrupt"]["action_requests"] for f in frames if f.get("status") == "interrupt"]
    assert hits, frames
    assert hits[0] == ["Approve deleting ALL files? (y/n)"]


# ── gh langstage-vscode #114: allowed_decisions honors the interrupt's own config ──


def _restricted_agent(config):
    def ask(state):
        decision = interrupt([{
            "action_request": {"action": "delete_database", "args": {"name": "prod"}},
            "config": config,
            "description": "Approve deleting the prod database?",
        }])
        return {"messages": [AIMessage(content=f"got {decision}")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize("wire", ["event", "chunk"])
@pytest.mark.parametrize(
    "config, expected",
    [
        # the vscode #114 repro: approve-only
        ({"allow_accept": True, "allow_edit": False, "allow_respond": False,
          "allow_ignore": False}, ["approve"]),
        ({"allow_ignore": True, "allow_accept": True}, ["reject", "approve"]),
        ({"allow_accept": True, "allow_respond": True}, ["respond", "approve"]),
    ],
)
def test_restricted_human_interrupt_advertises_only_its_verbs(wire, config, expected):
    """ag-ui-langgraph >= 0.0.43 strips every ``config`` key from the on_interrupt
    value (make_json_safe), so the HumanInterrupt's allow_* flags never reached the
    normalizer and every interrupt advertised all four verbs. They're now read from
    the checkpoint's original interrupt value, on BOTH wires."""
    agent = build_agent(_restricted_agent(config), name="HITL")
    it = iter_event_frames if wire == "event" else iter_chunk_frames
    frames = asyncio.run(_collect(it(agent, "go", f"v114-{wire}-{len(expected)}")))
    if wire == "event":
        hits = [f for f in frames if f.get("type") == "interrupt"]
    else:
        hits = [f["interrupt"] for f in frames if f.get("status") == "interrupt"]
    assert hits, frames
    assert hits[0]["allowed_decisions"] == expected
    # action_requests still come from the JSON-safe event value
    assert hits[0]["action_requests"] == [
        {"action": "delete_database", "args": {"name": "prod"}}
    ]


def test_hitl_middleware_review_configs_derive_allowed_decisions():
    """LangChain's HumanInTheLoopMiddleware (deepagents) shape carries the permitted
    verbs per review_config, not at the top level — derive their union."""
    payload = {
        "action_requests": [{"name": "rm", "args": {}}, {"name": "ls", "args": {}}],
        "review_configs": [
            {"action_name": "rm", "allowed_decisions": ["approve", "reject"]},
            {"action_name": "ls", "allowed_decisions": ["approve"]},
        ],
    }
    ars, rcs, decisions = _normalize_interrupt(payload)
    assert ars == payload["action_requests"]
    assert rcs == payload["review_configs"]
    assert decisions == ["reject", "approve"]
    # an explicit top-level allowed_decisions still wins
    assert _normalize_interrupt({**payload, "allowed_decisions": ["edit"]})[2] == ["edit"]
    # no verbs anywhere -> the full default set, as before
    assert _normalize_interrupt({"action_requests": [{"a": 1}]})[2] == [
        "reject", "edit", "respond", "approve"
    ]


# ── gh #144: resume rides RunAgentInput.resume[], never the deprecated wire ──


def _string_answer_agent():
    def ask(state):
        answer = interrupt({"question": "Approve?"})
        return {"messages": [AIMessage(content=f"Decision was: {answer!r}")]}

    b = StateGraph(MessagesState)
    b.add_node("ask", ask)
    b.add_edge(START, "ask")
    b.add_edge("ask", END)
    return b.compile(checkpointer=InMemorySaver())


_RESUME_NOISE = ("deprecated", "failed to parse")


@pytest.mark.parametrize("wire", ["event", "chunk"])
@pytest.mark.parametrize(
    "resume, expected",
    [
        (create_resume_input(decisions=[{"type": "approve"}]),
         "{'decisions': [{'type': 'approve'}]}"),
        ({"decisions": [{"type": "reject"}]}, "{'decisions': [{'type': 'reject'}]}"),
        ("yes go ahead", "'yes go ahead'"),   # plain text: cli #137's parse warning
        ('{"x": 1}', "{'x': 1}"),              # a JSON string still decodes, as before
        ("", "''"),                            # the --no-interactive empty auto-resume
    ],
)
def test_resume_emits_no_deprecation_or_parse_warning(wire, resume, expected, caplog):
    """Every documented resume used to log ``forwardedProps.command.resume is
    deprecated …`` (and, for a non-JSON string, ``failed to parse … resume_input as
    JSON …``) from ag_ui_langgraph — straight into the cli/vscode output (cli #126,
    #137, vscode #103). The resume now rides the adapter's standard
    ``RunAgentInput.resume[]``, delivering the same value with no warning."""
    import logging

    agent = build_agent(_string_answer_agent(), name="HITL")
    it = iter_event_frames if wire == "event" else iter_chunk_frames
    tid = f"r144-{wire}-{abs(hash(repr(resume)))}"
    asyncio.run(_collect(it(agent, "go", tid)))  # reach the interrupt
    with caplog.at_level(logging.DEBUG):
        frames = asyncio.run(_collect(it(agent, "", tid, resume=resume)))
    noisy = [r.getMessage() for r in caplog.records
             if r.name.startswith("ag_ui_langgraph")
             and any(n in r.getMessage() for n in _RESUME_NOISE)]
    assert noisy == [], noisy
    if wire == "event":
        text = "".join(f.get("content", "") for f in frames if f.get("type") == "content")
    else:
        text = "".join(f.get("chunk", "") for f in frames if f.get("status") == "streaming")
    assert text == f"Decision was: {expected}"


def test_resume_uses_the_standard_resume_field_when_supported():
    """Pins the wire itself: a single pending interrupt is answered with a
    ``ResumeEntry`` keyed by its id, and ``forwarded_props`` carries no command."""
    from langstage_core.agui import _resume_input_fields, _supports_agui_resume

    agent = build_agent(_string_answer_agent(), name="HITL")
    if not _supports_agui_resume(agent):
        pytest.skip("installed ag-ui-langgraph predates RunAgentInput.resume[]")
    asyncio.run(_collect(iter_event_frames(agent, "go", "r144-wire")))
    fields = asyncio.run(_resume_input_fields(agent, "r144-wire", {"ok": True}))
    assert fields["forwarded_props"] == {}
    (entry,) = fields["resume"]
    assert entry.status == "resolved" and entry.payload == {"ok": True}
    assert entry.interrupt_id
    # nothing pending -> the legacy wire, unchanged behavior
    fresh = asyncio.run(_resume_input_fields(agent, "never-interrupted", {"ok": True}))
    assert fresh == {"forwarded_props": {"command": {"resume": {"ok": True}}}}
    # no resume -> no command at all
    assert asyncio.run(_resume_input_fields(agent, "r144-wire", None)) == {"forwarded_props": {}}
