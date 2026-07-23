"""The keyless rich-frame demo agent (``langstage_core.demo.tools``) — gh #99.

This is the regression fixture the issue asks for: the echo stub only ever emits
``content``, so the entire rich-frame class (``tool_start`` / ``tool_end`` /
``extraction`` / ``reasoning`` / ``interrupt``) — the largest bug class in this
repo's history — had no keyless test exercising it end-to-end. These tests drive
the demo through **both** shipped mappings (``iter_event_frames`` and
``iter_chunk_frames``) and assert every documented frame type actually appears,
and that the interrupt path resumes.

Skipped unless the agui extra is installed (dev pulls it, so CI runs it).
"""
import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langstage_core import create_resume_input, load_agent_spec
from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames
from langstage_core.demo.tools import (
    DemoLookupExtractor,
    create_tool_demo_agent,
    demo_extractors,
)

pytestmark = pytest.mark.asyncio


async def _collect(aiter):
    return [frame async for frame in aiter]


def _agent():
    return build_agent(create_tool_demo_agent())


# ── the spec loads, so every surface can point at it keyless ─────────────────────


async def test_spec_loads_and_factory_builds():
    # The advertised spec string resolves to a compiled graph (the --demo=tools path).
    graph = load_agent_spec("langstage_core.demo.tools:graph")
    assert graph is not None
    assert getattr(graph, "name", None) == "Tool Demo Agent"


# ── content (normal message) — token-by-token, exactly like the echo stub ────────


async def test_content_frame_event_wire():
    frames = await _collect(iter_event_frames(_agent(), "hello there", "c-e"))
    content = [f for f in frames if f.get("type") == "content"]
    assert content, frames
    assert "hello there" in "".join(f["content"] for f in content)
    assert frames[-1] == {"type": "complete"}
    # token-by-token: the reply arrives as multiple content frames, not one blob.
    assert len(content) > 1


async def test_content_frame_chunk_wire():
    frames = await _collect(iter_chunk_frames(_agent(), "hello there", "c-c"))
    chunks = [f for f in frames if f.get("status") == "streaming" and "chunk" in f]
    assert chunks and "hello there" in "".join(f["chunk"] for f in chunks)
    assert frames[-1] == {"status": "complete"}


# ── reasoning ("think") — surfaced as a `reasoning` frame, separate from content ──


async def test_reasoning_frame_event_wire():
    frames = await _collect(iter_event_frames(_agent(), "think about it", "r-e"))
    reasoning = [f for f in frames if f.get("type") == "reasoning"]
    content = [f for f in frames if f.get("type") == "content"]
    assert reasoning, f"no reasoning frame: {[f.get('type') for f in frames]}"
    assert content, "reasoning turn must still produce a content answer"
    # reasoning is the chain-of-thought, never the answer text.
    assert "reason" in "".join(f["content"] for f in reasoning).lower()
    assert frames[-1] == {"type": "complete"}


async def test_reasoning_frame_chunk_wire():
    frames = await _collect(iter_chunk_frames(_agent(), "think about it", "r-c"))
    reasoning = [f for f in frames if "reasoning" in f]
    chunks = [f for f in frames if "chunk" in f]
    assert reasoning, f"no reasoning chunk: {frames}"
    assert chunks, "reasoning turn must still stream a content answer"
    assert frames[-1] == {"status": "complete"}


# ── tool_start → tool_end → extraction ("use a tool"), via a REAL ToolNode ───────


async def test_tool_frames_event_wire():
    frames = await _collect(
        iter_event_frames(_agent(), "use a tool now", "t-e", extractors=demo_extractors())
    )
    starts = [f for f in frames if f.get("type") == "tool_start"]
    ends = [f for f in frames if f.get("type") == "tool_end"]
    extraction = [f for f in frames if f.get("type") == "extraction"]
    content = [f for f in frames if f.get("type") == "content"]
    assert starts and starts[0]["name"] == "demo_lookup", frames
    assert ends and ends[0]["name"] == "demo_lookup" and ends[0]["status"] == "success"
    assert extraction, "no extraction frame — the paired extractor never fired"
    assert extraction[0]["extracted_type"] == "demo_fact"
    assert extraction[0]["data"]["answer"] == "42"
    assert content, "the tool turn must end with a content summary"
    assert frames[-1] == {"type": "complete"}


async def test_tool_frames_chunk_wire():
    frames = await _collect(
        iter_chunk_frames(_agent(), "use a tool now", "t-c", extractors=demo_extractors())
    )
    calls = [f for f in frames if "tool_calls" in f]
    results = [f for f in frames if "tool_result" in f]
    extraction = [f["extraction"] for f in frames if "extraction" in f]
    chunks = [f for f in frames if "chunk" in f]
    assert calls and calls[0]["tool_calls"][0]["name"] == "demo_lookup", frames
    assert results, "no tool_result chunk"
    assert extraction and extraction[0]["extracted_type"] == "demo_fact"
    assert extraction[0]["data"]["answer"] == "42"
    assert chunks, "the tool turn must end with a content summary"
    assert frames[-1] == {"status": "complete"}


# ── interrupt ("ask me") + resume ────────────────────────────────────────────────


async def test_interrupt_frame_event_wire():
    frames = await _collect(iter_event_frames(_agent(), "ask me a question", "i-e"))
    interrupts = [f for f in frames if f.get("type") == "interrupt"]
    assert interrupts, f"no interrupt frame: {[f.get('type') for f in frames]}"
    assert not any(f.get("type") == "error" for f in frames)
    assert interrupts[0]["action_requests"] == [
        {"action": "ask_user", "args": {"question": "What should I call you?"}}
    ]
    assert "respond" in interrupts[0]["allowed_decisions"]


async def test_interrupt_frame_chunk_wire():
    frames = await _collect(iter_chunk_frames(_agent(), "ask me a question", "i-c"))
    interrupts = [f for f in frames if f.get("status") == "interrupt"]
    assert interrupts, f"no interrupt frame: {frames}"
    assert not any(f.get("status") == "error" for f in frames)
    assert interrupts[0]["interrupt"]["action_requests"][0]["action"] == "ask_user"


async def test_interrupt_resumes_event_wire():
    # The full HITL round-trip: hit the interrupt, then resume the SAME thread with a
    # decision via the documented create_resume_input path; the turn must complete.
    agent = _agent()
    first = await _collect(iter_event_frames(agent, "ask me a question", "i-resume"))
    assert any(f.get("type") == "interrupt" for f in first)

    resumed = await _collect(
        iter_event_frames(
            agent, "", "i-resume", resume=create_resume_input(decisions=[{"type": "approve"}])
        )
    )
    assert not any(f.get("type") == "error" for f in resumed), resumed
    content = "".join(f["content"] for f in resumed if f.get("type") == "content")
    assert "approve" in content, f"resume decision not reflected: {content!r}"
    assert resumed[-1] == {"type": "complete"}


async def test_interrupt_resumes_chunk_wire():
    agent = _agent()
    first = await _collect(iter_chunk_frames(agent, "ask me a question", "i-resume-c"))
    assert any(f.get("status") == "interrupt" for f in first)

    resumed = await _collect(
        iter_chunk_frames(
            agent, "", "i-resume-c", resume=create_resume_input(decisions=[{"type": "approve"}])
        )
    )
    assert not any(f.get("status") == "error" for f in resumed), resumed
    text = "".join(f["chunk"] for f in resumed if "chunk" in f)
    assert "approve" in text
    assert resumed[-1] == {"status": "complete"}


# ── the headline regression fixture: EVERY documented frame type is reachable ─────

_EVENT_FRAME_TYPES = {
    "content",
    "tool_start",
    "tool_end",
    "extraction",
    "reasoning",
    "interrupt",
    "complete",
}


async def test_demo_covers_every_documented_frame_type_event_wire():
    """Union of frame types across the demo's triggers covers the full taxonomy —
    the copy-paste example + smoke test gh #99 asks for."""
    agent = _agent()
    seen: set[str] = set()
    seen.update(f["type"] for f in await _collect(iter_event_frames(agent, "hi", "cov1")))
    seen.update(f["type"] for f in await _collect(iter_event_frames(agent, "think", "cov2")))
    seen.update(
        f["type"]
        for f in await _collect(
            iter_event_frames(agent, "use a tool", "cov3", extractors=demo_extractors())
        )
    )
    seen.update(f["type"] for f in await _collect(iter_event_frames(agent, "ask me", "cov4")))
    missing = _EVENT_FRAME_TYPES - seen
    assert not missing, f"demo never emitted: {missing} (saw {seen})"


async def test_demo_covers_every_documented_frame_type_chunk_wire():
    agent = _agent()

    def _kinds(frames):
        kinds = set()
        for f in frames:
            if f.get("status") == "interrupt":
                kinds.add("interrupt")
            elif f.get("status") == "complete":
                kinds.add("complete")
            elif f.get("status") == "streaming":
                for key in ("chunk", "reasoning", "tool_calls", "tool_result", "extraction"):
                    if key in f:
                        kinds.add(key)
        return kinds

    seen: set[str] = set()
    seen |= _kinds(await _collect(iter_chunk_frames(agent, "hi", "kc1")))
    seen |= _kinds(await _collect(iter_chunk_frames(agent, "think", "kc2")))
    seen |= _kinds(
        await _collect(iter_chunk_frames(agent, "use a tool", "kc3", extractors=demo_extractors()))
    )
    seen |= _kinds(await _collect(iter_chunk_frames(agent, "ask me", "kc4")))
    expected = {"chunk", "reasoning", "tool_calls", "tool_result", "extraction", "interrupt", "complete"}
    missing = expected - seen
    assert not missing, f"demo never emitted (chunk wire): {missing} (saw {seen})"


# ── contrast: the echo stub CANNOT reach the rich frames (proves the asserts real) ─


async def test_echo_stub_lacks_rich_frames():
    """The same 'use a tool' + 'think' + 'ask me' triggers against the plain echo
    stub emit only content/complete — so the assertions above are testing a real
    capability difference, not a tautology. This is the gh #99 gap, made explicit."""
    stub = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    seen: set[str] = set()
    for msg in ("use a tool", "think", "ask me"):
        seen.update(
            f["type"]
            for f in await _collect(
                iter_event_frames(stub, msg, f"stub-{msg}", extractors=demo_extractors())
            )
        )
    assert seen <= {"content", "complete"}, f"echo stub unexpectedly rich: {seen}"
    assert "tool_start" not in seen and "reasoning" not in seen and "interrupt" not in seen


# ── the paired extractor in isolation ────────────────────────────────────────────


async def test_demo_lookup_extractor_unit():
    ex = DemoLookupExtractor()
    assert ex.tool_name == "demo_lookup"
    assert ex.extracted_type == "demo_fact"
    assert ex.extract('{"query": "q", "answer": "42"}') == {"query": "q", "answer": "42"}
    # non-matching / unparseable content yields no extraction frame.
    assert ex.extract("not json") is None
    assert ex.extract('{"unrelated": 1}') is None


async def test_demo_extractors_returns_the_paired_extractor():
    exts = demo_extractors()
    assert len(exts) == 1 and isinstance(exts[0], DemoLookupExtractor)
