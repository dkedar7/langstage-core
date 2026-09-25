"""``extractors=None`` / a single extractor are accepted everywhere (gh #178).

Both used to raise a raw ``TypeError: ... object is not iterable`` out of the ``iter_*``
generators (and so out of ``run_turn``), because the dispatch table was built from
``extractors`` before the guarded run. ``None`` now means "no extractors", a single
extractor is wrapped, and a genuinely wrong value is a clear ``TypeError`` naming the
argument.
"""
import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
from langstage_core import SessionAdapter  # noqa: E402
from langstage_core.agui import (  # noqa: E402
    collect_chunk_frames,
    iter_event_frames,
    run_turn,
)
from langstage_core.demo.stub import graph as stub_graph  # noqa: E402
from langstage_core.demo.tools import create_tool_demo_agent, demo_extractors  # noqa: E402


def test_run_turn_accepts_none():
    r = run_turn(stub_graph, "hi", extractors=None)
    assert r.outcome == "complete" and r.extractions == []


def test_run_turn_accepts_a_single_extractor():
    single = demo_extractors()[0]
    r = run_turn(create_tool_demo_agent(), "use a tool", extractors=single)
    assert r.outcome == "complete"
    assert [e["tool_name"] for e in r.extractions] == [single.tool_name]


async def test_chunk_collector_accepts_none():
    r = await collect_chunk_frames(stub_graph, "hi", "t1", extractors=None)
    assert r.outcome == "complete"


async def test_event_wire_accepts_none():
    frames = [f async for f in iter_event_frames(stub_graph, "hi", "t1", extractors=None)]
    assert frames[-1]["type"] == "complete"


async def test_non_iterable_is_a_clear_type_error():
    with pytest.raises(TypeError, match="extractors="):
        [f async for f in iter_event_frames(stub_graph, "hi", "t1", extractors=42)]


def test_session_adapter_accepts_none_and_rejects_garbage():
    SessionAdapter(graph=stub_graph, extractors=None)
    with pytest.raises(TypeError, match="extractors="):
        SessionAdapter(graph=stub_graph, extractors=42)
