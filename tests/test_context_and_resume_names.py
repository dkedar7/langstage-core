"""Run context from `configurable` (gh #217) and tool names on a HITL reject resume (gh #215)."""
import asyncio
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

pytest.importorskip("ag_ui_langgraph")

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames


class _Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


@dataclass
class DC:
    user_id: str = "dflt"


class TD(TypedDict, total=False):
    user_id: str


class PM(BaseModel):
    user_id: str = "dflt"


def _run(schema, config):
    seen = {}

    @tool
    def who(runtime: ToolRuntime) -> str:
        """Report the run context."""
        seen["ctx"] = runtime.context
        return "ok"

    msgs = iter([AIMessage(content="", tool_calls=[{"name": "who", "args": {}, "id": "c1"}]),
                 AIMessage(content="done")])
    kw = {"context_schema": schema} if schema else {}
    graph = create_agent(_Fake(messages=msgs, disable_streaming=True), tools=[who],
                         checkpointer=InMemorySaver(), **kw)

    async def go():
        return [f async for f in iter_event_frames(build_agent(graph, config=config), "hi", "t1")]

    frames = asyncio.run(go())
    saved = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    return frames, seen.get("ctx"), saved


@pytest.mark.parametrize("schema", [DC, TD, PM, None])
def test_configurable_reaches_context_without_langgraph_keys(schema):
    frames, ctx, saved = _run(schema, {"configurable": {"user_id": "alice"}})
    assert not [f for f in frames if f.get("type") == "error"], frames
    if schema is DC:
        assert ctx == DC(user_id="alice")
    elif schema is PM:
        assert ctx == PM(user_id="alice")
    else:
        assert ctx == {"user_id": "alice"}  # no thread_id
    assert len(saved) == 4  # thread_id still reached the checkpointer


@pytest.mark.parametrize("schema", [DC, TD, PM, None])
def test_no_configurable_passes_no_context(schema):
    frames, ctx, saved = _run(schema, None)
    assert not [f for f in frames if f.get("type") == "error"], frames
    assert ctx is None  # what plain LangGraph gives without a context
    assert len(saved) == 4


def _hitl_agent():
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    msgs = iter([AIMessage(content="Calling add.", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "c1"}]),
                 AIMessage(content="OK, I won't add them.")])
    graph = create_agent(_Fake(messages=msgs, disable_streaming=True), tools=[add],
                         middleware=[HumanInTheLoopMiddleware(interrupt_on={"add": True})],
                         checkpointer=InMemorySaver())
    return build_agent(graph)


@pytest.mark.parametrize("decision", ["reject", "approve"])
def test_resume_names_the_tool_on_both_wires(decision):
    async def go(wire):
        agent = _hitl_agent()
        async for _ in wire(agent, "add 2 and 3", "s1"):
            pass
        dec = {"type": decision, **({"message": "no"} if decision == "reject" else {})}
        return [f async for f in wire(agent, "", "s1", resume={"decisions": [dec]})]

    ends = [f for f in asyncio.run(go(iter_event_frames)) if f.get("type") == "tool_end"]
    results = [f for f in asyncio.run(go(iter_chunk_frames)) if "tool_result" in f]
    assert [f["name"] for f in ends] == ["add"]
    assert [f["name"] for f in results] == ["add"]


def test_package_summary_does_not_advertise_the_retired_event_layer():
    meta = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    summary = meta["project"]["description"]
    assert "parser" not in summary.lower() and "typed stream events" not in summary
