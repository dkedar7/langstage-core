"""SessionAdapter's experimental AG-UI mode (ADR 0002, web surface).

Verifies that streaming through the in-process AG-UI adapter emits the SAME
``event_to_dict`` frames + terminal ``session.outcome`` as the StreamParser path.
Skipped unless the agui extra is installed (the dev extra pulls it, so CI runs it).
"""
import asyncio
from typing import Iterator, List

import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langgraph_stream_parser import load_agent_spec
from langgraph_stream_parser.adapters import SessionAdapter

pytestmark = pytest.mark.asyncio


def _drain(queue: asyncio.Queue) -> List[dict]:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


async def test_text_and_outcome_on_demo_stub():
    adapter = SessionAdapter(graph=load_agent_spec("langgraph_stream_parser.demo.stub:graph"), agui=True)
    session = adapter.submit_message(None, "hello web")
    await session.current_task
    frames = _drain(session.event_queue)
    assert frames[-1]["type"] == "complete"
    assert session.outcome == "complete"
    assert "hello web" in "".join(f["content"] for f in frames if f["type"] == "content")


@tool
def get_weather(city: str) -> str:
    """Get the weather."""
    return "Sunny, 72F"


class _FakeToolModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        if any(isinstance(m, ToolMessage) for m in messages):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Sunny."))
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", tool_call_chunks=[{"name": "get_weather", "args": "", "id": "c1", "index": 0}]))
            for seg in ('{"city": ', '"PDX"}'):
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="", tool_call_chunks=[{"name": None, "args": seg, "id": None, "index": 0}]))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        chunks = list(self._stream(messages))
        msg = chunks[0].message
        for c in chunks[1:]:
            msg = msg + c.message
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=msg.content, tool_calls=getattr(msg, "tool_calls", [])))])


async def test_tool_frames():
    from langgraph.prebuilt import create_react_agent

    adapter = SessionAdapter(graph=create_react_agent(_FakeToolModel(), [get_weather]), agui=True)
    session = adapter.submit_message(None, "weather?")
    await session.current_task
    frames = _drain(session.event_queue)
    starts = [f for f in frames if f["type"] == "tool_start"]
    ends = [f for f in frames if f["type"] == "tool_end"]
    assert starts and starts[0]["name"] == "get_weather" and starts[0]["args"] == {"city": "PDX"}
    assert ends and ends[0]["result"] == "Sunny, 72F"


async def test_interrupt_then_resume_sets_outcomes():
    def gate(state):
        d = interrupt({"action_requests": [{"tool": "approve", "args": {"x": 1}}]})
        return {"messages": [AIMessage(content=f"ok {d}")]}

    b = StateGraph(MessagesState)
    b.add_node("gate", gate)
    b.add_edge(START, "gate")
    b.add_edge("gate", END)
    adapter = SessionAdapter(graph=b.compile(checkpointer=InMemorySaver()), agui=True)

    session = adapter.submit_message("s1", "go")
    await session.current_task
    frames = _drain(session.event_queue)
    assert any(f["type"] == "interrupt" for f in frames)
    assert session.outcome == "interrupted"
    assert session.interrupt and session.interrupt["action_requests"][0]["tool"] == "approve"

    session = adapter.submit_decisions("s1", [{"type": "accept"}])
    await session.current_task
    resumed = _drain(session.event_queue)
    assert session.outcome == "complete"
    assert "ok" in "".join(f["content"] for f in resumed if f["type"] == "content")
