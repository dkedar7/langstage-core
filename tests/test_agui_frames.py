"""The two shared AG-UI mapping helpers in the agui module (ADR 0002 dedupe).

``iter_event_frames`` -> event_to_dict wire (vscode + web).
``iter_chunk_frames`` -> stream_graph_updates chunk-dict wire (cli + jupyter).
Skipped unless the agui extra is installed (dev pulls it, so CI runs it).
"""
import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langgraph_stream_parser import load_agent_spec
from langgraph_stream_parser.agui import build_agent, iter_chunk_frames, iter_event_frames

pytestmark = pytest.mark.asyncio


async def _collect(aiter):
    return [frame async for frame in aiter]


async def test_iter_chunk_frames_shape():
    agent = build_agent(load_agent_spec("langgraph_stream_parser.demo.stub:graph"))
    frames = await _collect(iter_chunk_frames(agent, "chunk wire", "t1"))
    assert frames[-1] == {"status": "complete"}
    content = [f for f in frames if f.get("status") == "streaming" and "chunk" in f]
    assert content and all(set(f) == {"status", "chunk", "node"} for f in content)
    assert "chunk wire" in "".join(f["chunk"] for f in content)


async def test_iter_event_frames_shape():
    agent = build_agent(load_agent_spec("langgraph_stream_parser.demo.stub:graph"))
    frames = await _collect(iter_event_frames(agent, "event wire", "t2"))
    assert frames[-1] == {"type": "complete"}
    content = [f for f in frames if f.get("type") == "content"]
    assert content and all(set(f) == {"type", "content", "role", "node"} for f in content)
    assert "event wire" in "".join(f["content"] for f in content)


async def test_iter_event_frames_runs_extractors():
    """extractors= runs a matching extractor over each tool result and emits an
    `extraction` frame (ADR 0003 Stage 1, productionized)."""
    from typing import Iterator, List

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def note(text: str) -> str:
        """Write a note."""
        return '{"saved": true}'

    class M(BaseChatModel):
        @property
        def _llm_type(self):
            return "m"

        def bind_tools(self, tools, **k):
            return self

        def _stream(self, messages: List[BaseMessage], stop=None, run_manager=None, **k) -> Iterator[ChatGenerationChunk]:
            if any(isinstance(m, ToolMessage) for m in messages):
                yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": "note", "args": '{"text": "hi"}', "id": "c1", "index": 0}]))

        def _generate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
            cs = list(self._stream(messages)); msg = cs[0].message
            for c in cs[1:]:
                msg = msg + c.message
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=msg.content, tool_calls=getattr(msg, "tool_calls", [])))])

    class NoteExtractor:
        tool_name = "note"
        extracted_type = "note_saved"

        def extract(self, content):
            import json
            d = json.loads(content) if isinstance(content, str) else content
            return {"ok": bool(d.get("saved"))} if isinstance(d, dict) else None

    agent = build_agent(create_react_agent(M(), [note]))
    frames = await _collect(iter_event_frames(agent, "note hi", "tE", extractors=[NoteExtractor()]))
    extraction = [f for f in frames if f["type"] == "extraction"]
    assert extraction and extraction[0]["extracted_type"] == "note_saved"
    assert extraction[0]["data"] == {"ok": True}


async def test_iter_event_frames_state_passthrough():
    """state= seeds the graph input so agents with a richer contract than
    `messages` (e.g. hermes' iteration_budget) get their extra keys."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph
    from typing import Any as _Any
    from typing_extensions import TypedDict

    class S(TypedDict):
        messages: _Any
        budget: int

    def node(st):
        return {"messages": [AIMessage(content=f"budget={st.get('budget')}")]}

    b = StateGraph(S); b.add_node("n", node); b.add_edge(START, "n"); b.add_edge("n", END)
    agent = build_agent(b.compile(checkpointer=InMemorySaver()))
    frames = await _collect(iter_event_frames(agent, "hi", "tS", state={"budget": 7}))
    text = "".join(f["content"] for f in frames if f.get("type") == "content")
    assert "budget=7" in text, frames
