"""The two shared AG-UI mapping helpers in the agui module (ADR 0002 dedupe).

``iter_event_frames`` -> event_to_dict wire (vscode + web).
``iter_chunk_frames`` -> stream_graph_updates chunk-dict wire (cli + jupyter).
Skipped unless the agui extra is installed (dev pulls it, so CI runs it).
"""
import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langstage_core import load_agent_spec
from langstage_core.agui import build_agent, iter_chunk_frames, iter_event_frames

pytestmark = pytest.mark.asyncio


async def _collect(aiter):
    return [frame async for frame in aiter]


async def test_iter_chunk_frames_shape():
    agent = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    frames = await _collect(iter_chunk_frames(agent, "chunk wire", "t1"))
    assert frames[-1] == {"status": "complete"}
    content = [f for f in frames if f.get("status") == "streaming" and "chunk" in f]
    assert content and all(set(f) == {"status", "chunk", "node"} for f in content)
    assert "chunk wire" in "".join(f["chunk"] for f in content)


async def test_iter_event_frames_shape():
    agent = build_agent(load_agent_spec("langstage_core.demo.stub:graph"))
    frames = await _collect(iter_event_frames(agent, "event wire", "t2"))
    assert frames[-1] == {"type": "complete"}
    content = [f for f in frames if f.get("type") == "content"]
    assert content and all(set(f) == {"type", "content", "role", "node"} for f in content)
    assert "event wire" in "".join(f["content"] for f in content)


def _two_node_graph():
    """A graph with two distinct content-emitting nodes ("first", "second")."""
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("first", lambda s: {"messages": [AIMessage(content="from first.")]})
    b.add_node("second", lambda s: {"messages": [AIMessage(content="from second.")]})
    b.add_edge(START, "first")
    b.add_edge("first", "second")
    b.add_edge("second", END)
    return b.compile()


async def test_chunk_frames_carry_real_node_names():
    # gh #43: a multi-node graph's chunks used to all report node="agent", so the
    # CLI (which separates output on a node change) rendered them as one run-on. The
    # node now comes from the langgraph step, so distinct nodes are distinguishable.
    frames = await _collect(iter_chunk_frames(build_agent(_two_node_graph()), "hi", "tn1"))
    by_text = {f["chunk"]: f["node"] for f in frames if "chunk" in f}
    assert by_text.get("from first.") == "first"
    assert by_text.get("from second.") == "second"


async def test_event_frames_carry_real_node_names():
    frames = await _collect(iter_event_frames(build_agent(_two_node_graph()), "hi", "tn2"))
    by_text = {f["content"]: f["node"] for f in frames if f.get("type") == "content"}
    assert by_text.get("from first.") == "first"
    assert by_text.get("from second.") == "second"


def _echo_graph():
    """A non-token (snapshot-delivered) agent that echoes the last user message."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    b = StateGraph(MessagesState)
    b.add_node("echo", lambda s: {"messages": [AIMessage(content=f"reply to: {s['messages'][-1].content}")]})
    b.add_edge(START, "echo")
    b.add_edge("echo", END)
    return b.compile(checkpointer=InMemorySaver())


async def test_snapshot_agent_does_not_replay_history():
    # gh #67: a non-token agent delivers content via the final snapshot, which is the
    # FULL thread. Turn 2 must emit only turn 2's reply, not re-render turn 1.
    agent = build_agent(_echo_graph())

    # chunk-frames turn
    t1 = [f["chunk"] for f in await _collect(iter_chunk_frames(agent, "ONE", "t67")) if "chunk" in f]
    t2 = [f["chunk"] for f in await _collect(iter_chunk_frames(agent, "TWO", "t67")) if "chunk" in f]
    assert any("TWO" in c for c in t2)
    assert not any("ONE" in c for c in t2), f"replayed history: {t2}"
    # event-frames turn (separate thread)
    e1 = [f["content"] for f in await _collect(iter_event_frames(agent, "ONE", "t67e")) if f.get("type") == "content"]
    e2 = [f["content"] for f in await _collect(iter_event_frames(agent, "TWO", "t67e")) if f.get("type") == "content"]
    assert any("TWO" in c for c in e2) and not any("ONE" in c for c in e2), f"replayed history: {e2}"


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


# The mappers dispatch on type(ev).__name__, so the fakes just need the AG-UI class
# names + a `.delta` — name the classes exactly what ag-ui-langgraph emits.
class ReasoningMessageContentEvent:
    def __init__(self, delta):
        self.delta = delta


class TextMessageContentEvent:
    def __init__(self, delta):
        self.delta = delta


class _FakeReasoningAgent:
    """An agent whose run() emits the exact event sequence ag-ui-langgraph produces
    for a reasoning model: reasoning deltas, then the answer text."""

    async def run(self, run_input):
        for d in ("Let me think... ", "2+2=4. "):
            yield ReasoningMessageContentEvent(d)
        yield TextMessageContentEvent("The answer is 4.")


async def test_iter_event_frames_emits_reasoning():
    # gh #71: reasoning-model chain-of-thought must surface as a `reasoning` frame,
    # not be dropped — and stay separate from the `content` answer.
    frames = await _collect(iter_event_frames(_FakeReasoningAgent(), "think", "tr"))
    reasoning = [f for f in frames if f.get("type") == "reasoning"]
    assert reasoning, frames
    assert "".join(f["content"] for f in reasoning) == "Let me think... 2+2=4. "
    content = [f for f in frames if f.get("type") == "content"]
    assert "".join(f["content"] for f in content) == "The answer is 4."


async def test_iter_chunk_frames_emits_reasoning():
    frames = await _collect(iter_chunk_frames(_FakeReasoningAgent(), "think", "tr"))
    reasoning = [f["reasoning"] for f in frames if "reasoning" in f]
    assert "".join(reasoning) == "Let me think... 2+2=4. "
    chunks = [f["chunk"] for f in frames if "chunk" in f]
    assert "".join(chunks) == "The answer is 4."
