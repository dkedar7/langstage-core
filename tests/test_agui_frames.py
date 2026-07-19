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


# --- gh #89: mixed-mode turn (an earlier node streams, a later node returns a
# finished, non-streamed AIMessage) must render both, not silently drop the finished
# message once anything streamed. ---

def _streaming_model():
    """A keyless, deterministic BaseChatModel that streams tokens ("hi from model")."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    class Streamer(BaseChatModel):
        @property
        def _llm_type(self):
            return "streamer"

        def _generate(self, messages, stop=None, run_manager=None, **k):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hi from model"))])

        def _stream(self, messages, stop=None, run_manager=None, **k):
            for tok in ("hi ", "from ", "model"):
                ch = ChatGenerationChunk(message=AIMessageChunk(content=tok))
                if run_manager:
                    run_manager.on_llm_new_token(tok, chunk=ch)
                yield ch

    return Streamer()


def _mixed_mode_graph():
    """`model` streams tokens; a later `finalize` node appends a finished AIMessage."""
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    model = _streaming_model()
    b = StateGraph(MessagesState)
    b.add_node("model", lambda s: {"messages": [model.invoke(s["messages"])]})
    b.add_node("finalize", lambda s: {"messages": [AIMessage(content="[Reviewed by policy engine]")]})
    b.add_edge(START, "model")
    b.add_edge("model", "finalize")
    b.add_edge("finalize", END)
    return b.compile()


async def test_event_frames_mixed_mode_keeps_finished_message():
    # gh #89: the finished `finalize` message used to vanish — once `model` streamed,
    # the whole final snapshot was suppressed (`and not streamed_text`), dropping every
    # later non-streamed message. It must now render, mapped to its node.
    frames = await _collect(iter_event_frames(build_agent(_mixed_mode_graph()), "hi", "m89e"))
    text = "".join(f["content"] for f in frames if f.get("type") == "content")
    assert "hi from model" in text
    assert "[Reviewed by policy engine]" in text, f"finished message dropped: {text!r}"
    finalize = [
        f for f in frames
        if f.get("type") == "content" and "[Reviewed by policy engine]" in f["content"]
    ]
    assert finalize and finalize[0]["node"] == "finalize", finalize


async def test_chunk_frames_mixed_mode_keeps_finished_message():
    frames = await _collect(iter_chunk_frames(build_agent(_mixed_mode_graph()), "hi", "m89c"))
    text = "".join(f["chunk"] for f in frames if "chunk" in f)
    assert "hi from model" in text
    assert "[Reviewed by policy engine]" in text, f"finished message dropped: {text!r}"


def _single_streaming_graph():
    from langgraph.graph import END, START, MessagesState, StateGraph

    model = _streaming_model()
    b = StateGraph(MessagesState)
    b.add_node("model", lambda s: {"messages": [model.invoke(s["messages"])]})
    b.add_edge(START, "model")
    b.add_edge("model", END)
    return b.compile()


async def test_event_frames_fully_streamed_no_duplicate():
    # Guard the #89 fix: dropping the coarse `not streamed_text` guard must not make a
    # fully-streamed turn re-emit its streamed message from the final snapshot — the
    # streamed message id is deduped.
    frames = await _collect(iter_event_frames(build_agent(_single_streaming_graph()), "hi", "m89dupE"))
    text = "".join(f["content"] for f in frames if f.get("type") == "content")
    assert text == "hi from model", f"duplicated streamed content: {text!r}"


async def test_chunk_frames_fully_streamed_no_duplicate():
    frames = await _collect(iter_chunk_frames(build_agent(_single_streaming_graph()), "hi", "m89dupC"))
    text = "".join(f["chunk"] for f in frames if "chunk" in f)
    assert text == "hi from model", f"duplicated streamed content: {text!r}"


# --- gh #90: GenericToolExtractor's "*" tool_name is the fallback, reachable through
# the supported `extractors=[...]` API. ---

def _custom_tool_graph():
    """A graph that calls one tool with no dedicated extractor, then replies."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool
    def my_custom_tool(q: str) -> str:
        """A custom tool the parser has no dedicated extractor for."""
        return "custom-tool-output"

    class ToolThenDone(BaseChatModel):
        @property
        def _llm_type(self):
            return "t"

        def _generate(self, messages, stop=None, run_manager=None, **k):
            if any(isinstance(m, ToolMessage) for m in messages):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])
            m = AIMessage(content="", tool_calls=[{"name": "my_custom_tool", "args": {"q": "x"}, "id": "c1"}])
            return ChatResult(generations=[ChatGeneration(message=m)])

    model = ToolThenDone()
    b = StateGraph(MessagesState)
    b.add_node("model", lambda s: {"messages": [model.invoke(s["messages"])]})
    b.add_node("tools", ToolNode([my_custom_tool]))
    b.add_edge(START, "model")
    b.add_conditional_edges(
        "model",
        lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
        {"tools": "tools", END: END},
    )
    b.add_edge("tools", "model")
    return b.compile()


async def test_generic_tool_extractor_fires_as_fallback():
    # gh #90: GenericToolExtractor (tool_name == "*") was inert — "*" was a plain
    # dict key no real tool name matched. It's now the fallback for any tool without a
    # dedicated extractor, so it fires here and emits a generic `extraction` frame.
    from langstage_core import GenericToolExtractor

    agent = build_agent(_custom_tool_graph())
    frames = await _collect(
        iter_event_frames(agent, "go", "g90", extractors=[GenericToolExtractor()])
    )
    extraction = [f for f in frames if f["type"] == "extraction"]
    assert extraction, f"generic fallback never fired: {[f['type'] for f in frames]}"
    assert extraction[0]["tool_name"] == "my_custom_tool"
    assert extraction[0]["extracted_type"] == "tool_call"
    assert extraction[0]["data"] == {"content": "custom-tool-output"}


async def test_specific_extractor_wins_over_generic_fallback():
    # A dedicated extractor must still take precedence over the "*" fallback when both
    # are registered — the fallback only applies to tools without a specific match.
    from langstage_core import GenericToolExtractor

    class MyToolExtractor:
        tool_name = "my_custom_tool"
        extracted_type = "custom_specific"

        def extract(self, content):
            return {"specific": content}

    agent = build_agent(_custom_tool_graph())
    frames = await _collect(
        iter_event_frames(
            agent, "go", "g90b", extractors=[MyToolExtractor(), GenericToolExtractor()]
        )
    )
    extraction = [f for f in frames if f["type"] == "extraction"]
    assert extraction and extraction[0]["extracted_type"] == "custom_specific", extraction
    assert extraction[0]["data"] == {"specific": "custom-tool-output"}


# --- gh #92: `extractors=[...]` works on the chunk wire too (the README advertises it
# for *both* `iter_*` mappings, and the CLI/Jupyter surfaces are on this wire). It emits
# an `extraction` chunk — parity with iter_event_frames' `extraction` frame. ---

async def test_iter_chunk_frames_accepts_extractors_param():
    # The documented call from the README (`extractors=[...]` on an `iter_*` mapping) must
    # not raise TypeError on the chunk mapping — the exact adopter crash in the report.
    import inspect

    assert "extractors" in inspect.signature(iter_chunk_frames).parameters


async def test_iter_chunk_frames_runs_extractors():
    from langstage_core import GenericToolExtractor

    agent = build_agent(_custom_tool_graph())
    frames = await _collect(
        iter_chunk_frames(agent, "go", "c92", extractors=[GenericToolExtractor()])
    )
    extraction = [f["extraction"] for f in frames if "extraction" in f]
    assert extraction, f"no extraction chunk emitted: {[f for f in frames]}"
    assert extraction[0]["tool_name"] == "my_custom_tool"
    assert extraction[0]["extracted_type"] == "tool_call"
    assert extraction[0]["data"] == {"content": "custom-tool-output"}
    # The extraction chunk rides the normal stream — it must still terminate cleanly.
    assert frames[-1] == {"status": "complete"}


async def test_iter_chunk_frames_specific_extractor_wins_over_generic_fallback():
    from langstage_core import GenericToolExtractor

    class MyToolExtractor:
        tool_name = "my_custom_tool"
        extracted_type = "custom_specific"

        def extract(self, content):
            return {"specific": content}

    agent = build_agent(_custom_tool_graph())
    frames = await _collect(
        iter_chunk_frames(
            agent, "go", "c92b", extractors=[MyToolExtractor(), GenericToolExtractor()]
        )
    )
    extraction = [f["extraction"] for f in frames if "extraction" in f]
    assert extraction and extraction[0]["extracted_type"] == "custom_specific", extraction
    assert extraction[0]["data"] == {"specific": "custom-tool-output"}


async def test_iter_chunk_frames_without_extractors_emits_no_extraction():
    # Opt-in and backward compatible: with no extractors, the chunk wire is byte-for-byte
    # its old self — no `extraction` chunk — so existing CLI/Jupyter consumers are unchanged.
    agent = build_agent(_custom_tool_graph())
    frames = await _collect(iter_chunk_frames(agent, "go", "c92c"))
    assert not any("extraction" in f for f in frames), frames


# --- gh #93: a node/graph exception during streaming surfaces as the documented terminal
# `error` frame instead of propagating out of the iterator and crashing the consumer's
# `async for` (the README Quick start / shipped WebSocket example rely on the `error`
# frame). Parity with the resilience build_app / SessionAdapter already provide. ---

def _raising_graph():
    """A graph whose only node raises mid-run — the common failure path (a node/tool/model
    call that throws) the documented consumer loop must survive as an `error` frame."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    def boom(state):
        raise RuntimeError("model call failed")

    b = StateGraph(MessagesState)
    b.add_node("boom", boom)
    b.add_edge(START, "boom")
    b.add_edge("boom", END)
    return b.compile()


async def test_event_frames_node_exception_becomes_terminal_error_frame():
    # Before the fix, `_collect` re-raises the propagated RuntimeError (the consumer crash);
    # after, the exception is a terminal `error` frame and iteration ends cleanly.
    frames = await _collect(iter_event_frames(build_agent(_raising_graph()), "hi", "e93"))
    assert frames[-1]["type"] == "error", frames
    assert "RuntimeError" in frames[-1]["error"] and "model call failed" in frames[-1]["error"]
    # Terminal: the error frame is the last thing yielded — no misleading trailing `complete`.
    assert not any(f.get("type") == "complete" for f in frames), frames


async def test_chunk_frames_node_exception_becomes_terminal_error_frame():
    frames = await _collect(iter_chunk_frames(build_agent(_raising_graph()), "hi", "c93"))
    assert frames[-1]["status"] == "error", frames
    assert "RuntimeError" in frames[-1]["error"] and "model call failed" in frames[-1]["error"]
    assert not any(f.get("status") == "complete" for f in frames), frames


# --- gh #102: `max_result_len` is honored on BOTH `iter_*` mappings. The event wire has
# always capped its `tool_end` result at 500 with a `…(truncated)` marker; the chunk wire —
# the one the README assigns to the CLI and Jupyter surfaces — emitted the full result and
# had no `max_result_len` parameter at all, so a large tool blob flooded the terminal with
# no knob to bound it. Same parity-gap class as #92 (`extractors`). ---

def _big_result_graph(size=2000):
    """A graph whose single tool returns a `size`-char blob — the file read / search dump /
    API response that floods a terminal when the wire doesn't cap it."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool
    def big(q: str) -> str:
        """Return a big result."""
        return "X" * size

    def call(state):
        return {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "big", "id": "b1", "args": {"q": "go"}}])
            ]
        }

    b = StateGraph(MessagesState)
    b.add_node("call", call)
    b.add_node("tools", ToolNode([big]))
    b.add_edge(START, "call")
    b.add_edge("call", "tools")
    b.add_edge("tools", END)
    return b.compile()


async def _chunk_tool_results(agent, thread_id, **kw):
    frames = await _collect(iter_chunk_frames(agent, "go", thread_id, **kw))
    return [f["tool_result"] for f in frames if "tool_result" in f]


async def test_iter_chunk_frames_accepts_max_result_len_param():
    # The knob the report asks for: `max_result_len` must exist on the chunk mapping, with
    # the same default as its counterpart, so the two signatures agree.
    import inspect

    params = inspect.signature(iter_chunk_frames).parameters
    assert "max_result_len" in params
    assert params["max_result_len"].default == 500
    assert (
        params["max_result_len"].default
        == inspect.signature(iter_event_frames).parameters["max_result_len"].default
    )


async def test_iter_chunk_frames_truncates_tool_result_by_default():
    # Before the fix this yielded all 2000 characters, uncapped, straight to the terminal.
    results = await _chunk_tool_results(build_agent(_big_result_graph()), "c102a")
    assert results, "no tool_result chunk emitted"
    assert len(results[0]) == 500 + len("…(truncated)")
    assert results[0].endswith("…(truncated)")
    assert results[0][:500] == "X" * 500


async def test_iter_chunk_frames_truncation_matches_iter_event_frames():
    # The whole point of #102: the counterparts must produce the SAME capped string, so
    # neither marker nor boundary can drift again.
    agent = build_agent(_big_result_graph())
    chunk = (await _chunk_tool_results(agent, "c102b"))[0]
    event_frames = await _collect(iter_event_frames(agent, "go", "e102b"))
    event = [f["result"] for f in event_frames if f.get("type") == "tool_end"][0]
    assert chunk == event


async def test_iter_chunk_frames_honors_custom_max_result_len():
    results = await _chunk_tool_results(build_agent(_big_result_graph()), "c102c", max_result_len=10)
    assert results[0] == "X" * 10 + "…(truncated)"


async def test_iter_chunk_frames_leaves_short_result_untouched():
    # The boundary: a result at or under the cap is emitted byte-identical, with no marker —
    # so every existing CLI/Jupyter consumer of a normal-sized result is unchanged.
    results = await _chunk_tool_results(build_agent(_big_result_graph(size=500)), "c102d")
    assert results[0] == "X" * 500
    assert "(truncated)" not in results[0]


async def test_iter_chunk_frames_extractor_still_sees_the_full_result():
    # Truncation is a *display* concern: the extractor parses the payload, so it must keep
    # receiving the untruncated content (iter_event_frames feeds its extractor the raw
    # content for the same reason). Capping the extractor's input would corrupt parsed data.
    from langstage_core import GenericToolExtractor

    frames = await _collect(
        iter_chunk_frames(
            build_agent(_big_result_graph()), "go", "c102e", extractors=[GenericToolExtractor()]
        )
    )
    extraction = [f["extraction"] for f in frames if "extraction" in f]
    assert extraction, f"no extraction chunk emitted: {frames}"
    assert extraction[0]["data"] == {"content": "X" * 2000}


async def test_truncate_result_passes_non_str_through_untouched():
    # A non-str tool result (a dict/list/None a fake or future adapter yields) must not be
    # sliced — `{...}[:500]` is a TypeError. AG-UI types `content` as str, so this is the
    # defensive path both mappings share.
    from langstage_core.agui import _truncate_result

    assert _truncate_result({"a": 1}, 1) == {"a": 1}
    assert _truncate_result([1, 2, 3], 1) == [1, 2, 3]
    assert _truncate_result(None, 1) is None
