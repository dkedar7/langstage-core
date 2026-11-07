"""A finished message is synthesized once, under its final id (gh #229).

The served wire synthesizes TEXT_MESSAGE_* for finished messages from checkpoint
reads (gh #140). A read of the latest state also applies the running step's pending
writes, re-running the messages reducer, so an id-less ``AIMessage("...")`` got a new
uuid on every such read. A flush that landed in that window emitted the message under
one id, and a later flush, reading the committed checkpoint, emitted it again under
another: ``test_served_and_in_process_wires_agree`` intermittently saw
``ANSWER: the sky is blue.`` twice on the served side.

A thread whose step failed in one branch while a sibling branch finished holds that
sibling's writes as pending writes indefinitely, so it reproduces the window
deterministically.
"""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from langstage_core.agui import _TurnStream  # noqa: E402


def _thread_with_pending_writes():
    def answer(state):
        return {"messages": [AIMessage(content="ANSWER: the sky is blue.")]}  # no id

    async def boom(state):
        await asyncio.sleep(0.2)  # let the sibling finish and record its writes first
        raise RuntimeError("second branch failed")

    b = StateGraph(MessagesState)
    b.add_node("answer", answer)
    b.add_node("boom", boom)
    b.add_edge(START, "answer")
    b.add_edge(START, "boom")
    b.add_edge("answer", END)
    b.add_edge("boom", END)
    graph = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-229"}}

    async def run():
        with pytest.raises(RuntimeError):
            await graph.ainvoke({"messages": [("user", "hi")]}, cfg)
        state = await graph.aget_state(cfg)
        # The precondition: the latest-state read applies the sibling's pending write.
        assert any(t.result for t in state.tasks)
        return graph

    return asyncio.run(run())


def _stream(graph):
    ts = _TurnStream(SimpleNamespace(graph=graph), SimpleNamespace(thread_id="t-229"))
    ts._pre_ids = set()
    return ts


def test_checkpoint_reads_give_stable_ids():
    graph = _thread_with_pending_writes()
    ts = _stream(graph)

    async def reads():
        return [[m.id for m in await ts._checkpoint_messages()] for _ in range(3)]

    first, second, third = asyncio.run(reads())
    assert first == second == third


def test_repeated_flushes_never_emit_a_message_twice():
    graph = _thread_with_pending_writes()
    ts = _stream(graph)

    async def flushes():
        out = []
        for _ in range(3):
            out += [ev async for ev in ts._flush()]
        return out

    events = asyncio.run(flushes())
    texts = [e.delta for e in events if type(e).__name__ == "TextMessageContentEvent"]
    assert texts.count("ANSWER: the sky is blue.") <= 1
    starts = [e.message_id for e in events if type(e).__name__ == "TextMessageStartEvent"]
    assert len(starts) == len(set(starts))
