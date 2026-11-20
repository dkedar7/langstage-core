"""TaskRunner.retry continues a run that stopped partway from its checkpoint (gh #249).

retry() used to re-queue the task and send its prompt again: the graph started over
from START, so steps that had already finished (and their side effects) ran twice,
and the prompt appeared twice in the thread.
"""
import asyncio

import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from langstage_core import SessionAdapter  # noqa: E402
from langstage_core.tasks import (  # noqa: E402
    CANCELLED,
    DONE,
    FAILED,
    ONGOING,
    REVIEW_NEEDED,
    TERMINAL_STATES,
    InMemoryTaskStore,
    TaskRunner,
)


async def _wait(store, tid, states, timeout=10.0):
    states = set(states)
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        row = await store.get(tid)
        if row["state"] in states:
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f"task {tid} never reached {states}: {row}")


def _two_step_graph(side_effects, attempts, *, checkpointer=True):
    """do_work (side effect) -> finish (raises on its first call)."""

    def do_work(state):
        side_effects.append("work done")
        return {"messages": [AIMessage(content="step 1 done.")]}

    def finish(state):
        attempts["finish"] += 1
        if attempts["finish"] == 1:
            raise RuntimeError("transient 500")
        return {"messages": [AIMessage(content="step 2 done.")]}

    g = StateGraph(MessagesState)
    g.add_node("do_work", do_work)
    g.add_node("finish", finish)
    g.add_edge(START, "do_work")
    g.add_edge("do_work", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=InMemorySaver()) if checkpointer else g.compile()


async def _thread_messages(adapter, thread_id):
    st = await adapter._agui_agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [(m.type, m.content) for m in st.values["messages"]]


@pytest.mark.parametrize("checkpointer", [True, False], ids=["own-saver", "default-saver"])
async def test_retry_after_failure_continues_from_checkpoint(checkpointer):
    side_effects, attempts = [], {"finish": 0}
    adapter = SessionAdapter(graph=_two_step_graph(side_effects, attempts, checkpointer=checkpointer))
    store = InMemoryTaskStore()
    runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
    await runner.start()
    try:
        tid = await runner.enqueue(title="job", prompt="do the job")
        row = await _wait(store, tid, TERMINAL_STATES)
        assert row["state"] == FAILED and "transient 500" in row["error"]
        assert await adapter.has_unfinished_run(row["thread_id"]) is True

        assert await runner.retry(tid) is True
        row = await _wait(store, tid, {DONE, FAILED})
        assert row["state"] == DONE, row
        # The finished step did not run again ...
        assert side_effects == ["work done"]
        assert attempts["finish"] == 2
        # ... the prompt is in the thread once, and the result is the resumed run's.
        msgs = await _thread_messages(adapter, row["thread_id"])
        assert [m for m in msgs if m[0] == "human"] == [("human", "[Async task: job]\n\ndo the job")]
        assert msgs[-2:] == [("ai", "step 1 done."), ("ai", "step 2 done.")]
        assert row["result"] == "step 2 done."
        assert await adapter.has_unfinished_run(row["thread_id"]) is False
    finally:
        await runner.shutdown()


async def test_retry_of_task_cancelled_while_queued_is_a_fresh_turn():
    side_effects, attempts = [], {"finish": 1}  # finish never fails here
    adapter = SessionAdapter(graph=_two_step_graph(side_effects, attempts))
    store = InMemoryTaskStore()
    runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)  # not started
    tid = await runner.enqueue(title="job", prompt="do the job")
    assert await runner.cancel(tid) is True
    assert (await store.get(tid))["state"] == CANCELLED
    assert await runner.retry(tid) is True
    await runner.start()
    try:
        row = await _wait(store, tid, {DONE, FAILED})
        assert row["state"] == DONE
        assert side_effects == ["work done"]
        msgs = await _thread_messages(adapter, row["thread_id"])
        assert [m for m in msgs if m[0] == "human"] == [("human", "[Async task: job]\n\ndo the job")]
    finally:
        await runner.shutdown()


async def test_retry_after_cancel_mid_run_continues():
    side_effects = []
    gate = asyncio.Event()
    calls = {"slow": 0}

    def do_work(state):
        side_effects.append("work done")
        return {"messages": [AIMessage(content="step 1 done.")]}

    async def slow(state):
        calls["slow"] += 1
        if calls["slow"] == 1:
            gate.set()
            await asyncio.sleep(30)  # cancelled here
        return {"messages": [AIMessage(content="step 2 done.")]}

    g = StateGraph(MessagesState)
    g.add_node("do_work", do_work)
    g.add_node("slow", slow)
    g.add_edge(START, "do_work")
    g.add_edge("do_work", "slow")
    g.add_edge("slow", END)
    adapter = SessionAdapter(graph=g.compile(checkpointer=InMemorySaver()))
    store = InMemoryTaskStore()
    runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
    await runner.start()
    try:
        tid = await runner.enqueue(title="job", prompt="do the job")
        await _wait(store, tid, {ONGOING})
        await asyncio.wait_for(gate.wait(), 10)
        assert await runner.cancel(tid) is True
        await _wait(store, tid, {CANCELLED})
        await asyncio.sleep(0.1)
        assert await runner.retry(tid) is True
        row = await _wait(store, tid, {DONE, FAILED})
        assert row["state"] == DONE
        assert side_effects == ["work done"] and calls["slow"] == 2
        msgs = await _thread_messages(adapter, row["thread_id"])
        assert sum(1 for m in msgs if m[0] == "human") == 1
    finally:
        await runner.shutdown()


async def test_retry_of_task_cancelled_at_an_approval_brings_the_approval_back():
    side_effects = []

    def do_work(state):
        side_effects.append("work done")
        return {"messages": [AIMessage(content="step 1 done.")]}

    def ask(state):
        answer = interrupt({"question": "go ahead?"})
        return {"messages": [AIMessage(content=f"answer: {answer}")]}

    g = StateGraph(MessagesState)
    g.add_node("do_work", do_work)
    g.add_node("ask", ask)
    g.add_edge(START, "do_work")
    g.add_edge("do_work", "ask")
    g.add_edge("ask", END)
    adapter = SessionAdapter(graph=g.compile(checkpointer=InMemorySaver()))
    store = InMemoryTaskStore()
    runner = TaskRunner(adapter, store, concurrency=1, poll_interval=0.05)
    await runner.start()
    try:
        tid = await runner.enqueue(title="job", prompt="do the job")
        row = await _wait(store, tid, {REVIEW_NEEDED, FAILED, DONE})
        assert row["state"] == REVIEW_NEEDED and row["interrupt"]
        assert await runner.cancel(tid) is True
        assert await runner.retry(tid) is True
        row = await _wait(store, tid, {REVIEW_NEEDED, FAILED, DONE})
        assert row["state"] == REVIEW_NEEDED and row["interrupt"], row
        assert side_effects == ["work done"]
        msgs = await _thread_messages(adapter, row["thread_id"])
        assert sum(1 for m in msgs if m[0] == "human") == 1
    finally:
        await runner.shutdown()
