"""TaskRunner.shutdown() must not hang when its cancel races a wake (Python 3.11).

On Python 3.11, ``asyncio.wait_for`` returns normally, dropping the cancel, when the
cancel lands just as the awaited future completes (CPython gh-86296; 3.12 rewrote
``wait_for``). The idle worker waited on its wake event through ``wait_for``, so a
cancel that raced a wake was lost and ``shutdown()``'s ``gather`` waited forever. A
server that starts and stops at once (its lifespan wakes the workers) hung on the
way down, which is what stalled langstage's Python 3.11 CI.
"""

import asyncio

import pytest
from langgraph.graph import END, START, MessagesState, StateGraph

from langstage_core.adapters.session import SessionAdapter
from langstage_core.tasks import InMemoryTaskStore, TaskRunner


def _graph():
    b = StateGraph(MessagesState)
    b.add_node("noop", lambda state: {"messages": []})
    b.add_edge(START, "noop")
    b.add_edge("noop", END)
    return b.compile()


# Each value lands the cancel at a different step of the wake's hand-off, so one of
# them hits the window where 3.11's wait_for swallows it.
@pytest.mark.parametrize("steps", range(8))
async def test_shutdown_survives_a_cancel_that_races_the_wake(steps):
    runner = TaskRunner(
        SessionAdapter(graph=_graph()), InMemoryTaskStore(), concurrency=1, poll_interval=60
    )
    await runner.start()
    for _ in range(20):  # let the worker find no work and park on the wake
        await asyncio.sleep(0)
    runner._wake.set()
    for _ in range(steps):
        await asyncio.sleep(0)
    await asyncio.wait_for(runner.shutdown(), timeout=5)
