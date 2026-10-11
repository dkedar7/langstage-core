"""A turn's separate AIMessages join with a paragraph break, not inline (gh #200).

The README's frame contract: a change of ``message_id`` between two text frames is
a message boundary, joined with a paragraph break. The CLI and the VS Code panel
already follow it (langstage-cli#43, langstage-vscode#108); these tests hold core's
own one-shot results to it: ``TurnResult.text`` from both collectors and
``run_turn``, ``langstage-agui -m --json``, and the task engine's ``task["result"]``.
"""

import asyncio
import json
import textwrap

import pytest

from langstage_core._text import TextJoiner

ROUTER_THEN_ANSWER = "Routing your request.\n\nFinal canned answer."

# The issue's repro: a router AIMessage with a tool call, the tool's reply, then
# the answer AIMessage. Two assistant messages from two nodes in one turn.
FIN_AGENT = textwrap.dedent(
    '''
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def router(state):
        return {"messages": [AIMessage(
            content="Routing your request.",
            tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1"}],
        )]}

    def tool(state):
        return {"messages": [ToolMessage(content="found it", tool_call_id="c1", name="lookup")]}

    def answer(state):
        return {"messages": [AIMessage(content="Final canned answer.")]}

    g = StateGraph(MessagesState)
    g.add_node("router", router)
    g.add_node("tool", tool)
    g.add_node("answer", answer)
    g.add_edge(START, "router")
    g.add_edge("router", "tool")
    g.add_edge("tool", "answer")
    g.add_edge("answer", END)
    graph = g.compile()
    '''
)


def _fin_graph():
    ns: dict = {}
    exec(compile(FIN_AGENT, "fin_agent.py", "exec"), ns)
    return ns["graph"]


# ── The joining rule ─────────────────────────────────────────────────────────


def _join(*deltas):
    j = TextJoiner()
    for text, node, mid in deltas:
        j.add(text, node=node, message_id=mid)
    return j.text()


def test_deltas_of_one_message_join_unbroken():
    assert _join(("Hel", "agent", "m1"), ("lo", "agent", "m1"), (" there", "agent", "m1")) == "Hello there"


def test_a_new_message_id_is_a_paragraph_break():
    assert _join(("Let me add those.", "agent", "m1"), ("The sum is 42.", "agent", "m2")) == (
        "Let me add those.\n\nThe sum is 42."
    )


def test_a_new_node_is_a_boundary_without_message_ids():
    assert _join(("a", "router", None), ("b", "answer", None)) == "a\n\nb"


def test_missing_message_ids_in_one_node_join_unbroken():
    assert _join(("a", "agent", None), ("b", "agent", None)) == "ab"


def test_existing_trailing_newlines_are_topped_up_not_doubled():
    assert _join(("a\n", "x", "m1"), ("b", "x", "m2")) == "a\n\nb"
    assert _join(("a\n\n", "x", "m1"), ("b", "x", "m2")) == "a\n\nb"


def test_no_break_before_the_first_text_or_after_whitespace_only_text():
    assert _join(("", "x", "m1"), ("a", "x", "m2")) == "a"
    assert _join(("  ", "x", "m1"), ("a", "x", "m2")) == "  a"


# ── TurnResult.text on both collectors, and run_turn ────────────────────────


@pytest.fixture
def agui():
    pytest.importorskip("ag_ui_langgraph")
    pytest.importorskip("fastapi")
    import langstage_core.agui as agui

    return agui


async def test_event_collector_breaks_between_messages(agui):
    r = await agui.collect_event_frames(agui.build_agent(_fin_graph()), "hi", "s1")
    assert r.outcome == "complete"
    assert r.text == ROUTER_THEN_ANSWER


async def test_chunk_collector_breaks_between_messages(agui):
    r = await agui.collect_chunk_frames(agui.build_agent(_fin_graph()), "hi", "s1")
    assert r.outcome == "complete"
    assert r.text == ROUTER_THEN_ANSWER


def test_run_turn_breaks_between_messages(agui):
    assert agui.run_turn(_fin_graph(), "hi").text == ROUTER_THEN_ANSWER


def test_message_json_breaks_between_messages(agui, tmp_path, capsys):
    from langstage_core.agui.__main__ import main

    (tmp_path / "fin_agent.py").write_text(FIN_AGENT, encoding="utf-8")
    rc = main(["--agent", f"{tmp_path / 'fin_agent.py'}:graph", "-m", "hi", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["text"] == ROUTER_THEN_ANSWER


# ── task["result"] ──────────────────────────────────────────────────────────


async def test_task_result_breaks_between_messages():
    from langstage_core.adapters.session import SessionAdapter
    from langstage_core.tasks import DONE, InMemoryTaskStore, TaskRunner

    store = InMemoryTaskStore()
    runner = TaskRunner(
        SessionAdapter(graph=_fin_graph(), stream_mode="updates"), store,
        concurrency=1, poll_interval=0.05,
    )
    await runner.start()
    try:
        tid = await runner.enqueue(title="route", prompt="hi")
        loop = asyncio.get_running_loop()
        end = loop.time() + 5.0
        row = await store.get(tid)
        while row["state"] != DONE and loop.time() < end:
            await asyncio.sleep(0.02)
            row = await store.get(tid)
        assert row["state"] == DONE, row
        assert row["result"] == ROUTER_THEN_ANSWER
    finally:
        await runner.shutdown()
