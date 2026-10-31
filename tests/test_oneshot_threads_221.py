"""`langstage-agui -m` / `--json` / `--verify` run each invocation on a fresh thread (gh #221).

They used the constant threads "oneshot" / "verify". With an agent that brings its own
durable checkpointer, every run then saw the earlier runs' history, and one paused run
left every later `-m` replaying that interrupt (exit 2). The agent here keeps one
module-level saver across `main()` calls, standing in for a SQLite/Postgres store.
"""
import json
import sys
import types

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langstage_core.agui import averify, build_agent
from langstage_core.agui.__main__ import main

MOD = "_lc221_durable_agent"


def _node(state):
    if state["messages"][-1].content == "ask me":
        interrupt({"question": "name?"})
    humans = [m.content for m in state["messages"] if m.type == "human"]
    return {"messages": [AIMessage(content=f"seen {len(humans)}: {humans}")]}


@pytest.fixture
def spec(monkeypatch):
    g = StateGraph(MessagesState)
    g.add_node("n", _node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    saver = InMemorySaver()
    mod = types.ModuleType(MOD)
    mod.graph = g.compile(checkpointer=saver)
    mod.saver = saver
    monkeypatch.setitem(sys.modules, MOD, mod)
    return f"{MOD}:graph"


def _threads(saver):
    return {c.config["configurable"]["thread_id"] for c in saver.list(None)}


def test_message_runs_do_not_share_history(spec, capsys):
    assert main(["-a", spec, "-m", "run 1"]) == 0
    assert main(["-a", spec, "-m", "run 2"]) == 0
    out = capsys.readouterr().out
    assert "seen 1: ['run 1']" in out
    assert "seen 1: ['run 2']" in out  # was "seen 2: ['run 1', 'run 2']"


def test_a_paused_run_does_not_stick_later_runs(spec, capsys):
    assert main(["-a", spec, "-m", "ask me"]) == 2
    assert main(["-a", spec, "-m", "hello"]) == 0  # was 2: the old interrupt replayed
    assert main(["-a", spec, "--json", "-m", "hello again"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["outcome"] == "complete"
    assert payload["text"] == "seen 1: ['hello again']"


def test_each_invocation_uses_its_own_oneshot_thread(spec, capsys):
    for m in ("a", "b"):
        main(["-a", spec, "-m", m])
    main(["-a", spec, "--json", "-m", "c"])
    threads = _threads(sys.modules[MOD].saver)
    assert len(threads) == 3
    assert all(t.startswith("oneshot-") for t in threads)


def test_verify_uses_a_fresh_thread_per_probe(spec, capsys):
    assert main(["-a", spec, "--verify"]) == 0
    assert main(["-a", spec, "--verify"]) == 0
    threads = _threads(sys.modules[MOD].saver)
    assert len(threads) == 2 and all(t.startswith("verify-") for t in threads)


async def test_averify_keeps_an_explicit_thread_id(spec):
    agent = build_agent(sys.modules[MOD].graph)
    assert (await averify(agent, thread_id="probe-1")).ok
    assert (await averify(agent, thread_id="probe-1")).ok
    assert _threads(sys.modules[MOD].saver) == {"probe-1"}
