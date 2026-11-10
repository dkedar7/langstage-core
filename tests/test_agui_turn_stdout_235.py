"""gh #235: on `langstage-agui -m` / `--json` / `--verify`, stdout carries only the
reply, the JSON document or the verdict. Whatever a node or tool prints DURING the
turn goes to stderr; the import-time guard (load_agent_spec) never covered the turn.
"""
from __future__ import annotations

import io
import json
import sys
import textwrap

import pytest

from langstage_core.agui.__main__ import main
from langstage_core.console import agent_stdout_to_stderr

NOISY = textwrap.dedent(
    """
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def respond(state):
        print("debug: respond node called \u2713")
        return {"messages": [AIMessage(content="hi")]}

    g = StateGraph(MessagesState)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    graph = g.compile()
    """
)


@pytest.fixture
def noisy_spec(tmp_path):
    p = tmp_path / "noisy.py"
    p.write_text(NOISY, encoding="utf-8")
    return f"{p}:graph"


def test_json_stdout_is_valid_json_when_a_node_prints(noisy_spec, capsys):
    rc = main(["--agent", noisy_spec, "-m", "hello", "--json"])
    cap = capsys.readouterr()
    assert rc == 0
    doc = json.loads(cap.out)  # the whole of stdout parses: nothing ahead of the JSON
    assert doc["text"] == "hi" and doc["outcome"] == "complete"
    assert "debug: respond node called" in cap.err


def test_message_stdout_is_only_the_reply(noisy_spec, capsys):
    rc = main(["--agent", noisy_spec, "-m", "hello"])
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.strip() == "hi"
    assert "debug: respond node called" in cap.err


def test_verify_stdout_is_only_the_verdict(noisy_spec, capsys):
    rc = main(["--agent", noisy_spec, "--verify"])
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.startswith("ok:") and "debug" not in cap.out
    assert "debug: respond node called" in cap.err


def test_helper_restores_stdout_and_yields_the_real_one(capsys):
    before = sys.stdout
    with agent_stdout_to_stderr() as out:
        assert out is before
        print("agent noise")
        out.write("result\n")
    assert sys.stdout is before
    cap = capsys.readouterr()
    assert cap.out == "result\n"
    assert cap.err == "agent noise\n"


def test_helper_restores_stdout_when_the_turn_raises():
    before = sys.stdout
    with pytest.raises(RuntimeError):
        with agent_stdout_to_stderr():
            raise RuntimeError("boom")
    assert sys.stdout is before


def test_diverted_print_escapes_what_stderr_cannot_encode(monkeypatch):
    # An agent printing a char stderr can't encode (cp1252) no longer kills the turn.
    raw = io.BytesIO()
    err = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", err)
    with agent_stdout_to_stderr():
        print("done \u2713")
    err.flush()
    assert raw.getvalue().decode("cp1252") == "done \\u2713\n"
