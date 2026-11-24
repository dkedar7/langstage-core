"""gh #252: on `langstage-agui -m` / `--json` / `--verify`, stdout carries only the
reply, the JSON document or the verdict, even when the agent's process tree writes
to file descriptor 1 (a child from `os.system` / `subprocess.run` without
`capture_output`, or native code). gh #235 only swapped `sys.stdout`.

These run the real CLI in a subprocess, since fd 1 is what's under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import langstage_core

FD1 = textwrap.dedent(
    """
    import os, subprocess, sys
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def respond(state):
        os.write(1, b"raw fd1 no newline")
        subprocess.run([sys.executable, "-c",
                        "import sys; sys.stdout.write('child no newline')"])
        os.system("echo 42")  # sh and cmd both have echo
        return {"messages": [AIMessage(content="ok")]}

    g = StateGraph(MessagesState)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    graph = g.compile()
    """
)


@pytest.fixture
def fd1_spec(tmp_path):
    p = tmp_path / "fd1_agent.py"
    p.write_text(FD1, encoding="utf-8")
    return f"{p}:graph"


def _env() -> dict[str, str]:
    # The child imports the langstage_core under test (a worktree or src checkout may
    # differ from what the interpreter has installed).
    src = str(Path(langstage_core.__file__).resolve().parents[1])
    path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    return {**os.environ, "PYTHONPATH": path, "PYTHONIOENCODING": "utf-8"}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "langstage_core.agui", *args],
        capture_output=True, text=True, encoding="utf-8", env=_env(), timeout=120,
    )


def _assert_leaks_went_to_stderr(err: str) -> None:
    assert "raw fd1 no newline" in err
    assert "child no newline" in err
    assert "42" in err


def test_json_stdout_parses_when_the_agent_writes_to_fd1(fd1_spec):
    r = _run("--agent", fd1_spec, "-m", "hello", "--json")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)  # the whole of stdout is the one JSON document
    assert doc["text"] == "ok" and doc["outcome"] == "complete"
    _assert_leaks_went_to_stderr(r.stderr)


def test_message_stdout_is_only_the_reply_when_the_agent_writes_to_fd1(fd1_spec):
    r = _run("--agent", fd1_spec, "-m", "hello")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
    _assert_leaks_went_to_stderr(r.stderr)


def test_verify_stdout_is_only_the_verdict_when_the_agent_writes_to_fd1(fd1_spec):
    r = _run("--agent", fd1_spec, "--verify")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok:") and r.stdout.count("\n") == 1
    _assert_leaks_went_to_stderr(r.stderr)


HELPER = textwrap.dedent(
    r"""
    import os, sys
    from langstage_core.console import agent_stdout_to_stderr

    print("before", flush=True)
    with agent_stdout_to_stderr() as out:
        os.write(1, b"[leak]")
        print("[print]")
        out.write("result\n")
    print("after", flush=True)
    os.write(1, b"fd1 back\n")
    """
)


def test_helper_restores_fd1_on_exit(tmp_path):
    script = tmp_path / "helper.py"
    script.write_text(HELPER, encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=_env(), timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["before", "result", "after", "fd1 back"]
    assert "[leak]" in r.stderr and "[print]" in r.stderr


def test_helper_restores_fd1_when_the_turn_raises(tmp_path):
    script = tmp_path / "raises.py"
    script.write_text(textwrap.dedent(
        r"""
        import os
        from langstage_core.console import agent_stdout_to_stderr

        try:
            with agent_stdout_to_stderr():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        os.write(1, b"fd1 back\n")
        """
    ), encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=_env(), timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "fd1 back\n"
