"""An agent module that raises at import time is one clean ``error:`` line (gh #244).

``langstage-agui`` wrapped ``load_agent_spec()`` in ``except (ImportError, AttributeError,
OSError, ValueError, TypeError)``: the spec mistakes of gh #100. Anything else the agent
module raises while its body runs escaped as a raw traceback on ``--verify``, ``-m``,
``-m --json`` and bare serve. The usual case is the README's own real-model pattern,
``api_key=os.environ["OPENAI_API_KEY"]`` at module level, run with the key unset.

The README's promise is "a clean one-line ``error:`` on stderr, never a traceback (set
``LANGSTAGE_DEBUG=1`` for one)", and the exit code is 1 on every command (ADR 0007).
"""
import subprocess
import sys

import pytest

import langstage_core.agui as agui_pkg
from langstage_core.agui.__main__ import main

MODULES = {
    # name: (module body, what the one line must say)
    "missing-env-key": ('import os\nKEY = os.environ["SOME_MISSING_KEY_244"]\n',
                        "KeyError: 'SOME_MISSING_KEY_244'"),
    "runtime-error": ('raise RuntimeError("boom")\n', "RuntimeError: boom"),
    "zero-division": ("x = 1 / 0\n", "ZeroDivisionError: division by zero"),
    "custom-exception": ('class ProviderError(Exception):\n    pass\n'
                         'raise ProviderError("no credentials")\n',
                         "ProviderError: no credentials"),
}
COMMANDS = {
    "verify": ["--verify"],
    "message": ["-m", "hi"],
    "message-json": ["-m", "hi", "--json"],
    "serve": ["--port", "18244"],
}


class _FakeSock:
    def close(self):
        pass


@pytest.fixture
def no_serve(monkeypatch):
    """Never bind a port or start a server; record whether serve() was reached."""
    calls: list = []
    monkeypatch.setattr(agui_pkg, "serve", lambda graph, **kw: calls.append(graph))
    monkeypatch.setattr(agui_pkg, "_bind_socket", lambda host, port: _FakeSock())
    monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
    monkeypatch.delenv("DEEPAGENT_DEBUG", raising=False)
    monkeypatch.delenv("SOME_MISSING_KEY_244", raising=False)
    return calls


def _spec(tmp_path, monkeypatch, body):
    (tmp_path / "agent244.py").write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return "agent244.py:graph"


@pytest.mark.parametrize("command", list(COMMANDS))
@pytest.mark.parametrize("module", list(MODULES))
def test_import_time_error_is_one_clean_line(module, command, tmp_path, monkeypatch, no_serve, capsys):
    body, needle = MODULES[module]
    spec = _spec(tmp_path, monkeypatch, body)
    rc = main(["--agent", spec, *COMMANDS[command]])
    captured = capsys.readouterr()

    assert rc == 1, "a load failure is a failure on every command (ADR 0007)"
    assert captured.err.strip() == f"error: could not load agent {spec!r}: {needle}"
    assert "Traceback" not in captured.err
    assert captured.out == "", "nothing on stdout: no banner, no reply, no JSON"
    assert no_serve == [], "serve() must not be reached"


@pytest.mark.parametrize("command", ["verify", "message-json"])
def test_debug_adds_the_traceback_after_the_line(command, tmp_path, monkeypatch, no_serve, capsys):
    body, needle = MODULES["missing-env-key"]
    spec = _spec(tmp_path, monkeypatch, body)
    monkeypatch.setenv("LANGSTAGE_DEBUG", "1")
    rc = main(["--agent", spec, *COMMANDS[command]])
    err = capsys.readouterr().err

    assert rc == 1
    lines = err.splitlines()
    assert lines[0] == f"error: could not load agent {spec!r}: {needle}"
    assert "Traceback (most recent call last)" in err
    assert "agent244.py" in err, "the traceback names the user's file"


def test_spec_mistakes_keep_their_message(tmp_path, monkeypatch, no_serve, capsys):
    # The gh #100 cases are unchanged: load_agent_spec's own descriptive message, with
    # no exception-type prefix.
    monkeypatch.chdir(tmp_path)
    rc = main(["--agent", "not_a_real_module_244:graph", "--verify"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.strip() == ("error: could not load agent 'not_a_real_module_244:graph': "
                           "No module named 'not_a_real_module_244'")


def test_missing_import_inside_the_module_gets_a_debug_traceback(tmp_path, monkeypatch, no_serve, capsys):
    spec = _spec(tmp_path, monkeypatch, "import not_a_real_pkg_244\n")
    assert main(["--agent", spec, "--verify"]) == 1
    assert "Traceback" not in capsys.readouterr().err
    monkeypatch.setenv("LANGSTAGE_DEBUG", "1")
    assert main(["--agent", spec, "--verify"]) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[0].startswith("error: could not load agent")
    assert "Traceback (most recent call last)" in err


def test_the_console_script_prints_no_traceback(tmp_path):
    # End to end through the real entry point, as the issue ran it.
    (tmp_path / "nokey.py").write_text(MODULES["missing-env-key"][0], encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "langstage_core.agui", "-a", "nokey.py:graph", "--verify"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={k: v for k, v in __import__("os").environ.items()
             if k not in ("LANGSTAGE_DEBUG", "DEEPAGENT_DEBUG", "SOME_MISSING_KEY_244")},
    )
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.strip() == (
        "error: could not load agent 'nokey.py:graph': KeyError: 'SOME_MISSING_KEY_244'")
