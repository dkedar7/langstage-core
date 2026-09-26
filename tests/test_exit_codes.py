"""The family exit-code scheme (ADR 0007).

0 success / 1 failure / 2 paused on a HITL interrupt / 64 usage error. argparse's own
usage-error code is 2, which collides with "paused" (the gh #174 fail-open class), so
every console script uses ``langstage_core.cli.ArgumentParser``, whose ``error()`` exits 64.
"""
import socket
import subprocess
import sys

import pytest

import langstage_core.agui as agui_pkg
from langstage_core import cli
from langstage_core.agui.__main__ import main


class _FakeSock:
    def close(self):
        pass


# ---- the public helper ---------------------------------------------------------------


def test_constants_are_the_family_scheme():
    assert (cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_PAUSED, cli.EXIT_USAGE) == (0, 1, 2, 64)


@pytest.mark.parametrize(
    "outcome, code",
    [("complete", 0), ("error", 1), ("interrupted", 2), ("something-else", 1), (None, 1)],
)
def test_exit_code_for_outcome(outcome, code):
    assert cli.exit_code_for_outcome(outcome) == code


def test_argument_parser_usage_error_exits_64(capsys):
    p = cli.ArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--n", "notanint"])
    assert exc.value.code == 64
    err = capsys.readouterr().err
    assert "usage:" in err and "x: error:" in err


def test_argument_parser_unknown_flag_exits_64(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.ArgumentParser(prog="x").parse_args(["--bogus"])
    assert exc.value.code == 64


def test_argument_parser_subparsers_inherit_64(capsys):
    p = cli.ArgumentParser(prog="x")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("run")
    s.add_argument("--n", type=int)
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["run", "--n", "zz"])
    assert exc.value.code == 64


def test_argument_parser_help_still_exits_0(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.ArgumentParser(prog="x").parse_args(["--help"])
    assert exc.value.code == 0


def test_usage_error_helper_prints_and_returns_64(capsys):
    p = cli.ArgumentParser(prog="x")
    assert cli.usage_error(p, "--a and --b are mutually exclusive") == 64
    err = capsys.readouterr().err
    assert "x: error: --a and --b are mutually exclusive" in err


# ---- langstage-agui ------------------------------------------------------------------


@pytest.fixture
def no_spec(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGSTAGE_AGENT_SPEC", raising=False)
    monkeypatch.delenv("DEEPAGENT_AGENT_SPEC", raising=False)
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "no-global"))


@pytest.mark.parametrize("argv", [["--demo=bogus"], ["--bogus"], ["--port", "abc"]])
def test_agui_argparse_usage_errors_exit_64(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 64


@pytest.mark.parametrize("extra", [[], ["--verify"], ["-m", "hi"], ["--show-config"]])
def test_agui_demo_agent_conflict_is_usage_64(extra, capsys):
    assert main(["--demo", "--agent", "x.py:g", *extra]) == 64
    assert "mutually exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--verify"], ["-m", "hi"]])
def test_agui_no_spec_is_failure_1_on_every_command(extra, no_spec, capsys):
    assert main(extra) == 1
    assert "no agent spec" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--verify"], ["-m", "hi"]])
def test_agui_load_failure_is_1_on_every_command(extra, capsys):
    assert main(["--agent", "/nope/x.py:graph", *extra]) == 1
    assert "could not load agent" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--verify"], ["-m", "hi"]])
def test_agui_missing_extra_is_1_on_every_command(extra, monkeypatch, capsys):
    def _raise():
        raise RuntimeError("AG-UI support needs the 'agui' extra")

    monkeypatch.setattr(agui_pkg, "ensure_available", _raise)
    assert main(["--demo", *extra]) == 1


def test_agui_busy_port_is_failure_1(monkeypatch, capsys):
    monkeypatch.setattr(agui_pkg, "ensure_available", lambda: None)
    monkeypatch.setattr(agui_pkg, "serve", lambda graph, **kw: None)
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        rc = main(["--demo", "--host", "127.0.0.1", "--port", str(port)])
    finally:
        holder.close()
    assert rc == 1
    assert "cannot serve" in capsys.readouterr().err


def test_agui_serve_ok_is_0(monkeypatch, capsys):
    monkeypatch.setattr(agui_pkg, "ensure_available", lambda: None)
    monkeypatch.setattr(agui_pkg, "serve", lambda graph, **kw: None)
    monkeypatch.setattr(agui_pkg, "_bind_socket", lambda host, port: _FakeSock())
    assert main(["--demo"]) == 0


def test_agui_verify_ok_0_and_message_codes(capsys):
    pytest.importorskip("ag_ui_langgraph")
    assert main(["--demo", "--verify"]) == 0
    assert main(["--demo", "-m", "hi"]) == 0
    assert main(["--demo=tools", "-m", "ask me first", "--json"]) == 2


def test_agui_verify_failure_is_1(monkeypatch, capsys):
    from langstage_core.agui.verify import VerifyResult

    monkeypatch.setattr(
        agui_pkg, "verify", lambda graph: VerifyResult(ok=False, reason="boom", frames=0, content_chars=0)
    )
    assert main(["--demo", "--verify"]) == 1


# ---- python -m langstage_core.host ---------------------------------------------------


def test_host_module_unknown_flag_exits_64():
    r = subprocess.run(
        [sys.executable, "-m", "langstage_core.host", "--bogus"], capture_output=True, text=True
    )
    assert r.returncode == 64
    assert "unrecognized arguments" in r.stderr


def test_host_module_ok_exits_0():
    r = subprocess.run([sys.executable, "-m", "langstage_core.host"], capture_output=True, text=True)
    assert r.returncode == 0


def test_readme_exit_codes_snippet_runs_as_written():
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Exit codes"):text.index("## Configuration")]
    [code] = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    ns: dict = {}
    exec(compile(code, "README:exit-codes", "exec"), ns)
    assert ns["parser"].prog == "my-surface"
    assert ns["EXIT_USAGE"] == 64
