"""agui CLI: --version and host/port consistency (dogfood cluster 3).

These paths return before importing the AG-UI server, so they don't need the
[agui] extra. Regression: `--show-config` advertised localhost:8050 while the
server actually bound 127.0.0.1:8000 — the shown config and the real bind
disagreed. host/port now come from the resolved HostConfig (CLI flags override),
so --show-config reflects exactly what serve() binds.
"""
import pytest

import langstage_core.agui as agui_pkg
from langstage_core.agui.__main__ import main


def test_version_returns_zero_and_prints_pkg(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "langstage-core" in out


def test_show_config_reflects_port_override(capsys):
    rc = main(["--port", "9123", "--host", "0.0.0.0", "--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    # The CLI flags appear as overrides — the same values serve() will bind.
    assert "9123" in out
    assert "0.0.0.0" in out
    assert "[override]" in out


def test_show_config_reflects_agent_override(capsys):
    # gh #60: --show-config resolved without the --agent override and reported
    # agent_spec = None while serve() honored the flag (advertised != honored).
    # --agent is now applied before the --show-config branch, so the shown config
    # matches the real run.
    rc = main(["--show-config", "--agent", "my_agent.py:graph"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "my_agent.py:graph" in out
    assert "[override]" in out


def test_show_config_default_host_port_present(capsys):
    rc = main(["--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    # host/port are shown (from HostConfig) — what the server will bind by default.
    assert "host" in out and "port" in out


def test_demo_show_config_resolves_echo_spec(capsys):
    # Bare --demo is unchanged: it serves the echo stub.
    rc = main(["--demo", "--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "langstage_core.demo.stub:graph" in out


def test_demo_tools_show_config_resolves_tools_spec(capsys):
    # gh #99: --demo=tools serves the rich-frame demo instead of the echo stub.
    rc = main(["--demo=tools", "--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "langstage_core.demo.tools:graph" in out


def test_demo_tools_is_still_mutually_exclusive_with_agent(capsys):
    rc = main(["--demo=tools", "--agent", "my_agent.py:graph"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_demo_rejects_unknown_value(capsys):
    # An unknown --demo value is an argparse error (SystemExit 2), not a silent fall-through.
    with pytest.raises(SystemExit) as exc:
        main(["--demo=bogus"])
    assert exc.value.code == 2


def test_show_config_omits_keys_the_server_ignores(capsys):
    # The AG-UI server consumes only agent_spec/host/port; workspace_root/debug/
    # title are inherited but inert on this surface, so --show-config must not
    # advertise them (gh #39).
    rc = main(["--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "agent_spec" in out and "host" in out and "port" in out
    for inert in ("workspace_root", "debug", "title", "LANGSTAGE_TITLE", "LANGSTAGE_DEBUG"):
        assert inert not in out, inert


class TestInvalidSpecFailsBeforeBanner:
    """gh #100: an unloadable --agent spec printed the "Serving '<spec>' over AG-UI
    at <url>" success banner and *then* died with a raw traceback, because the spec
    was only loaded inside serve(). That is the exact fake-success-line-then-traceback
    failure the two sibling paths in the same function already avoid (the missing
    [agui] extra, and the no-spec-at-all `error: no agent spec ...` / exit 2). The
    spec now resolves before the banner: clean one-line `error: ...` on stderr,
    exit 2, no banner, no traceback.
    """

    @pytest.fixture
    def stub_serve(self, monkeypatch):
        """Neutralise the [agui]-extra gate and capture what serve() is handed.

        main() does `from . import ... ensure_available, serve` at call time, so
        patching the package attributes is what the CLI actually resolves.
        """
        calls: list = []
        monkeypatch.setattr(agui_pkg, "ensure_available", lambda: None)
        monkeypatch.setattr(agui_pkg, "serve", lambda graph, **kw: calls.append((graph, kw)))
        return calls

    @pytest.mark.parametrize(
        "spec, needle",
        [
            # The three variants from the issue, one per leaked exception type.
            ("myagent:graph", "No module named 'myagent'"),                    # ModuleNotFoundError
            ("langstage_core.demo.stub:not_a_real_attr", "not_a_real_attr"),   # AttributeError
            ("./no_such_file.py:graph", "Agent file not found"),               # FileNotFoundError
        ],
    )
    def test_unloadable_spec_is_a_clean_stderr_error(self, spec, needle, stub_serve, capsys):
        rc = main(["--agent", spec])
        captured = capsys.readouterr()

        assert rc == 2, "must exit non-zero, matching the sibling error paths"
        assert "error: could not load agent" in captured.err
        assert needle in captured.err
        assert "Traceback" not in captured.err, "the raw traceback must not leak"
        # The regression itself: no fake success line claiming the server is up.
        assert "Serving" not in captured.out
        assert stub_serve == [], "serve() must not be reached with a bad spec"
        captured.err.encode("ascii")  # cp1252-safe

    def test_malformed_spec_without_attr_suffix_is_also_clean(self, stub_serve, capsys):
        # load_agent_spec raises ValueError (not Import/Attribute/OSError) when the
        # required ':attr' suffix is missing — same clean treatment, no traceback.
        rc = main(["--agent", "myagent"])
        captured = capsys.readouterr()
        assert rc == 2
        assert "error: could not load agent" in captured.err
        assert "Serving" not in captured.out
        assert stub_serve == []

    def test_valid_spec_still_prints_banner_and_serves_the_loaded_graph(
        self, stub_serve, tmp_path, capsys
    ):
        # The happy path must be untouched: banner still printed, server still started.
        agent = tmp_path / "good_agent.py"
        agent.write_text("graph = {'marker': 'the-real-graph'}\n")
        spec = f"{agent}:graph"

        rc = main(["--agent", spec, "--port", "9111"])
        out = capsys.readouterr().out

        assert rc == 0
        assert f"Serving {spec!r} over AG-UI at" in out
        assert "9111" in out
        assert len(stub_serve) == 1
        graph, kwargs = stub_serve[0]
        # serve() accepts a spec string OR a loaded graph; it is handed the already
        # loaded graph so the spec is not resolved a second time.
        assert graph == {"marker": "the-real-graph"}
        assert not isinstance(graph, str)
        assert kwargs["port"] == 9111

    def test_spec_module_is_imported_exactly_once(self, stub_serve, tmp_path, capsys):
        """Pre-loading must not double-run the agent module's import side effects.

        A file-path spec gets a fresh unique module name per load, so loading twice
        would execute the module body twice — this pins that it happens once.
        """
        marker = tmp_path / "imports.log"
        agent = tmp_path / "side_effect_agent.py"
        agent.write_text(
            "from pathlib import Path\n"
            f"Path(r'{marker}').open('a').write('imported\\n')\n"
            "graph = object()\n"
        )

        rc = main(["--agent", f"{agent}:graph"])
        capsys.readouterr()

        assert rc == 0
        assert marker.read_text().count("imported") == 1
        assert len(stub_serve) == 1


class TestMessage:
    """--message: one-shot 'run my prompt and print the reply' (gh #120).

    Uses the keyless tools demo, so these need the agui extra (skip without it).
    """

    def test_message_prints_reply_and_exits_zero(self, capsys):
        pytest.importorskip("ag_ui_langgraph")
        rc = main(["--demo=tools", "--message", "use a tool"])
        out = capsys.readouterr().out
        assert rc == 0
        assert out.strip(), "expected the reply text on stdout"

    def test_message_json_emits_typed_turnresult(self, capsys):
        pytest.importorskip("ag_ui_langgraph")
        import json

        rc = main(["--demo=tools", "-m", "use a tool", "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert data["outcome"] == "complete"
        assert set(data) >= {
            "text", "outcome", "tool_calls", "extractions", "reasoning", "interrupt", "error",
        }
        assert data["tool_calls"] and data["tool_calls"][0]["name"] == "demo_lookup"

    def test_message_interrupt_exits_two(self, capsys):
        # A HITL turn that pauses -> outcome interrupted -> exit 2 (gh #120), the
        # 0/1/2 vocabulary --verify / no-spec use. "ask me" is the tools-demo trigger.
        pytest.importorskip("ag_ui_langgraph")
        import json

        rc = main(["--demo=tools", "-m", "ask me first", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["outcome"] == "interrupted"
        assert rc == 2


class TestShowConfigJsonAndExitCodes:
    """gh #125 (--show-config --json) + #124 (load-fail exit code respects the command)."""

    def test_show_config_json_emits_config_dict(self, capsys):
        import json

        rc = main(["--demo=tools", "--show-config", "--json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "config" in data and "toml" in data
        assert "unknown_keys" in data["toml"]
        assert data["demo"]["agent_spec"] == "langstage_core.demo.tools:graph"

    def test_load_failure_exit_code_respects_command(self, capsys):
        # gh #124: a failed load is "failed"/"error" (exit 1) under --verify / --message,
        # never 2 (which --message reads as interrupted); the serve path keeps 2.
        assert main(["--agent", "/nope/x.py:graph", "--verify"]) == 1
        capsys.readouterr()
        assert main(["--agent", "/nope/x.py:graph", "-m", "hi"]) == 1
        capsys.readouterr()
        assert main(["--agent", "/nope/x.py:graph"]) == 2
        capsys.readouterr()


def test_missing_agui_extra_exit_code_respects_command(monkeypatch, capsys):
    # gh #134: a missing [agui] extra is a "can't run" failure — 1 under --verify /
    # --message (never 2, which --message reads as interrupted), 2 on the serve path,
    # like the agent-load-failure path (#124).
    import langstage_core.agui as agui_pkg

    def _raise():
        raise RuntimeError("AG-UI support needs the 'agui' extra")

    monkeypatch.setattr(agui_pkg, "ensure_available", _raise)
    assert main(["--demo", "--verify"]) == 1
    capsys.readouterr()
    assert main(["--demo", "-m", "hi"]) == 1
    capsys.readouterr()
    assert main(["--demo"]) == 2  # serve path keeps the usage/can't-start code
    capsys.readouterr()
