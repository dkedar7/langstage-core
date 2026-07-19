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
