"""agui CLI: --version and host/port consistency (dogfood cluster 3).

These paths return before importing the AG-UI server, so they don't need the
[agui] extra. Regression: `--show-config` advertised localhost:8050 while the
server actually bound 127.0.0.1:8000 — the shown config and the real bind
disagreed. host/port now come from the resolved HostConfig (CLI flags override),
so --show-config reflects exactly what serve() binds.
"""
from langgraph_stream_parser.agui.__main__ import main


def test_version_returns_zero_and_prints_pkg(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "langgraph-stream-parser" in out


def test_show_config_reflects_port_override(capsys):
    rc = main(["--port", "9123", "--host", "0.0.0.0", "--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    # The CLI flags appear as overrides — the same values serve() will bind.
    assert "9123" in out
    assert "0.0.0.0" in out
    assert "[override]" in out


def test_show_config_default_host_port_present(capsys):
    rc = main(["--show-config"])
    assert rc == 0
    out = capsys.readouterr().out
    # host/port are shown (from HostConfig) — what the server will bind by default.
    assert "host" in out and "port" in out
