"""Serving a spec that loads but isn't a runnable graph is one clean ``error:`` line (gh #247).

``langstage-agui --agent unc.py:graph`` (serve mode, the default command) printed the
``Serving … at <url>`` success banner and THEN a raw traceback when the spec loaded but
the object wasn't a runnable graph: a ``StateGraph`` nobody called ``.compile()`` on, or
any non-graph value. ``build_agent()`` only ran inside ``serve()`` → ``build_app()``,
after the banner. ``--verify`` and ``--message`` already answered the same object with
one line (gh #180); this is the serve-mode sibling of gh #100 and gh #143.

Now the agent is built before the port is bound and before the banner: no banner, one
``error:`` line with the same detail ``--verify`` gives, exit 1, a traceback only under
``LANGSTAGE_DEBUG``. The agent built there is the one served, so nothing is built twice.
"""
import os
import subprocess
import sys

import pytest

import langstage_core.agui as agui_pkg
from langstage_core.agui.__main__ import main

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("langgraph")

UNCOMPILED = (
    "from langgraph.graph import StateGraph, MessagesState, START, END\n"
    "g = StateGraph(MessagesState)\n"
    "g.add_node('n', lambda s: {})\n"
    "g.add_edge(START, 'n')\n"
    "g.add_edge('n', END)\n"
    "graph = g          # forgot .compile()\n"
)
COMPILED = UNCOMPILED.replace("graph = g          # forgot .compile()", "graph = g.compile()")
UNCOMPILED_DETAIL = ("TypeError: build_agent expected a compiled LangGraph graph, but got an "
                     "uncompiled StateGraph; call .compile() on it first.")
NOT_RUNNABLE = {
    # name: (module body, the detail every command reports for this object)
    "uncompiled": (UNCOMPILED, UNCOMPILED_DETAIL),
    "int": ("graph = 42\n",
            "TypeError: build_agent expected a compiled LangGraph graph (CompiledStateGraph) "
            "or an agent spec string, but got int."),
    "function": ("def graph():\n    return None\n",
                 "TypeError: build_agent expected a compiled LangGraph graph "
                 "(CompiledStateGraph) or an agent spec string, but got function."),
}


class _FakeSock:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clean_config(tmp_path, monkeypatch):
    """No user-level config or debug switch leaks into these runs."""
    gdir = tmp_path / "global-config"
    gdir.mkdir()
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(gdir))
    monkeypatch.delenv("DEEPAGENTS_CONFIG_HOME", raising=False)
    for var in ("LANGSTAGE_DEBUG", "DEEPAGENT_DEBUG", "LANGSTAGE_AGENT_SPEC",
                "LANGSTAGE_HOST", "LANGSTAGE_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def stubs(monkeypatch):
    """Never bind a real port or start a server; record what was bound and served."""
    seen = {"bound": [], "served": []}

    def bind(host, port):
        sock = _FakeSock()
        seen["bound"].append(sock)
        return sock

    monkeypatch.setattr(agui_pkg, "_bind_socket", bind)
    monkeypatch.setattr(agui_pkg, "serve", lambda agent, **kw: seen["served"].append((agent, kw)))
    return seen


def _spec(tmp_path, monkeypatch, body, name="agent247"):
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return f"{name}.py:graph"


@pytest.mark.parametrize("kind", list(NOT_RUNNABLE))
def test_serving_a_non_runnable_object_is_one_clean_line(kind, tmp_path, monkeypatch, stubs, capsys):
    body, detail = NOT_RUNNABLE[kind]
    spec = _spec(tmp_path, monkeypatch, body)
    rc = main(["--agent", spec, "--port", "18247"])
    captured = capsys.readouterr()

    assert rc == 1, "can't serve is a failure (ADR 0007)"
    assert captured.err.strip() == f"error: cannot serve agent {spec!r}: {detail}"
    assert "Traceback" not in captured.err
    assert captured.out == "", "no false-green banner"
    assert stubs["served"] == [], "serve() must not be reached"
    assert stubs["bound"] == [], "the port is never bound, so no socket is left open"


@pytest.mark.parametrize("kind", list(NOT_RUNNABLE))
def test_the_detail_matches_verify_and_message(kind, tmp_path, monkeypatch, stubs, capsys):
    # One object, three commands: the same `Type: message` after each command's prefix.
    body, detail = NOT_RUNNABLE[kind]
    spec = _spec(tmp_path, monkeypatch, body)
    lines = {}
    for command, argv in (("verify", ["--verify"]), ("message", ["-m", "hi"]), ("serve", [])):
        assert main(["--agent", spec, *argv]) == 1
        lines[command] = capsys.readouterr().err.strip()
    assert lines["verify"] == f"error: agent did not complete a turn: {detail}"
    assert lines["message"] == lines["verify"]
    assert lines["serve"] == f"error: cannot serve agent {spec!r}: {detail}"


def test_debug_adds_the_traceback_after_the_line(tmp_path, monkeypatch, stubs, capsys):
    spec = _spec(tmp_path, monkeypatch, UNCOMPILED)
    monkeypatch.setenv("LANGSTAGE_DEBUG", "1")
    rc = main(["--agent", spec])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == f"error: cannot serve agent {spec!r}: {UNCOMPILED_DETAIL}"
    assert "Traceback (most recent call last)" in captured.err
    assert lines[-1] == UNCOMPILED_DETAIL, "the traceback ends on the same exception"


def test_any_build_failure_is_one_line_with_its_type(tmp_path, monkeypatch, stubs, capsys):
    # Not only the TypeError: whatever stops the agent being built is named, not dumped.
    spec = _spec(tmp_path, monkeypatch, COMPILED)

    def boom(graph, **kw):
        raise RuntimeError("adapter refused this graph")

    monkeypatch.setattr(agui_pkg, "build_agent", boom)
    rc = main(["--agent", spec])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err.strip() == (
        f"error: cannot serve agent {spec!r}: RuntimeError: adapter refused this graph")
    assert captured.out == "" and stubs["served"] == [] and stubs["bound"] == []


def test_a_bad_agent_is_reported_before_a_busy_port(tmp_path, monkeypatch, capsys):
    # Both wrong at once: the agent error comes first (it has to be fixed either way),
    # and the port is not touched.
    import socket

    spec = _spec(tmp_path, monkeypatch, UNCOMPILED)
    monkeypatch.setattr(agui_pkg, "serve", lambda *a, **k: pytest.fail("must not serve"))
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        rc = main(["--agent", spec, "--host", "127.0.0.1", "--port", str(holder.getsockname()[1])])
    finally:
        holder.close()
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error: cannot serve agent ") and "cannot serve at" not in err


# ── a valid graph is served exactly as before ────────────────────────


def test_a_valid_graph_prints_the_banner_once_and_serves_the_built_agent(
        tmp_path, monkeypatch, stubs, capsys):
    (tmp_path / "langstage.toml").write_text('[configurable]\ngreeting = "Bonjour"\n',
                                             encoding="utf-8")
    spec = _spec(tmp_path, monkeypatch, COMPILED)
    rc = main(["--agent", spec, "--port", "18247", "--name", "Acme Agent"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.count("Serving ") == 1
    assert captured.out.strip() == f"Serving {spec!r} over AG-UI at http://localhost:18247/"
    assert captured.err == ""
    assert len(stubs["bound"]) == 1 and stubs["bound"][0].closed, "the bound socket is closed after"
    (agent, kw), = stubs["served"]
    # serve() is handed the agent main() built: the display name and the [configurable]
    # table are already on it, and add_agui_endpoint() takes a built agent as-is.
    assert agui_pkg._is_langgraph_agent(agent)
    assert agent.name == "Acme Agent"
    assert agent.config["configurable"] == {"greeting": "Bonjour"}
    assert kw["sock"] is stubs["bound"][0]
    assert kw["name"] == "Acme Agent" and kw["config"] == {"configurable": {"greeting": "Bonjour"}}


def test_the_agent_is_built_once_and_its_module_imported_once(tmp_path, monkeypatch, capsys):
    # Through the real serve() and build_app(); only uvicorn's run loop is stubbed.
    import uvicorn

    body = ("import pathlib\n"
            "with open(pathlib.Path(__file__).with_name('imports.log'), 'a') as f:\n"
            "    f.write('x')\n" + COMPILED)
    spec = _spec(tmp_path, monkeypatch, body)
    builds = []
    real_build = agui_pkg.build_agent

    def counting_build(graph, **kw):
        builds.append(kw)
        return real_build(graph, **kw)

    ran = []
    monkeypatch.setattr(agui_pkg, "build_agent", counting_build)
    monkeypatch.setattr(agui_pkg, "_bind_socket", lambda host, port: _FakeSock())
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: ran.append(self.config.app))

    assert main(["--agent", spec, "--port", "18247"]) == 0
    assert len(builds) == 1, "main() builds the agent; build_app() must not build it again"
    assert len(ran) == 1, "the server was started once"
    assert (tmp_path / "imports.log").read_text() == "x", "the agent module ran once"
    assert capsys.readouterr().out.count("Serving ") == 1


def test_a_prebuilt_agent_in_the_spec_is_served_as_is(tmp_path, monkeypatch, stubs, capsys):
    body = COMPILED + ("from langstage_core.agui import build_agent\n"
                       "graph = build_agent(graph, name='Prebuilt')\n")
    spec = _spec(tmp_path, monkeypatch, body)
    builds = []
    real_build = agui_pkg.build_agent
    monkeypatch.setattr(agui_pkg, "build_agent",
                        lambda graph, **kw: builds.append(kw.get("name")) or real_build(graph, **kw))
    assert main(["--agent", spec]) == 0
    (agent, _kw), = stubs["served"]
    assert agent.name == "Prebuilt"
    assert builds == ["Prebuilt"], "only the module's own build; main() passed it through"


# ── the Python API is unchanged ──────────────────────────────────────


def _uncompiled():
    from langgraph.graph import END, START, MessagesState, StateGraph

    g = StateGraph(MessagesState)
    g.add_node("n", lambda s: {})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g


@pytest.mark.parametrize("call", ["build_agent", "build_app", "serve"])
def test_python_callers_still_get_the_typeerror(call, monkeypatch):
    # serve() must raise before it binds anything, as it always has.
    monkeypatch.setattr(agui_pkg, "_bind_socket", lambda *a: pytest.fail("must not bind"))
    with pytest.raises(TypeError, match="uncompiled StateGraph"):
        getattr(agui_pkg, call)(_uncompiled())


# ── end to end through the real entry point ──────────────────────────


@pytest.mark.parametrize("kind", ["uncompiled", "int"])
def test_the_console_script_prints_no_banner_and_no_traceback(kind, tmp_path):
    body, detail = NOT_RUNNABLE[kind]
    (tmp_path / "unc.py").write_text(body, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "langstage_core.agui", "--agent", "unc.py:graph",
         "--host", "127.0.0.1", "--port", "18247"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items()
             if k not in ("LANGSTAGE_DEBUG", "DEEPAGENT_DEBUG", "LANGSTAGE_AGENT_SPEC")},
    )
    assert proc.returncode == 1
    assert proc.stdout == "", "no banner"
    assert proc.stderr.strip() == f"error: cannot serve agent 'unc.py:graph': {detail}"
