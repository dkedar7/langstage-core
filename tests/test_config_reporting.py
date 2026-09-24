"""Wave 2 config-reporting root causes (gh #175, #167, #179, #137, #139, #170, #171,
#153, and the surface reports that share them).

One file per root-cause family so the contract each surface relies on is pinned in
core, where they all resolve config:

- a present-but-malformed TOML is reported as *malformed*, never as *absent*;
- every degraded / rejected value is exposed as data (``config_issues()``) so a
  surface's ``--strict`` can fail on it;
- every legacy alias prints exactly ONE visible stderr notice per process;
- ``debug`` is honored through the resolved config, and a bare key swallowed by a
  preceding ``[table]`` is called out at resolve time;
- quoted-string booleans in TOML are coerced with the env-bool semantics;
- ``[configurable]`` reaches the graph on the core CLI paths and is shown;
- ``config_dict()`` lists every contributing TOML file;
- ``langstage_core.console.safe_print`` never crashes on an unencodable console.
"""
import io
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from langstage_core.host import HostConfig
from langstage_core.host import config as core_config


@pytest.fixture
def isolated_global(tmp_path, monkeypatch):
    gdir = tmp_path / "global"
    gdir.mkdir()
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(gdir))
    monkeypatch.delenv("DEEPAGENTS_CONFIG_HOME", raising=False)
    return gdir


def _proj(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "proj"
    d.mkdir(exist_ok=True)
    (d / "langstage.toml").write_text(body, encoding="utf-8")
    return d


# ── 1. malformed TOML is reported as malformed, not absent ───────────


class TestMalformedTomlReported:
    BAD = '[server]\nport = 8000\nhost = localhost\n'  # unquoted string -> TOMLDecodeError

    def test_config_dict_reports_found_and_malformed(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, self.BAD)
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        toml = cfg.config_dict()["toml"]
        assert toml["found"] is True
        assert toml["path"] == str(proj / "langstage.toml")
        assert toml["malformed"] is True
        assert toml["paths"] == []  # nothing was actually READ
        [entry] = toml["malformed_files"]
        assert entry["path"] == str(proj / "langstage.toml")
        assert "TOMLDecodeError" in entry["error"]
        assert "line 3" in entry["error"]

    def test_describe_says_malformed_not_not_found(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, self.BAD)
        text = HostConfig.resolve(env={}, toml_start=proj).describe()
        assert "no langstage.toml" not in text
        assert "MALFORMED" in text and str(proj / "langstage.toml") in text
        assert "TOMLDecodeError" in text

    def test_valid_global_plus_malformed_project(self, isolated_global, tmp_path):
        (isolated_global / "config.toml").write_text('[ui]\ntitle = "G"\n')
        proj = _proj(tmp_path, self.BAD)
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        toml = cfg.config_dict()["toml"]
        assert toml["found"] is True
        assert toml["path"] == str(isolated_global / "config.toml")  # the file that DID load
        assert toml["malformed"] is True
        text = cfg.describe()
        assert "TOML read from:" in text and "MALFORMED" in text

    def test_clean_and_absent_unchanged(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        toml = cfg.config_dict()["toml"]
        assert toml["found"] is False and toml["path"] is None
        assert toml["malformed"] is False and toml["malformed_files"] == []
        assert "no langstage.toml" in cfg.describe()

    def test_config_issues_exposes_malformed_file(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, self.BAD)
        issues = HostConfig.resolve(env={}, toml_start=proj).config_issues()
        assert [i["kind"] for i in issues] == ["malformed_toml"]
        assert issues[0]["path"] == str(proj / "langstage.toml")

    def test_host_cli_reports_malformed(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, self.BAD)
        r = subprocess.run(
            [sys.executable, "-m", "langstage_core.host"],
            capture_output=True, text=True, cwd=proj, env={**os.environ},
        )
        assert r.returncode == 0
        assert "no langstage.toml" not in r.stdout
        assert "MALFORMED" in r.stdout


# ── strict: degraded values are exposed as data ─────────────────────


class TestConfigIssues:
    def test_clean_config_has_no_issues(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, "[server]\nport = 9000\n")
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.config_issues() == []
        assert cfg.config_dict()["issues"] == []

    def test_malformed_env_value_is_an_issue(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(env={"LANGSTAGE_PORT": "abc"}, toml_start=tmp_path)
        [issue] = cfg.config_issues()
        assert issue["kind"] == "malformed_value"
        assert issue["field"] == "port" and issue["source"] == "env:LANGSTAGE_PORT"
        assert issue["value"] == "abc"

    def test_malformed_toml_value_is_an_issue(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, '[server]\nport = "warm"\n')
        [issue] = HostConfig.resolve(env={}, toml_start=proj).config_issues()
        assert issue["kind"] == "malformed_value"
        assert issue["source"] == "toml:server.port"

    def test_invalid_value_is_an_issue(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(env={"LANGSTAGE_PORT": "70000"}, toml_start=tmp_path)
        [issue] = cfg.config_issues()
        assert issue["kind"] == "invalid_value" and issue["field"] == "port"

    def test_unknown_key_is_an_issue(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, "[server]\nprot = 1\n")
        [issue] = HostConfig.resolve(env={}, toml_start=proj).config_issues()
        assert issue["kind"] == "unknown_toml_key" and issue["key"] == "server.prot"

    def test_issues_are_per_resolve_not_process_global(self, isolated_global, tmp_path):
        # The stderr notes are deduped per process; the issue list must not be, or a
        # second resolve (e.g. the one --strict runs) would read clean.
        HostConfig.resolve(env={"LANGSTAGE_PORT": "abc"}, toml_start=tmp_path)
        again = HostConfig.resolve(env={"LANGSTAGE_PORT": "abc"}, toml_start=tmp_path)
        assert len(again.config_issues()) == 1
        assert HostConfig.resolve(env={}, toml_start=tmp_path).config_issues() == []


# ── 2. legacy aliases: exactly one visible notice ───────────────────


def _no_pytest_env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.pop("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", None)
    env.update(extra)
    return env


class TestLegacyAliasNotices:
    def test_deepagents_config_home_warns(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LANGSTAGE_CONFIG_HOME", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CONFIG_HOME", str(tmp_path))
        core_config._warned_legacy_env.discard("DEEPAGENTS_CONFIG_HOME")
        with pytest.warns(DeprecationWarning, match="DEEPAGENTS_CONFIG_HOME"):
            HostConfig.resolve(env={}, toml_start=tmp_path)

    def test_canonical_config_home_wins_silently(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("DEEPAGENTS_CONFIG_HOME", str(tmp_path / "old"))
        core_config._warned_legacy_env.discard("DEEPAGENTS_CONFIG_HOME")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert core_config._global_toml_path() == tmp_path / "config.toml"

    def test_workspace_root_legacy_env_warns(self, tmp_path, monkeypatch):
        from langstage_core.host import workspace as ws

        monkeypatch.setattr(ws, "_ACTIVE", None)
        monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
        monkeypatch.setenv("DEEPAGENT_WORKSPACE_ROOT", str(tmp_path))
        core_config._warned_legacy_env.discard("DEEPAGENT_WORKSPACE_ROOT")
        with pytest.warns(DeprecationWarning, match="DEEPAGENT_WORKSPACE_ROOT"):
            assert ws.workspace_root() == tmp_path.resolve()

    def test_workspace_root_legacy_env_expands_tilde_and_warns(self, tmp_path, monkeypatch):
        from langstage_core.host import workspace as ws

        monkeypatch.setattr(ws, "_ACTIVE", None)
        monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
        monkeypatch.setenv("DEEPAGENT_WORKSPACE_ROOT", "~")
        core_config._warned_legacy_env.discard("DEEPAGENT_WORKSPACE_ROOT")
        with pytest.warns(DeprecationWarning, match="DEEPAGENT_WORKSPACE_ROOT"):
            assert ws.workspace_root() == Path.home().resolve()

    @pytest.mark.parametrize(
        "var,value",
        [
            ("DEEPAGENT_AGENT_SPEC", "x.py:graph"),
            ("DEEPAGENTS_CONFIG_HOME", "."),
        ],
    )
    def test_exactly_one_visible_notice_per_process(self, tmp_path, var, value):
        # Default warning filters (as a spawned surface runs): the polished `note:` is
        # the one visible signal; the DeprecationWarning must not ALSO surface by
        # being attributed to the user's __main__ (gh langstage-vscode #112).
        env = _no_pytest_env(**{var: value})
        env.pop("LANGSTAGE_CONFIG_HOME", None)
        env.pop("PYTHONWARNINGS", None)
        # resolve() called from a main() in __main__ — the shape of every console
        # script, where the old stacklevel=4 landed the warning on the user's module.
        code = (
            "from langstage_core.host import HostConfig\n"
            "def main():\n"
            "    HostConfig.resolve(); HostConfig.resolve()\n"
            "main()\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=tmp_path, env=env,
        )
        assert r.returncode == 0, r.stderr
        assert r.stderr.count(var) == 1, r.stderr
        assert "DeprecationWarning" not in r.stderr


# ── 3. debug honored through config; misplaced bare key; bool coercion ──


class TestDebugThroughConfig:
    def test_legacy_env_debug_enables_traceback(self, monkeypatch, tmp_path):
        from langstage_core.agui import _debug_traceback_extra

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "g"))
        monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
        monkeypatch.setenv("DEEPAGENT_DEBUG", "1")
        try:
            raise ValueError("kaboom")
        except ValueError:
            extra = _debug_traceback_extra()
        assert "kaboom" in extra.get("traceback", "")

    def test_toml_debug_enables_traceback(self, monkeypatch, tmp_path):
        from langstage_core.agui import _debug_traceback_extra

        (tmp_path / "langstage.toml").write_text("debug = true\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "g"))
        monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
        monkeypatch.delenv("DEEPAGENT_DEBUG", raising=False)
        try:
            raise ValueError("kaboom")
        except ValueError:
            extra = _debug_traceback_extra()
        assert "kaboom" in extra.get("traceback", "")

    def test_debug_off_by_default(self, monkeypatch, tmp_path):
        from langstage_core.agui import _debug_traceback_extra

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "g"))
        monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
        monkeypatch.delenv("DEEPAGENT_DEBUG", raising=False)
        try:
            raise ValueError("kaboom")
        except ValueError:
            assert _debug_traceback_extra() == {}


class TestMisplacedBareKey:
    def test_debug_after_table_gets_a_runtime_note(self, isolated_global, tmp_path, capsys):
        proj = _proj(tmp_path, '[server]\nport = 9222\ndebug = true\n')
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.debug is False
        err = capsys.readouterr().err
        assert "server.debug" in err and "before any [table]" in err
        [issue] = cfg.config_issues()
        assert issue["kind"] == "unknown_toml_key"
        assert issue["key"] == "server.debug" and issue["did_you_mean"] == "debug"

    def test_describe_hints_the_fix(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, '[server]\ndebug = true\n')
        text = HostConfig.resolve(env={}, toml_start=proj).describe()
        assert "server.debug" in text and "top of the file" in text

    def test_top_level_debug_still_works(self, isolated_global, tmp_path, capsys):
        proj = _proj(tmp_path, 'debug = true\n[server]\nport = 9222\n')
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.debug is True and cfg.config_issues() == []
        assert "note:" not in capsys.readouterr().err


class TestTomlBoolCoercion:
    @pytest.mark.parametrize("raw,expected", [
        ('"false"', False), ('"no"', False), ('"0"', False), ('"off"', False),
        ('"true"', True), ('"YES"', True), ("true", True), ("false", False),
        ("1", True), ("0", False),
    ])
    def test_quoted_bools_coerce(self, isolated_global, tmp_path, raw, expected):
        proj = _proj(tmp_path, f"debug = {raw}\n")
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.debug is expected
        assert cfg.sources["debug"].startswith("toml")

    def test_unrecognized_string_degrades_with_note(self, isolated_global, tmp_path, capsys):
        proj = _proj(tmp_path, 'debug = "enabled"\n')
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.debug is False and cfg.sources["debug"] == "default"
        assert "ignoring malformed debug" in capsys.readouterr().err
        assert cfg.config_issues()[0]["kind"] == "malformed_value"

    def test_non_bool_scalar_degrades(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, "debug = 2\n")
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.debug is False and cfg.sources["debug"] == "default"


# ── 4. [configurable] passthrough ───────────────────────────────────


class TestConfigurable:
    def test_configurable_table_exposed(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, '[configurable]\nmodel = "m"\ntemperature = 0.7\n')
        cfg = HostConfig.resolve(env={}, toml_start=proj)
        assert cfg.configurable() == {"model": "m", "temperature": 0.7}
        assert cfg.config_dict(configurable=cfg.configurable())["configurable"] == {
            "model": "m", "temperature": 0.7,
        }

    def test_absent_or_non_table_configurable_is_empty(self, isolated_global, tmp_path):
        assert HostConfig.resolve(env={}, toml_start=tmp_path).configurable() == {}
        proj = _proj(tmp_path, 'configurable = "nope"\n')
        assert HostConfig.resolve(env={}, toml_start=proj).configurable() == {}

    def test_host_cli_shows_configurable(self, isolated_global, tmp_path):
        proj = _proj(tmp_path, '[configurable]\ngreeting = "Bonjour"\n')
        r = subprocess.run(
            [sys.executable, "-m", "langstage_core.host"],
            capture_output=True, text=True, cwd=proj, env={**os.environ},
        )
        assert "LangGraph configurable:" in r.stdout and "greeting: Bonjour" in r.stdout

    def test_agui_show_config_shows_configurable(self, isolated_global, tmp_path, monkeypatch, capsys):
        from langstage_core.agui.__main__ import main

        proj = _proj(tmp_path, '[configurable]\ngreeting = "Bonjour"\n')
        monkeypatch.chdir(proj)
        assert main(["--show-config"]) == 0
        assert "greeting: Bonjour" in capsys.readouterr().out
        assert main(["--show-config", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["configurable"] == {"greeting": "Bonjour"}

    def test_agui_message_forwards_configurable_to_graph(
        self, isolated_global, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("ag_ui_langgraph")
        agent_py = tmp_path / "cfg_agent.py"
        agent_py.write_text(
            "from langchain_core.messages import AIMessage\n"
            "from langgraph.graph import StateGraph, START, END, MessagesState\n"
            "def respond(state, config):\n"
            "    g = (config or {}).get('configurable', {}).get('greeting', '<none>')\n"
            "    return {'messages': [AIMessage(content=f'greeting={g}')]}\n"
            "_g = StateGraph(MessagesState)\n"
            "_g.add_node('respond', respond)\n"
            "_g.add_edge(START, 'respond'); _g.add_edge('respond', END)\n"
            "graph = _g.compile()\n"
        )
        (tmp_path / "langstage.toml").write_text('[configurable]\ngreeting = "Bonjour"\n')
        monkeypatch.chdir(tmp_path)
        from langstage_core.agui.__main__ import main

        rc = main(["--agent", f"{agent_py}:graph", "--message", "hi", "--json"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["text"] == "greeting=Bonjour"

    def test_serve_forwards_configurable(self, isolated_global, tmp_path, monkeypatch):
        from langstage_core import agui
        from langstage_core.agui.__main__ import main

        (tmp_path / "langstage.toml").write_text('[configurable]\ngreeting = "Bonjour"\n')
        monkeypatch.chdir(tmp_path)
        seen = {}
        monkeypatch.setattr(agui, "serve", lambda graph, **kw: seen.update(kw))
        assert main(["--demo"]) == 0
        assert seen["config"] == {"configurable": {"greeting": "Bonjour"}}


# ── 5. config_dict lists every contributing TOML file ────────────────


def test_config_dict_lists_all_toml_paths(isolated_global, tmp_path):
    (isolated_global / "config.toml").write_text('[workspace]\nroot = "/from/global"\n')
    proj = _proj(tmp_path, "[server]\nport = 9123\n")
    cfg = HostConfig.resolve(env={}, toml_start=proj)
    toml = cfg.config_dict()["toml"]
    assert toml["paths"] == [
        str(isolated_global / "config.toml"),
        str(proj / "langstage.toml"),
    ]
    assert toml["path"] == str(proj / "langstage.toml")  # back-compat: highest precedence


# ── 6. console-safe output ──────────────────────────────────────────


class _Cp1252(io.TextIOWrapper):
    def __init__(self):
        super().__init__(io.BytesIO(), encoding="cp1252", errors="strict", newline="\n")

    def text(self):
        self.flush()
        return self.buffer.getvalue().decode("cp1252")


class TestSafePrint:
    def test_unencodable_chars_are_escaped_not_raised(self):
        from langstage_core.console import safe_print

        out = _Cp1252()
        safe_print("title = 日本語 \U0001F680 café", file=out)
        assert out.text() == "title = \\u65e5\\u672c\\u8a9e \\U0001f680 café\n"

    def test_encodable_text_untouched(self):
        from langstage_core.console import safe_print, safe_write

        out = _Cp1252()
        safe_print("a", "b", sep="-", end="!", file=out)
        safe_write("xyz", file=out)
        assert out.text() == "a-b!xyz"

    def test_stream_without_encoding(self):
        from langstage_core.console import safe_print

        out = io.StringIO()
        safe_print("日本", file=out)
        assert out.getvalue() == "日本\n"

    @pytest.mark.parametrize("args", [[], ["--show-config"]])
    def test_host_cli_survives_cp1252(self, tmp_path, args):
        env = _no_pytest_env(PYTHONIOENCODING="cp1252", LANGSTAGE_TITLE="日本語 \U0001F680")
        r = subprocess.run(
            [sys.executable, "-m", "langstage_core.host", *args],
            capture_output=True, cwd=tmp_path, env=env,
        )
        assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
        assert b"\\u65e5" in r.stdout

    def test_agui_show_config_survives_cp1252(self, tmp_path):
        env = _no_pytest_env(
            PYTHONIOENCODING="cp1252", LANGSTAGE_AGENT_SPEC="C:/Users/田中/agent.py:graph"
        )
        r = subprocess.run(
            [sys.executable, "-m", "langstage_core.agui", "--show-config"],
            capture_output=True, cwd=tmp_path, env=env,
        )
        assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
        assert b"\\u7530\\u4e2d" in r.stdout

    def test_agui_message_survives_cp1252(self, tmp_path):
        pytest.importorskip("ag_ui_langgraph")
        agent_py = tmp_path / "emoji_agent.py"
        agent_py.write_text(
            "from langchain_core.messages import AIMessage\n"
            "from langgraph.graph import StateGraph, START, END, MessagesState\n"
            "def r(state):\n"
            "    return {'messages': [AIMessage(content='Result: 42 \\U0001F389')]}\n"
            "_g = StateGraph(MessagesState)\n"
            "_g.add_node('r', r)\n"
            "_g.add_edge(START, 'r'); _g.add_edge('r', END)\n"
            "graph = _g.compile()\n"
        )
        env = _no_pytest_env(PYTHONIOENCODING="cp1252")
        r = subprocess.run(
            [sys.executable, "-m", "langstage_core.agui", "-a", f"{agent_py}:graph", "-m", "hi"],
            capture_output=True, cwd=tmp_path, env=env,
        )
        assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
        assert b"Result: 42 \\U0001f389" in r.stdout
