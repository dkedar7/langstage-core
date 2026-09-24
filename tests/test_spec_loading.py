"""Agent-spec loading + config-relative path resolution (Wave 2).

Covers the root causes reported across the family:

- sibling imports from a ``file.py:attr`` agent (core #150, cli #145, web #167)
- a project-local package via ``module:attr`` (core #147, cli #141)
- whitespace / ``~`` normalization of the spec (vscode #135, #125)
- a colon-less spec is a clear error, never a silent default (jupyter #151)
- a ``str`` attribute is never re-read as another spec (cli #149)
- import-time ``print`` kept off stdout on machine-readable paths (cli #136, web #140)
- relative ``[agent] spec`` / ``[workspace] root`` resolve against the toml's own
  directory, for the project AND the global file (cli #132, #133, vscode #123, #126)
"""
import os
import sys
import textwrap
from pathlib import Path

import pytest

from langstage_core.host import HostConfig, apply_workspace, load_agent_spec
from langstage_core.host import workspace as ws_mod


@pytest.fixture
def clean_sys_path(monkeypatch):
    """Snapshot sys.path / sys.modules so path inserts and throwaway modules don't leak."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ── sys.path: file form + dotted form ────────────────────────────────


class TestSiblingImports:
    def test_file_spec_can_import_sibling_module(self, tmp_path, clean_sys_path):
        # core #150 / cli #145 / web #167: a multi-file agent.
        _write(tmp_path / "w2_helpers_b.py", "VALUE = {'ok': 1}\n")
        agent = _write(tmp_path / "my_agent.py", "from w2_helpers_b import VALUE\ngraph = VALUE\n")
        assert load_agent_spec(f"{agent}:graph") == {"ok": 1}

    def test_repeated_loads_do_not_duplicate_path_entry(self, tmp_path, clean_sys_path):
        agent = _write(tmp_path / "my_agent.py", "graph = {'n': 1}\n")
        load_agent_spec(f"{agent}:graph")
        load_agent_spec(f"{agent}:graph")
        norm = os.path.normcase(str(tmp_path.resolve()))
        assert sum(os.path.normcase(os.path.abspath(p)) == norm for p in sys.path if p) == 1

    def test_dotted_spec_finds_project_local_package(self, tmp_path, monkeypatch, clean_sys_path):
        # core #147 / cli #141: `mypkg.agents:chatbot` from the project dir, with cwd NOT
        # on sys.path (a console-script entry point).
        _write(tmp_path / "w2pkg" / "__init__.py", "")
        _write(tmp_path / "w2pkg" / "agents.py", "chatbot = {'pkg': True}\n")
        monkeypatch.chdir(tmp_path)
        cwd_norm = os.path.normcase(str(tmp_path.resolve()))
        sys.path[:] = [
            p for p in sys.path
            if p not in ("", ".") and os.path.normcase(os.path.abspath(p)) != cwd_norm
        ]
        assert load_agent_spec("w2pkg.agents:chatbot") == {"pkg": True}

    def test_dotted_spec_base_dir(self, tmp_path, clean_sys_path):
        # A toml-sourced dotted spec resolves against the toml dir, passed as base_dir.
        _write(tmp_path / "w2pkg_b" / "__init__.py", "")
        _write(tmp_path / "w2pkg_b" / "agents.py", "chatbot = {'base': True}\n")
        assert load_agent_spec("w2pkg_b.agents:chatbot", base_dir=tmp_path) == {"base": True}

    def test_dotted_spec_missing_module_still_raises(self, tmp_path, monkeypatch, clean_sys_path):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ModuleNotFoundError, match="w2_nope"):
            load_agent_spec("w2_nope.agents:chatbot")

    def test_dotted_real_import_error_inside_module_not_masked(
        self, tmp_path, monkeypatch, clean_sys_path
    ):
        # The module IS found but itself imports something missing: surface THAT name.
        _write(tmp_path / "w2pkg_c" / "__init__.py", "")
        _write(tmp_path / "w2pkg_c" / "agents.py", "import w2_definitely_missing_dep\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ModuleNotFoundError, match="w2_definitely_missing_dep"):
            load_agent_spec("w2pkg_c.agents:chatbot")

    def test_relative_file_spec_base_dir(self, tmp_path, clean_sys_path):
        _write(tmp_path / "sub" / "agent.py", "graph = {'rel': 1}\n")
        assert load_agent_spec("sub/agent.py:graph", base_dir=tmp_path) == {"rel": 1}


# ── spec normalization + validation ──────────────────────────────────


class TestSpecNormalization:
    def test_whitespace_is_stripped(self, tmp_path, clean_sys_path):
        # vscode #135: a copy-paste trailing space.
        agent = _write(tmp_path / "agent.py", "graph = {'ws': 1}\n")
        assert load_agent_spec(f"  {agent}:graph \n") == {"ws": 1}

    def test_whitespace_around_colon_is_stripped(self, tmp_path, clean_sys_path):
        agent = _write(tmp_path / "agent.py", "graph = {'ws': 2}\n")
        assert load_agent_spec(f"{agent} : graph") == {"ws": 2}

    def test_tilde_is_expanded(self, tmp_path, monkeypatch, clean_sys_path):
        # vscode #125: `~/agents/agent.py:graph` from a toml/env value (never shell-expanded).
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _write(tmp_path / "agents" / "agent.py", "graph = {'home': 1}\n")
        assert load_agent_spec("~/agents/agent.py:graph") == {"home": 1}

    def test_colon_less_py_spec_is_clear_error_with_hint(self):
        # jupyter #151: never a silent fallback; the error names the fix.
        with pytest.raises(ValueError, match=r"my_agent\.py:graph"):
            load_agent_spec("my_agent.py")

    def test_colon_less_windows_path_is_clear_error(self):
        # `C:\x\agent.py` (no :attr) must not be split at the drive colon and imported
        # as a module named "C".
        with pytest.raises(ValueError, match="object name"):
            load_agent_spec(r"C:\x\agent.py")

    def test_parse_agent_spec(self):
        from langstage_core.host import parse_agent_spec

        assert parse_agent_spec(" pkg.mod : graph ") == ("pkg.mod", "graph")
        assert parse_agent_spec(r"C:\x\agent.py:graph") == (r"C:\x\agent.py", "graph")
        with pytest.raises(ValueError):
            parse_agent_spec("pkg.mod")
        with pytest.raises(ValueError):
            parse_agent_spec("   ")

    def test_str_attribute_is_rejected_not_reinterpreted(self, tmp_path, clean_sys_path):
        # cli #149: the VALUE of `graph` must never be imported as another spec.
        agent = _write(tmp_path / "agent.py", "graph = 'openai:gpt-4o-mini'\n")
        with pytest.raises(TypeError, match="str"):
            load_agent_spec(f"{agent}:graph")

    def test_build_agent_str_attribute_not_reinterpreted(self, tmp_path, clean_sys_path):
        pytest.importorskip("ag_ui_langgraph")
        from langstage_core.agui import build_agent

        agent = _write(tmp_path / "agent.py", "graph = 'math:pi'\n")
        with pytest.raises(TypeError, match="str"):
            build_agent(f"{agent}:graph")


# ── import-time stdout ───────────────────────────────────────────────


class TestImportTimeStdout:
    def test_default_leaves_stdout_alone(self, tmp_path, capsys, clean_sys_path):
        agent = _write(tmp_path / "noisy.py", "print('BANNER')\ngraph = 1\n")
        load_agent_spec(f"{agent}:graph")
        assert "BANNER" in capsys.readouterr().out

    def test_stdout_to_stderr(self, tmp_path, capsys, clean_sys_path):
        # cli #136 / web #140: machine-readable paths keep stdout clean.
        agent = _write(tmp_path / "noisy.py", "print('BANNER')\ngraph = 1\n")
        assert load_agent_spec(f"{agent}:graph", stdout_to_stderr=True) == 1
        out, err = capsys.readouterr()
        assert "BANNER" not in out
        assert "BANNER" in err

    def test_agui_message_json_stdout_is_only_json(self, tmp_path, capsys, clean_sys_path):
        pytest.importorskip("ag_ui_langgraph")
        import json

        from langstage_core.agui.__main__ import main

        agent = _write(
            tmp_path / "noisy.py",
            "print('BANNER')\nfrom langstage_core.demo.stub import graph\n",
        )
        rc = main(["--agent", f"{agent}:graph", "-m", "hi", "--json"])
        out, err = capsys.readouterr()
        assert rc == 0, err
        assert "BANNER" in err
        json.loads(out)  # nothing but the JSON object on stdout


# ── config-relative paths ────────────────────────────────────────────


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "confhome"
    home.mkdir()
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(home))
    for var in ("LANGSTAGE_AGENT_SPEC", "DEEPAGENT_AGENT_SPEC",
                "LANGSTAGE_WORKSPACE_ROOT", "DEEPAGENT_WORKSPACE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    return home


class TestConfigRelativePaths:
    def _project(self, tmp_path: Path, toml: str) -> Path:
        proj = tmp_path / "proj"
        (proj / "src" / "deep").mkdir(parents=True)
        _write(proj / "langstage.toml", toml)
        return proj

    def test_relative_spec_resolves_against_project_toml_dir(self, tmp_path, isolated_home):
        # vscode #123: run from a subdirectory; the toml is discovered by walking up.
        proj = self._project(tmp_path, '[agent]\nspec = "agent.py:graph"\n')
        cfg = HostConfig.resolve(toml_start=proj / "src" / "deep", env={})
        assert cfg.agent_spec == f"{proj.resolve() / 'agent.py'}:graph"

    def test_relative_workspace_root_resolves_against_project_toml_dir(self, tmp_path, isolated_home):
        # cli #132 / vscode #126.
        proj = self._project(tmp_path, '[workspace]\nroot = "wsdir"\n')
        cfg = HostConfig.resolve(toml_start=proj / "src" / "deep", env={})
        assert Path(cfg.workspace_root) == proj.resolve() / "wsdir"

    def test_relative_paths_in_global_toml_resolve_against_its_dir(self, tmp_path, isolated_home):
        # cli #133: the same rule for the global file (documented; use ~ or absolute there).
        _write(isolated_home / "config.toml",
               '[agent]\nspec = "agents/a.py:graph"\n[workspace]\nroot = "ws"\n')
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        cfg = HostConfig.resolve(toml_start=elsewhere, env={})
        assert cfg.agent_spec == f"{isolated_home / 'agents' / 'a.py'}:graph"
        assert Path(cfg.workspace_root) == isolated_home / "ws"

    def test_tilde_in_toml_expands(self, tmp_path, isolated_home, monkeypatch):
        # vscode #125, both keys.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        proj = self._project(
            tmp_path, '[agent]\nspec = "~/agents/a.py:graph"\n[workspace]\nroot = "~/thing"\n'
        )
        cfg = HostConfig.resolve(toml_start=proj, env={})
        assert cfg.agent_spec == f"{tmp_path / 'agents' / 'a.py'}:graph"
        assert Path(cfg.workspace_root) == tmp_path / "thing"

    def test_module_spec_in_toml_unchanged_but_base_dir_exposed(self, tmp_path, isolated_home):
        proj = self._project(tmp_path, '[agent]\nspec = "mypkg.agents:chatbot"\n')
        cfg = HostConfig.resolve(toml_start=proj / "src", env={})
        assert cfg.agent_spec == "mypkg.agents:chatbot"
        assert cfg.toml_dir_for("agent_spec") == proj.resolve()

    def test_absolute_values_unchanged(self, tmp_path, isolated_home):
        target = (tmp_path / "abs" / "a.py").resolve()
        proj = self._project(
            tmp_path, f'[agent]\nspec = "{target.as_posix()}:graph"\n'
        )
        cfg = HostConfig.resolve(toml_start=proj / "src", env={})
        assert Path(cfg.agent_spec.rpartition(":")[0]) == target

    def test_env_and_override_stay_cwd_relative(self, tmp_path, isolated_home):
        # Only TOML-sourced values are rebased; "where you typed it" keeps the cwd base.
        proj = self._project(tmp_path, '[agent]\nspec = "agent.py:graph"\n[workspace]\nroot = "w"\n')
        cfg = HostConfig.resolve(
            toml_start=proj / "src",
            env={"LANGSTAGE_AGENT_SPEC": "other.py:graph"},
            overrides={"workspace_root": "cli_ws"},
        )
        assert cfg.agent_spec == "other.py:graph"
        assert str(cfg.workspace_root) == "cli_ws"
        assert cfg.toml_dir_for("agent_spec") is None

    def test_env_workspace_root_tilde_expands(self, tmp_path, isolated_home, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        cfg = HostConfig.resolve(env={"LANGSTAGE_WORKSPACE_ROOT": "~/w"}, use_toml=False)
        assert Path(cfg.workspace_root) == tmp_path / "w"


class TestApplyWorkspaceTilde:
    def test_apply_workspace_expands_tilde(self, tmp_path, monkeypatch):
        # vscode #125: never mkdir a literal "~" directory under cwd.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ws_mod, "_ACTIVE", None)
        monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("DEEPAGENT_WORKSPACE_ROOT", raising=False)
        ws = apply_workspace("~/thing")
        assert ws.root == (tmp_path / "home" / "thing").resolve()
        assert not (tmp_path / "~").exists()


class TestAguiCliFromSubdir:
    def test_toml_dotted_spec_verifies_from_subdirectory(
        self, tmp_path, isolated_home, monkeypatch, clean_sys_path
    ):
        # End to end: a project-local package named in a walked-up langstage.toml loads
        # from a subdirectory (toml_dir_for -> base_dir), cwd not on sys.path.
        pytest.importorskip("ag_ui_langgraph")
        from langstage_core.agui.__main__ import main

        proj = tmp_path / "proj"
        _write(proj / "w2pkg_d" / "__init__.py", "")
        _write(proj / "w2pkg_d" / "agents.py", "from langstage_core.demo.stub import graph\n")
        _write(proj / "langstage.toml", '[agent]\nspec = "w2pkg_d.agents:graph"\n')
        (proj / "sub").mkdir()
        monkeypatch.chdir(proj / "sub")
        assert main(["--verify"]) == 0
