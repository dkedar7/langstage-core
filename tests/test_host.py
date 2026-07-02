"""Tests for the host submodule: load_agent_spec, HostConfig, Workspace."""
import sys
from pathlib import Path

import pytest

from langstage_core.host import HostConfig, Workspace, load_agent_spec


# ── load_agent_spec ──────────────────────────────────────────────────


class TestLoadAgentSpec:
    def test_load_from_file(self, tmp_path: Path):
        agent_file = tmp_path / "my_agent.py"
        agent_file.write_text("agent = {'name': 'demo'}\n")

        loaded = load_agent_spec(f"{agent_file}:agent")
        assert loaded == {"name": "demo"}

    def test_load_from_dotted_module(self):
        # math:pi is a stand-in for a dotted module path.
        import math

        assert load_agent_spec("math:pi") == math.pi

    def test_missing_separator_raises_value_error(self):
        with pytest.raises(ValueError, match="suffix is required"):
            load_agent_spec("just_a_path")

    def test_empty_object_name_raises_value_error(self):
        with pytest.raises(ValueError):
            load_agent_spec("module:")

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_agent_spec("does_not_exist.py:agent")

    def test_missing_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute"):
            load_agent_spec("math:not_a_real_attr")

    def test_no_implicit_fallback(self, tmp_path: Path):
        # A file defining only `graph` must NOT be loadable as `:agent`.
        agent_file = tmp_path / "graph_only.py"
        agent_file.write_text("graph = 1\n")
        with pytest.raises(AttributeError):
            load_agent_spec(f"{agent_file}:agent")

    def test_colon_in_windows_path_uses_last_separator(self, tmp_path: Path):
        # rpartition(':') must split on the final colon so drive letters
        # like C:\... in a path still resolve the object name correctly.
        agent_file = tmp_path / "win_agent.py"
        agent_file.write_text("graph = 'ok'\n")
        loaded = load_agent_spec(f"{agent_file}:graph")
        assert loaded == "ok"


# ── HostConfig ───────────────────────────────────────────────────────


class TestHostConfig:
    def test_defaults(self):
        cfg = HostConfig()
        assert cfg.agent_spec is None
        assert cfg.workspace_root == Path(".")
        assert cfg.host == "localhost"
        assert cfg.port == 8050
        assert cfg.debug is False
        assert cfg.title == "LangStage"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "agent.py:graph")
        monkeypatch.setenv("DEEPAGENT_WORKSPACE_ROOT", "/tmp/ws")
        monkeypatch.setenv("DEEPAGENT_HOST", "0.0.0.0")
        monkeypatch.setenv("DEEPAGENT_PORT", "9000")
        monkeypatch.setenv("DEEPAGENT_DEBUG", "true")
        monkeypatch.setenv("DEEPAGENT_TITLE", "My Agent")

        cfg = HostConfig.from_env()
        assert cfg.agent_spec == "agent.py:graph"
        assert cfg.workspace_root == Path("/tmp/ws")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.debug is True
        assert cfg.title == "My Agent"

    def test_from_env_empty_uses_defaults(self, monkeypatch):
        for key in (
            "DEEPAGENT_AGENT_SPEC", "DEEPAGENT_WORKSPACE_ROOT", "DEEPAGENT_HOST",
            "DEEPAGENT_PORT", "DEEPAGENT_DEBUG", "DEEPAGENT_TITLE",
        ):
            monkeypatch.delenv(key, raising=False)
        cfg = HostConfig.from_env()
        assert cfg.port == 8050
        assert cfg.workspace_root == Path(".")

    def test_merge_applies_non_none(self):
        cfg = HostConfig(port=8050, title="A")
        merged = cfg.merge(port=9001, title=None)
        assert merged.port == 9001
        assert merged.title == "A"  # None override ignored
        assert cfg.port == 8050  # original unchanged

    def test_merge_ignores_unknown_keys(self):
        cfg = HostConfig()
        merged = cfg.merge(not_a_field="x", port=1234)
        assert merged.port == 1234
        assert not hasattr(merged, "not_a_field")

    def test_subclass_extends(self, monkeypatch):
        from dataclasses import dataclass

        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"

        cfg = WebConfig.from_env().merge(theme="dark")
        assert cfg.theme == "dark"
        assert cfg.port == 8050


# ── Workspace ────────────────────────────────────────────────────────


class TestWorkspace:
    def test_ensure_creates_dir(self, tmp_path: Path):
        target = tmp_path / "ws" / "nested"
        ws = Workspace(target).ensure()
        assert target.is_dir()
        assert ws.root == target

    def test_subpath_joins_under_root(self, tmp_path: Path):
        ws = Workspace(tmp_path).ensure()
        p = ws.subpath("a", "b.txt")
        assert p == (tmp_path / "a" / "b.txt").resolve()

    def test_subpath_rejects_escape(self, tmp_path: Path):
        ws = Workspace(tmp_path / "ws").ensure()
        with pytest.raises(ValueError, match="escapes workspace"):
            ws.subpath("..", "..", "etc", "passwd")

    def test_name(self, tmp_path: Path):
        ws = Workspace(tmp_path / "project")
        assert ws.name == "project"

    def test_str_root_coerced_to_path(self):
        ws = Workspace(".")
        assert isinstance(ws.root, Path)
