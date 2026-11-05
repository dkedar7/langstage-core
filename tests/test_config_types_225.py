"""gh #225: text fields type-check their TOML value, and overrides / merge() go
through the same coercion as TOML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from langstage_core import HostConfig


def _toml(tmp_path: Path, body: str) -> Path:
    (tmp_path / "langstage.toml").write_text(body, encoding="utf-8")
    return tmp_path


def _resolve(tmp_path: Path, **kw) -> HostConfig:
    return HostConfig.resolve(toml_start=tmp_path, env={}, **kw)


@pytest.mark.parametrize("raw", ["true", "1.5", "[1, 2]", "{ a = 1 }", "2026-11-05"])
def test_non_text_toml_value_for_text_field_degrades(tmp_path, raw):
    _toml(tmp_path, f"[server]\nhost = {raw}\n")
    cfg = _resolve(tmp_path)
    assert cfg.host == "localhost"
    assert cfg.sources["host"] == "default"
    (issue,) = [i for i in cfg.config_issues() if i["kind"] == "malformed_value"]
    assert issue["field"] == "host"
    assert issue["source"] == "toml:server.host"
    assert "quote it" in issue["error"]


def test_integer_for_text_field_is_kept_as_its_digits(tmp_path):
    # An unquoted numeric token/password means the text the user typed; dropping it
    # to the default would turn auth off.
    _toml(tmp_path, '[agent]\nspec = "x.py:graph"\n[server]\nhost = 127\n')
    cfg = _resolve(tmp_path)
    assert cfg.host == "127"
    assert cfg.config_issues() == []


def test_bad_text_falls_back_to_the_layer_beneath(tmp_path):
    _toml(tmp_path, "[server]\nhost = false\n")
    cfg = HostConfig.resolve(toml_start=tmp_path, env={"LANGSTAGE_HOST": "0.0.0.0"})
    assert cfg.host == "0.0.0.0"  # env still wins; toml value only noted
    cfg = _resolve(tmp_path, overrides={"host": "example.org"})
    assert cfg.host == "example.org"


def test_optional_text_field_accepts_none_and_rejects_bool(tmp_path):
    _toml(tmp_path, "[agent]\nspec = true\n")
    cfg = _resolve(tmp_path)
    assert cfg.agent_spec is None
    assert [i["field"] for i in cfg.config_issues()] == ["agent_spec"]


@pytest.mark.parametrize("raw,want", [("false", False), ("off", False), ("yes", True), (0, False)])
def test_override_bool_is_coerced(tmp_path, raw, want):
    cfg = _resolve(tmp_path, overrides={"debug": raw})
    assert cfg.debug is want
    assert cfg.sources["debug"] == "override"
    assert cfg.config_issues() == []
    assert cfg.config_dict()["config"]["debug"]["value"] is want


def test_malformed_override_degrades_with_issue(tmp_path, capsys):
    _toml(tmp_path, "debug = true\n")
    cfg = _resolve(tmp_path, overrides={"debug": "maybe-225", "port": "80x"})
    assert cfg.debug is True
    assert cfg.sources["debug"].startswith("toml")
    assert cfg.port == 8050
    issues = {i["field"]: i for i in cfg.config_issues()}
    assert issues["debug"]["source"] == "override"
    assert issues["debug"]["kind"] == "malformed_value"
    assert issues["port"]["source"] == "override"
    assert "malformed override debug='maybe-225'" in capsys.readouterr().err


def test_override_numeric_and_path_coerced(tmp_path):
    cfg = _resolve(tmp_path, overrides={"port": "9001", "workspace_root": str(tmp_path)})
    assert cfg.port == 9001
    assert isinstance(cfg.workspace_root, Path)


def test_override_path_object_for_text_field(tmp_path):
    spec = tmp_path / "agent.py"
    cfg = _resolve(tmp_path, overrides={"agent_spec": spec})
    assert isinstance(cfg.agent_spec, str)


def test_merge_coerces_and_keeps_current_on_bad_value(capsys):
    cfg = HostConfig(port=8050, debug=True)
    assert cfg.merge(debug="off").debug is False
    assert cfg.merge(port="9002").port == 9002
    bad = cfg.merge(port="nope-225", title="T")
    assert bad.port == 8050
    assert bad.title == "T"
    (issue,) = bad.config_issues()
    assert issue["field"] == "port" and issue["source"] == "override"
    assert "malformed override port='nope-225'" in capsys.readouterr().err


def test_subclass_text_fields_are_checked(tmp_path):
    @dataclass
    class StageConfig(HostConfig):
        theme: str = "auto"
        _TOML = {"theme": "ui.theme"}

    _toml(tmp_path, "[ui]\ntheme = true\n")
    cfg = StageConfig.resolve(toml_start=tmp_path, env={})
    assert cfg.theme == "auto"
    assert [i["field"] for i in cfg.config_issues()] == ["theme"]
