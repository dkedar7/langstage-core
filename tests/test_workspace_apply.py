"""apply_workspace / workspace_root — the single workspace source of truth (ADR 0005).

A surface calls apply_workspace() once after resolving config; everything downstream
reads workspace_root() instead of a private global. These replace the five bespoke
"apply" mechanisms that each drifted into a workspace bug.
"""

import os
from pathlib import Path

import pytest

from langstage_core import Workspace, apply_workspace, workspace_root
from langstage_core.host import workspace as ws_mod


@pytest.fixture(autouse=True)
def _isolate_active_workspace(monkeypatch, tmp_path):
    """apply_workspace sets PROCESS-global state (a module global + env + maybe cwd).
    Snapshot and restore it around each test so invocations stay isolated."""
    monkeypatch.chdir(tmp_path)  # a safe cwd; monkeypatch restores the real one
    saved_active = ws_mod._ACTIVE
    monkeypatch.setattr(ws_mod, "_ACTIVE", None)
    monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("DEEPAGENT_WORKSPACE_ROOT", raising=False)
    yield
    ws_mod._ACTIVE = saved_active


def test_apply_workspace_publishes_the_source_of_truth(tmp_path):
    target = tmp_path / "ws"
    ws = apply_workspace(target)
    assert isinstance(ws, Workspace)
    # Created, active, and published to both env names — all pointing at the resolved dir.
    assert target.is_dir()
    assert workspace_root() == target.resolve()
    assert os.environ["LANGSTAGE_WORKSPACE_ROOT"] == str(target.resolve())
    assert os.environ["DEEPAGENT_WORKSPACE_ROOT"] == str(target.resolve())


def test_apply_workspace_no_chdir_by_default(tmp_path):
    before = Path.cwd()
    apply_workspace(tmp_path / "ws")
    assert Path.cwd() == before  # servers must not have their cwd moved


def test_apply_workspace_chdir_opt_in(tmp_path):
    target = tmp_path / "ws"
    apply_workspace(target, chdir=True)
    assert Path.cwd().resolve() == target.resolve()


def test_workspace_root_falls_back_to_env_then_cwd(tmp_path, monkeypatch):
    # No apply_workspace() this test -> fall back to the env a parent process set...
    monkeypatch.setenv("LANGSTAGE_WORKSPACE_ROOT", str(tmp_path / "from_env"))
    assert workspace_root() == (tmp_path / "from_env").resolve()
    # ...and to cwd when nothing is set.
    monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
    assert workspace_root() == Path.cwd().resolve()


def test_in_process_active_beats_stale_env(tmp_path, monkeypatch):
    # A stale env from a previous run must not override the value applied here.
    monkeypatch.setenv("LANGSTAGE_WORKSPACE_ROOT", str(tmp_path / "stale"))
    apply_workspace(tmp_path / "current")
    assert workspace_root() == (tmp_path / "current").resolve()
