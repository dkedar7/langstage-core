"""The top-level public API matches what the README documents (gh #80).

The README's "What's in the box" says everything is re-exported from the top-level
`langstage_core` package (except the AG-UI helpers under `langstage_core.agui`). Two docs
pointed `SessionAdapter` / `Session` at import paths that raised ImportError; these tests
pin the documented surface so it can't silently drift again.
"""

import langstage_core


def test_session_adapter_is_importable_top_level():
    # gh #80: `from langstage_core import SessionAdapter, Session` was promised by the
    # README's blanket top-level claim but used to ImportError.
    from langstage_core import Session, SessionAdapter

    assert SessionAdapter is not None and Session is not None


def test_top_level_matches_the_adapters_submodule():
    from langstage_core import Session, SessionAdapter
    from langstage_core.adapters import Session as S2
    from langstage_core.adapters import SessionAdapter as SA2

    assert SessionAdapter is SA2
    assert Session is S2


def test_documented_top_level_names_are_all_exported():
    # A representative slice of the README "What's in the box" top-level surface.
    expected = {
        "load_agent_spec",
        "HostConfig",
        "Workspace",
        "apply_workspace",
        "workspace_root",
        "SessionAdapter",
        "Session",
        "prepare_agent_input",
        "create_resume_input",
        "ToolExtractor",
        "TaskRunner",
    }
    missing = expected - set(langstage_core.__all__)
    assert not missing, f"missing from __all__: {missing}"
    # everything advertised in __all__ actually imports off the package
    for name in langstage_core.__all__:
        assert hasattr(langstage_core, name), f"{name} in __all__ but not importable"
