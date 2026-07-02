"""Tests for the demo default-agent factory.

These avoid actually building an agent (that needs deepagents + an API key).
They verify the import is lazy and the missing-extra error is helpful.
"""
import sys

import pytest

from langstage_core.demo import create_default_agent


def _deepagents_installed() -> bool:
    try:
        import deepagents  # noqa: F401
        return True
    except ImportError:
        return False


def test_factory_is_importable_and_callable():
    assert callable(create_default_agent)


def test_import_does_not_eagerly_pull_deepagents():
    # The factory imports deepagents lazily; importing the module must not.
    if _deepagents_installed():
        pytest.skip("deepagents installed — laziness check is vacuous")
    assert "deepagents" not in sys.modules


def test_missing_extra_raises_helpful_error():
    if _deepagents_installed():
        pytest.skip("deepagents installed — cannot exercise the missing-extra path")
    with pytest.raises(RuntimeError, match=r"deepagents.*extra|extra.*deepagents"):
        create_default_agent(".")
