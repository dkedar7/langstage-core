"""Regression for the agent-name -> OpenAI message `name` 400 (langstage-jupyter #23).

OpenAI-compatible providers require the message `name` field to match
^[^\\s<|\\\\/>]+$. The agent's display name flows into that field, so a name with a
space 400s on the second turn. create_default_agent now slugifies the name.
"""
import inspect
import warnings

import pytest

from langstage_core.demo.agent import _safe_agent_name, create_default_agent


def test_default_name_is_provider_safe():
    default = inspect.signature(create_default_agent).parameters["name"].default
    assert " " not in default
    assert _safe_agent_name(default) == default  # default is already a no-op slug


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("deep-agent", "deep-agent"),
        ("Default Agent", "Default-Agent"),
        ("My Cool Agent", "My-Cool-Agent"),
        ("a/b\\c|d>e<f", "a-b-c-d-e-f"),
        ("  spaced  ", "spaced"),
        ("   ", "agent"),
    ],
)
def test_safe_agent_name(raw, expected):
    assert _safe_agent_name(raw) == expected


def test_warns_when_name_changed():
    with pytest.warns(UserWarning, match="name"):
        _safe_agent_name("Has Space")


def test_no_warning_when_already_safe():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        assert _safe_agent_name("already-safe") == "already-safe"
