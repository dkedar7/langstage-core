"""gh #253: a missing provider key that fails as a *connection* error names the unset
variable, on every error line core formats (event and chunk wires, served RUN_ERROR,
`langstage-agui -m` / `--json` / `--verify`, SessionAdapter tasks).

With no key, langchain-anthropic sends an empty ``Authorization`` header and the user
sees ``AnthropicConnectionError: Connection error.``, which reads like a network
problem. The provider classes are faked here (same class names and modules as the
real ones) so the tests need no provider package and no network.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import langstage_core
from langstage_core.keyhint import describe_error, missing_key_hint, with_key_hint


def _cls(name: str, module: str, base: type = Exception) -> type:
    return type(name, (base,), {"__module__": module})


AnthropicConnectionError = _cls("AnthropicConnectionError", "langchain_anthropic.chat_models")
AnthropicAuthenticationError = _cls("AnthropicAuthenticationError", "langchain_anthropic.chat_models")
SdkAPIConnectionError = _cls("APIConnectionError", "anthropic._exceptions")
OpenAIAPIConnectionError = _cls("APIConnectionError", "openai._exceptions")


@pytest.fixture
def no_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ── the helper ───────────────────────────────────────────────────────────


def test_wrapped_connection_error_names_the_unset_key(no_keys):
    exc = AnthropicConnectionError("Connection error.")
    assert missing_key_hint(exc) == "hint: ANTHROPIC_API_KEY is not set"
    assert describe_error(exc) == (
        "AnthropicConnectionError: Connection error. (hint: ANTHROPIC_API_KEY is not set)"
    )


def test_sdk_error_found_through_the_cause_chain(no_keys):
    try:
        try:
            raise SdkAPIConnectionError("Connection error.")
        except Exception as inner:
            raise RuntimeError("node failed") from inner
    except RuntimeError as outer:
        assert missing_key_hint(outer) == "hint: ANTHROPIC_API_KEY is not set"


def test_provider_from_the_module_when_the_class_name_is_generic(no_keys):
    assert missing_key_hint(OpenAIAPIConnectionError("x")) == "hint: OPENAI_API_KEY is not set"


@pytest.mark.parametrize("value", ["sk-ant-secret-value-123"])
def test_no_hint_when_the_key_is_set_and_the_value_is_never_printed(monkeypatch, value):
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    exc = AnthropicConnectionError("Connection error.")
    assert missing_key_hint(exc) is None
    assert value not in describe_error(exc)


def test_blank_key_counts_as_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert missing_key_hint(AnthropicConnectionError("x")) == "hint: ANTHROPIC_API_KEY is not set"


def test_no_hint_for_auth_errors_unknown_providers_or_plain_errors(no_keys):
    assert missing_key_hint(AnthropicAuthenticationError("401")) is None  # a wrong key is already clear
    assert missing_key_hint(ConnectionError("refused")) is None           # no provider to name
    assert missing_key_hint(ValueError("boom")) is None
    assert missing_key_hint(None) is None


def test_formatted_strings_from_a_run_error(no_keys):
    msg = "AnthropicConnectionError: Connection error."
    assert with_key_hint(msg) == msg + " (hint: ANTHROPIC_API_KEY is not set)"
    assert with_key_hint(with_key_hint(msg)) == with_key_hint(msg)  # appended once
    assert with_key_hint("ValueError: Connection error.") == "ValueError: Connection error."


# ── every core error line carries it ─────────────────────────────────────

NOKEY = textwrap.dedent(
    """
    from langgraph.graph import END, START, MessagesState, StateGraph

    AnthropicConnectionError = type(
        "AnthropicConnectionError", (Exception,), {"__module__": "langchain_anthropic.chat_models"}
    )

    def respond(state):
        raise AnthropicConnectionError("Connection error.")

    g = StateGraph(MessagesState)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    graph = g.compile()
    """
)

EXPECTED = "AnthropicConnectionError: Connection error. (hint: ANTHROPIC_API_KEY is not set)"


def _graph():
    ns: dict = {}
    exec(NOKEY, ns)
    return ns["graph"]


async def test_event_and_chunk_wires(no_keys):
    from langstage_core.agui import iter_chunk_frames, iter_event_frames

    frames = [f async for f in iter_event_frames(_graph(), "hi", "t-253-ev")]
    errors = [f["error"] for f in frames if f.get("type") == "error"]
    assert errors and EXPECTED in errors[-1]

    chunks = [c async for c in iter_chunk_frames(_graph(), "hi", "t-253-ch")]
    errors = [c["error"] for c in chunks if c.get("status") == "error"]
    assert errors and EXPECTED in errors[-1]


async def test_verify_reason(no_keys):
    from langstage_core.agui import averify

    result = await averify(_graph())
    assert not result.ok
    assert "hint: ANTHROPIC_API_KEY is not set" in (result.error_message or result.reason or "")


@pytest.fixture
def nokey_spec(tmp_path):
    p = tmp_path / "nokey_agent.py"
    p.write_text(NOKEY, encoding="utf-8")
    return f"{p}:graph"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    src = str(Path(langstage_core.__file__).resolve().parents[1])
    path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "PYTHONPATH": path, "PYTHONIOENCODING": "utf-8"}
    for var in ("ANTHROPIC_API_KEY", "LANGSTAGE_DEBUG"):
        env.pop(var, None)
    return subprocess.run(
        [sys.executable, "-m", "langstage_core.agui", *args],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
    )


def test_cli_message_prints_the_hint_on_its_one_error_line(nokey_spec):
    r = _run("-a", nokey_spec, "-m", "hi")
    assert r.returncode == 1
    lines = [ln for ln in r.stderr.splitlines() if "AnthropicConnectionError" in ln]
    assert len(lines) == 1 and EXPECTED in lines[0]


def test_cli_json_carries_the_hint(nokey_spec):
    r = _run("-a", nokey_spec, "--json", "-m", "hi")
    assert r.returncode == 1
    doc = json.loads(r.stdout)
    assert EXPECTED in json.dumps(doc)


def test_cli_verify_prints_the_hint(nokey_spec):
    r = _run("-a", nokey_spec, "--verify")
    assert r.returncode == 1
    assert "hint: ANTHROPIC_API_KEY is not set" in r.stdout + r.stderr
