"""gh #255: on a bare install (no langgraph) the resume helpers say what to install
instead of raising a raw ``ModuleNotFoundError: No module named 'langgraph'``.
The decision helpers never import langgraph and keep working.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import langstage_core
from langstage_core import create_resume_input, normalize_decision, prepare_agent_input


@pytest.fixture
def no_langgraph(monkeypatch):
    # A None entry in sys.modules makes `import langgraph.types` raise ImportError,
    # which is what a bare install does.
    monkeypatch.setitem(sys.modules, "langgraph.types", None)


@pytest.mark.parametrize(
    "call",
    [
        lambda: create_resume_input(decisions=[{"type": "approve"}]),
        lambda: create_resume_input(value=True),
        lambda: prepare_agent_input(decisions=[{"type": "approve"}]),
    ],
    ids=["decisions", "value", "prepare_agent_input"],
)
def test_resume_helpers_name_the_install(no_langgraph, call):
    with pytest.raises(ModuleNotFoundError) as info:
        call()
    msg = str(info.value)
    assert 'pip install "langstage-core[agui]"' in msg
    assert "langgraph" in msg
    assert info.value.name == "langgraph"


def test_argument_errors_still_come_first(no_langgraph):
    with pytest.raises(ValueError):
        create_resume_input()


def test_message_input_and_decision_helpers_need_no_langgraph(no_langgraph):
    assert prepare_agent_input(message="hi") == {"messages": [{"role": "user", "content": "hi"}]}
    assert normalize_decision("accept") == "approve"


def test_bare_install_in_a_fresh_interpreter():
    """The real import path, with langgraph blocked the way a bare install lacks it."""
    code = textwrap.dedent(
        """
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "langgraph" or name.startswith("langgraph."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        sys.meta_path.insert(0, _Block())
        from langstage_core import create_resume_input
        try:
            create_resume_input(decisions=[{"type": "approve"}])
        except ModuleNotFoundError as e:
            print("MSG:", e)
        """
    )
    src = str(Path(langstage_core.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert 'MSG: the resume helpers' in r.stdout
    assert 'pip install "langstage-core[agui]"' in r.stdout
