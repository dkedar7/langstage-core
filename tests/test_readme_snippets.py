"""README snippets touched in the 1.0.37 sweep run as written.

- "Delegate work to a background task" must terminate on a HITL agent (gh #166).
- "Connect a real model" builds with no deprecation warning (gh #151). The live turn needs
  a key, so this runs the snippet up to ``asyncio.run(main())``.
- The HITL decision-verb helper example returns what its comments say (vscode #114/#117).
"""
import asyncio
import contextlib
import io
import re
import warnings
from pathlib import Path

import pytest

_README = Path(__file__).resolve().parent.parent / "README.md"


def _python_blocks_under(heading: str) -> list[str]:
    text = _README.read_text(encoding="utf-8")
    start = text.index(heading)
    nxt = re.search(r"^#{2,4} ", text[start + len(heading):], flags=re.M)
    section = text[start: start + len(heading) + (nxt.start() if nxt else len(text))]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def test_delegate_snippet_terminates_on_a_hitl_agent():
    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    [code] = _python_blocks_under("### Delegate work to a background task")
    # The snippet ends with asyncio.run(main()); run main() under a timeout instead so a
    # regression to the #166 infinite loop fails the test rather than hanging it.
    body = code.replace("asyncio.run(main())", "")
    ns: dict = {}
    exec(compile(body, "README:delegate", "exec"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        asyncio.run(asyncio.wait_for(ns["main"](), timeout=30))
    lines = out.getvalue().splitlines()
    assert lines[0] == "['respond', 'approve']"
    assert lines[1] == "done"
    assert "approve" in lines[2]


def test_real_model_snippet_builds_without_deprecation(monkeypatch):
    pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
    pytest.importorskip("langchain_openai", reason="needs the 'real' extra")
    pytest.importorskip("langchain.agents", reason="needs langchain>=1")
    [code] = _python_blocks_under("### Connect a real model")
    assert "create_react_agent" not in code
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    body = code.replace("asyncio.run(main())", "")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from langgraph.warnings import LangGraphDeprecatedSinceV10

        warnings.simplefilter("error", LangGraphDeprecatedSinceV10)
        ns: dict = {}
        exec(compile(body, "README:real-model", "exec"), ns)
    assert hasattr(ns["agent"], "run")


def test_decision_helper_snippet():
    blocks = [b for b in _python_blocks_under("### Human-in-the-loop (interrupt → resume)")
              if "normalize_decision" in b]
    [code] = blocks
    ns: dict = {}
    exec(compile(code, "README:decisions", "exec"), ns)
    for line in code.splitlines():
        m = re.match(r"^(normalize_decision|is_allowed_decision)\((.*)\)\s+#\s+(\S+)", line)
        if m:
            got = eval(f"{m.group(1)}({m.group(2)})", ns)
            assert repr(got) == m.group(3), line


def test_install_section_does_not_promise_tasks_on_a_bare_install():
    # gh #138: SessionAdapter (the only task driver) needs the [agui] extra.
    text = _README.read_text(encoding="utf-8")
    assert "host/config/tasks layer" not in text
    assert "The task engine needs `[agui]` too" in text
