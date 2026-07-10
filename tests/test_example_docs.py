"""Every shipped doc/example resumes an interrupt with the HITL *decision envelope*,
not a bare list (gh #85, gh #87).

The README, the Jupyter notebook, and the FastAPI WebSocket example all show how to answer
an `interrupt` frame by calling `iter_chunk_frames`/`iter_event_frames` again with `resume=`.
The real `HumanInTheLoopMiddleware` reads the resumed value back as
``interrupt(...)["decisions"]``, so the resume value must be a **dict** with a ``"decisions"``
list — exactly what ``create_resume_input(decisions=[...])`` builds. A bare list
(``resume=[{"type": "approve"}]``) is forwarded verbatim into ``Command(resume=[...])`` and
crashes a real HITL agent with ``TypeError: list indices must be integers or slices, not
str``.

#85 fixed the notebook; #87 caught the *same* bug still present in the FastAPI example — the
two had drifted. So this guard now scans **every** shipped doc AND every ``examples/*.py``
file, not just the two it originally covered, so no shipped surface can regress to the bare
list again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langstage_core import create_resume_input

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _ROOT / "examples"
_README = _ROOT / "README.md"
_NOTEBOOK = _EXAMPLES / "jupyter_example.ipynb"

# Any `resume=` assignment in a doc/example. We only care that when the value opens with a
# literal, it opens with a `{` (the envelope) — never `[` (the bare-list bug).
_RESUME_LITERAL = re.compile(r"resume=\s*([\[{])")


def _doc_sources() -> list[tuple[str, str]]:
    """(label, text) for every shipped doc/example that could carry a resume snippet:
    the README, the Jupyter notebook, and every ``examples/*.py`` file."""
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    nb_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    sources = [
        ("jupyter_example.ipynb", nb_text),
        ("README.md", _README.read_text(encoding="utf-8")),
    ]
    for py in sorted(_EXAMPLES.glob("*.py")):
        sources.append((f"examples/{py.name}", py.read_text(encoding="utf-8")))
    return sources


def test_docs_never_show_a_bare_list_resume():
    # Covers the README, the notebook, AND every examples/*.py — the #87 drift was a .py
    # example the original (docs-only) guard didn't scan.
    for label, text in _doc_sources():
        for m in _RESUME_LITERAL.finditer(text):
            opener = m.group(1)
            assert opener == "{", (
                f"{label}: `resume=` opens with a `{opener}` — a bare list crashes the HITL "
                f"middleware (it reads `interrupt(...)['decisions']`). Use the decision "
                f'envelope: resume={{"decisions": [...]}} (gh #85 / #87).'
            )


def test_resume_snippets_use_the_decision_envelope():
    # Any source that actually uses a `resume={...}` literal must use the `"decisions"`
    # envelope (not some other dict). Sources with no resume literal (e.g. agent.py) are
    # skipped rather than forced to mention it.
    seen_any = False
    for label, text in _doc_sources():
        compact = text.replace(" ", "")
        if "resume={" not in compact:
            continue
        seen_any = True
        assert 'resume={"decisions"' in compact, (
            f'{label}: a `resume={{...}}` literal must use the decision envelope '
            f'`resume={{"decisions": [...]}}` (gh #85 / #87).'
        )
    assert seen_any, "expected at least one shipped doc/example to demonstrate resume="


def test_documented_envelope_matches_create_resume_input():
    # The literal the docs show is exactly what the blessed helper builds, so the two can't
    # drift: if the envelope key ever changes, this fails alongside the doc guard above.
    cmd = create_resume_input(decisions=[{"type": "approve"}])
    assert cmd.resume == {"decisions": [{"type": "approve"}]}
