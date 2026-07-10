"""The resume snippets in the docs use the HITL *decision envelope*, not a bare list (gh #85).

The Jupyter example notebook and the README both show how to answer an `interrupt`
frame by calling `iter_chunk_frames`/`iter_event_frames` again with `resume=`. The real
`HumanInTheLoopMiddleware` reads the resumed value back as ``interrupt(...)["decisions"]``,
so the resume value must be a **dict** with a ``"decisions"`` list — exactly what
``create_resume_input(decisions=[...])`` builds. An earlier draft of the notebook passed a
bare list (``resume=[{"type": "approve"}]``); copy-pasting it crashed HITL with a
``TypeError``/``KeyError`` because a list has no ``["decisions"]``. These tests pin the
documented shape so it can't silently drift back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langstage_core import create_resume_input

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_README = Path(__file__).resolve().parent.parent / "README.md"
_NOTEBOOK = _EXAMPLES / "jupyter_example.ipynb"

# Any `resume=` assignment in a doc snippet. We only care that when the value opens with a
# literal, it opens with a `{` (the envelope) — never `[` (the bare-list bug).
_RESUME_LITERAL = re.compile(r"resume=\s*([\[{])")


def _doc_sources() -> list[tuple[str, str]]:
    """(label, text) for every doc that carries a resume snippet."""
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    nb_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    return [("jupyter_example.ipynb", nb_text), ("README.md", _README.read_text(encoding="utf-8"))]


def test_docs_never_show_a_bare_list_resume():
    for label, text in _doc_sources():
        for m in _RESUME_LITERAL.finditer(text):
            opener = m.group(1)
            assert opener == "{", (
                f"{label}: `resume=` opens with a `{opener}` — a bare list crashes the HITL "
                f"middleware (it reads `interrupt(...)['decisions']`). Use the decision "
                f'envelope: resume={{"decisions": [...]}} (gh #85).'
            )


def test_docs_show_the_decision_envelope():
    # Both docs should actually demonstrate the envelope, not just avoid the bare list.
    for label, text in _doc_sources():
        compact = text.replace(" ", "")
        assert 'resume={"decisions"' in compact, f'{label}: expected a `resume={{"decisions": ...}}` snippet'


def test_documented_envelope_matches_create_resume_input():
    # The literal the docs show is exactly what the blessed helper builds, so the two can't
    # drift: if the envelope key ever changes, this fails alongside the doc guard above.
    cmd = create_resume_input(decisions=[{"type": "approve"}])
    assert cmd.resume == {"decisions": [{"type": "approve"}]}
