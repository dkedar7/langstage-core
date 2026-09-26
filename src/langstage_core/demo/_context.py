"""Separate what the user typed from the context surfaces add to it (gh #192).

Surfaces attach host context to the human message before the agent sees it:

- the web app (and anything using ``prepare_agent_input(context_parts=...)``)
  prepends one ``[...]`` line per part, then a blank line, then the message —
  ``[Current time: ...]``, ``[Working directory: ...]``, ``[File browser folder: ...]``;
- JupyterLab appends a blank line and a block starting ``Current directory: ...``,
  ``Currently focused...`` or ``User has selected the following text...``.

A real model should see all of it. The keyless demo agents echo the message back
(and route on trigger words in it), so they use :func:`user_text` to see only what
was typed. Stdlib only, so importing it keeps the demo modules dependency-free.
"""
from __future__ import annotations

import re

# One whole line that is a single bracketed context part.
_CONTEXT_LINE = re.compile(r"\[[^\n\]]*\]")
# JupyterLab's trailing context block (langstage_jupyter.agent_wrapper), which always
# follows the message after a blank line and opens with one of these.
_TRAILING_CONTEXT = re.compile(
    r"\n\n(?:Current directory: |Currently focused|User has selected the following text)"
)


def user_text(content: str) -> str:
    """The user's own text from a human message that may carry surface context.

    Drops leading lines that are entirely ``[...]`` (plus the blank line after
    them) and a trailing JupyterLab context block. Anything else, including
    brackets inside the typed text, is kept as is.
    """
    lines = content.split("\n")
    i = 0
    while i < len(lines) and _CONTEXT_LINE.fullmatch(lines[i].strip()):
        i += 1
    if i:
        while i < len(lines) and not lines[i].strip():
            i += 1
        content = "\n".join(lines[i:])
    match = _TRAILING_CONTEXT.search(content)
    if match:
        content = content[: match.start()]
    return content
