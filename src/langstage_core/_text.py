"""Join a turn's streamed assistant text the way the frame contract says (gh #200)."""

from __future__ import annotations

from typing import Any


class TextJoiner:
    """Accumulate assistant text deltas into one reply string.

    A change of ``message_id`` between two text deltas is a message boundary (two
    AIMessages of one turn, e.g. a router's note and the final answer), and the
    README's frame contract says to join it with a paragraph break, not inline.
    Frames without a ``message_id`` fall back to a change of ``node``. Deltas of one
    message share an id and join unbroken. This is the rule the CLI and the VS Code
    panel already apply (langstage-cli#43/#119, langstage-vscode#108), so core's
    one-shot results (``TurnResult.text``, ``task["result"]``) match them (gh #200).
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._has_text = False
        self._node: Any = None
        self._message_id: Any = None

    def add(self, text: str, *, node: Any = None, message_id: Any = None) -> str:
        """Append one delta; return what was appended (any break, then ``text``),
        so a caller streaming to a console can write exactly that."""
        if not text:
            return ""
        new_message = self._has_text and (
            node != self._node or (message_id is not None and message_id != self._message_id)
        )
        brk = ""
        if new_message:
            tail = self._parts[-1]
            if not tail.endswith("\n\n"):
                brk = "\n" if tail.endswith("\n") else "\n\n"
        self._parts.append(brk + text)
        self._node = node
        self._message_id = message_id
        self._has_text = self._has_text or bool(text.strip())
        return brk + text

    def text(self) -> str:
        return "".join(self._parts)
