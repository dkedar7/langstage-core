"""Console-safe output for every LangStage CLI (gh #153, #171; langstage #146).

A default Windows console (cp1252), or any stdout a surface is spawned with a
non-UTF-8 ``PYTHONIOENCODING`` / piped encoding, raises ``UnicodeEncodeError`` the
moment a CLI prints a character outside that codec: a CJK ``title``, an
``agent_spec`` under a localized user folder, an emoji in a model reply. The
config diagnostic (``--show-config`` / ``python -m langstage_core.host``) and
``langstage-agui --message`` both crashed that way, and each surface had grown its
own one-off guard (``_print`` / ``_write_safe`` / ``_echo_streamsafe``).

This is the one shared helper. It **escapes** what the stream can't encode
(``backslashreplace``: ``日`` -> ``\\u65e5``) instead of crashing, so a path or
value stays unambiguous in a diagnostic, and it leaves encodable text byte-for-byte
untouched. It never mutates ``sys.stdout`` / ``sys.stderr`` (no ``reconfigure``):
a library must not change the host process's streams (OBJECTIVES.md anti-scope),
and a leaf CLI that *wants* to reconfigure can still do so at its own entrypoint.

    from langstage_core.console import safe_print, safe_write

    safe_print(cfg.describe())             # instead of print(...)
    safe_write(chunk["chunk"])             # instead of sys.stdout.write(...)
    safe_print("oops", file=sys.stderr)    # any text stream

``click.echo`` users: ``safe_print(text)`` is a drop-in for ``click.echo(text)`` on
stdout (click's own echo raises on an unencodable char just like ``print``).
"""
from __future__ import annotations

import sys
from typing import IO, Any

__all__ = ["safe_print", "safe_write", "console_safe"]


def console_safe(text: str, stream: IO[str] | None = None) -> str:
    """Return ``text`` made encodable for ``stream`` (default ``sys.stdout``).

    Characters the stream's encoding can't represent become ``backslashreplace``
    escapes; everything else is unchanged. A stream with no ``encoding`` attribute
    (``io.StringIO``, a test capture) or an unknown codec gets ``text`` back as-is.
    """
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, "backslashreplace").decode(encoding)
    except LookupError:  # an exotic/unknown codec name — nothing sensible to do
        return text


def safe_write(text: str, file: IO[str] | None = None, *, flush: bool = False) -> None:
    """``file.write(text)`` that escapes unencodable characters instead of raising.

    The streaming counterpart of :func:`safe_print` (no separator, no newline) — for
    writing a model reply chunk by chunk. The text is made safe *before* the write, so
    a failure can't leave a half-written line behind.
    """
    stream = sys.stdout if file is None else file
    if stream is None:  # pythonw / a detached process has no stdout
        return
    stream.write(console_safe(text, stream))
    if flush:
        stream.flush()


def safe_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: IO[str] | None = None,
    flush: bool = False,
) -> None:
    """``print()`` that escapes unencodable characters instead of raising.

    Same signature as the builtin. Encodable output is identical to ``print``;
    only characters the target stream can't encode are ``backslashreplace``-escaped.
    """
    safe_write(sep.join(str(v) for v in values) + end, file, flush=flush)
