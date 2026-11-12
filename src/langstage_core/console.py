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

import contextlib
import sys
from collections.abc import Iterator
from typing import IO, Any

__all__ = [
    "safe_print",
    "safe_write",
    "console_safe",
    "agent_stdout_to_stderr",
    "is_closed_pipe",
    "quiet_closed_stdout",
]


def is_closed_pipe(exc: BaseException) -> bool:
    """True when ``exc`` means the reader of a pipe went away (gh #238).

    POSIX raises ``BrokenPipeError``; Windows raises ``OSError`` with ``EINVAL`` for a
    write to a pipe whose reader closed (the langstage-vscode #163 finding).
    """
    import errno

    if isinstance(exc, BrokenPipeError):
        return True
    return sys.platform == "win32" and isinstance(exc, OSError) and exc.errno == errno.EINVAL


def quiet_closed_stdout(stream: IO[str] | None = None) -> None:
    """Point ``stream``'s file descriptor (default stdout) at the null device (gh #238).

    Call it after catching a closed-pipe error at a CLI entry point, so the
    interpreter's shutdown flush doesn't raise a second ``BrokenPipeError`` (and print
    ``Exception ignored ... BrokenPipeError``). A stream without a real descriptor
    (test capture) is left alone.
    """
    import os

    stream = sys.stdout if stream is None else stream
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stream.fileno())
        os.close(devnull)
    except (OSError, ValueError, AttributeError):
        pass


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
    try:
        stream.write(console_safe(text, stream))
        if flush:
            stream.flush()
    except OSError as exc:
        # The reader closed the pipe (``| head``). Report it as BrokenPipeError on every
        # OS (Windows says EINVAL), so a CLI entry point can end quietly (gh #238).
        if is_closed_pipe(exc) and not isinstance(exc, BrokenPipeError):
            raise BrokenPipeError(32, "the reader closed the output pipe") from exc
        raise


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


class _SafeStderr:
    """A text stream that forwards writes to ``target`` (stderr), escaping what it
    can't encode (gh #235). Anything else (``fileno``, ``isatty``, ``encoding``,
    ``buffer``...) is delegated, so a library that pokes at ``sys.stdout`` still works.
    """

    def __init__(self, target: IO[str]) -> None:
        self._target = target

    def write(self, text: str) -> int:
        self._target.write(console_safe(text, self._target))
        return len(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


@contextlib.contextmanager
def agent_stdout_to_stderr() -> Iterator[IO[str]]:
    """Run an agent turn with its prints sent to stderr; yield the real stdout.

    A machine-readable one-shot path (``langstage-agui -m`` / ``--json`` /
    ``--verify``) owns stdout: it carries only the reply, the JSON document or the
    verdict. ``load_agent_spec(stdout_to_stderr=True)`` already keeps import-time
    prints off it; this covers the turn itself, where a node, a tool, a chatty SDK or
    a ``verbose=True`` chain prints (gh #235). Write the result to the yielded stream::

        with agent_stdout_to_stderr() as out:
            result = run_the_turn()
            safe_print(json.dumps(result), file=out)

    The diverted text is escaped like :func:`safe_write`, so an agent printing a
    character stderr can't encode no longer kills the turn on cp1252.
    ``sys.stdout`` is restored on exit; nothing is reconfigured.
    """
    real = sys.stdout
    with contextlib.redirect_stdout(_SafeStderr(sys.stderr)):  # type: ignore[arg-type]
        yield real
