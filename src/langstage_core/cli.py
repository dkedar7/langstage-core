"""The LangStage family's command-line exit codes (ADR 0007).

Every console script in the family (``langstage-agui``, ``langstage``, the cli, hermes,
jupyter and vscode sidecar entry points) exits with the same four codes, so a CI gate or
wrapper script reads them the same way on every surface:

====  ================  ===========================================================
code  name              meaning
====  ================  ===========================================================
0     ``EXIT_OK``       success
1     ``EXIT_FAIL``     failure: not configured / no agent spec, load or import
                        error, turn error, a verify / selfcheck / doctor check
                        failed, the server can't start (including a busy port)
2     ``EXIT_PAUSED``   the turn paused on a human-in-the-loop interrupt (the run
                        is fine but needs input)
64    ``EXIT_USAGE``    usage error: bad or conflicting command-line arguments
====  ================  ===========================================================

argparse exits 2 on a usage error, which collides with "paused": a typo'd flag would
read as a benign HITL pause (the gh #174 fail-open class). :class:`ArgumentParser` is a
drop-in ``argparse.ArgumentParser`` whose usage errors exit 64 (``EX_USAGE`` from BSD
``sysexits.h``). Sub-parsers made with ``add_subparsers()`` inherit it.

Stdlib-only, so importing it costs nothing.
"""
from __future__ import annotations

import argparse
import sys
from typing import NoReturn

__all__ = [
    "EXIT_OK",
    "EXIT_FAIL",
    "EXIT_PAUSED",
    "EXIT_USAGE",
    "ArgumentParser",
    "exit_code_for_outcome",
    "usage_error",
]

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PAUSED = 2
EXIT_USAGE = 64

_OUTCOME_CODES = {"complete": EXIT_OK, "error": EXIT_FAIL, "interrupted": EXIT_PAUSED}


def exit_code_for_outcome(outcome: str | None) -> int:
    """Map a turn outcome to its exit code: ``complete`` -> 0, ``interrupted`` -> 2,
    ``error`` or anything unrecognized -> 1 (fail closed)."""
    return _OUTCOME_CODES.get(outcome, EXIT_FAIL) if isinstance(outcome, str) else EXIT_FAIL


class ArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` whose usage errors exit 64 instead of 2.

    ``--help`` / ``--version`` actions still exit 0; the message format is unchanged
    (``usage: ...`` then ``prog: error: ...`` on stderr).
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def usage_error(parser: argparse.ArgumentParser, message: str) -> int:
    """Report a usage error the parser itself can't detect (e.g. two flags that conflict
    only after config resolution) in argparse's format, and *return* 64 so ``main()``
    can ``return usage_error(parser, "...")`` without raising ``SystemExit``."""
    parser.print_usage(sys.stderr)
    sys.stderr.write(f"{parser.prog}: error: {message}\n")
    return EXIT_USAGE
