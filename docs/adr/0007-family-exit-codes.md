# ADR 0007 — One exit-code scheme for every LangStage command line

**Status:** Accepted — 2026-09-25
**Decision owner:** Kedar Dabhadkar
**Relates to:** gh #174 (fail-open no-spec exit), #124, #134, #143; every surface's CLI

## Context

The family ships several console scripts: `langstage-agui` and `python -m
langstage_core.host` here, plus `langstage` (web), the cli, hermes, jupyter and vscode
sidecar entry points. Each grew its own exit codes one issue at a time:

| Situation | What surfaces did before this ADR |
|---|---|
| turn paused on a HITL interrupt | `2` (agui `--message`, vscode `--selfcheck`) |
| no agent spec / can't load | `1` under agui `--verify`/`--message`, but `2` on the agui serve path |
| server can't start (busy port) | `2` (agui, "the serve path's can't-start code") |
| a verify command failed | `1` on most surfaces, `2` on hermes `verify` |
| bad or conflicting arguments | argparse's default `2`; agui's `--demo` + `--agent` was `1` or `2` depending on the command |

`2` meant three different things: *paused*, *can't start*, and *usage error*. #174 is
what that costs. With no agent configured, `langstage-agui --verify` / `--message` exited
`2`. A CI gate written against the README's "`--message` exits 2 on interrupt" read an
unconfigured agent as a benign HITL pause and passed it. The same collision still exists
wherever argparse's default is in play: a typo'd flag in a CI script exits `2`, which
reads as "paused, fine". Each fix so far patched one path ("map to 1 under `--verify` and
`--message`, keep 2 on serve"). That kept the codes command-dependent and left the next
surface free to drift.

## Decision

Every console script and subcommand in the family exits with these codes:

| Code | Name | Meaning |
|---|---|---|
| `0` | `EXIT_OK` | success |
| `1` | `EXIT_FAIL` | failure: not configured / no agent spec, load or import error, turn error, a verify / selfcheck / doctor / check failed, can't start (including a busy port) |
| `2` | `EXIT_PAUSED` | the run paused on a human-in-the-loop interrupt: it is fine but needs input |
| `64` | `EXIT_USAGE` | usage error: bad or conflicting command-line arguments |

1. **`2` means only "paused".** Nothing else may exit 2. A tool that treats 2 as "OK, needs
   a human" can't then mistake a broken setup for a pause.
2. **Every can't-run or can't-start condition is `1`, whatever the command.** No more
   "1 under `--verify`, 2 when serving".
3. **Usage errors are `64`** (`EX_USAGE` from BSD `sysexits.h`). argparse's default of 2
   collides with paused, so every console script overrides it. Conflicts that are only
   detected after parsing (for example `--demo` with `--agent`) are usage errors too.
4. **Core ships the vocabulary.** `langstage_core.cli` provides `EXIT_OK` / `EXIT_FAIL` /
   `EXIT_PAUSED` / `EXIT_USAGE`, `exit_code_for_outcome()` (complete → 0, interrupted → 2,
   error or anything unrecognized → 1, so it fails closed), an `ArgumentParser` subclass
   whose `error()` exits 64 (sub-parsers inherit it), and `usage_error(parser, msg)` for
   post-parse conflicts. It is stdlib-only. Surfaces may use it or hard-code the same
   numbers. The numbers are the contract, not the helper.

## Consequences

- **Changed in core 1.0.38:** `langstage-agui` exits `1` (was `2`) on the serve path for
  no spec, an unloadable spec, a missing `[agui]` extra and a busy port. `--demo` with
  `--agent` is `64` on every command (was `1` under `--verify`/`--message`, `2` when
  serving). Any argparse error (`--demo=bogus`, `--port abc`, an unknown flag) is `64`
  (was `2`) for both `langstage-agui` and `python -m langstage_core.host`.
- **Breaking for scripts that matched the old codes.** A wrapper that checked for `2` to
  detect "can't serve" or "bad flag" must now check `1` or `64`. That is the point: those
  scripts were also treating a HITL pause as the same failure.
- **Surfaces follow in their own releases.** Each surface audits its entry points
  (`verify`, `doctor`, `check`, `chat`, `--selfcheck`, `--ask`, start failures, usage
  errors), documents the table in its README with a link here, and pins the codes in
  tests.
- **Shell conventions still apply outside our control.** A process killed by a signal, or
  an uncaught exception that escapes `main()` (Python exits `1`), is outside this table.
  An uncaught exception still lands in "failure", which is the safe side.
