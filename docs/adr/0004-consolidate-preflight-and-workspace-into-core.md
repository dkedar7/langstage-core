# ADR 0004 — Consolidate live preflight and workspace application into the core

**Status:** Accepted — 2026-07-03
**Decision owner:** Kedar Dabhadkar

## Context

Five days of dogfooding across the six LangStage packages closed ~36 issues.
Sorting them by root cause rather than by repo, ~40% are one meta-bug: **a
surface reports a config / health / version state it never actually applied or
verified** — a clean `doctor` for a missing extra (hermes #41), `[ok] loads` for
a non-graph (web #39), a green `--selfcheck` for a path that errors
(vscode #28, historically), a labextension version that doesn't match the wheel
(jupyter #38 *then* #48), and a recurring family of "workspace root silently
dropped" (vscode #19, jupyter #45/#36, web #44, the hermes factory path).

A read-only audit of all six packages found *why* these recur. Core already ships
one genuinely shared abstraction — `HostConfig.resolve()` (`defaults < TOML < env
< overrides`), which every surface subclasses. **Two things that should also be
shared are not**, and every recurring bug lives in exactly those two seams:

| Surface | Config resolution | Workspace **application** | Live preflight (runs a real turn?) |
|---|---|---|---|
| **core** | **owns** `HostConfig.resolve()` | resolves the *value*; never applies it (`Workspace` helper unused) | **none** — parts exist (`build_agent` + `iter_event_frames`), uncomposed |
| cli | delegates (`CodeConfig`) | `os.chdir` only — never reaches the agent's tools | ✗ static `--show-config` |
| vscode | delegates fully | push value → `os.environ` | ✓ `--selfcheck` drives a real turn |
| jupyter | delegates (`LabConfig`) | mutate module-global + rebuild backend | ✗ health = `agent is not None` |
| web | delegates (`AppConfig`) | dataclass (browser) **+** module-global (tools), hand-synced | ✗ `check` asserts `callable(astream)`, never invokes |
| hermes | **forks** `resolve()` (2nd TOML) | factory path drops it; only `verify` forwards | ✓ `verify`; `doctor` static → #41 |

Read the columns, not the rows. The **config** column is uniform — and the lone
fork (hermes) is the only surface that had a config-resolution bug (#24). The
**workspace-application** and **preflight** columns are a zoo: five surfaces apply
workspace root five different ways, and the "run a real turn" preflight was
independently reinvented twice (vscode, hermes) and is static or absent the other
four times. This is a natural experiment: the abstraction we consolidated doesn't
drift; the two we copied per-surface generate ~half the backlog.

## Decision

Finish the consolidation. Two additive core capabilities, adopted by surfaces
incrementally (same "additive now, migrate surface-by-surface" discipline as
ADR 0001).

1. **Shared live preflight — `langstage_core.agui.verify` / `averify`.**
   Build the agent, stream **one real turn** through `iter_event_frames`, return a
   structured `VerifyResult` (`ok` iff a `complete` frame arrived with no `error`
   frame and no raised exception). vscode `--selfcheck` and hermes `verify` are the
   reference behaviour; this is the primitive they were each hand-rolling. Every
   surface's `doctor` / `check` / `selfcheck` / `health` delegates to it, so
   "healthy" means the same thing everywhere: *the agent actually completed a
   turn.* This kills the false-green class (#41, #39, #28). **Shipped in 1.0.6.**

2. **Workspace *application* in core — not just resolution.** Core resolves
   `workspace_root` today but stops there; each surface invents how to make it the
   agent's actual root (`chdir` / env-push / backend-arg / global-mutate). Core
   should own a single "apply the resolved workspace" path — root the agent's
   `FilesystemBackend`/tools *and* the surface from one source (the unused
   `host/workspace.py` `Workspace` is the seed) — so file-browser and agent tools
   can never disagree (#44) and a pinned root can't be silently re-rooted (#45).
   *Staged — higher blast radius; each surface migrates in its own PR.*

Two supporting guardrails, tracked alongside:

3. **Version-parity CI gate (jupyter):** assert the built labextension version
   equals the wheel version, so the `skip-if-exists` drift (#38, #48) can't recur
   silently. A pre-merge clean-room first-run smoke (install the wheel with no
   extras, run one turn via the new `verify`) generalises this across the family.

4. **Deprecation sunset:** every package carries `deepagent-*` scripts + alias
   trees, `DEEPAGENT_*` env, legacy `.toml` fallbacks, and no-op `[agui]`/`[demo]`
   extras — all "kept for one transition window," **none with a removal version.**
   Pick a family-wide target (proposed: the next core minor, **1.1.0**, with
   surfaces dropping their aliases in the release that pins `langstage-core>=1.1`)
   and collect the removals into one deliberate pass.

## Consequences

- **New core API:** `langstage_core.agui.verify(agent_or_graph) -> VerifyResult`,
  its `averify` coroutine, and the `VerifyResult` dataclass — behind the existing
  `[agui]` extra (it drives the AG-UI adapter). No change to any surface until it
  opts in; adoption is one thin PR per surface (replace the bespoke check body with
  a `verify()` call plus any surface-specific extras, e.g. hermes' side-effect
  assertions).
- **"Healthy" becomes a single definition** across the family, removing the
  false-green failure mode by construction rather than by per-surface patching.
- **Workspace application (2)** is the higher-value, higher-risk half; it is
  deliberately staged behind (1) so the low-risk win lands first and the tools-vs-
  browser rewiring gets its own reviewed PR per surface.
- **The dogfood matures** from "here are N bugs" to also flagging consolidation
  candidates: any fix that had to ship to ≥2 repos, any health check that asserts
  static state instead of running a turn, any recurrence, and any deprecation with
  no removal date. The synthesis that produced this ADR becomes the closing stage's
  standing checklist.
