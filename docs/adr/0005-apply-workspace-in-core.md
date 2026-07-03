# ADR 0005 — Apply the resolved workspace in core (one source of truth)

**Status:** Accepted — 2026-07-03
**Decision owner:** Kedar Dabhadkar
**Relates to:** ADR 0004 (this is its staged second half), backlog item 37

## Decisions (resolved)

1. **BYO boundary endorsed.** Core roots agents it builds; a bring-your-own graph's
   tools opt in by reading `core.workspace_root()`. `check` warns when a BYO agent's
   tools are unlikely to honor the workspace. (Open question 1.)
2. **`chdir` kept as opt-in** (`chdir=True`) for the single-process surfaces (cli, the
   vscode sidecar) only; servers (web, jupyter) never chdir. (Open question 2.)
3. **Shipped as core 1.0.7** — additive, immediate, decoupled from the 1.1.0
   deprecation sunset. (Open question 3.)

## Context

ADR 0004 established: config *resolution* is consolidated in core
(`HostConfig.resolve()`) and does not drift, while the two things that were *not*
consolidated — live preflight and workspace **application** — generate ~half the
dogfood backlog. Preflight is now shared (`core.verify`, adopted by cli/web/jupyter).
This ADR proposes the harder half: workspace application.

**The distinction.** Core resolves `workspace_root` (the *value*) through the shared
precedence chain, so every surface gets the value identically. But core **stops
there** — no `chdir`, no wiring of the value into the agent's filesystem tools. The
`host/workspace.py` `Workspace` helper (ensure + safe-join) exists for this and is
used by **nobody**. So each surface invented its own "apply":

| Surface | Mechanism to make the resolved root the agent's root | Bug it produced |
|---|---|---|
| cli | `os.chdir()` — process cwd only; never reaches the agent's tools | — |
| vscode | write `os.environ["LANGSTAGE_WORKSPACE_ROOT"]` | #19 (`setdefault` dropped `--workspace`) |
| hermes | `workspace=` → `FilesystemBackend(root_dir=...)`; the `chat` factory path **drops** it | factory-path gap (only `verify` forwarded it) |
| jupyter | mutate module-global `config.WORKSPACE_ROOT` + rebuild backend | #45 (pinned root discarded), #36 (dead re-root / bad import) |
| web | file-browser reads the dataclass value; agent tools read a mutable module global; hand-synced in `CoworkApp.__init__` | #44 (split-brain — tools ignored `--workspace`) |

Five surfaces, five mechanisms, three shipped bugs. One missing abstraction, copied
five ways and drifting.

### The crux (why this is not a dive-in)

There is **no single mechanism that uniformly re-roots an arbitrary bring-your-own
graph's tools**, because core does not own those tools. A BYO compiled graph
(`-a my_agent.py:graph`, supported by cli/vscode/jupyter/web) is already built with
whatever tools the user wired; core cannot reach inside it and change where its
tools resolve paths. Any honest design must split "agents core/the surface builds"
(rootable) from "BYO graphs" (not rootable by core — opt-in only). The five current
mechanisms all blur this, which is part of why they drift.

Candidate mechanisms, and why none is a silver bullet:

- **`chdir`** — universal for cwd-relative tools, but process-global (not per-request);
  a blunt instrument a multi-workspace server couldn't use.
- **Env var** — crosses process boundaries (vscode subprocess), but only tools that
  *read* it honor it (the bundled deepagents tools do; BYO tools don't), and it's
  process-global. It's the "lowest-priority hack" #44 called out.
- **`FilesystemBackend(root_dir=...)` at build time** — explicit and thread-safe, but
  only works for agents *we* build; can't touch a pre-built BYO graph.
- **`RunnableConfig` `configurable.workspace_root`** — LangGraph-native, per-invocation,
  but again only tools wired to read it honor it.

## Decision (proposed)

Make core own a **single active-workspace source of truth**, applied once at startup
by each surface, and be explicit about the BYO boundary. Concretely:

1. **`apply_workspace(root, *, chdir=False)`** — the one function every surface calls
   after resolving config. It ensures the dir exists, publishes it as the single
   source of truth (module state + the `LANGSTAGE_WORKSPACE_ROOT` env var, plus the
   legacy `DEEPAGENT_WORKSPACE_ROOT` until the ADR-0004 sunset), optionally `chdir`s,
   and returns the `Workspace`. This *replaces* the five bespoke mechanisms.
2. **`workspace_root()`** — the one accessor tools and surfaces read, instead of each
   maintaining a private global (web's `config.WORKSPACE_ROOT`, jupyter's, …).
3. **For agents core/the surface builds** (the bundled deepagents agent — web default,
   hermes, jupyter default), root the `FilesystemBackend` from `workspace_root()` at
   build time (mechanism C). This is where most of the actual bugs live (web #44,
   the hermes factory drop), and it's fully in core's control.
4. **For BYO graphs**, core cannot re-root the user's tools. It publishes the value
   through `workspace_root()` + the env var and **documents the contract**: a BYO tool
   that should honor the workspace reads `core.workspace_root()`. `check`/`--verify`
   can warn when a BYO agent's tools are unlikely to honor it. This limitation is
   named, not papered over.
5. **`chdir` is opt-in**, taken only by single-process, single-agent surfaces (cli,
   the vscode sidecar). Servers (web, jupyter) do **not** chdir; they root the built
   agent's backend and serve the file-browser from the same `workspace_root()`.

### Explicit assumption (the one thing that could bite later)

`apply_workspace` sets **process-global** state. That is correct today because every
surface runs **one workspace per process** — including the servers: web/jupyter have
a single workspace root for the whole app, and the task board runs copies of the
*same* agent in the *same* workspace. If a future surface needs **per-request/per-agent
workspaces**, that is a `RunnableConfig`-based extension (mechanism D), explicitly out
of scope here. This assumption is stated so a future maintainer sees the boundary.

## Proposed API (builds on the existing `Workspace`)

```python
# langstage_core/host/workspace.py  (extends today's Workspace: root/ensure/subpath/name)

_ACTIVE: Workspace | None = None

def apply_workspace(root, *, chdir=False):
    """Make `root` the active resolved workspace: ensure it exists, publish it as
    the single source of truth (module state + LANGSTAGE_WORKSPACE_ROOT env, plus
    legacy DEEPAGENT_WORKSPACE_ROOT until sunset), optionally chdir. Returns the
    Workspace. Call ONCE, after HostConfig.resolve(), before building the agent."""

def workspace_root() -> Path:
    """The active resolved workspace root — the ONE accessor tools/surfaces read
    instead of a private global. Falls back to cwd if apply_workspace wasn't called."""
```

Exported top-level (`langstage_core`) alongside `Workspace`.

## Per-surface migration (one PR each, staged; web last)

- **cli**: `os.chdir(workspace_path)` → `apply_workspace(cfg.workspace_root, chdir=True)`.
- **vscode**: the manual `os.environ[...] = ...` block → `apply_workspace(cfg.workspace_root)`.
- **hermes**: pass `cfg.workspace_root` into the factory on the **`chat`** path (today only
  `verify` forwards it); the factory roots `FilesystemBackend` from `workspace_root()`.
- **jupyter**: the `config.WORKSPACE_ROOT` mutation + rebuild dance → `apply_workspace()`;
  a re-root on a new `server_root_dir` calls `apply_workspace()` again.
- **web** (highest risk): replace the hand-synced `_config.WORKSPACE_ROOT = …` + env
  writes in `CoworkApp.__init__` with one `apply_workspace()` call; **both** the
  file-browser (`FileManager`) and the agent tools derive from `workspace_root()` —
  the split-brain (#44) is closed by construction (one source, not two hand-synced).

Old env-var readers keep working throughout, because `apply_workspace` sets the same
env vars — so migration is additive, surface-by-surface (ADR 0004 discipline).

## Risks & mitigations

- **Re-opening the web split-brain (#44).** Mitigation: a regression test that sets
  `--workspace <tmp>`, runs a turn whose agent writes a file via its bash/fs tool, and
  asserts the file lands under `<tmp>` **and** that the file-browser lists it — proving
  one source feeds both. This test is the acceptance gate for the web PR.
- **Order of operations.** `apply_workspace` must run *before* the agent is built and
  the file-browser is constructed. Encode as the first step of each surface's startup.
- **Process-global test pollution** (same shape as `_QUIET`): a conftest fixture resets
  `_ACTIVE` + the env vars between tests.
- **BYO tools silently unrooted.** Not a regression (they were never rooted), but name
  it: document the `workspace_root()` contract and consider a `check` warning.

## Verification

A pm-dogfood-able cross-surface acceptance: for each surface, set `--workspace <tmp>`,
drive one turn that writes a file via the agent's filesystem tool, and assert the file
appears under `<tmp>` (not the launch cwd). This is the single behavior all five bugs
were about; passing it on all five surfaces is "done."

## Staging

1. Ship `apply_workspace` / `workspace_root` in core (additive; a 1.0.x minor).
2. Migrate cli, vscode, hermes, jupyter (each a thin PR + the acceptance test).
3. Migrate web last, gated on the split-brain regression test.
4. (Optional, later) a `RunnableConfig` path if per-request workspaces ever appear.

## Open questions for the decision

1. **Endorse the BYO boundary?** Core roots agents it builds; BYO graphs opt in via
   `workspace_root()`. This is honest but means `-a my_agent.py` with hand-rolled tools
   isn't auto-rooted. Acceptable, or do we want a `check` warning / a documented
   `LANGSTAGE_TOOLS`-style helper that reads `workspace_root()` for BYO authors?
2. **`chdir` for cli/vscode — keep it?** It's the bluntest mechanism but the most
   universal for those single-process surfaces. Keep as an opt-in `chdir=True`, or drop
   chdir entirely and rely on backend-rooting + the env var only?
3. **Version:** ship the core API as **1.0.7** (additive, immediate) or fold into the
   **1.1.0** that ADR 0004 earmarked for the deprecation sunset?
