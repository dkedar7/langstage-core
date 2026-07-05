# ADR 0006 — Operate the agent from the workspace as its cwd (single-workspace surfaces)

**Status:** Accepted — 2026-07-04
**Decision owner:** Kedar Dabhadkar
**Relates to:** ADR 0005 (workspace application + the BYO boundary); dogfood finding F7

## Context

A fresh-user dogfood found that a bring-your-own agent that writes files the normal
Python way — `Path("notes.md").write_text(...)`, a raw **cwd-relative** path — lands
those files in different places per surface, and mostly *not* where the user expects:

| Surface | BYO agent's relative write lands in | Shows in that surface's file browser? |
|---|---|---|
| cli | the **workspace** (it `chdir`s) | (n/a) |
| web | the server **launch cwd** | ❌ browser is rooted at the workspace |
| vscode | the launch cwd | (n/a) |
| jupyter | the launch cwd (= JupyterLab's notebook root) | ✅ (that root *is* JupyterLab's browser) |

On the web console the agent replied *"see notes.md in the workspace"* while the file
browser showed **"Workspace is empty."** This breaks the family's headline promise —
*"the same spec runs unchanged on every stage"* — on the filesystem, and it makes the
flagship "operate any agent + file browser" story feel broken.

ADR 0005 established the honest boundary: core resolves + publishes `workspace_root()`,
but **cannot re-root an arbitrary BYO graph's tools** — a raw `Path(...)` doesn't read
`workspace_root()`. The *only* mechanism that redirects a raw cwd-relative write is the
**process cwd**. cli already `chdir`s (ADR 0005, `chdir=True`); the servers do not.

## Decision

**Every single-workspace surface operates the agent with the resolved workspace as the
process cwd.** So a BYO agent's raw relative file ops land in the workspace — where that
surface's file browser shows them — without the agent having to know about
`workspace_root()`.

1. **cli** — already `chdir`s. No change.
2. **vscode sidecar** — `chdir` to the resolved workspace, *after* resolving the agent
   spec (a relative `-a ./x.py:graph` must resolve against the launch dir, cf. cli
   gh #30). Single-process, single-agent — clean.
3. **web** — `chdir` to the resolved workspace in `CoworkApp.run()`, *after* the agent is
   built and `create_server()` has wired the file browser (which uses the **absolute**
   resolved workspace, so it is unaffected). A langstage server serves exactly one
   workspace per process (ADR 0005 assumption), so moving its cwd to that workspace is
   safe: the bundled frontend is served from the package dir, and the file browser +
   uploads use absolute paths. `chdir` is done in `run()` (not `__init__`) so *embedding*
   `CoworkApp` programmatically has no cwd side effect.
4. **jupyter — the documented exception.** JupyterLab owns the process: its own file
   browser and notebook root are the launch dir, and `chdir`ing the server out from under
   it would desync them. So in jupyter the agent's cwd stays JupyterLab's launch dir —
   which *is* JupyterLab's file browser, so BYO files are visible there. If you want a
   separate agent workspace in jupyter, the **bundled** agent's `FilesystemBackend`
   honors `LANGSTAGE_WORKSPACE_ROOT` (rooted absolutely); a BYO raw-relative agent uses
   JupyterLab's dir. This is called out in the jupyter docs.

The `[Working directory: …]` context every surface prepends already reports the real
`workspace_root()` (web fixed in the same dogfood), so a well-behaved BYO agent that
*reads* its working directory also does the right thing — belt and suspenders.

## Consequences

- A BYO agent that writes `Path("out.txt")` now produces a file the user can see in the
  web/cli file view, consistently. The bundled agent is unaffected (its
  `FilesystemBackend` is rooted at an absolute `workspace_root()`, immune to cwd).
- `chdir` is process-global. This is acceptable under the one-workspace-per-process model
  (ADR 0005). A future per-request-workspace surface would need the `RunnableConfig` path
  instead and must **not** rely on cwd.
- Placement matters: resolve the agent spec (and, for web, build the server) **before**
  `chdir`, so relative agent paths and startup file lookups still resolve against the
  launch dir.
- jupyter remains intentionally different; documented rather than forced into parity.
