# ADR 0002 — Execute the event-layer retirement; scope the core to the host layer

**Status:** Accepted — 2026-07-01
**Decision owner:** Kedar Dabhadkar
**Supersedes the "staged / later" clauses of:** [ADR 0001](0001-adopt-ag-ui-for-the-wire.md)

## Context

ADR 0001 (2026-06-14) set the direction: adopt `ag-ui-langgraph` for the wire,
declare the **host layer** (`HostConfig` + `load_agent_spec` + demo stub) the
core's durable identity, and let the **event-translation layer** (the typed
`StreamEvent` dataclasses, `StreamParser`, `event_to_dict`, and the tool-output
extractors in `events.py` + `handlers/`) become "legacy/optional," retired
**surface-by-surface only after AG-UI earns it.** That was deliberately
additive: at 0.3.0 we had not yet shipped the AG-UI base, so deleting the event
layer would have broken all six surfaces at once.

Three things have changed since, and together they close ADR 0001's "not yet":

1. **The AG-UI base shipped and the surfaces are on it.** `langgraph_stream_parser.agui` + the `[agui]` extra + `langstage-agui --agent <spec>` exist; all six packages have re-released against the AG-UI bridge. The precondition ADR 0001 named ("after AG-UI earns it") is materially met.
2. **The event layer has no external consumers — and it is not free.** Nothing outside our own surfaces consumes `events.py`. It is pure internal plumbing, and the nightly dogfood routine keeps *hardening* it — dict-form message rendering, `tool_end` name resolution, extracted-type mapping. Every one of those is behaviour `ag-ui-langgraph` already implements and maintains for us. We are paying maintenance to keep a private copy of a solved problem.
3. **External validation that the wire converged.** LangChain shipped `deepagents-code` (the `dcode` terminal agent) on the Deep Agents SDK. Like every other app in this ecosystem, it renders its own stream internally and reaches frontends via the standard protocol, not via our event dicts. The market is not adopting a bespoke LangGraph event vocabulary; it is converging on AG-UI. Maintaining our own is swimming upstream.

The honest counter-argument (raised and not dismissed): **in-process renderers
do not need a wire.** Pulling an HTTP/serialization protocol into the terminal
and Jupyter render paths would add weight and external-schema coupling those
surfaces don't otherwise need. The resolution is that **AG-UI is a data model,
not only a transport**: `ag-ui-langgraph` can emit AG-UI event *objects* from a
LangGraph stream in-process, with no socket. We reuse the vocabulary and the
encoder; we do not serialize over HTTP unless a surface actually serves a
remote frontend. That dissolves the objection — *if* the in-process encoder is
usable without dragging the web-server stack (see Open questions).

## Decision (proposed)

1. **Execute the staged retirement now.** Stop treating "retire the event layer"
   as indefinitely deferred. Commit to a concrete end-state and timeline.
2. **Target end-state for the core** = host layer + AG-UI bridge only:
   `HostConfig`, `load_agent_spec`, the demo stub, and `langgraph_stream_parser.agui`.
   The dependency floor stays `langchain-core` for the host layer; AG-UI deps stay behind the `[agui]` extra.
3. **`events.py` / `StreamParser` / extractors become a deprecated compat shim.**
   They keep working through one deprecation window so nothing breaks mid-migration,
   emit a `DeprecationWarning`, and are removed at the core's next major.
4. **Surfaces consume AG-UI events in-process.** Each first-party surface
   (web, cli, jupyter, vscode) migrates its renderer from `StreamParser` dicts to
   the in-process AG-UI event stream, one surface per PR, behind the existing
   per-surface test suites. No surface is migrated until its replacement renders
   the demo stub + a real agent at parity (text, tool calls, interrupts, usage).
5. **Sequence — cheapest/lowest-risk first:** `langstage-cli` (smallest render
   surface, fastest to verify) → `langstage-jupyter` → `langstage-vscode` →
   `langstage` (web; largest, migrate last). Each step is independently shippable
   and reversible.

## Consequences

- The core's name `langgraph-stream-parser` will, after this, describe a job it
  no longer primarily does (it won't be a stream *parser* — it'll be the host +
  bridge). ADR 0001 anticipated a `langstage-core`-style rename. **Proposed:**
  bundle that rename with the major release that removes `events.py`, so users
  absorb one rename, not two. (Decision deferred to that release; flagged here so
  it isn't a surprise.)
- One event vocabulary across the family instead of two. A rendering fix lands
  once, upstream in `ag-ui-langgraph`, instead of in our `handlers/` + each
  surface. The nightly dogfood stops generating `events.py` hardening issues.
- We deepen the ADR 0001 dependency bet on a CopilotKit-led spec — now in the
  in-process render paths, not just the wire. Accepted on the same grounds as
  0001 (MIT, ~14k★, multi-framework, active), with the added mitigation that the
  host layer — the actual differentiator — has **zero** AG-UI coupling and is
  unaffected if we ever need to swap the wire.
- Short-term cost is four renderer migrations. They are sequenced, isolated, and
  each gated on parity, so the risk is paid down incrementally rather than in a
  big-bang cutover.

## Alternatives considered

- **Keep a thin bespoke in-process event model, AG-UI only on the wire.** This is
  the status quo of ADR 0001. Rejected as the *end* state because it permanently
  carries two vocabularies and the maintenance that comes with them, for a model
  that has no external consumers. (It remains the *transition* state — that's
  exactly the deprecation window in Decision 3.)
- **Delete `events.py` in one swing.** Rejected for the same reason ADR 0001
  rejected it: it breaks all six surfaces at once. The per-surface sequence is
  the small-impact-surface path.

## Open questions / validation gates (must clear before step 4 lands)

> **Status:** gate (1) cleared 2026-06-30, gate (2) cleared 2026-07-01 — the cli
> ships text + tools + interrupts on the AG-UI path (langstage-cli 0.5.17). The
> resume mechanism is proven and generic, so the HITL surfaces (web, vscode) are
> unblocked. (Usage-event parity is the one remaining spot-check.)

1. **Does `ag-ui-langgraph` expose an in-process event generator without the
   FastAPI/uvicorn stack?** — **RESOLVED ✓ (2026-06-30, probe).**
   `ag_ui_langgraph.LangGraphAgent.run(RunAgentInput)` is an `AsyncGenerator` of
   AG-UI event objects (`TextMessage*`, `ToolCall*`, `State*`, `Reasoning*`, run/
   step lifecycle) — consumed by iterating it in-process, no ASGI app. A probe
   drove a keyless echo graph (text recovered from `MessagesSnapshotEvent`) and a
   keyless **streaming** fake model (11× `TextMessageContentEvent` deltas
   reconstructing the text), confirming the streaming terminal UX survives.
   `uvicorn` is **never imported**; a base `ag-ui-langgraph` install pulls
   `ag-ui-protocol` + `langgraph` and **no fastapi/uvicorn/starlette**.
   - *One caveat, cheap:* `import ag_ui_langgraph` today eagerly imports its
     FastAPI endpoint, so the top-level import currently forces `fastapi`+
     `starlette` (still not `uvicorn`). The adapter modules (`agent`, `types`,
     `utils`) are themselves fastapi-free — verified by guarding that import,
     **uninstalling fastapi entirely**, and re-running the probe green.
   - *Action:* land a one-line upstream lazy-import in `ag-ui-langgraph`'s
     `__init__` (filed upstream: ag-ui-protocol/ag-ui#2067, with the patch +
     evidence ready to PR). Until/unless it merges, the fallback is to accept
     `fastapi`+`starlette` as transitive deps of
     the in-process path — light (no `uvicorn`, no C extensions, `pydantic`
     already present via langchain), far from "a web server in the terminal install."
   - *Not covered by this probe:* interrupt/resume + usage parity — that's gate (2).
2. **Interrupt/resume + usage parity.** — **RESOLVED ✓ (2026-07-01, cli).**
   A LangGraph `interrupt()` surfaces in-process as `CustomEvent(name="on_interrupt",
   value=<payload>)` (maps to the display), and **resume** works by delivering the
   decision as **`forwarded_props.command.resume`** — the field the `ag-ui-langgraph`
   adapter converts to LangGraph's `Command(resume=...)`. Passing `{"decisions": […]}`
   there (the exact value the default path builds via `prepare_agent_input(decisions=…)`)
   drives the graph past the interrupt instead of re-interrupting. Shipped in
   `langstage-cli` 0.5.17 (`--agui` runs the full display→decide→resume loop,
   including `--no-interactive` auto-approve) with a resume round-trip test.
   - *Earlier dead end:* the top-level `RunAgentInput.resume` (typed `ResumeEntry`)
     did **not** drive a raw `interrupt()` — the adapter reads
     `forwarded_props.command.resume`, not that field.
   - *Generic:* the same mechanism unblocks the web board and vscode HITL surfaces.
   - *Remaining spot-check:* token-**usage** events (not exercised by the cli path).

## Not doing (now)

- Renaming the core package — bundled with the major that removes `events.py`, decided then.
- Touching the host layer — it is the keeper; this ADR does not change it.

**Migration progress: ALL FOUR SURFACES DONE** — cli (0.5.17), jupyter (0.5.13),
vscode (0.4.9), web (langstage 0.12.3, on core 0.6.16). Each ships an opt-in
`[agui]` path with full text + tools + interrupts parity (frame/chunk shape AND
terminal outcome), default path untouched. Both gates cleared; each surface
migrated one PR at a time, gated on parity.

**Shared-mapping note (two format-families).**
- *chunk-dict* (`stream_graph_updates` shape): **cli + jupyter** — byte-identical
  copies in `langstage_cli.agui_stream` / `langstage_jupyter.agui_stream`.
- *`event_to_dict` frames*: **vscode + web** — web uses the core's
  `agui.iter_event_frames` (0.6.16); the vscode sidecar still has its own copy.

**Remaining work (post-migration):**
1. **Dedupe:** point the vscode sidecar at `agui.iter_event_frames` (delete its
   copy); consider a core `agui` helper for the chunk-dict family (cli + jupyter).
2. **Deprecate the event layer (Decision 3):** now that every first-party surface
   has an AG-UI path, mark `StreamParser` / `events.py` / extractors deprecated,
   then remove at the next major — *keeping* `extractors/` (hermes extends it).
3. **Rename** the core to a `langstage-core`-style identity, bundled with that major.
4. Flip each surface's default to AG-UI once the opt-in paths have soaked.
