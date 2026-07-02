# ADR 0003 — Deprecate and retire the event layer

**Status:** Implemented — 2026-07-02 (Accepted 2026-07-01)
**Decision owner:** Kedar Dabhadkar
**Builds on:** [ADR 0001](0001-adopt-ag-ui-for-the-wire.md), [ADR 0002](0002-execute-event-layer-retirement.md)

## Context

ADR 0002 is done: all four render surfaces (cli, jupyter, vscode, web) ship an
**opt-in** in-process AG-UI path, and the two shared mappings live in the core
(`agui.iter_event_frames`, `agui.iter_chunk_frames`). The dedupe is complete.

ADR 0002 listed the endgame as "deprecate `events.py` → remove at a major →
rename." That line was too glib. Executing it surfaces two facts that change the
plan, and this ADR exists to get them right *before* anything irreversible ships.

### Fact 1 — the AG-UI paths are opt-in; the event layer is still the default

Every surface still defaults to `StreamParser`. You cannot bolt a loud
`DeprecationWarning` onto `StreamParser.__init__` while it is the default code
path — it would fire on every normal run. **Deprecation must follow a
default-flip, not precede it.** (ADR 0002's remaining-work ordering had this
backwards.)

### Fact 2 — hermes is an un-migrated *fifth* surface, entangled with extractors

`langstage-hermes` was never migrated. Its `chat` command renders through
`StreamParser` + `PrintAdapter`, and its reflection/skill value is *designed* to
ride on **four custom extractors** (`langstage_hermes/extractors.py`) that
implement `extractors.base.ToolExtractor` and emit `ToolExtractedEvent`s (e.g.
`skill_created`), which the cli's `_pretty_extraction` renders as typed callouts.

So ADR 0002's "keep `extractors/`, remove the rest" is **incoherent**: extractors
are *driven by* `StreamParser` and *emit* an `events.py` type. You cannot remove
`StreamParser` / `events.py` while keeping extractors functional for hermes.

**Latent-bug finding (2026-07-01):** those four extractors are **never
registered** — the only reference outside `extractors.py` is its own docstring,
and `chat` uses a bare `StreamParser()`. So the skill/memory/compression callouts
are **dead code that doesn't fire today**. This *de-risks* the migration (there's
no live behavior to preserve) and means bridging extractors to AG-UI (Stage 1)
makes them fire for the first time — a fix, not just a port. (Whether to also fix
the legacy path by registering them is a separate, optional hermes bug-fix.)

## What "the event layer" actually is (inventory)

**Retire** (event-translation): `StreamParser`, the `events.py` dataclasses +
`event_to_dict`, `handlers/`, `stream_graph_updates` / `astream_graph_updates`,
and the render adapters `CLIAdapter` / `FastAPIAdapter` / `JupyterDisplay` /
`PrintAdapter`.

**Keep** (not event-translation, already load-bearing for AG-UI): the host layer
(`HostConfig`, `load_agent_spec`, demo), `tasks/`, the `agui/` module (`iter_*`,
`build_*`), `SessionAdapter` (now dual-mode — it keeps its AG-UI path), and the
input helper `prepare_agent_input` (the AG-UI paths reuse it for context-combining).

**Entangled** (the blocker): `extractors/` + hermes' dependency on
`StreamParser` + `PrintAdapter` + `ToolExtractedEvent`.

## Decision (proposed)

Retire the event layer in **five ordered stages**, gated so nothing user-visible
regresses and no warning fires on a default path.

1. **Solve the extractor → AG-UI story.** — **VALIDATED ✓ (2026-07-01, prototype).**
   Simpler than first proposed: no AG-UI `CustomEvent` is needed. The extractors
   run in the **mapping layer** — an `extractors=` param on `iter_event_frames` /
   `iter_chunk_frames`. On each `ToolCallResultEvent` the bridge looks up the
   extractor by `tool_name` (already tracked from `ToolCallStart`), calls
   `extract(content)`, and if non-None emits an `extraction` frame **byte-identical
   to `event_to_dict(ToolExtractedEvent)`**. The prototype
   (`deepagent-hermes @ spike/extractor-agui-bridge`) ran hermes' *real*
   `SkillManageExtractor` over an AG-UI stream and produced
   `{"type":"extraction","tool_name":"skill_manage","extracted_type":"skill_event",
   "data":{...,"extracted_subtype":"skill_created"}}` — exactly what hermes'
   `_pretty_extraction` renderer reads. The extractor protocol is preserved as-is.
2. **Migrate hermes** (the fifth surface) onto the AG-UI path using (1), reaching
   parity for its reflection/skill callouts. Only after this do all five surfaces
   have an AG-UI path.
3. **Soak, then flip defaults.** Let the opt-in paths run in the wild; then flip
   each surface's default from `StreamParser` to AG-UI, one surface per release,
   each gated on parity, with the old path still reachable via an escape hatch
   (`--no-agui` / env) for one release.
4. **Deprecate.** Once nothing defaults to the event layer: `PendingDeprecationWarning`
   (silent) on the retire-set for one minor, then `DeprecationWarning` the next.
   Keep the extractor *protocol* (`ToolExtractor`) if (1) preserves it as the
   extension point; deprecate only the `StreamParser`-driven plumbing.
5. **Remove at the next major + rename.** Delete the retire-set and rename the
   core to a `langstage-core`-style identity in the same major, so users absorb
   one break, not two. `event_to_dict` frames remain producible (that shape is the
   AG-UI mapping's output) even though `StreamParser` is gone.

## Consequences

- **Longer than ADR 0002 implied.** This is a multi-release arc (hermes migration →
  default-flips × 5 → two-step deprecation → major). That is the cost of retiring a
  layer that five surfaces and one external-ish contract depend on, without churn.
- **hermes gets first-class AG-UI support** — the reflection/skill callouts, today
  a bespoke extractor path, become a documented AG-UI extension point. Net upgrade.
- The `event_to_dict` **wire shape survives** the death of `StreamParser` (it's what
  `iter_event_frames` emits), so external consumers of that JSON shape are unaffected
  by the internal retirement — only direct importers of `StreamParser`/`events` break,
  and only at the major, after two warning stages.

## Alternatives considered

- **Deprecate now, keep a frozen extractor sub-layer forever.** Rejected: it
  permanently keeps `StreamParser` + `events.py` + `extractors/` alive (most of the
  layer) to serve one surface — the opposite of the goal, and it strands hermes on
  the legacy path.
- **Drop hermes' extractor callouts.** Rejected: they are hermes' core value
  (reflection/skill visibility). Retiring infrastructure must not delete a feature.
- **Flip defaults before soaking.** Rejected: the opt-in paths are days old; flipping
  the default across five surfaces before real-world soak is how you ship a
  regression to everyone at once.

## Open questions / gates (must clear before Stage 3)

1. **Extractor → AG-UI fidelity.** — **RESOLVED ✓ (prototype, see Stage 1).** The
   `extractors=`-param bridge reproduces the extraction frame at `event_to_dict`
   parity against hermes' real `SkillManageExtractor`. The remaining extractors are
   the same protocol; wiring them is mechanical.
2. **Escape hatch lifetime.** How long do we keep `--no-agui` after a default-flip?
   Proposed: one minor per surface, then remove with the event layer at the major.
3. **usage events.** ADR 0002 left token-usage parity as a spot-check; confirm the
   AG-UI paths carry it before flipping the surfaces that display cost.

## Not doing (now)

- Any `DeprecationWarning` on `StreamParser` / `events` — premature until Stage 4.
- The rename — bundled with the Stage 5 major.
- Removing `prepare_agent_input` or `SessionAdapter` — both are keepers.

## Outcome — 2026-07-02 (Implemented)

The full arc shipped. The event layer is gone and AG-UI is the sole streaming
path on every surface. There is no `--no-agui` escape hatch: rather than a
default-flip-then-soak, the event layer was removed outright at the core major and
each surface repointed to the AG-UI-only adapter in one release.

- **Core:** `langgraph-stream-parser` renamed to **`langstage-core` 1.0.0**.
  Removed `StreamParser`, `events.py`, `event_to_dict`, `compat.py`, `handlers/`,
  and the message/interrupt extractors; the `SessionAdapter` is now AG-UI-only.
  **Kept:** `prepare_agent_input`, `create_resume_input`, `host/`, `tasks/`,
  `extractors/` (base + builtins, event-layer-free), and the two shared mappings
  `agui.iter_event_frames` / `agui.iter_chunk_frames`. A final
  `langgraph-stream-parser 1.0.0` compat shim re-exports `langstage_core` under
  the old import name with a `DeprecationWarning`.
- **Surfaces (all AG-UI-only, repointed to `langstage-core[agui]` in base deps):**
  langstage-cli **0.6.1**, langstage-hermes **0.4.1**, langstage-vscode **0.5.0**,
  langstage-jupyter **0.6.0**, langstage (web) **0.13.0**.
- **Open questions, resolved:** (1) extractor→AG-UI fidelity — shipped (hermes'
  four extractors are wired via the `extractors=` bridge). (2) escape-hatch
  lifetime — moot; no escape hatch shipped (clean removal at the major). (3)
  usage-event parity — the surfaces that display cost run on the AG-UI paths.
- **One regression caught + fixed during rollout:** the Jupyter React frontend
  rendered AG-UI **token deltas** as fragmented grey per-token "intermediates"
  (it was built for the old cumulative per-message chunks). Fixed by accumulating
  same-node deltas into one message (langstage-jupyter 0.6.0). The other frontends
  were already delta-native.
