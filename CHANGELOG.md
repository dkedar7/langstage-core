# Changelog

## [1.0.36] - 2026-09-24

Wave 2 of the 2026-09 family sweep: root causes that were filed separately in several
surfaces, fixed once here (#183, #184, #185, #186).

### Fixed
- **Agent-spec loading.** A `file.py:attr` agent can import its sibling modules (its directory is
  put on `sys.path`, idempotently) (#150; cli#145, web#167); a project-local
  `package.module:attr` resolves from the cwd (or `base_dir=`) when not otherwise importable (#147;
  cli#141). Specs are whitespace-stripped and `~`-expanded; a colon-less spec errors with a
  `file.py:graph` hint; a `str` attribute raises `TypeError` instead of being re-read as another
  spec (vscode#135, #125; jupyter#151; cli#149).
- **Config-relative paths.** A relative `[agent] spec` / `[workspace] root` in a TOML file resolves
  against that file's directory (project and global), not the cwd; `~` expands from every source,
  including `apply_workspace` and the `workspace_root()` env fallback (cli#132, #133; vscode#123,
  #126, #125).
- **Import-time stdout.** `langstage-agui --verify` / `-m` / `--json` keep an agent's import-time
  prints off stdout (cli#136, web#140).
- **Honest config reporting.** `--show-config` / `python -m langstage_core.host` report a
  present-but-malformed `langstage.toml` as MALFORMED (with the parse error) instead of "not found"
  (#175; cli#140, hermes#151, vscode#110, web#170).
- **Legacy aliases.** `DEEPAGENTS_CONFIG_HOME` and `DEEPAGENT_WORKSPACE_ROOT` emit the one-time
  legacy notice (#167, #179); each legacy alias is announced exactly once (vscode#112).
- **`debug` and booleans.** Error-frame tracebacks honor the resolved `debug` (legacy env, TOML)
  (#137); a bare top-level key bound to a preceding `[table]` gets a note (#139); quoted TOML
  booleans are coerced with the env-bool rules, unrecognized values degrade with a note
  (jupyter#133).
- **`[configurable]`** is forwarded to the graph by `langstage-agui` (serve and `--message`) and
  shown by `--show-config` (#170; vscode#127).
- **cp1252 consoles.** `--show-config`, `python -m langstage_core.host`, and `langstage-agui -m` no
  longer crash with `UnicodeEncodeError` (#171, #153; web#146).
- **Served wire parity.** `build_app` / `serve` stream TEXT_MESSAGE_* / TOOL_CALL_* for finished
  (non-token-streamed) messages, matching the in-process wires (#140). Finished messages are emitted
  when their node finishes, text before the message's own tool calls (cli#119); content from
  earlier nodes is no longer dropped when a later node errors (vscode#105).
- **Frame consistency.** Chunk-wire `error` is terminal (#161); chunk tool results carry error
  status (#168) and tool calls carry `id` (#149); no `extraction` frame for a failed tool (#177);
  `TurnResult.interrupt` is identical across collectors (#154).
- **Isolated one-shots.** `build_agent` no longer mutates the caller's graph and `run_turn` uses a
  fresh thread id per call (#163).
- **stdlib `TypedDict` state** no longer fails every turn on Python 3.11 (web#166).
- **HITL resume** rides ag-ui-langgraph's standard `RunAgentInput.resume` (0.0.43+,
  feature-detected) instead of the deprecated `forwarded_props.command.resume`: no more
  deprecation / "failed to parse resume_input" warnings on every resume; public `resume=` unchanged,
  legacy wire kept as a fallback (#144; cli#126, #137; vscode#103).
- **`allowed_decisions`** reflects the interrupt's own config (re-read from the checkpoint, since
  ag-ui-langgraph >= 0.0.43 strips HumanInterrupt `config`); HumanInTheLoopMiddleware
  `review_configs[*].allowed_decisions` honored (vscode#114).
- **`langstage-agui` errors.** `-m` / `--json` report a non-runnable agent cleanly with exit 1, like
  `--verify` (#180); the port is bound before "Serving ... at <url>" prints, so a busy port is a
  clean error and exit 2, and `serve()` raises `OSError` (#143).

### Added
- `load_agent_spec(..., base_dir=, stdout_to_stderr=)`, `parse_agent_spec()`,
  `HostConfig.toml_dir_for()`.
- `langstage_core.console.safe_print` / `safe_write` / `console_safe`: console-safe output
  (unencodable characters are backslash-escaped, never raised).
- `HostConfig.config_issues()`, `HostConfig.malformed_toml()`, `HostConfig.configurable()`;
  `config_dict()["toml"]` gains `malformed`, `malformed_files`, `paths`; `config_dict()["issues"]`;
  `serve(config=...)`.
- Additive frame keys: `message_id` on `content` / `chunk` frames (vscode#108); real `duration_ms`
  on `tool_end` / chunk `tool_result` (web#160); chunk `tool_result` siblings `id`, `name`,
  `tool_status`, `duration_ms`; `outcome` (`complete` | `interrupted`) on the terminal `complete`
  frame (#152).

### Changed
- Relative paths from TOML resolve to absolute paths in `HostConfig` (visible in `--show-config`).
  In the global `~/.langstage/config.toml`, relative paths resolve against `~/.langstage/`; use
  `~/...` or an absolute path there.
- A failed tool no longer emits an `extraction` frame; repeated one-shot calls on a bare graph
  without a checkpointer no longer share state.

### Docs
- README frame reference for both wires (#169).

## [1.0.35] - 2026-09-23

### Fixed
- **`langstage-agui --verify` / `--message` exit `1`, not `2`, when no agent spec resolves
  (gh #174).** With no `--agent`, `LANGSTAGE_AGENT_SPEC`, or `[agent].spec`, both commands exited
  `2` — outside `--verify`'s documented `0`/`1` and, under `--message`, the code that means
  *interrupted*. A CI/deploy gate following the README therefore passed a completely unconfigured
  agent as a benign HITL pause (fail-open). The no-spec path (and the `--demo`/`--agent` conflict)
  now gets the same command-aware mapping as a bad spec (#124) and a missing extra (#134): `1`
  under `--verify`/`--message`, `2` kept on the serve path.
- **One `build_agent(...)` agent is now safe to drive from concurrent turns (gh #165).** The
  README and `examples/fastapi_websocket.py` build one agent and reuse it across sessions, but
  `iter_event_frames` / `iter_chunk_frames` (and so the collectors, `run_turn`, and `verify`) drove
  the shared `LangGraphAgent` directly — and it keeps per-run state on the instance (`active_run`).
  Two interleaved turns crashed the later one mid-stream with `TypeError: 'NoneType' object does
  not support item assignment` (a lost answer). Both wires now `clone()` the agent per run — the
  isolation `build_app` and `SessionAdapter` already used; the graph + checkpointer (thread state)
  are shared as before.

## [1.0.34] - 2026-08-08

### Fixed
- **A missing `[agui]` extra now maps to each command's exit-code contract (gh #134).** The
  `ensure_available()` failure path hardcoded exit `2`, so the documented bare-install preflight
  (`langstage-agui --demo --verify`) returned `2` — outside `--verify`'s `0`/`1` vocabulary and, under
  `--message`, the code that means *interrupted*. It now returns `1` under `--verify`/`--message`
  (a can't-run failure) and keeps `2` on the serve path — the same command-aware treatment the
  agent-load-failure path got in #124.
- **`run_turn()` raises an actionable error inside a running event loop instead of a raw asyncio
  internal + a leaked coroutine (gh #135).** Called from a Jupyter cell (or any async context),
  `run_turn` failed with `RuntimeError: asyncio.run() cannot be called from a running event loop`
  plus a `coroutine ... was never awaited` warning. It now checks for a running loop up front (before
  building the coroutine, so nothing leaks) and raises a clear error pointing at the documented
  alternative: `await collect_event_frames(...)` / `collect_chunk_frames(...)`.

## [1.0.33] - 2026-08-06

### Fixed
- **An out-of-range port now degrades to the default with a note instead of a silent misbind
  (support for langstage #123).** `port` was type-checked but never range-checked, so an in-range
  *integer* out of the valid *port* range (`70000`) sailed through, was advertised by
  `--show-config` and the startup banner, and then uvicorn silently masked it to 16 bits
  (`70000 & 0xFFFF == 4464`) — binding a different port than everything advertised, with no error.
  A new extensible `_VALIDATORS` map (MRO-merged like `_ENV`/`_TOML`) validates resolved values;
  `port` degrades an out-of-range value (from env, TOML, or an override) to the default + a
  `note:`, exactly like a malformed numeric value.
- **Doc: `iter_chunk_frames`'s docstring no longer names the removed pre-1.0 `stream_graph_updates`
  helper as the current wire (gh #130).** It described the live chunk wire with a symbol that was
  removed in 1.0 and isn't importable; the docstring (and the shipped Jupyter example prose) now
  describe the actual `status`-keyed chunk dicts.
- **Doc: the README now marks `TurnResult.frames` as an `int` count, not the frame list (gh #131).**
  Listed among the collected-data fields, `frames` read as "the frames"; it's a count
  (`for f in result.frames:` crashed).

### Added
- **`TurnResult.traceback` — the crash traceback on the one-shot path, under `LANGSTAGE_DEBUG`
  (gh #132).** 1.0.32 put the traceback on the streaming error frame, but the one-shot collectors
  (`run_turn`/`collect_*`) and `langstage-agui --message` — the funnel adopters actually use —
  dropped it. `collect_event_frames`/`collect_chunk_frames` now capture it into
  `TurnResult.traceback` (populated only on an `error` outcome with debug on, `None` otherwise), and
  `langstage-agui --message` prints it (text mode → stderr; `--json` → a `traceback` field). The
  one-shot path now reaches the same *where* the streaming wires do.

## [1.0.32] - 2026-08-03

### Fixed
- **`--show-config` / `config_dict` no longer misattribute a global-config value to the project
  file (gh langstage #119).** `resolve()` labeled every TOML value with the last file read (the
  project `langstage.toml`), so a value set only in the global `~/.langstage/config.toml` was
  reported as coming from `langstage.toml`. Each value is now attributed to the highest-precedence
  file that actually defines its key. The layered `_load_toml_files` retains per-file data;
  `load_toml_config`'s public signature is unchanged.
- **`langstage-agui --show-config --json` now emits JSON instead of silently printing the human
  table (gh #125).** The `--json` flag was defined but only honored by `--message`; `--show-config`
  ignored it. It now prints the machine-readable `config_dict` (the structured twin of `describe`),
  so a tooling consumer gets JSON rather than scraping `[source]` brackets.
- **A failed agent load respects each command's exit-code contract (gh #124).** The load-failure
  path always exited `2`, which collided with `--verify` (0 ok / 1 failed) and `--message`
  (2 = interrupted). A load failure now exits `1` under `--verify`/`--message` (it's a failure/error,
  not an interrupt) and keeps `2` on the serve path (a can't-start usage error).

### Added
- **`HostConfig.unknown_toml_keys()` + `--show-config` surfacing of unknown/typo'd TOML keys
  (support for langstage #120 / langstage-vscode #82).** The layered config silently ignored keys
  that map to no field — the most common config mistake, which `config`/`--show-config` couldn't
  catch. `describe()` now lists them and `config_dict()["toml"]["unknown_keys"]` carries them. Keys
  under a passthrough table (`_TOML_PASSTHROUGH`, default `("configurable",)`) are never flagged;
  a subclass with its own passthrough table widens the tuple.
- **The terminal `error` frame optionally carries a `traceback` under `LANGSTAGE_DEBUG` (support
  for langstage-vscode #83).** A node crash surfaces only `Type: message`; with `LANGSTAGE_DEBUG`
  enabled, the `error` frame (both wires) also includes the exception traceback so a surface's
  `--traceback`/debug mode can show *where* the agent crashed. Off by default — the frame is
  byte-identical unless debug is explicitly on.

## [1.0.31] - 2026-07-31

### Fixed
- **`verify` / `averify` now fail an EMPTY completed turn (gh #119).** A graph that loaded and
  reached a `complete` frame but produced zero content / tool calls / reasoning reported `ok: one
  turn completed cleanly (0 chars)` and exited 0 — the exact "false green" the module exists to
  prevent, and one of the two failures the README says `--verify` catches ("a graph that loads but
  yields an empty ... turn"). The verdict now requires the turn to produce some output; a
  tool-call-only or reasoning turn still counts (it did work), a bare empty complete does not, with
  reason `"turn completed but produced no content (0 chars)"`.
- **`verify` / `averify` now treat a HITL interrupt as a HEALTHY preflight, not a failure (support
  for langstage-jupyter #95).** A turn that reaches a well-formed `interrupt` is the human-in-the-loop
  feature *working* — the agent ran and paused for a decision, which is neither an error nor an empty
  turn. It now reports `ok=True` (`"turn paused cleanly on an interrupt (HITL agent)"`), so
  preflighting an approval-gated agent in CI no longer red-fails it. This intentionally supersedes the
  earlier "an interrupt isn't a clean pass" stance (gh #110), which the nightly routine showed broke
  the advertised HITL preflight for an entire class of healthy agents. (`_terminal_outcome` and the
  collectors are unchanged — only verify's pass/fail derivation.)

### Added
- **`langstage-agui --message "..."` / `-m` — a one-shot "run my prompt and print the reply" (gh
  #120).** The terminal smoke-test companion to `--verify`: `--verify` proves the agent *runs* with a
  canned, discarded probe; `--message` runs your OWN prompt and streams the answer to stdout, with no
  Python and no server. `--json` prints the typed `TurnResult` (text / tool_calls / extractions /
  reasoning / outcome / interrupt / error) for scripting. Exit code mirrors the turn outcome
  (`complete`=0 / `error`=1 / `interrupted`=2), consistent with `--verify`. A thin wrapper over the
  shipped chunk wire — the natural `--show-config` → `--verify` → `--message` progression.

### Docs
- **"Connect a real model" README quickstart + the `[real]` extra is now documented (gh #121).** The
  docs took adopters through the keyless demos and stopped exactly at "now point this at my real
  model"; the shipped `[real]` extra (`langchain-openai` + `langgraph`) was referenced nowhere. Added a
  minimal, provider-neutral `pip install "langstage-core[agui,real]"` + `create_react_agent` example
  (OpenAI / OpenRouter / any OpenAI-compatible endpoint), noting the deepagents + Anthropic stack as
  the alternative.
- **"Delegate work to a background task" README quickstart + fixed the `tasks/__init__.py` docstring
  (gh #122).** The task engine was a headline capability with no runnable example, and the only worked
  snippet (the module docstring) referenced an undefined `adapter` and stopped at `enqueue` without
  reading the result. Added a verified end-to-end enqueue → poll `TERMINAL_STATES` → read
  `task["result"]` example (a `Task` is a `TypedDict`), in both the README and the docstring.

## [1.0.30] - 2026-07-26

### Fixed
- **`iter_event_frames` / `iter_chunk_frames` now accept a bare `CompiledGraph` (gh #117).** The two
  headline streaming mappings required a `build_agent(...)`-wrapped agent; passing a raw compiled
  graph — which the collector siblings (`collect_event_frames`, `run_turn`) and the README's "Stream
  any `CompiledGraph`" claim both accept — surfaced a single `error` frame carrying a leaked-LangGraph
  `AttributeError: 'CompiledStateGraph' object has no attribute 'run'`. Both mappings now auto-wrap
  through `build_agent` (driving anything already exposing `.run` directly), so a raw graph streams;
  a genuinely bad input degrades to a clean terminal `error` frame instead of a cryptic one.
- **`build_agent` rejects a non-graph / uncompiled `StateGraph` with an actionable `TypeError`
  (support for langstage-jupyter #92).** A wrong-type object (a `dict`, `None`, a function) reached
  `LangGraphAgent(...)` and leaked `'X' object has no attribute 'nodes'`; an uncompiled `StateGraph`
  errored mid-turn with `'StateGraph' object has no attribute 'aget_state'`. `build_agent` now
  validates it holds a compiled graph (exposes `aget_state`) up front and raises a clear message —
  `"...got dict."` or `"...uncompiled StateGraph; call .compile() on it first."` — the same DX fix
  gh #112 (spec strings) and gh #100 made elsewhere.
- **`verify` / `averify` return a clean `ok=False` verdict for a bad agent, never a raised exception
  (langstage-jupyter #92).** The agent was built *outside* `averify`'s guarded run, so a build
  failure escaped `verify()` and crashed the caller with a raw traceback (the exact class of bug the
  primitive exists to prevent). The build now runs inside the try — the same "build inside the guard"
  fix as `SessionAdapter._produce` (gh #115) — so a wrong-type or uncompiled export is reported as a
  failed preflight with an actionable reason.
- **A finished multi-text-block `AIMessage` no longer loses all but its first block (gh
  langstage-vscode #75).** ag-ui's `resolve_message_content` flattens a list-content message to only
  its first `text` block, so the final `MessagesSnapshotEvent` a non-streamed turn relies on carried
  a silently-truncated assistant `content` (wrong-but-plausible output, no error). The snapshot walk
  now re-reads the original LangChain messages from the graph checkpoint and uses the message's full
  `.text` (all text blocks joined, reasoning/thinking blocks excluded — those already surface as
  `reasoning` frames), on both the event and chunk wires. Best-effort: any failure falls back to
  ag-ui's content, never fatal.

## [1.0.29] - 2026-07-25

### Fixed
- **A build/clone failure in `SessionAdapter._produce` is now a clean `error` frame, not a silent
  hang (gh #115).** `build_agent(self._graph)` and `self._agui_agent.clone()` ran *outside* the
  `try/except` that records the terminal outcome — it wrapped only the `iter_event_frames` loop. So
  if building or cloning the agent raised (a graph `build_agent` can't wrap, a version-incompatible
  `ag-ui-langgraph`, or the documented bare install without the `[agui]` extra), the exception
  escaped `_produce` uncaught: the session pushed no terminal frame, `session.outcome`/`error` stayed
  `None`, and the background task was orphaned. Because `SessionAdapter` drives the task board, the
  web SSE stream, and the VS Code sidecar, a recoverable error became a **silent infinite hang** — a
  task wedged in `ongoing` forever with `error=None`, or an SSE client getting keepalives with no
  `error` frame. Build/clone now run inside the `try`, so such a failure degrades exactly like a
  stream error: an `error` frame plus `outcome="error"`.
- **A malformed *boolean* config env var now degrades with a note, like a malformed numeric one
  (support for langstage-hermes #92).** The boolean env caster silently coerced any unrecognized
  value to `False` — so `LANGSTAGE_DEBUG=enabled` (a natural way to try to turn something *on*)
  flipped a default off with no warning, and `--show-config` credited `[env:...]` as if honored. New
  `_env_bool_strict` (the caster for boolean config *fields*) raises `ValueError` on an unrecognized
  value so `resolve()`'s guard emits the same one-line `note:` a malformed numeric env already gets
  (gh #83/#104) and falls back to the field default — booleans and numbers degrade consistently. The
  plain `_env_bool` stays lenient for direct flag reads (`if _env_bool(os.getenv("…SUPPRESS…"))`),
  where a typo should mean "off", not crash a diagnostic. Recognized: `1/true/yes/on` / `0/false/no/off`.
  Hosts with their own boolean fields (langstage-hermes) adopt `_env_bool_strict` on their next release.

## [1.0.28] - 2026-07-25

### Fixed
- **The scalar-interrupt fix (cli #95) also covers the chunk wire now.** 1.0.27 fixed the
  `on_interrupt` handler on the event wire but missed the identical handler on the chunk wire —
  the one the CLI actually consumes — so a bare-string `interrupt("Approve X?")` still rendered
  "(no action details provided)" in `langstage-cli` on 1.0.27. Both handlers now keep a non-JSON
  string, and the end-to-end regression test is parametrized over both wires so this can't recur.

## [1.0.27] - 2026-07-25

### Fixed
- **A bare-string `interrupt("Approve deleting X?")` — the canonical LangGraph HITL form —
  no longer vanishes; the human sees what they're approving (gh langstage-cli #95).** The
  `on_interrupt` handler ran `json.loads()` on a string interrupt value and, when it wasn't valid
  JSON (a plain question almost never is), fell back to `{}` — so `_normalize_interrupt` produced
  an **empty** `action_requests` and every surface rendered "(no action details provided)", asking
  the human to Approve/Reject blind. That defeats the entire point of HITL. Now a non-JSON string
  is kept as the string, and `_normalize_interrupt` surfaces any scalar/string payload as a single
  action request (renderers already handle a scalar — cli's `format_interrupt_request` returns
  `str(action)`). Dict-payload interrupts are unaffected. Fixed on both wires, so cli, the web
  app, and the vscode sidecar all benefit.

### Added
- **The one-shot Python surface accepts a spec string, matching the CLI (gh #112).** `build_agent`,
  `run_turn`, `verify`, `collect_event_frames`, and `collect_chunk_frames` took only a compiled
  graph or a prebuilt agent — so passing the library's own headline input form, a `module:attr` /
  `path/to/file.py:attr` spec, failed with a cryptic leaked-LangGraph `AttributeError: 'str' object
  has no attribute 'nodes'`, even though every CLI surface (`langstage-agui --agent …`, `--verify`,
  `--show-config`) already resolves specs. `build_agent` now resolves a `str` through the same
  `load_agent_spec()`, and because `run_turn`/`verify`/`collect_*` route any non-agent through
  `build_agent`, the whole one-shot surface inherits spec support from one point — the eval/batch/CI
  use case the collectors were built for (agents identified by spec).
- **`HostConfig.config_dict()` — a machine-readable twin of `describe()` for `--show-config --json`
  (support for langstage-jupyter #88 and langstage-vscode #71).** `--show-config` is the family's
  config-diagnostic verb but emitted only aligned human text, so a CI/tooling consumer had to regex
  the `[source]` bracket out of formatted output — brittle, and config-source correctness is the
  most bug-prone area in the family (a dense history of precedence/advertising regressions).
  `config_dict()` returns the same value + provenance (`{value, source, env, legacy_env, toml}` per
  key, plus a `toml: {found, path}` block) that `describe()` formats, so a surface can emit
  `--show-config --json` and a pipeline can assert on the precedence contract. A test pins that
  `config_dict()` and `describe()` never drift on keys or sources.

## [1.0.26] - 2026-07-23

### Added
- **A one-shot "run a turn, get a typed result" helper over the `iter_*` mappings (gh #110).**
  The library exposed exactly two consumers — `iter_event_frames` and `iter_chunk_frames`, both
  *streaming* async generators. That's the right primitive for a live UI, but a large share of
  real usage is *not* a live UI: a test, an eval/grading harness, a batch job, a `@tool` that
  delegates to a sub-agent, a "run my agent once, give me the answer" script. Every one of those
  wanted a single call that runs one turn and returns the result, so each hand-rolled the same
  accumulator loop over `iter_event_frames` (collect `content` deltas, watch for `interrupt`,
  map `complete` → outcome, catch `error`) — four copies in one dogfooding session. New
  `langstage_core.agui.collect_event_frames(agent, message, thread_id, *, resume, max_result_len,
  extractors, state)` (async) drives one turn and returns a typed `TurnResult` dataclass:
  `text` (joined `content`), `tool_calls` (`[{name, args, id}]`), `extractions`
  (`[{tool_name, extracted_type, data}]`), `reasoning`, `outcome`
  (`"complete"`/`"interrupted"`/`"error"`), `interrupt` (the interrupt frame when paused),
  `error` (the message when it failed), and `frames` (total seen). It takes the *same* kwargs as
  `iter_event_frames` and forwards them unchanged, so it's a drop-in "I don't need the stream,
  just the result." A chunk-wire counterpart `collect_chunk_frames` (same `TurnResult`, for
  CLI/Jupyter parity) and a sync convenience `run_turn(graph_or_agent, message, *, thread_id=...)`
  — which accepts a compiled graph **or** a prebuilt `LangGraphAgent` like `verify`, builds it if
  needed, and runs under `asyncio.run` — round out the set, mirroring the `averify`/`verify`
  async+sync pairing. Exported from `langstage_core.agui` (alongside `iter_event_frames` /
  `verify`, not top-level). This makes asserting "message X yields answer Y / calls tool Z /
  raises an interrupt" a one-liner in a test, directly serving the largest bug class in this
  repo's history (the turn is wrong/empty).
- **The `complete` / `interrupted` / `error` terminal-outcome rule now lives in ONE place
  (gh #110).** `SessionAdapter._produce` already implemented the exact state machine (`error` ⇒
  `error`; a `complete` after a pending `interrupt` ⇒ `interrupted`; else `complete`), but it was
  welded to the session/queue object, so every non-streaming consumer reimplemented it and could
  drift. Factored a small shared `_terminal_outcome(*, saw_interrupt, saw_error)` in
  `langstage_core.agui`; `_produce`, both new collectors, and `verify()` all call it now, so the
  rule is defined and tested once. (`_produce`'s `cancelled` outcome stays where it is — it's a
  transport concern set on `asyncio.CancelledError`, not part of the frame-driven rule. `verify()`
  is unchanged on every turn a probe actually produces: `outcome == "complete"` iff `not saw_error`
  there, so its `ok` verdict is identical, now derived from the shared rule instead of an inline
  `saw_complete and not saw_error`.)

### Note
- The sibling `langstage` package's `oneturn.py` (`complete_turn` / `run_turn_sync`) is a
  *different* layer — it buffers a `SessionAdapter` for the web server's one-turn HTTP endpoint,
  where the session/queue is needed. These core `collect_*` / `run_turn` helpers are the
  lower-level, **session-free** collectors: they just iterate the `iter_*` mappings, no
  `SessionAdapter`. Use these in tests/evals/scripts; the web endpoint keeps using its own.

## [1.0.25] - 2026-07-23

### Added
- **A keyless, network-free demo agent that exercises *every* rich frame type, not just
  text echo (gh #99).** The shared demo stub (`langstage_core.demo.stub:graph`) behind every
  surface's `--demo` and the README Quick Start only ever emitted `content` frames — a plain
  echo. But `content` is one of *eight* documented frame types, and the rich ones
  (`tool_start` / `tool_end` / `extraction` / `reasoning` / `interrupt`) are the library's
  whole selling point, reachable before now only by hand-building a LangGraph graph *and*
  correctly wiring a real `ToolNode`. That was the single biggest dogfooding friction, and it
  left the largest bug class in this repo's history — rich-frame emission defects (#41/#43/#45,
  #50, #71, #89, #90, #91) — with no keyless regression fixture. `langstage_core.demo.tools:graph`
  (built by `create_tool_demo_agent()`) closes the gap: a real compiled graph — a router node
  plus a real `langgraph.prebuilt.ToolNode` — that routes on a trigger in the user's message:
  a normal message streams a token-by-token `content` reply (as the echo stub does); `"use a
  tool"` calls a built-in keyless `demo_lookup` tool through the `ToolNode`, producing
  `tool_start` → `tool_end` plus an `extraction` frame via the paired built-in
  `DemoLookupExtractor` (pass `demo_extractors()` to the `iter_*` mappings' `extractors=`);
  `"think"` streams a `reasoning` delta (kept separate from the answer); and `"ask me"` raises
  a resumable `interrupt(...)`. Everything is deterministic and offline — the "model" is a local
  fake, no API key, no network, no model call. The tool path deliberately goes through a **real
  `ToolNode`**, because the AG-UI adapter emits streaming `ToolCall*` events only when a tool
  runs through one — a hand-rolled `AIMessage(tool_calls=...)` + manually-appended `ToolMessage`
  arrives via the snapshot path instead (the distinction 1.0.24 / #91 hinges on); the demo bakes
  the correct wiring in so no adopter has to rediscover it. Doubles as a live keyless smoke test
  and the regression fixture the issue asked for.
- **`langstage-agui --demo=tools` serves the rich demo over AG-UI, keyless (gh #99).** `--demo`
  now takes an optional value: bare `--demo` still serves the echo stub unchanged, and
  `--demo=tools` serves `langstage_core.demo.tools:graph`. `--show-config`, `--verify`, and
  mutual-exclusion with `--agent` all behave for both. The README's frame-type list now points
  at a runnable keyless snippet that prints each of `content` / `reasoning` / `tool_start` /
  `tool_end` / `extraction` / `interrupt` / `complete`.

## [1.0.24] - 2026-07-23

### Fixed
- **The snapshot (non-token) path now surfaces tool calls and tool results, not just text
  (gh #91, filed via langstage-cli).** A graph whose messages are produced *without token
  streaming* — a custom node calling `model.invoke()`, a non-streaming provider, a rule-based
  node that returns a finished `AIMessage`/`ToolMessage` (the "Creating Your Own Agent" shape) —
  delivers everything through the final `MessagesSnapshotEvent`, and that branch of **both**
  `iter_*` mappings yielded text only: it filtered to `role=="assistant" and content`, so a
  `ToolMessage` result was dropped, a tool call carried on a snapshot `AIMessage` was never
  surfaced, and a tool-call-only `AIMessage(content="", tool_calls=[...])` was dropped whole —
  rendering a completely empty turn while `verify()`/the exit code reported success. The headline
  "tool-call rendering" feature was invisible on this entire class of agent. (A tool executed via
  a real `ToolNode` emits streaming `ToolCall*` events and rendered fine, which is exactly why the
  gap went unnoticed.) The snapshot branch now emits `tool_calls`/`tool_result` (chunk wire) and
  `tool_start`/`tool_end` (event wire) for the not-yet-streamed messages, via a shared
  `_snapshot_items` walk so the two wires can't drift. Dedup keeps a fully-streamed turn from
  double-rendering: content by message id, tool calls by `tool_call_id` (populated only on the
  streaming `ToolCallStart`), tool results by a new `streamed_result_ids` set. The snapshot
  `tool_result` honors `max_result_len` and runs extractors, at parity with the streaming path.
- **The `extraction` frame no longer defeats `max_result_len` for the generic tool-callout card
  (gh #106).** `GenericToolExtractor` echoes the tool result verbatim as a display card
  (`{"content": <raw>}`), and the `iter_*` mappings ran every extractor on the *raw* content — so
  with `GenericToolExtractor` in `extractors=[...]` the wire carried a 512-char capped `tool_end`
  **and** the full 5000-char blob in the adjacent `extraction` frame, reintroducing exactly the
  unbounded-terminal-output harm `#102`'s cap exists to prevent. A `caps_content` marker now
  distinguishes a display-passthrough extractor (fed the already-truncated result) from a
  *structured* extractor that parses content (still fed the raw content, because capping its input
  would corrupt the parse — the deliberate #102 decision). `GenericToolExtractor` sets it; custom
  structured extractors do not.

### Added
- **`langstage-agui --verify` runs one keyless turn against the agent and reports whether it
  actually works (gh #105).** The already-shipped, exported `verify()` preflight was reachable only
  from Python and wired to no CLI flag, so the question every adopter asks right after `--agent` —
  "did it load *and* produce a turn?" — had no one-command answer (`--show-config` proves the spec
  *resolves*, not that it *runs*; a typo'd `module:attr` or a graph that yields an empty/erroring
  turn both pass `--show-config` and only surface at first chat). `--verify` loads the spec with the
  same clean error handling as the serve path, drives `verify()`, prints the result, and exits
  `0` ok / `1` failed — keyless, so it fits a CI/deploy gate. Documented in the README.

## [1.0.23] - 2026-07-23

### Fixed
- **A malformed numeric env var no longer crashes every entrypoint that resolves config
  (gh #104).** `HostConfig.resolve()` cast environment values with no error handling, so a
  single bad char in `LANGSTAGE_PORT` — `abc`, an unexpanded `"$PORT"`, a copy-paste
  `"localhost:8050"`, a stray `"8050 x"` — raised an uncaught `ValueError` straight out of
  `resolve()` and took down `langstage-agui --show-config`, `--demo`, `--agent ...`, and
  `python -m langstage_core.host`. The **TOML** layer already degraded this exact mistake to a
  one-line `note:` and kept resolving; the **env** layer did not — the two paths were asymmetric
  for the same input, and the module's own docstrings *advertised* that "the numeric env casters
  already emit" the graceful note when in fact they crashed. Now they do: a malformed numeric env
  var is caught, a `note: ignoring malformed <VAR>=<value> (…); using … instead.` goes to stderr
  (ASCII-only, deduped per (var, value)), and resolution continues.

  Crucially the rejected env var falls through to **the layer beneath it, not straight to the
  built-in default**: if a valid `langstage.toml` sets the same key, that value is kept and
  `--show-config` still credits `toml (…)` — a malformed env var can no longer silently clobber a
  user's explicit config with a hardcoded default. This is the root-cause fix for the crash side
  of langstage-hermes #83 and both halves of langstage-jupyter #83 (the value-discard and the
  `--show-config` source-mislabel); those hosts inherit it by resolving through this method and
  can drop their own local env-leniency shims.

## [1.0.22] - 2026-07-19

### Fixed
- **`iter_chunk_frames` ignored `max_result_len`, so the CLI/Jupyter wire emitted tool results
  uncapped with no knob to bound them (gh #102).** The two `iter_*` mappings are documented
  counterparts on the same wire vocabulary, but they disagreed on tool-result truncation:
  `iter_event_frames` capped each result at `max_result_len` (default 500) and appended a
  `…(truncated)` marker, while `iter_chunk_frames` yielded `ToolCallResultEvent.content` straight
  through and **had no `max_result_len` parameter at all** — there was no way to cap it, not even
  by opting in. The asymmetry pointed exactly the wrong way: the README assigns `iter_chunk_frames`
  to *"the CLI and Jupyter surfaces"* ("terminal-friendly chunk dicts"), so the one wire that renders
  directly into a terminal or notebook cell was the one with no bound, while the event wire —
  consumed by the web/VS Code surfaces that already paginate and scroll — was protected. A tool
  returning a large blob (a file read, a search result set, an API dump) therefore flooded the
  render loop unbounded: a 2 MB result is a 2 MB wall of text in the user's terminal, and the
  documented fix (pass `max_result_len=`) raised `TypeError: unexpected keyword argument`.
  `iter_chunk_frames` now takes the same `max_result_len: int = 500` and truncates its
  `tool_result` chunk identically, so the counterparts agree by default and the terminal surfaces
  inherit the protection `SessionAdapter` has always had. The truncation itself moved into one
  shared `_truncate_result()` that both mappings call, rather than a second inline copy of the
  slice-and-mark — this is the third parity gap between the pair (after `extractors` in #92 and
  the `error` frame in #93), and a duplicated implementation is what let the two drift in the
  first place. Two boundaries are deliberate. A result at or under the cap is emitted
  byte-identical with no marker, so every existing CLI/Jupyter consumer of a normal-sized result
  is untouched. And the *extractor* still receives the full, untruncated content: truncation is a
  display concern, so capping the extractor's input would corrupt the payload it parses — the
  event wire feeds its extractor the raw content for the same reason. A non-`str` result is passed
  through rather than sliced (AG-UI types `content` as `str`, so this is defensive — but the chunk
  wire yields `content` as-is where the event wire coerces with `str()`, and slicing a dict would
  raise `TypeError`).

## [1.0.21] - 2026-07-18

### Fixed
- **`langstage-agui --agent <spec>` printed a fake `Serving '<spec>' over AG-UI at <url>` success
  banner and *then* died with a raw traceback whenever the spec was unloadable (gh #100).** The
  banner was printed unconditionally, and the spec was only resolved later — inside `serve()`, which
  is where `load_agent_spec()` runs. So the single most likely CLI mistake (a typo'd module, a
  missing attribute, a nonexistent file) produced the worst possible output: a line claiming the
  server was already up at a URL, immediately followed by an unhandled `ModuleNotFoundError` /
  `AttributeError` / `FileNotFoundError` and exit 1. That is precisely the
  fake-success-then-traceback failure the *same function* already went out of its way to prevent for
  its two sibling cases — the missing `[agui]` extra (`ensure_available()` fails fast to stderr
  "so the user doesn't see a fake success line followed by a traceback") and the no-spec-at-all path
  (`error: no agent spec — ...`, exit 2) — leaving the CLI internally inconsistent on its most
  common error. `main()` now resolves the spec *before* announcing anything: on failure it prints a
  clean one-line `error: could not load agent '<spec>': <reason>` to stderr and exits 2, matching
  both siblings in style and exit code, with no banner and no traceback. `load_agent_spec()` already
  raises descriptive errors, so the reason needs no embellishment. The happy path is unchanged — the
  banner still prints and the server still starts — and because `serve()` accepts either a spec
  string or an already-compiled graph, the pre-loaded graph is handed straight through, so the agent
  module is imported exactly once and no import side effect runs twice.
- **A wrong-*typed* value in `langstage.toml` was accepted verbatim, so `execute_timeout = "300"`
  resolved to the Python `str` `'300'` for a field declared `float` (gh langstage-jupyter #78).**
  `_coerce` coerced `Path` fields only and passed `int`/`float`/`bool` fields through untouched, so a
  syntactically-valid TOML value of the wrong type was never checked against the field it landed in.
  Quoting a number — one of the most common TOML mistakes — was therefore silently accepted, and
  `--show-config` *strips the quotes*, making the misconfiguration visually indistinguishable from a
  correct one in the very tool built to inspect it. The defect then surfaced far from its cause as a
  raw `TypeError: unsupported operand type(s) for +: 'float' and 'str'` the first time a consumer did
  arithmetic on the value, with nothing pointing back at `langstage.toml` or the offending key. The
  prior fix for the identical hazard on the env side (langstage-jupyter #75) added a `_lenient_number`
  wrapper but applied it only to that stage's env casters; its own suggested fix anticipated this
  sibling and named the right layer, so the repair lands here in core where every stage inherits it
  instead of in one host. Numeric fields are now cast to their declared type, so a coercible value is
  coerced (`"300"` -> `300.0`, `"8123"` -> `8123`, and an int literal widens to `1.0` for a `float`
  field). An uncoercible one (`"warm"`, a table, an array) keeps the default *and* the `default`
  source attribution — so `--show-config` can never present an unusable value as a live TOML setting —
  and emits the same one-line `note: ignoring malformed <key>=<value> in <file> (<Error>: ...); using
  default <default> instead.` that the numeric env casters and malformed-syntax TOML (#42) already
  print, now naming the file so the user is pointed back at their config. `bool` is handled
  deliberately: because `bool` subclasses `int` in Python, `temperature = true` would otherwise
  become `1.0`, so a bool supplied *for* a numeric field is treated as malformed — while a genuine
  bool field keeps accepting TOML `true`/`false` untouched. A **fractional** value for an `int`
  field is likewise malformed rather than truncated: `port = 8050.7` silently becoming `8050` would
  be the very defect this fix closes, a wrong-typed value quietly accepted as something the user did
  not write. An integral float (`8123.0`) is unambiguous and still coerces. The note is deduped per
  (key, value), as several surfaces each resolve the config in one process. Correctly-typed configs
  behave exactly as before and emit no notes.

## [1.0.20] - 2026-07-16

### Fixed
- **`SessionAdapter` silently dropped the `extractors` argument, so the web / task-board surface
  could never emit `extraction` frames (gh #96).** `SessionAdapter` is the streaming engine behind
  the web app + task board, yet its `__init__` accepted only `graph` / `max_result_len` and funneled
  everything else into `**_legacy` — so `SessionAdapter(graph=g, extractors=[...])` was accepted
  without error but the extractors were discarded, and `_produce` called `iter_event_frames(...)`
  with no `extractors=`. Every *other* surface (CLI/Jupyter via `iter_chunk_frames`, VS Code via
  `iter_event_frames`) could pass `extractors=[...]` and render skill/memory/todo/`display_inline`
  callouts, but the headline web/task-board surface could not — a documented public feature was
  unreachable there, and the misconfiguration failed silently. `SessionAdapter.__init__` now accepts
  an `extractors=[...]` iterable and forwards it into `iter_event_frames`, so the SSE stream carries
  `extraction` frames for matching tools — parity with `iter_event_frames` / `iter_chunk_frames`
  (same by-tool-name dispatch, same `"*"`-sentinel `GenericToolExtractor` fallback). It stays opt-in:
  with no `extractors` (the default), the stream is unchanged, so existing consumers see no new
  frames. The stale class docstring (which still referenced the retired
  `stream_mode` / `parser_kwargs` / `StreamParser` / `event_to_dict` layer) now documents the real
  `extractors` argument.

## [1.0.19] - 2026-07-14

### Fixed
- **`iter_chunk_frames` had no `extractors` parameter, so the README's "pass `extractors=[...]`
  to the `iter_*` mappings" raised `TypeError` for the CLI/Jupyter surfaces it names (gh #92).**
  The README advertises tool extractors as a feature of *both* `iter_*` mappings and points the
  CLI/Jupyter surfaces at `iter_chunk_frames`, but only `iter_event_frames` accepted
  `extractors` — so a CLI/Jupyter adopter following the docs hit
  `iter_chunk_frames() got an unexpected keyword argument 'extractors'`, and the chunk wire had
  no extraction shape at all (those surfaces couldn't render skill/memory/todo callouts).
  `iter_chunk_frames` now accepts the same `extractors=[...]` iterable as `iter_event_frames`
  (same by-tool-name dispatch, same `"*"`-sentinel `GenericToolExtractor` fallback) and, after
  a tool result, emits an `extraction` chunk
  `{"status": "streaming", "extraction": {"tool_name", "extracted_type", "data"}}` — the
  chunk-wire parallel of the event wire's `extraction` frame. It stays opt-in: with no
  `extractors`, the chunk stream is byte-for-byte unchanged, so existing consumers see no new
  frames.

## [1.0.18] - 2026-07-14

### Fixed
- **A node/agent exception during streaming propagated out of `iter_event_frames` /
  `iter_chunk_frames` and crashed the documented consumer loop, instead of surfacing as the
  advertised terminal `error` frame (gh #93).** Both iterators document — and the README
  Quick start + shipped `examples/fastapi_websocket.py` rely on — an `error` frame among
  their outputs (`content` / `tool_start` / `tool_end` / `interrupt` / `complete` /
  `error`). But the `error` frame was only reachable from a `RunErrorEvent`; the *common*
  failure path — a node/tool/model call that raises — propagated straight out of
  `agent.run()`, so a consumer written exactly as the docs show (a bare `async for` relying
  on the `error` frame) crashed with the raw exception and never got the frame. The two
  other consumers of `agent.run()` already hardened this (`build_app` emits a terminal
  `RUN_ERROR`; `SessionAdapter._produce` wraps the iterator) — but the in-process iterators
  the Quick start / WebSocket example hand to CLI/Jupyter/WebSocket adopters were left bare.
  Both iterators now wrap the `agent.run(...)` loop in `try/except`, yielding the terminal
  `error` frame (`{"type": "error", "error": "..."}` / `{"status": "error", "error":
  "..."}`) — the same treatment `build_app.gen()` applies — so a real agent error renders a
  failed turn instead of killing the consumer. The frame vocabulary is unchanged.

## [1.0.17] - 2026-07-12

### Fixed
- **A mixed-mode turn silently dropped a finished (non-streamed) `AIMessage` from a
  later node once any earlier node had streamed tokens (gh #89).** Both
  `iter_event_frames` and `iter_chunk_frames` rendered non-streamed assistant content
  only through the final `MessagesSnapshotEvent`, but that branch was guarded by
  `and not streamed_text` — so a turn that streamed one node (e.g. a model node) and
  then appended a finished message from a later node (a guardrail / disclaimer /
  formatting / fallback node) suppressed the *entire* snapshot, and the later node's
  reply vanished with no error. The guard is replaced by per-message dedup keyed on
  the streamed message ids: the snapshot now always runs and emits the assistant
  messages it did **not** already stream token-by-token. A fully-streamed turn still
  emits nothing extra (every id is deduped, so no duplication) and a fully-snapshot
  turn is unchanged (nothing was streamed, so all messages emit) — preserving the
  #67 history-slicing and #43 node-mapping behavior.
- **`GenericToolExtractor` was unreachable on the 1.0 AG-UI path — a shipped, top-level
  importable public built-in that was 100% dead code (gh #90).** Its documented purpose
  is to be the *fallback* extractor (any tool without a dedicated extractor emits a
  generic `tool_call` extraction), but its only registration API —
  `StreamParser(default_extractor=…)` — was removed in 1.0 (ADR 0003), and the
  replacement `iter_event_frames(…, extractors=[…])` dispatched **strictly by
  `tool_name`**, so its `"*"` sentinel was registered under the literal key `"*"` and no
  real tool name ever matched it. `iter_event_frames` now treats an extractor whose
  `tool_name == "*"` as the fallback (consulted when no specific extractor matches), so
  the built-in fires through the supported public API; a dedicated extractor still wins
  over the fallback. Also refreshed the stale docstrings that pointed at the removed
  `StreamParser` API — the public `ToolExtractor` protocol Example (`base.py`) and
  `GenericToolExtractor` (`builtins.py`) now document the `extractors=[…]` path.

## [1.0.16] - 2026-07-10

### Fixed
- **`examples/fastapi_websocket.py` resumed an interrupt with a bare list, crashing a real
  HITL agent — the same bug as #85, missed in the FastAPI example (gh #87).** The `decision`
  branch forwarded the client's bare `decisions` list straight into `resume=[...]`, but the
  HITL middleware reads it back as `interrupt(...)["decisions"]`, so a real deepagents/HITL
  agent (the swap-in the example invites) crashed with `TypeError: list indices must be
  integers or slices, not str` on the first Approve. The example now wraps it in the decision
  envelope `resume={"decisions": [...]}`, matching the README, the Jupyter example (#86), and
  `create_resume_input`. The `tests/test_example_docs.py` guard now scans **every**
  `examples/*.py` file too — not just the notebook + README — so no shipped surface can
  regress to the bare list again.

## [1.0.15] - 2026-07-09

### Fixed
- **The Jupyter example notebook's HITL resume snippet used a bare list, which crashes the
  middleware (gh #85).** `examples/jupyter_example.ipynb` showed
  `resume=[{"type": "approve"}]`, but `HumanInTheLoopMiddleware` reads the resumed value back
  as `interrupt(...)["decisions"]` — a list has no `["decisions"]`, so copy-pasting the
  snippet crashed the resume. Corrected to the **decision envelope**
  `resume={"decisions": [{"type": "approve"}]}` (the same shape `create_resume_input(...)`
  builds and the README already documents). A new `tests/test_example_docs.py` pins the
  notebook and README to the envelope form — it forbids a bare-list `resume=` in either and
  ties the documented literal to `create_resume_input(...)`'s output so they can't drift.

## [1.0.14] - 2026-07-08

### Changed
- **`HostConfig.describe()` is now the single, complete config diagnostic — it renders the
  `[configurable]` table too (new `configurable=` arg).** A run of nightly issues
  (#55/#57/#61/#64/#66) were all "config-diagnostic drift": each surface assembled its own
  `--show-config` / interactive `/config` output (base dump + a separately-rendered
  `[configurable]` table + footer tweaks), and the pieces drifted. Folding the
  `[configurable]` table into `describe()` means the whole diagnostic comes from one method,
  so every surface's two config views render identically by construction. Purely additive —
  callers that don't pass `configurable` are unchanged.

## [1.0.13] - 2026-07-08

### Fixed
- **`create_resume_input(...)` no longer double-wraps under `iter_event_frames` /
  `iter_chunk_frames`, crashing HITL resume (gh #82).** `create_resume_input()` returns a
  LangGraph `Command`, but the iterators' `resume=` builds the `Command` themselves
  (`forwarded_props.command.resume`), so passing the helper's output produced
  `Command(resume=Command(...))` — the graph's `interrupt()` then returned the inner
  `Command` and a realistic HITL node crashed with `'Command' object is not subscriptable`.
  The iterators now unwrap a `Command`'s `.resume`, so **both** `resume=create_resume_input(...)`
  and a raw `resume={"decisions": [...]}` converge on a single, correct wrap. Also refreshed
  `create_resume_input`'s docstring (its examples used the removed `StreamParser.parse` API).
- **A malformed TOML config is no longer listed as "read", and its warning prints once
  (gh langstage-hermes #61).** `_read_toml` caught a `TOMLDecodeError`, returned `{}`, and
  warned — but `load_toml_config` still appended the file to `sources` (so `--show-config`
  printed `TOML read from: <it>`, contradicting the "ignoring malformed" note), and the note
  was emitted twice (the loader plus the source-labeling re-read each warned). `_read_toml`
  now records malformed paths and dedupes the warning; loaders skip listing a malformed file.

## [1.0.12] - 2026-07-07

### Fixed
- **`from langstage_core import SessionAdapter` / `Session` now works, matching the docs
  (gh #80).** The README's "What's in the box" says everything is re-exported at the top
  level (except the AG-UI helpers), and lists `SessionAdapter` / `Session` — but they were
  only importable from `langstage_core.adapters`, so the documented top-level import raised
  `ImportError`. They are now re-exported at the top level (still available under
  `langstage_core.adapters` too). Also corrected the package docstring, which had
  miscategorized `SessionAdapter` under the `agui` bullet (implying
  `from langstage_core.agui import SessionAdapter`, which never worked).

## [1.0.11] - 2026-07-06

### Docs
- **The README no longer over-promises the `langgraph_stream_parser` compat shim (gh
  #77).** The banner and migration guide said the old import "still works via a compat
  shim," which read as "install `langstage-core` and `import langgraph_stream_parser`
  keeps working" — but the shim is a **separate** distribution (`langgraph-stream-parser`
  1.0, which re-exports `langstage_core`) that `langstage-core` neither bundles nor
  depends on (depending on it would be circular). So a fresh `pip install langstage-core`
  hit `ModuleNotFoundError` on the old import. The README now states the old name keeps
  working only while the separate `langgraph-stream-parser` package stays installed
  (kept on an in-place upgrade; add it explicitly, or just `import langstage_core`, on a
  fresh install).

## [1.0.10] - 2026-07-06

### Fixed
- **The standard HumanInterrupt shape no longer crashes an interrupt turn (gh
  langstage-vscode #40).** The `on_interrupt` handler in `iter_event_frames` /
  `iter_chunk_frames` assumed the interrupt value was a dict keyed `action_requests`
  and did `payload.get(...)` — so the **list of HumanInterrupt dicts** that deepagents /
  langchain HITL actually emit (`[{"action_request": {...}, "config": {...}}, ...]`)
  raised `'list' object has no attribute 'get'` and failed the turn, while any other
  plain dict returned an empty `action_requests` (the advertised field never populated).
  A shared `_normalize_interrupt` now handles all three shapes: the HumanInterrupt list
  (unwrapping each `action_request` and deriving `allowed_decisions` from the `config`
  flags), our own `action_requests`-keyed dict, and a plain dict (surfaced as a single
  action request). Fixes the crash on every surface — the vscode sidecar and web
  (`iter_event_frames`) and the cli (which reads `frame["interrupt"]["action_requests"]`
  off `iter_chunk_frames` and would otherwise hit the same list crash).

## [1.0.9] - 2026-07-05

### Fixed
- **A relative workspace no longer doubles the agent's working directory (gh #66).**
  `apply_workspace()` stored the root *as given* (e.g. `./ws`), and `workspace_root()`
  re-`.resolve()`d it against the current cwd on every call — so a surface that then
  chdir'd *into* the workspace (the cli, or the web app's `_enter_workspace`) made every
  subsequent `workspace_root()` re-resolve `./ws` against the new cwd and double it to
  `ws/ws`, splitting the agent's working directory from the file browser root. The root
  is now resolved to absolute **once**, at apply time, so `workspace_root()` is idempotent
  across a later chdir. Absolute and default (`.`) workspaces were unaffected.

### Docs
- **The two shipped `examples/` run again on a clean 1.0 install (gh #75).** Both still
  imported symbols removed in the 1.0 rename (`FastAPIAdapter`, `JupyterDisplay`,
  `StreamParser`) and crashed on the first line. `fastapi_websocket.py` now drives
  `agui.iter_event_frames` over the WebSocket (the same typed frames the web UI renders;
  the HTML client is unchanged), and `jupyter_example.ipynb` was rewritten around
  `agui.iter_chunk_frames`. Both run keyless. Also dropped the stale `FastAPIAdapter`
  cross-reference in `SessionAdapter`'s docstring.

## [1.0.8] - 2026-07-04

### Fixed
- **`iter_event_frames` / `iter_chunk_frames` now emit the advertised `reasoning`
  frame (gh #71).** The mappers had no branch for the `Reasoning*` / `Thinking*`
  AG-UI events that `ag-ui-langgraph` emits for reasoning-capable models (Anthropic
  extended thinking, OpenAI o-series, DeepSeek R1, Qwen, xAI, …), so the model's
  chain-of-thought was silently dropped and no `reasoning` frame was ever produced —
  contradicting the README. `ReasoningMessageContentEvent` /
  `ThinkingTextMessageContentEvent` now map to `{"type": "reasoning", "content": …}`
  (event wire) / `{"status": "streaming", "reasoning": …}` (chunk wire), kept
  separate from the `content` answer so renderers can show or collapse the thinking.

## [1.0.7] - 2026-07-03

### Added
- **`apply_workspace()` / `workspace_root()` — the single workspace source of truth
  (ADR 0005).** Core resolved `workspace_root` (the value) but never *applied* it, so
  each surface invented its own mechanism (cli `chdir` / vscode env-push / hermes
  backend-arg / jupyter global-mutate / web hand-sync) and three drifted into a
  workspace bug. `apply_workspace(root, *, chdir=False)` — called once after
  `HostConfig.resolve()` — ensures the dir exists, records it as the process's active
  workspace, and publishes it as `LANGSTAGE_WORKSPACE_ROOT` (plus the legacy
  `DEEPAGENT_WORKSPACE_ROOT`); `workspace_root()` is the one accessor tools and
  surfaces read instead of a private global. `chdir` is opt-in (single-process
  surfaces only). Surfaces migrate to it incrementally; a bring-your-own graph's
  tools opt in by reading `workspace_root()`. Both exported top-level.

## [1.0.6] - 2026-07-03

### Added
- **`langstage_core.agui.verify` / `averify` — a shared live-preflight primitive
  (ADR 0004).** Runs ONE real turn through the AG-UI adapter (`iter_event_frames`)
  and returns a structured `VerifyResult` (`ok` / `saw_complete` / `saw_error` /
  `error_message` / `content_chars`). `ok` is True only if the turn *completed
  with no error* — a missing API key, broken tool, or bad state schema fails here,
  where a static "imports fine / loads / key is set" check gives a false green.
  Accepts a compiled graph or an already-built `LangGraphAgent`; `verify()` is the
  sync wrapper, `averify()` the coroutine. This is the primitive each surface's
  `doctor`/`check`/`selfcheck`/`health` was reinventing (vscode `--selfcheck` and
  hermes `verify` drive a real turn; cli/web/jupyter assert static state and gave
  the false-green class in the backlog). Surfaces adopt it incrementally.

## [1.0.5] - 2026-07-03

### Fixed
- **Non-token-streamed agents re-rendered the entire conversation history every turn
  (gh #67).** `iter_event_frames` / `iter_chunk_frames` emit content from the final
  `MessagesSnapshotEvent` for agents that don't token-stream — but that snapshot is
  the *full* thread (prior turns come from the checkpointer), so every turn re-emitted
  turns 1..n. Now the snapshot is sliced to the messages after the last user message,
  so only the current turn's replies are emitted. (Regression from the 1.0.4 snapshot
  node-mapping.)

### Known limitation
- **A failed tool call can still render as `status="success"` (gh #55).** The AG-UI
  `ToolCallResultEvent` carries no status field and its `raw_event` is empty, so the
  `ToolMessage`'s error status is dropped before the mapping sees it. `iter_event_frames`
  now flags a `tool_end` as an error when a preceding `on_tool_error` RawEvent named the
  same tool (best-effort, covers raised-and-handled errors), but a tool that *returns* a
  `status="error"` message without raising is still shown as success — that needs the
  status carried on the AG-UI event upstream (tracked in ag-ui-langgraph).

## [1.0.4] - 2026-07-02

### Fixed
- **AG-UI frames now carry the real langgraph node instead of a fixed `"agent"`
  (gh #43, langstage-cli).** `iter_event_frames` / `iter_chunk_frames` hardcoded
  `node="agent"` on every content/tool frame, so a multi-node graph's output was
  indistinguishable — the CLI (which starts a new marker on a node change) rendered
  two nodes' messages as one unreadable run-on. The node now tracks
  `StepStartedEvent.step_name`, and the non-streaming `MessagesSnapshotEvent` path
  maps trailing assistant messages back to their steps. Single-node graphs (node
  `"agent"`) are unchanged; renderers can now separate per-node output.

## [1.0.3] - 2026-07-02

### Fixed
- **A malformed config file crashed every entrypoint (gh #42).** Config resolves at
  import time on several surfaces, so a raw `TOMLDecodeError` (or an unreadable
  file) from a broken `langstage.toml` killed `--version` / `--help` / `--demo`,
  the `deepagent-*` aliases, the server extension, and even `import langstage_jupyter`
  — not just the command that needed the config. `_read_toml` now skips a bad file
  with a visible one-line notice and falls back to environment + defaults.

### Added
- **Legacy `deepagents.toml` now emits a deprecation notice (gh #25).** The legacy
  `DEEPAGENT_*` env vars already warned on use, but a legacy `deepagents.toml`
  (project) or `~/.deepagents/config.toml` (global) resolved silently. It now raises
  a once-per-file `DeprecationWarning` + a visible stderr notice (same
  `LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1` opt-out and pytest suppression as the env
  notice), closing the advertised-parity gap.

## [1.0.2] - 2026-07-02

### Fixed
- **`langstage-agui --show-config` ignored `--agent` (gh #60).** The `--show-config`
  branch resolved `HostConfig` with only the host/port overrides — the `--agent`
  spec was applied later, after the early return — so `--show-config --agent X`
  reported `agent_spec = None` while `serve()` honored the flag (advertised ≠
  honored). The `--agent`/`--host`/`--port` overrides are now resolved once, before
  the branch, so the shown config matches the real run. Regression-tested.

## [1.0.1] - 2026-07-02

### Fixed
- **README documented the removed event-layer API.** The 1.0.0 README was only
  name-renamed, so its body still showed `StreamParser`, `langstage_core.events`,
  `event_to_dict`, `stream_graph_updates`, and the removed display adapters — every
  example failed to import against the 1.0 wheel (caught by dogfooding the published
  docs). Rewritten as a slim quickstart for the actual 1.0 surface (host + AG-UI
  bridge + task engine + resume helpers), with a migration table. Docs-only.

## [1.0.0] - 2026-07-02

### Changed
- **Renamed `langgraph-stream-parser` → `langstage-core`; retired the event layer
  (ADR 0002 / 0003).** AG-UI is now the sole streaming wire across the LangStage
  family. Removed `StreamParser`, `events.py`, `event_to_dict`, `compat.py`,
  `handlers/`, the message/interrupt extractors, and the display adapters
  (`CLIAdapter`/`PrintAdapter`/`FastAPIAdapter`/`JupyterDisplay`). `SessionAdapter`
  is now AG-UI-only.
- **Kept:** `load_agent_spec`, `HostConfig`, `Workspace`, `prepare_agent_input`,
  `create_resume_input`, the `tasks` engine, `extractors` (`ToolExtractor` +
  built-ins), and the two shared mappings `agui.iter_event_frames` /
  `agui.iter_chunk_frames`.
- A `langgraph-stream-parser` 1.0.0 compat shim re-exports `langstage_core` under
  the old import name with a `DeprecationWarning`.

## [0.6.13] - 2026-06-27

### Fixed
- **`stream_mode="auto"` silently dropped all content for a pure `messages`-mode
  stream.** Auto-detect only recognized multi-mode `(mode, data)` tuples and v2
  parts; a `messages` stream's first chunk is `(message, metadata)`, which fell
  through to `"updates"` and matched none of its chunks — rendering an empty turn
  with no error (the explicit `stream_mode="messages"` rendered fine). `_peek_and_detect`
  / `_apeek_and_detect` now recognize a token-streaming first chunk and return
  `"messages"`. (Found by the dogfood routine, gh #41.)

## [0.6.12] - 2026-06-26

### Fixed
- **`langstage-agui --show-config` advertised `workspace_root` / `debug` /
  `title`, which the AG-UI server ignores.** The server consumes only
  `agent_spec`, `host`, and `port`, but `--show-config` printed all six inherited
  `HostConfig` rows with confident env-var sources — the exact inconsistency
  0.6.11's `describe(omit_keys=…)` was added to fix for sibling surfaces, never
  applied to the agui CLI. It now passes `omit_keys=["workspace_root", "debug",
  "title"]`, matching the stdio sidecar and JupyterLab launcher. (gh #39)

## [0.6.11] - 2026-06-25

### Fixed
- **`CLIAdapter` crashed with `UnicodeEncodeError` on a default Windows (cp1252)
  console.** It prints `⏺`, the braille spinner, box-drawing, and `✓ ✗ ⚠`
  without ever reconfiguring stdio, so the first styled line raised before any
  output appeared (and `use_colors=False` didn't help — that strips ANSI, not
  the glyphs). `CLIAdapter.run()` now reconfigures stdout/stderr to UTF-8
  (`errors="replace"`) up front, mirroring the `langstage-cli` entry point. The
  last surface that wasn't cp1252-safe. (gh #37)

### Added
- **`HostConfig.describe(omit_keys=[...])`** — hide inherited keys a stage
  doesn't actually honor, so `--show-config` never advertises an env var (with a
  confident source attribution) that has zero effect on that surface. Used by
  the stdio sidecar and the JupyterLab launcher to drop the web-only
  `host`/`port` rows. (gh: langstage-jupyter #30, langstage-vscode #14)

## [0.6.10] - 2026-06-22

### Fixed
- **The stub agent's missing-deps error over-claimed and gave no remedy.** When
  `langgraph` isn't installed, `create_stub_agent()` raised "needs langgraph +
  langchain-core (every deep-agent surface already installs them)" — which is
  false for a base `langstage-vscode` install, and never told the user how to
  fix it. The message now names the actual remedy: the lightweight
  `pip install "langgraph-stream-parser[stub]"` extra (or `pip install langgraph`).
  (Found by the dogfood routine.)
- **`LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1` now silences the `DeprecationWarning`
  too**, not just the stderr notice. Previously a suppressed run could still leak
  a raw `DeprecationWarning` (e.g. into a VS Code output channel), making the
  "set … to silence" hint only half-true. Setting the env var now opts out of
  every legacy-env deprecation signal. (Found by the dogfood routine.)

## [0.6.9] - 2026-06-22

### Changed
- **Legacy `DEEPAGENT_*` env vars now emit a *visible* one-line deprecation
  notice to stderr**, not just a `DeprecationWarning` (which Python's default
  filter silently swallows, so CLI users never saw the nudge). Fires once per
  variable, from the shared resolver — so every surface (web, CLI, JupyterLab,
  VS Code, Hermes) gets it for free, no per-surface change. ASCII-only
  (cp1252-safe); suppressed under pytest and via
  `LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1`. (Found by the dogfood routine: the
  canonical-vs-legacy contract advertised a warning the runtime never showed.)

## [0.6.8] - 2026-06-22

### Fixed
- **The keyless Quick Start failed *silently*.** `create_stub_agent()` compiled
  the echo demo with a default `MemorySaver`, so streaming it the way the README
  Quick Start shows — with no `config`/`thread_id` — raised a "checkpointer
  requires thread_id" error that the parser surfaced as a lone `ErrorEvent` and a
  **blank reply (exit 0)** for anyone copy-pasting the docs. The stub now compiles
  **without** a checkpointer by default, so the documented config-free path just
  works; pass `checkpointer=...` (or rely on AG-UI's auto-attach) when you want
  threaded state. (Found by the dogfood routine.)
- **A UTF-8 BOM in `langstage.toml` crashed config loading on Windows.** Notepad
  and PowerShell's `Out-File -Encoding utf8` both write a BOM by default;
  `tomllib.load()` (binary) rejects it with a cryptic "Invalid statement (at line
  1, column 1)". Because surfaces resolve config eagerly, this could brick a whole
  stage at import (notably `langstage-jupyter`). `_read_toml` now decodes with
  `utf-8-sig`, stripping the BOM. Fixes every stage at once. (Found by the dogfood
  routine.)

### Docs
- `python -m langgraph_stream_parser.host --help` no longer prints an em-dash that
  rendered as mojibake (`�`) on a default Windows (cp1252) console.
- README: `langstage-agui` serves at `http://localhost:8050` (the host-config
  default), not the stale `127.0.0.1:8000`.

## [0.6.7] - 2026-06-21

### Fixed
- **`tool_end` reported `name="unknown"`** when a `ToolMessage` lacked a `name`,
  even though the correlated `tool_start` (same id) carried it. The end event now
  backfills the name from the tracked start event. (Found via langstage-vscode.)
- **`langstage-agui --demo` without the `[agui]` extra** printed a success-looking
  `Serving … at http://…` banner and *then* a traceback. It now pre-flights the
  AG-UI deps and exits 2 with a clean install hint (no fake banner). New
  `agui.ensure_available()`.
- **`python -m langgraph_stream_parser.host` ignored all args** — `--help` was a
  silent no-op. Added a tiny argparse so `-h/--help` works and unknown flags error.

### Added
- A lightweight **`[stub]` extra** (`langgraph` only) for the keyless stub agent /
  `--demo` path, so trying the echo demo no longer drags in the full `[demo]` /
  deepagents ML stack (~50 pkgs). README + `demo/__init__` docstring now point the
  keyless path at `[stub]`.

## [0.6.6] - 2026-06-21

### Fixed
- **Dict-form messages rendered nothing** — breaking the "runs any CompiledGraph"
  promise. LangGraph's `add_messages` reducer accepts dict messages
  (`{"role": "assistant", "content": ...}` or `{"type": "ai", ...}`), but the
  updates handler dispatched on the message's class name (`"dict"`), which matched
  no branch, so a node returning a dict message produced no `ContentEvent`. The
  handler now coerces dict messages to LangChain Message objects
  (`convert_to_messages`) before dispatch; existing Message objects pass through
  unchanged. Fixes blank output for dict-returning agents in langstage-cli /
  langstage (web) / langstage-vscode. (gh #-dogfood)

## [0.6.5] - 2026-06-20

### Fixed
- **`langstage-agui --show-config` disagreed with the real bind.** It printed
  HostConfig's `localhost:8050` while the server bound argparse's
  `127.0.0.1:8000`. host/port now come from the resolved HostConfig (so env /
  `langstage.toml` host/port work), with `--host`/`--port` applied as overrides —
  so `--show-config` reflects exactly what `serve()` binds. (Default bind is now
  the host-config default, `localhost:8050`, instead of `127.0.0.1:8000`.)

### Added
- `langstage-agui --version` prints the package version (it had no version flag).

### Docs
- README: corrected the "Zero Dependencies" claim (`langchain-core` is a hard
  runtime dep) and made the Quick Start runnable — it parses *your* compiled
  graph (needs `langgraph`), with a keyless `create_stub_agent()` path via the
  `[demo]` extra.

## [0.6.4] - 2026-06-20

### Fixed
- **Dual-mode (`["updates","messages"]`) dropped content from non-token-streaming
  nodes.** The updates handler hard-suppressed all content, so a `CompiledGraph`
  whose node returns a finished `AIMessage` (rule-based / router / retrieval
  agents, or any LLM call made outside a token-streaming node) produced **no
  `ContentEvent` at all** — chat surfaces rendered a blank reply. The updates
  handler now emits a finished message's content as a **fallback** when the
  messages/token stream did not already deliver it, deduplicated by message id
  (falling back to node name when a message has no id) so token-streamed content
  is never doubled. Fixes blank replies in `langstage` (web) and
  `langstage-vscode` for non-streaming agents.

### Changed
- `__version__` is now derived from the installed distribution metadata
  (`importlib.metadata.version`) instead of a hard-coded constant, so it can
  never drift out of sync with `pyproject.toml`. No behavior change — it still
  reports the installed version.

## [0.6.3] - 2026-06-18

### Fixed
- `create_default_agent` now slugifies the agent `name` for the LLM message
  `name` field, and its default name is space-free (`"Deep Agent"` -> `"deep-agent"`).
  OpenAI-compatible providers (incl. OpenRouter) require that field to match
  `^[^\s<|\/>]+$`, so an agent named with a space hit a cryptic `400` on the
  second turn. Names with unsafe characters are now slugified with a warning, so
  any human-readable name works. (Surfaced via langstage-jupyter #23, whose shipped
  default `"Default Agent"` tripped it.)

## [0.6.2] - 2026-06-16

### Fixed
- **Declare `langchain-core` as a dependency.** `langgraph_stream_parser.tasks.tools`
  imports `langchain_core.tools` at module top level (reached by a plain
  `import langgraph_stream_parser`), but the package declared `dependencies = []` — so a
  bare `pip install langgraph-stream-parser` failed with `ModuleNotFoundError:
  langchain_core`. Now a hard dependency. (Found by rolling the minimal-install CI guard
  across the family; `langgraph` stays optional — its imports are lazy.)

### CI
- Added a `minimal-install` job (install with no extras + import smoke) to guard against
  undeclared dependencies going forward.

## [0.6.1] - 2026-06-15

### Changed
- Sharpened the `start_async_task` tool description so models reliably prefer
  async delegation for long-running / parallelizable / background work (instead
  of doing it inline or via a blocking sub-agent). Docstring-only; no behavior
  change.

## [0.6.0] - 2026-06-14

Make delegated tasks **interactive** — a per-task event transcript, agent
self-delegation tools, and talk-back. Builds on the 0.5.0 task engine.

### Added
- **Event transcript per task.** `TaskRunner` now *streams* each run's events to
  the store as they arrive (instead of draining at the end), via two new
  `TaskStore` methods — `append_events` / `get_events`. This is what makes a
  per-task detail/replay view (and live-tailing) possible. `InMemoryTaskStore`
  implements both.
- **`TASK_TOOLS`** (`langgraph_stream_parser.tasks.tools`) — five agent-facing
  delegation tools (`start_async_task`, `check_async_task`, `list_async_tasks`,
  `update_async_task`, `cancel_async_task`) so an agent can spawn async copies of
  itself against the local runner (no remote server). Mirrors the deepagents
  async-subagent contract.
- **`current_task_id`** context var — set while a task's agent runs, so a
  sub-task it spawns is automatically linked to it (`parent_id`), forming a tree.
- **`TaskRunner.followup(task_id, message)`** — send a follow-up to a finished
  task; continues its thread (it remembers prior work) and re-runs in the
  background. `TaskRunner.store` property for read access from tools.

### Notes
- Additive; the 0.5.0 API is unchanged. The streaming rewrite preserves the
  same terminal-outcome → board-state mapping and cancel/shutdown semantics
  (regression-tested).

## [0.5.0] - 2026-06-14

An **async task-delegation engine** — the reusable core behind a "delegate a
task, it runs in the background, track it on a board" surface. Single-process,
dependency-free; surfaces provide a concrete store.

### Added
- **`langgraph_stream_parser.tasks`** — `TaskRunner` (an asyncio worker pool that
  drives the shared `SessionAdapter`), a `TaskStore` protocol with a dependency-free
  `InMemoryTaskStore` reference impl, the `Task` record + `TaskState` machine
  (`queued → ongoing → review_needed → done/failed/cancelled`), and
  `set_runner`/`get_runner` so agent tools can reach the runner.
  `enqueue()` returns a `task_id` immediately (non-blocking); workers run each
  task as its own session and transition it by the run's outcome; `cancel`,
  `resume` (HITL), and `retry` round out the controls; orphaned `ongoing` tasks
  are requeued on `start()`.
- **`Session.outcome`** (+ `Session.interrupt` / `Session.error`) — a typed
  terminal-outcome signal set by the session adapter
  (`complete | interrupted | error | cancelled`). Headless consumers read this
  instead of re-inspecting the event stream. Correctly distinguishes a HITL
  pause from completion (the parser always emits a trailing `CompleteEvent`,
  even after an interrupt).

### Fixed
- `__version__` was stale at `0.2.1` (pyproject was already ahead); now `0.5.0`.

### Notes
- Additive and dependency-free. The engine depends only on the `TaskStore`
  protocol and the public `SessionAdapter` surface, so a surface can back it
  with any store (in-memory, SQLite, …) and later graduate to a remote Agent
  Protocol server without changing the engine.

## [0.4.1] - 2026-06-14

AG-UI bridge robustness — an edge-case audit (`tests/test_agui_matrix.py`, run against purpose-built tool-calling / interrupting / erroring / checkpointer-less agents) surfaced three real issues, now fixed:

### Fixed
- **Graphs compiled without a checkpointer no longer hard-crash.** The AG-UI adapter calls `graph.aget_state()`, which raises `No checkpointer set` — common for plain user graphs. `build_agent()` now auto-attaches an in-memory checkpointer when the graph lacks one (AG-UI needs threaded state for interrupts/resume regardless).
- **Agent exceptions mid-run now emit a terminal `RUN_ERROR`** instead of silently killing the stream / unhandled 500. The endpoint is now a resilient wrapper around the agent run.

### Verified (was previously only claimed)
- Tool calls map to `TOOL_CALL_START`/`ARGS`/`END` + `TOOL_CALL_RESULT`.
- Interrupts surface as a `CUSTOM` `on_interrupt` event; resume via `forwardedProps.command.resume` continues the run (HITL round-trip).
- Multi-turn thread state persists; concurrent requests are isolated (per-request agent clone). Note: AG-UI clients must use **unique message ids** per turn — the adapter dedupes by id.

## [0.4.0] - 2026-06-14

Adopt **AG-UI** for the wire (see `docs/adr/0001-adopt-ag-ui-for-the-wire.md`).

### Added
- **`langgraph_stream_parser.agui`** — a bridge that serves any LangGraph agent over the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) using the official MIT `ag-ui-langgraph` adapter. `build_agent()`, `add_agui_endpoint()`, `build_app()`, `serve()`.
- **`langstage-agui` console script** (and `python -m langgraph_stream_parser.agui`) — `langstage-agui --agent <spec>` / `--demo` serves any agent spec over AG-UI. The agent spec resolves through the shared host config chain.
- **`[agui]` extra** — `ag-ui-langgraph[fastapi]` + `uvicorn`.

### Changed
- **Requires Python ≥ 3.11** (was ≥ 3.10). The AG-UI adapter stack (`ag-ui-langgraph` → modern `langchain`/`langgraph`) does not stream correctly under 3.10, and the whole LangStage surface family already requires 3.11+. Dropped 3.10 from CI and classifiers.

### Notes
- This is additive. The typed-event layer (`StreamParser`, `event_to_dict`, the extractors) is unchanged and still backs the existing surfaces; per the ADR it becomes legacy/optional and surfaces migrate to AG-UI incrementally. The host layer (`HostConfig`, `load_agent_spec`, demo) is the durable core.

## [0.3.0] - 2026-06-10

The host family is renamed to **LangStage** ("every stage for your LangGraph agent"); this package keeps its name as the shared core.

### Added
- **`LANGSTAGE_*` is the canonical config vocabulary**: `LANGSTAGE_AGENT_SPEC` etc. env vars, project `langstage.toml`, global `~/.langstage/config.toml`, and `LANGSTAGE_CONFIG_HOME`. The legacy names (`DEEPAGENT_*`, `deepagents.toml`, `~/.deepagents/config.toml`, `DEEPAGENTS_CONFIG_HOME`) still resolve everywhere as deprecated fallbacks — canonical wins when both are set; legacy env use emits a once-per-var `DeprecationWarning`. Moving the global config out of `~/.deepagents/` also exits the schema collision with LangChain's dcode, which owns that directory.
- Host subclasses may declare either spelling in their `_ENV` maps; both names resolve (`_env_pair` derivation), so downstream hosts need no immediate change.

### Changed
- `HostConfig.title` default: `"Deep Agent"` → `"LangStage"`.
- `describe()` shows both vocabularies per field (`env: LANGSTAGE_X (legacy DEEPAGENT_X)`).
- README family table updated to the LangStage package names.

## [0.2.2] - 2026-06-10

### Added
- `demo.create_stub_agent()` / `langgraph_stream_parser.demo.stub:graph` — the keyless, deterministic echo agent behind every surface's `--demo` mode. Lazy imports; base install stays dependency-free.

## [0.2.1] - 2026-06-08

### Added
- `GenericToolExtractor` + `default_extractor` fallback for unknown tools (#16).
- Tag-driven Release workflow (#17).

## [0.2.0] - 2026-06-02

Repositions langgraph-stream-parser as the **shared runtime substrate** for the deep-agent host family (`cowork-dash`, `deepagent-lab`, `deepagent-code`, `deepagent-vscode`).

### Added
- **`host/` submodule** — shared host conventions:
  - `load_agent_spec("path.py:var" | "module:var")` — strict agent-spec loader (the `:object` suffix is required).
  - `HostConfig` — layered config resolver: `defaults < deepagents.toml < DEEPAGENT_* env < overrides`, with per-field source tracking. Subclass and extend the `_ENV` / `_TOML` maps (merged across the MRO) to add host-specific keys. `DEEPAGENT_AGENT_SPEC` is the canonical agent-spec env var.
  - `load_toml_config()` — loads + deep-merges global `~/.deepagents/config.toml` (override dir via `DEEPAGENTS_CONFIG_HOME`) and the nearest project `deepagents.toml`.
  - `Workspace` — workspace-root wrapper with traversal-safe `subpath()`.
  - `python -m langgraph_stream_parser.host` (and `HostConfig.describe()`) — prints each resolved value, its source, and the env var / TOML key that sets it.
- **`adapters.SessionAdapter`** — session-scoped streaming for web hosts: per-session event queue, cancellation, side-channel `push_event()`, and persistent SSE that survives client reconnects.
- **`demo.create_default_agent()`** — shared filesystem-backed default agent factory (behind the `[demo]` extra; lazy-imports `deepagents`).
- **Four built-in extractors** for the agentskills.io / Hermes pattern, wired into the default set so every host gets them through `compat`: `SkillManageExtractor` (`skill_manage`), `SkillViewExtractor` (`skill_view`), `CompressionExtractor` (`__compression__`), `MemoryExtractor` (`memory`).
- `event_to_dict(event, *, max_result_len=...)` — lets hosts drop bespoke serializer shims.

### Fixed
- `skip_tools` previously suppressed a tool's **extractor** as well as its lifecycle events, silently dropping `todo_list` / `reflection`. Extractors now run for skipped tools; only the lifecycle (start/end) events are suppressed.

## [0.1.9] - 2026-05-19

Compatibility refresh for **langgraph 1.2**, **langchain-core 1.4**, and **deepagents 0.6**.

### Added
- `UsageEvent.cache_read_tokens` and `UsageEvent.cache_creation_tokens` — populated from `usage_metadata.input_token_details` (`cache_read`, `cache_creation`). Default 0; omitted from `to_dict()` when zero.
- `ContentEvent.is_subagent` and `ReasoningEvent.is_subagent` — set to `True` when stream metadata carries `ls_agent_type == "subagent"` (deepagents >= 0.6). Lets consumers distinguish subagent output even when `lc_agent_name` is absent.
- `"respond"` decision support in `InterruptEvent.build_decisions()` / `create_resume()`: pass `response="..."` to send a text reply in place of the tool call. Matches the deepagents 0.6 decision verb set.
- `InterruptEvent.build_decisions(..., use_edited_action=False)` escape hatch for runtimes that still expect the legacy `{"type": "edit", "args": ...}` resume shape.

### Changed
- `extract_message_content()` now skips the full set of non-text content blocks defined by langchain-core 1.4 standard content blocks: `tool_call`, `tool_use`, `tool_call_chunk`, `server_tool_call`, `server_tool_call_chunk`, `server_tool_result`, `invalid_tool_call`, `image`, `audio`, `video`, `file` (plus the existing `reasoning` / `thinking`). Previously these could leak as stringified dicts into `ContentEvent.content`. Tool lifecycle and reasoning events are unchanged.
- `InterruptEvent.build_decisions("edit", ...)` now emits the modern `{"type": "edit", "edited_action": {"name", "args"}}` shape by default, matching LangGraph 1.1+ / deepagents 0.5+. Set `use_edited_action=False` for the legacy shape.
- `InterruptEvent.allowed_decisions` defaults to `{"approve", "reject", "edit", "respond"}` when no review configs are present (was `{"approve", "reject"}`).
- Dev dependency bumped: `langgraph>=1.1.0`, `langchain-core>=1.4.0`.

### Notes
- **No breaking change for default tool extractors**: deepagents 0.6 ships new built-in tools (`glob_search`, `grep_search`, `execute`, `start_async_task` / `check_async_task` / `update_async_task` / `cancel_async_task` / `list_async_tasks`, plus QuickJS `CodeInterpreterMiddleware`). These flow through the regular `ToolCallStartEvent` / `ToolCallEndEvent` lifecycle — no parser change required.
- **v3 `stream_events` typed projections** (LangGraph 1.2 beta) are not yet supported. v2 `StreamPart` parsing remains the recommended path for `stream()` / `astream()`.

## [0.1.8] - 2026-04-18

### Added
- `ReasoningEvent` dataclass for reasoning / thinking content; emitted from langchain-core `reasoning` and `thinking` content blocks, and from `think_tool` reflections. Carries a `source` field (`"content_block"` or `"think_tool"`) so UIs can distinguish provenance.
- `DisplayEvent` dataclass for rich inline content (dataframes, images, plotly, html, json) from `display_inline`-style tools. Carries `display_type`, `data`, `title`, `status`, `error`, `tool_name`, `tool_call_id`, `node`, `namespace`.
- `extract_reasoning_content()` helper in `extractors.messages` for parsing reasoning blocks from `AIMessageChunk.content`.
- `UpdatesHandler._event_from_extraction()` routes extractor output to typed events; unknown `extracted_type` values still flow through `ToolExtractedEvent` for custom extractors.
- README sections: "Reasoning & Thinking" and "Rich Inline Display" with typed matching examples.

### Changed
- `think_tool` output is now a `ReasoningEvent(source="think_tool")` instead of `ToolExtractedEvent(extracted_type="reflection")`. Legacy dict API (`stream_graph_updates`) still produces `{"chunk": text}` for backward compatibility.
- `display_inline` tool output is now a `DisplayEvent` instead of `ToolExtractedEvent(extracted_type="display_inline")`.
- `extract_message_content()` now skips reasoning blocks so they can be surfaced as `ReasoningEvent` separately.

### Fixed
- Removed dead `has_messages` variable in `_parse_v2`.

## [0.1.7] - 2026-04-18

### Added
- `FastAPIAdapter` for streaming LangGraph events over WebSocket and Server-Sent Events; stateless by design — conversation state is keyed by `session_id` used as LangGraph `thread_id`
- Per-session asyncio lock with refcounted cleanup to serialize concurrent turns on the same thread
- `BaseAdapter._text_prompt_interrupt()` helper, shared by `PrintAdapter` and `JupyterDisplay`
- `BaseAdapter._truncate()` helper for preview-length capping
- `fastapi` optional dependency group (`pip install langgraph-stream-parser[fastapi]`)

### Changed
- Hoisted `_last_rendered_count` incremental-render cursor from Print/CLI into `BaseAdapter`
- Slimmed `examples/fastapi_websocket.py` from ~455 to ~234 lines by using the new adapter

### Fixed
- `UsageEvent` now has an explicit case in `BaseAdapter._process_event` instead of silently falling through

## [0.1.6] - 2026-03-28

### Added
- v2 StreamPart parsing (`stream_mode="v2"`) with auto-detection of `{"type", "ns", "data"}` dict format
- `ValuesEvent` for full state snapshots from `stream_mode="values"` (v2)
- `DebugEvent` for debug, checkpoint, and task trace data from v2 streaming
- Routing for v2 stream types: updates, messages, custom, values, debug, checkpoints, tasks

## [0.1.5] - 2026-02-06

### Added
- Subgraph namespace preservation on events (`namespace` field on `ContentEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolExtractedEvent`, `InterruptEvent`, `StateUpdateEvent`, `UsageEvent`)
- `agent_name` field on `ContentEvent`, extracted from `lc_agent_name` metadata in messages mode (for deep agent subagents)
- `CustomEvent` for data emitted via `get_stream_writer()` (`stream_mode="custom"`)
- `stream_mode="custom"` support in single and multi-mode parsing

## [0.1.4] - 2026-02-06

### Added
- `context_parts` parameter on `prepare_agent_input()` for prepending context lines (e.g., timestamp, working directory) to user messages

## [0.1.3] - 2026-02-09

### Fixed
- Handle multi-element interrupt tuples from LangGraph subgraphs
- Aggregate `action_requests` and `review_configs` across all Interrupt objects in a tuple

## [0.1.2] - 2026-02-08

### Added
- Subgraph namespace stripping for `subgraphs=True` streams
- Automatic handling of single-mode `(namespace, data)` and multi-mode `(namespace, mode, data)` chunk formats
- All parent and subgraph chunks processed uniformly with namespace stripped

## [0.1.1] - 2026-02-07

### Added
- Dual stream mode support (`stream_mode=["updates", "messages"]`) with automatic deduplication
- Auto-detection mode (`stream_mode="auto"`) that inspects the first chunk
- `MessagesHandler` for token-level content streaming from `stream_mode="messages"`
- `UsageEvent` for token usage metadata from AIMessage `usage_metadata`
- `DisplayInlineExtractor` for extracting inline display artifacts
- Event serialization helpers (`InterruptEvent.build_decisions()`, `InterruptEvent.create_resume()`)

### Changed
- `stream_mode` is now a constructor parameter on `StreamParser` (moved from `parse()`/`aparse()`)
- `UpdatesHandler` accepts `suppress_content` flag for dual-mode deduplication

## [0.1.0] - 2026-02-01

### Initial Release

- Add `StreamParser` for parsing LangGraph stream outputs into typed events
- Add typed event classes: `ContentEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolExtractedEvent`, `InterruptEvent`, `StateUpdateEvent`, `CompleteEvent`, `ErrorEvent`
- Add tool lifecycle tracking (start → end)
- Add extensible extractor system with built-in `ThinkToolExtractor` and `TodoExtractor`
- Add interrupt handling with `create_resume_input()` and `prepare_agent_input()`
- Add async support via `aparse()`
- Add legacy dict-based API for backward compatibility (`stream_graph_updates`, `resume_graph_from_interrupt`)

### Display Adapters

- Add `BaseAdapter` abstract class for building custom display adapters
- Add `PrintAdapter` for plain text output in any Python environment
- Add `CLIAdapter` for styled terminal output with ANSI colors and spinner animation
- Add `JupyterDisplay` for rich notebook display with live updates
- Add configurable `reflection_types` and `todo_types` for custom tool rendering
- Add `**stream_kwargs` pass-through to `graph.stream()`
