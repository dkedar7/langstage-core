# Changelog

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
  bool field keeps accepting TOML `true`/`false` untouched. The note is deduped per (key, value), as
  several surfaces each resolve the config in one process. Correctly-typed configs behave exactly as
  before and emit no notes.

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
