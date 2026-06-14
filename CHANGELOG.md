# Changelog

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
