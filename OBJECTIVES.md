# Objectives & scope — langstage-core

*What this repo is for, who it serves, and what it deliberately is **not** — the yardstick
for deciding whether a proposed change or filed issue belongs here. When triaging an issue,
start here.*

## Objective

Turn **any** LangGraph `CompiledGraph` into something every LangStage surface can run
identically. langstage-core is the shared SDK/runtime behind the family:

- the **host layer** — spec-loading (`load_agent_spec`) + layered configuration (`HostConfig`);
- the in-process **AG-UI bridge** — `iter_event_frames` / `iter_chunk_frames`, the collectors,
  `run_turn`, `verify`, `serve`, `build_agent`;
- the async **task-delegation engine** — `TaskRunner` / `TaskStore` / `InMemoryTaskStore`;
- **extractors** and interrupt-aware input helpers.

## Who it's for

The other five family repos (its primary consumer), plus third-party developers embedding a
LangGraph agent behind AG-UI. It is a library, not an application.

## In scope

- Both wires (event + chunk) and their parity; the collectors; the preflight primitives.
- Config resolution and precedence; spec loading; keyless demos.
- The task engine and the extractor contract.
- Honest, actionable errors and preflight verdicts (no false green / false red).
- **Any behavior two or more surfaces would otherwise duplicate** — it belongs here.

## Out of scope (anti-scope)

- Surface-specific UX/rendering (colors, menus, terminal/editor chrome) — that lives in a leaf.
- Re-implementing or forking what `ag-ui-langgraph` or `langgraph` own. Core *wraps* them and
  mitigates their sharp edges narrowly; it does not reinvent them.
- Becoming an agent *framework* — it runs graphs, it does not prescribe how you build them.
- Model/provider-specific code, or heavy dependencies in the base install (extras exist for that).
- Global side effects on the host process (e.g. reconfiguring third-party loggers). A library
  must not; that belongs to the application consuming it.

## How this fits the family

langstage-core is the hub; the surfaces (web, cli, jupyter, vscode) are thin consumers, and
langstage-hermes is an opinionated agent built on top. The strongest signal that a
leaf-repo issue actually belongs here: its fix would otherwise be duplicated across surfaces.

## Using this to triage

Before acting on an issue or PR: does it serve the objective above? Is it in scope or
anti-scope? Weigh its value — **security > correctness > advertised-≠-honored > DX/docs >
polish > net-new feature** — against the cost of a manual release. Then **fix, defer, or
decline with a reason.** Not every filed issue is worth acting on.
