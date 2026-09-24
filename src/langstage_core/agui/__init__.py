"""Serve any LangGraph agent over the **AG-UI** protocol.

AG-UI (Agent-User Interaction Protocol) is the event-based wire format for
streaming rich agent interactions — text, tool calls, reasoning, state, and
human-in-the-loop interrupts — to frontends. This module is the LangStage
family's blessed bridge to it: the host layer (``load_agent_spec`` +
``HostConfig``) resolves *which* agent to run, and the official, MIT-licensed
``ag-ui-langgraph`` adapter owns the wire. The result is that every surface's
agent is reachable from any AG-UI client without each surface reimplementing a
protocol.

See ``docs/adr/0001-adopt-ag-ui-for-the-wire.md`` for the rationale.

Requires the ``agui`` extra::

    pip install "langstage-core[agui]"

Quick start::

    # Serve any agent spec over AG-UI:
    langstage-agui --agent my_agent.py:graph

    # Or in code:
    from langstage_core.agui import build_app
    app = build_app(my_compiled_graph)   # an ASGI app; run with uvicorn
"""
# NB: intentionally NOT `from __future__ import annotations`. The resilient
# endpoint below needs real (non-string) annotations so FastAPI can resolve
# RunAgentInput as the request body; PEP 604 unions work natively on >=3.11.
from typing import Any

__all__ = [
    "build_agent",
    "add_agui_endpoint",
    "build_app",
    "serve",
    "ensure_available",
    "iter_event_frames",
    "iter_chunk_frames",
    "collect_event_frames",
    "collect_chunk_frames",
    "run_turn",
    "TurnResult",
    "verify",
    "averify",
    "VerifyResult",
    "DEFAULT_AGENT_NAME",
]

DEFAULT_AGENT_NAME = "LangStage Agent"

_IMPORT_HINT = (
    "AG-UI support needs the 'agui' extra: "
    'pip install "langstage-core[agui]"'
)


def ensure_available() -> None:
    """Raise the agui-extra ``RuntimeError`` if the AG-UI server deps are missing.

    Lets a caller fail fast with the clean install hint *before* any user-facing
    output (e.g. a "Serving … at <url>" banner), instead of mid-serve.
    """
    try:
        import ag_ui_langgraph  # noqa: F401
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:  # pragma: no cover - only without the [agui] extra
        raise RuntimeError(_IMPORT_HINT) from e


def _is_langgraph_agent(obj: Any) -> bool:
    """True if obj is already an ``ag_ui_langgraph.LangGraphAgent`` (has the
    adapter contract), so callers may pass a prebuilt agent through."""
    return hasattr(obj, "clone") and hasattr(obj, "run") and hasattr(obj, "name")


def _schema_keys_without_pydantic(graph: Any, constant_keys: list) -> dict:
    """Schema keys read straight off the graph's state classes (their annotations),
    for when ag-ui's pydantic JSON-schema introspection can't build a schema.

    The case that motivates it (gh langstage#166): a state declared with the stdlib
    ``typing.TypedDict`` on Python < 3.12. LangGraph runs it fine, but
    ``graph.get_input_jsonschema()`` goes through ``pydantic.TypeAdapter``, which raises
    ``PydanticUserError`` ("use typing_extensions.TypedDict") — a ``RuntimeError``
    subclass that ag-ui-langgraph's ``get_schema_keys`` fallback doesn't catch, so
    EVERY turn errored before the graph ran. A TypedDict's keys are simply its
    annotations, so read them directly.
    """
    builder = getattr(graph, "builder", None)

    def keys(attr: str) -> list:
        schema = getattr(builder, attr, None) or getattr(builder, "state_schema", None)
        try:
            names = list(getattr(schema, "__annotations__", {}) or {})
        except Exception:  # pragma: no cover - defensive
            names = []
        return [*names, *[k for k in constant_keys if k not in names]]

    try:
        config_schema = graph.get_config_jsonschema()
        config_keys = list(config_schema.get("properties", {}))
    except Exception:  # noqa: BLE001 - the config schema is optional here
        config_keys = []
    return {
        "input": keys("input_schema"),
        "output": keys("output_schema"),
        "config": config_keys,
        "context": [],
    }


_AGENT_CLASSES: dict = {}


def _agent_class(base: Any) -> Any:
    """The ``LangGraphAgent`` subclass :func:`build_agent` instantiates (cached per base).

    Its one override degrades ag-ui's schema introspection instead of failing the turn
    when pydantic can't build a JSON schema for the graph's state (e.g. a stdlib
    ``typing.TypedDict`` on Python 3.11, gh langstage#166). ``clone()`` rebuilds via
    ``type(self)``, so every per-run clone keeps the override.
    """
    cls = _AGENT_CLASSES.get(base)
    if cls is None:

        class LangStageGraphAgent(base):  # type: ignore[misc, valid-type]
            def get_schema_keys(self, config):
                try:
                    return super().get_schema_keys(config)
                except Exception as exc:
                    try:
                        from pydantic.errors import PydanticErrorMixin
                    except ImportError:  # pragma: no cover - pydantic is always present
                        raise exc from None
                    if not isinstance(exc, PydanticErrorMixin):
                        raise
                    return _schema_keys_without_pydantic(
                        self.graph, list(getattr(self, "constant_schema_keys", ["messages", "tools"]))
                    )

        LangStageGraphAgent.__qualname__ = LangStageGraphAgent.__name__
        cls = _AGENT_CLASSES[base] = LangStageGraphAgent
    return cls


def build_agent(
    graph: Any,
    *,
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
) -> Any:
    """Wrap a compiled LangGraph graph in an ``ag-ui-langgraph`` ``LangGraphAgent``.

    Args:
        graph: A compiled LangGraph graph (``CompiledStateGraph``). Its state
            must include a ``messages`` key (the AG-UI adapter's only schema
            requirement) — true for ``MessagesState`` and deepagents graphs.
        name: Display name surfaced to AG-UI clients.
        description: Optional human description.
        config: Optional ``RunnableConfig`` / dict forwarded to the graph.

    Returns:
        A ``LangGraphAgent`` ready to attach to an ASGI app.

    Raises:
        RuntimeError: if the ``agui`` extra is not installed.
    """
    try:
        from ag_ui_langgraph import LangGraphAgent
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    # Accept the library's headline input form — a `module:attr` / `path/to/file.py:attr`
    # spec string — not just a compiled graph, so the Python one-shot surface matches
    # the CLI (`langstage-agui --agent my_agent.py:graph`). Passing a spec used to fail
    # with a cryptic leaked-LangGraph `AttributeError: 'str' object has no attribute
    # 'nodes'`; now it resolves through the same load_agent_spec() the CLI uses. Because
    # run_turn / verify / collect_* all route a non-agent through build_agent, they
    # inherit spec support from this one point. (gh #112)
    if isinstance(graph, str):
        from ..host import load_agent_spec

        graph = load_agent_spec(graph)
    # Validate we actually hold a COMPILED LangGraph graph before handing it to the
    # adapter. A compiled graph exposes ``aget_state`` (the adapter calls it); an
    # uncompiled ``StateGraph``, a ``dict``, ``None``, or a bare function do not — and
    # passing one used to reach ``LangGraphAgent(...)`` / ``graph.run()`` and surface a
    # cryptic leaked-LangGraph ``AttributeError`` ('X' object has no attribute 'nodes'/
    # 'run'/'aget_state') to the user. Fail fast with an actionable message instead —
    # the same DX fix gh #112 (spec strings) and gh #100 made elsewhere, here for the
    # wrong-type / forgot-to-``.compile()`` cases (gh langstage-jupyter #92).
    if not hasattr(graph, "aget_state"):
        if hasattr(graph, "compile") and hasattr(graph, "add_node"):
            raise TypeError(
                "build_agent expected a compiled LangGraph graph, but got an "
                "uncompiled StateGraph; call .compile() on it first."
            )
        raise TypeError(
            "build_agent expected a compiled LangGraph graph (CompiledStateGraph) "
            f"or an agent spec string, but got {type(graph).__name__}."
        )
    # AG-UI requires threaded state — the adapter calls graph.aget_state() and
    # supports interrupts/resume, both of which need a checkpointer. Many user
    # graphs are compiled without one (and would otherwise hard-crash with
    # "No checkpointer set"), so attach an in-memory default when absent.
    #
    # The saver goes on a COPY of the graph, never on the caller's object: assigning
    # ``graph.checkpointer`` in place meant the next ``run_turn(graph, ...)`` /
    # ``collect_*(graph, ...)`` found a checkpointer already attached and silently
    # resumed the previous call's thread, so "one-shot" calls accumulated state
    # (gh #163). Each build_agent(graph) now owns a fresh in-memory saver — build the
    # agent once and reuse it (the documented pattern) to keep state across turns. A
    # graph compiled WITH a checkpointer is used as-is (its state stays shared).
    if getattr(graph, "checkpointer", None) is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            saver = InMemorySaver()
            try:
                graph = graph.copy(update={"checkpointer": saver})
            except Exception:  # pragma: no cover - a graph-like without Pregel.copy()
                graph.checkpointer = saver
        except Exception:  # pragma: no cover - best-effort; LangGraphAgent will surface real issues
            pass
    return _agent_class(LangGraphAgent)(
        name=name, graph=graph, description=description, config=config
    )


def add_agui_endpoint(
    app: Any,
    graph: Any,
    *,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
) -> Any:
    """Attach an AG-UI endpoint for ``graph`` to an existing FastAPI ``app``.

    ``graph`` may be a compiled graph or an already-built ``LangGraphAgent``.
    Returns the same ``app`` for chaining.

    The endpoint is *resilient*: if the agent raises mid-run, a terminal
    ``RUN_ERROR`` event is emitted and the stream closes cleanly, rather than
    crashing the connection with an unhandled 500 (the bare upstream adapter
    lets node exceptions propagate). Each request runs on its own cloned agent.
    """
    try:
        from ag_ui.core import EventType, RunAgentInput, RunErrorEvent
        from ag_ui.encoder import EventEncoder
        from fastapi import Request
        from fastapi.responses import StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e

    agent = graph if _is_langgraph_agent(graph) else build_agent(
        graph, name=name, description=description, config=config
    )

    @app.post(path)
    async def _run(input_data: RunAgentInput, request: Request):
        accept = request.headers.get("accept")
        try:
            encoder = EventEncoder(accept=accept)
        except TypeError:  # pragma: no cover - older SDKs without the accept kwarg
            encoder = EventEncoder()
        media_type = getattr(encoder, "get_content_type", lambda: "text/event-stream")()
        run_agent = agent.clone()

        async def gen():
            try:
                # _TurnStream: a node that returns a finished (non-token-streamed)
                # AIMessage used to reach a served client ONLY inside the closing
                # MESSAGES_SNAPSHOT (zero TEXT_MESSAGE_* events), while the in-process
                # iter_* wires streamed it as content. The same step-level
                # reconstruction now feeds both, so served and in-process agree (gh #140).
                async for ev in _TurnStream(run_agent, input_data).events():
                    # run() yields SSE-encoded strings; encode objects defensively.
                    yield ev if isinstance(ev, (str, bytes)) else encoder.encode(ev)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as RUN_ERROR
                yield encoder.encode(
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

        return StreamingResponse(gen(), media_type=media_type)

    @app.get(path)
    async def _health():
        return {"status": "ok", "agent": {"name": getattr(agent, "name", name)}}

    return app


def build_app(
    graph: Any,
    *,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
    title: str | None = None,
) -> Any:
    """Build a standalone FastAPI ASGI app exposing ``graph`` over AG-UI.

    Run it with any ASGI server, e.g. ``uvicorn.run(app, ...)`` — or just use
    :func:`serve`.
    """
    try:
        from fastapi import FastAPI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e
    app = FastAPI(title=title or name)
    add_agui_endpoint(app, graph, path=path, name=name, description=description, config=config)
    return app


def serve(
    spec_or_graph: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
) -> None:
    """Load an agent (if given a spec string) and serve it over AG-UI.

    ``spec_or_graph`` is either an agent spec string (``module:attr`` or
    ``path/to/file.py:attr`` — resolved via the host layer's
    :func:`~langstage_core.host.load_agent_spec`) or an already
    compiled graph. Blocks running a uvicorn server.
    """
    if isinstance(spec_or_graph, str):
        from ..host import load_agent_spec  # the host layer feeds AG-UI

        graph = load_agent_spec(spec_or_graph)
    else:
        graph = spec_or_graph
    app = build_app(graph, path=path, name=name, description=description)
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e
    uvicorn.run(app, host=host, port=port)


_DEFAULT_DECISIONS = ["reject", "edit", "respond", "approve"]

# HumanInterrupt.config key -> the decision it permits (the deepagents / langchain
# HITL convention). Used to derive allowed_decisions from a HumanInterrupt list.
_CONFIG_DECISION = {
    "allow_accept": "approve",
    "allow_edit": "edit",
    "allow_respond": "respond",
    "allow_ignore": "reject",
}


def _normalize_interrupt(payload, default_decisions=_DEFAULT_DECISIONS):
    """Normalize a langgraph interrupt value to ``(action_requests, review_configs,
    allowed_decisions)``, tolerating the shapes real agents actually produce.

    The on_interrupt handler used to assume the value was a dict keyed
    ``action_requests``/``allowed_decisions`` and did ``payload.get(...)`` — which
    crashed on the **standard HumanInterrupt list** that deepagents / langchain HITL
    emit (``'list' object has no attribute 'get'``) and returned an empty
    ``action_requests`` for any other dict (gh langstage-vscode #40). Handled shapes:

    - a **list of HumanInterrupt dicts** —
      ``[{"action_request": {...}, "config": {...}, "description": ...}, ...]`` — the
      deepagents / langchain convention: each ``action_request`` becomes an action
      request, and ``config``'s ``allow_*`` flags derive the allowed decisions;
    - a dict **already keyed** ``action_requests`` (our own shape) — used as-is;
    - any other **plain dict** — treated as a single action request (so a
      ``interrupt({...})`` value surfaces instead of vanishing to ``[]``).
    """
    if isinstance(payload, list):
        action_requests = []
        allowed: set = set()
        for item in payload:
            if isinstance(item, dict):
                action_requests.append(item.get("action_request", item))
                cfg = item.get("config")
                if isinstance(cfg, dict):
                    allowed.update(
                        dec for key, dec in _CONFIG_DECISION.items() if cfg.get(key)
                    )
            else:
                action_requests.append(item)
        decisions = [d for d in default_decisions if d in allowed] or list(default_decisions)
        return action_requests, [], decisions
    if isinstance(payload, dict):
        if "action_requests" in payload:
            return (
                payload.get("action_requests", []),
                payload.get("review_configs", []),
                payload.get("allowed_decisions", default_decisions),
            )
        if payload:  # a plain dict interrupt value -> a single action request
            return [payload], [], list(default_decisions)
        return [], [], list(default_decisions)
    # A bare string / scalar — the canonical `interrupt("Approve deleting X?")` HITL
    # form — must surface as a single action request, not vanish to an empty list
    # that renders "(no action details provided)" and asks the human to approve
    # blind (gh langstage-cli #95). Renderers already handle a scalar request
    # (cli's format_interrupt_request returns str(action)); the bug was that this
    # normalizer dropped it before any renderer saw it.
    if payload is not None and payload != "":
        return [payload], [], list(default_decisions)
    return [], [], list(default_decisions)


def _truncate_result(result, max_result_len):
    """Cap a tool result at ``max_result_len``, appending the ``…(truncated)`` marker.

    The single implementation both ``iter_*`` mappings call, so the two counterparts
    can't drift on truncation the way they did before gh #102 (``iter_event_frames``
    capped; ``iter_chunk_frames`` — the CLI/Jupyter wire, where an unbounded blob is
    arguably *more* harmful — emitted the full result with no knob at all).

    Boundary: a result of exactly ``max_result_len`` or shorter is returned unchanged
    and unmarked; a longer one is sliced to ``max_result_len`` *and then* marked, so
    the yielded string is ``max_result_len`` + 12 characters.

    A non-``str`` result is passed through untouched rather than sliced. AG-UI types
    ``ToolCallResultEvent.content`` as ``str``, so this is defensive — but the chunk
    wire yields ``content`` as-is (only the event wire coerces with ``str()``), and
    slicing e.g. a dict a fake or future adapter produced would raise ``TypeError``.
    """
    if isinstance(result, str) and len(result) > max_result_len:
        return result[:max_result_len] + "…(truncated)"
    return result


async def _full_text_by_id(agent, thread_id):
    """Map ``message id -> full text`` (ALL text blocks joined) from the graph checkpoint.

    ag-ui's ``resolve_message_content`` flattens a multi-text-block ``AIMessage`` to
    only its FIRST text block (``ag_ui_langgraph/utils.py``), so the final
    ``MessagesSnapshotEvent`` that a non-streamed turn relies on carries a truncated
    assistant ``content`` — later text blocks vanish with no error, and the client shows
    a plausible-but-partial reply (gh langstage-vscode #75). We re-read the ORIGINAL
    LangChain messages from the graph's checkpoint and use ``message.text`` (which joins
    all ``text`` blocks like ``AIMessage.text``, and ignores reasoning/thinking blocks —
    those already surface as separate ``reasoning`` frames). Best-effort: any failure
    (no ``.graph``, no checkpointer, an odd state schema) returns ``{}`` and the snapshot
    walk falls back to ag-ui's (possibly truncated) content — never fatal.
    """
    graph = getattr(agent, "graph", None)
    aget_state = getattr(graph, "aget_state", None)
    if aget_state is None:
        return {}
    try:
        state = await aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:  # pragma: no cover - best-effort fidelity, never fatal
        return {}
    values = getattr(state, "values", None) or {}
    out: dict[str, str] = {}
    for m in values.get("messages", []) or []:
        mid = getattr(m, "id", None)
        if mid is None:
            continue
        text = getattr(m, "text", None)
        # langchain_core >=1.x exposes ``.text`` as a property returning a str (a
        # callable str subclass, for back-compat); older versions expose a ``.text()``
        # method. Read the property value; only CALL it when it isn't already a str (the
        # real-method case) — so we neither miss it nor trip the deprecation warning.
        if callable(text) and not isinstance(text, str):
            try:
                text = text()
            except Exception:  # pragma: no cover
                text = None
        if isinstance(text, str) and text:
            out[str(mid)] = str(text)
    return out


def _snapshot_items(messages, *, streamed_ids, tool_names, streamed_result_ids, step_nodes, current_node, full_text_by_id=None):
    """Yield the not-yet-emitted items from a final ``MessagesSnapshotEvent``.

    Shared by both ``iter_*`` mappings so the two wires can't drift on snapshot
    handling. The snapshot branch used to yield **only** assistant text and drop
    every message with empty content, so a turn whose messages are produced
    *without token streaming* (a custom node calling ``model.invoke()``, a
    non-streaming provider, a rule-based node) surfaced no tool call and no tool
    result, and a tool-call-only ``AIMessage(content="", tool_calls=[...])``
    rendered as a completely empty turn while ``--verify`` reported success
    (gh #91). This walks assistant **and** tool messages and yields normalized
    items each wire renders in its own vocabulary:

      ``{"kind": "content", "text", "node", "id"}``    — an assistant text message
      ``{"kind": "tool_call", "name", "args", "id", "message_id"}`` — a tool call not streamed
      ``{"kind": "tool_result", "name", "raw", "id", "message_id", "error"}`` — a result not streamed

    Within one assistant message its text is yielded BEFORE its tool calls, so a
    "narrate, then call a tool" message renders in causal order (gh langstage-cli #119).

    Dedup mirrors how a fully-streamed turn already emitted things during the run,
    so nothing double-renders: content by message id (``streamed_ids``), tool calls
    by ``tool_call_id`` (``tool_names`` is populated only on the streaming
    ``ToolCallStartEvent``), tool results by ``tool_call_id``
    (``streamed_result_ids``). A non-streaming turn streamed none of these, so the
    snapshot is the sole source and everything is emitted; a fully-streamed turn
    finds them all already emitted and yields nothing.
    """
    import json

    msgs = list(messages)
    last_user = max(
        (i for i, m in enumerate(msgs) if getattr(m, "role", None) in ("user", "human")),
        default=-1,
    )
    tail = msgs[last_user + 1:]
    # Node mapping aligns to assistant *content* messages only (gh #43), so keep the
    # index space the pre-#91 code used — interleaving tool messages must not shift it.
    content_msgs = [
        m for m in tail
        if getattr(m, "role", None) == "assistant" and getattr(m, "content", None)
    ]
    offset = max(0, len(content_msgs) - len(step_nodes))
    # Local name map so a tool result can still name its tool on a fully-snapshot turn,
    # where ToolCallStart never populated the shared tool_names.
    names = dict(tool_names)
    ci = 0
    for m in tail:
        role = getattr(m, "role", None)
        if role == "assistant":
            mid = getattr(m, "id", None)
            # In-message order: a message's own TEXT comes before its tool calls. The
            # #91 walk yielded the tool calls first, so a finished
            # ``AIMessage("Let me check the weather.", tool_calls=[...])`` rendered its
            # tool call BEFORE the narration that motivates it, and the tool result then
            # jammed onto the narration's line in the cli (gh langstage-cli #119).
            content = getattr(m, "content", None)
            if content:
                if mid not in streamed_ids:
                    idx = ci - offset
                    node = step_nodes[idx] if 0 <= idx < len(step_nodes) else current_node
                    # Prefer the ORIGINAL message's full text: ag-ui's
                    # resolve_message_content flattens a multi-text-block AIMessage to
                    # only its FIRST block, so the snapshot ``content`` silently drops
                    # later blocks (gh langstage-vscode #75). full_text_by_id carries the
                    # checkpoint message's joined ``.text`` keyed by id; fall back to
                    # ag-ui's content when absent (single-block, tool-only, or no state).
                    text = full_text_by_id.get(str(mid), content) if full_text_by_id else content
                    yield {"kind": "content", "text": text, "node": node, "id": mid}
                ci += 1
            for tc in getattr(m, "tool_calls", None) or []:
                tcid = getattr(tc, "id", None)
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "tool") if fn is not None else "tool"
                if tcid is not None:
                    names.setdefault(tcid, name)
                if tcid in tool_names:
                    continue  # already emitted via streaming ToolCall events
                raw_args = (getattr(fn, "arguments", "") if fn is not None else "") or ""
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                yield {"kind": "tool_call", "name": name, "args": args, "id": tcid,
                       "message_id": mid}
        elif role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid in streamed_result_ids:
                continue  # already emitted via the streaming ToolCallResultEvent
            yield {
                "kind": "tool_result",
                "name": names.get(tcid, "tool"),
                "raw": getattr(m, "content", "") or "",
                "id": tcid,
                "message_id": getattr(m, "id", None),
                "error": bool(getattr(m, "error", None)),
            }


def _message_text(m) -> str:
    """The joined text of a LangChain message (all ``text`` blocks, like ``AIMessage.text``).

    langchain_core >=1.x exposes ``.text`` as a property returning a str (a callable str
    subclass, for back-compat); older versions expose a ``.text()`` method. Read the
    property value; only CALL it when it isn't already a str. Falls back to a plain-str
    ``content``; returns ``""`` when there is no text.
    """
    text = getattr(m, "text", None)
    if callable(text) and not isinstance(text, str):
        try:
            text = text()
        except Exception:  # pragma: no cover
            text = None
    if isinstance(text, str):
        return str(text)
    content = getattr(m, "content", None)
    return content if isinstance(content, str) else ""


def _as_snapshot_message(m):
    """Adapt a checkpoint (LangChain) message to the attribute shape
    :func:`_snapshot_items` walks (the ag-ui ``MessagesSnapshotEvent`` message shape):
    ``role`` / ``id`` / ``content`` (+ ``tool_calls[*].id/.function.name/.arguments`` on
    an assistant message, ``tool_call_id`` / ``error`` on a tool message)."""
    import json
    from types import SimpleNamespace

    kind = getattr(m, "type", None)
    role = {"ai": "assistant", "AIMessageChunk": "assistant", "human": "user",
            "tool": "tool", "system": "system"}.get(kind, kind)
    mid = getattr(m, "id", None)
    out = SimpleNamespace(role=role, id=str(mid) if mid is not None else None,
                          content=_message_text(m))
    if role == "assistant":
        calls = []
        for tc in getattr(m, "tool_calls", None) or []:
            tc = tc if isinstance(tc, dict) else {}
            try:
                arguments = json.dumps(tc.get("args") or {})
            except (TypeError, ValueError):
                arguments = json.dumps({"_raw": str(tc.get("args"))})
            calls.append(SimpleNamespace(
                id=tc.get("id"),
                function=SimpleNamespace(name=tc.get("name") or "tool", arguments=arguments),
            ))
        out.tool_calls = calls
    elif role == "tool":
        content = getattr(m, "content", "")
        out.content = content if isinstance(content, str) else (_message_text(m) or str(content))
        out.tool_call_id = getattr(m, "tool_call_id", None)
        out.error = getattr(m, "status", None) == "error"
    return out


class _TurnStream:
    """Drive ``agent.run(run_input)`` and surface **finished** messages as real AG-UI
    events at the step that produced them: the one place both ``iter_*`` wires and the
    served endpoint get their "non-token-streamed message" handling from.

    ag-ui-langgraph only emits ``TEXT_MESSAGE_*`` / ``TOOL_CALL_*`` for what a model
    *streams*. A node that returns a finished ``AIMessage`` (``model.invoke()``, a
    router, a canned reply, a forwarded sub-agent result) produced nothing until the
    closing ``MESSAGES_SNAPSHOT``. The in-process ``iter_*`` mappings patched that by
    walking the snapshot at the very end (gh #89/#91), which left four holes:

    - the **served** wire (``build_app`` / ``serve``) forwards upstream events verbatim, so
      a finished message emitted zero ``TEXT_MESSAGE_*`` events there (gh #140);
    - everything arrived at the END, so a finished ``AIMessage`` carrying text + a tool
      call rendered its text AFTER the tool had already run (gh langstage-cli #119);
    - when a later node errored, the snapshot never came, so content an earlier node had
      already produced and committed was dropped (gh langstage-vscode #105);
    - consecutive finished messages carried no message identity (gh langstage-vscode #108).

    So after every ``StepFinishedEvent`` (and before ``MESSAGES_SNAPSHOT`` /
    ``RUN_ERROR`` / an exception) this reads the thread's checkpoint (LangGraph has
    already committed a step's writes by the time the step finishes) and synthesizes
    ``TEXT_MESSAGE_START/CONTENT/END``, ``TOOL_CALL_START/ARGS/END`` and
    ``TOOL_CALL_RESULT`` for each message of THIS turn that was not already streamed,
    using the checkpoint's own message / tool-call ids (the same ids the final
    ``MESSAGES_SNAPSHOT`` carries). Messages already in the checkpoint before the run
    (history, or the pre-interrupt part of a resumed turn) are never re-emitted. An
    upstream ``TOOL_CALL_START/ARGS/END`` for a call already synthesized (a ``ToolNode``
    re-announces the call when the tool finishes) is suppressed, so a client never sees
    one tool call twice.

    Best-effort: an agent without a readable ``.graph`` checkpoint (a test double, an
    odd graph) makes this a transparent pass-through, and the end-of-turn snapshot walk
    in the ``iter_*`` mappings remains the fallback.
    """

    def __init__(self, agent, run_input):
        self.agent = agent
        self.run_input = run_input
        self.thread_id = getattr(run_input, "thread_id", None)
        self.text_ids: set = set()
        self.call_ids: set = set()
        self.result_ids: set = set()
        self._synth_call_ids: set = set()
        self._synth_result_ids: set = set()
        # tool_call_ids whose checkpoint ToolMessage has status="error". The AG-UI
        # ToolCallResultEvent has no status field, so the iter_* mappings read it here.
        self.errored_result_ids: set = set()
        self._pre_ids: set | None = None

    async def _checkpoint_messages(self):
        graph = getattr(self.agent, "graph", None)
        aget_state = getattr(graph, "aget_state", None)
        if aget_state is None or self.thread_id is None:
            return None
        try:
            state = await aget_state({"configurable": {"thread_id": self.thread_id}})
        except Exception:  # noqa: BLE001 - best-effort; never fatal to the turn
            return None
        values = getattr(state, "values", None)
        if values is None:
            return []
        if not isinstance(values, dict):
            return None
        msgs = values.get("messages") or []
        return list(msgs) if isinstance(msgs, (list, tuple)) else None

    async def _flush(self):
        if self._pre_ids is None:
            return
        msgs = await self._checkpoint_messages()
        if not msgs:
            return
        new = [
            _as_snapshot_message(m) for m in msgs
            if getattr(m, "id", None) is not None and str(m.id) not in self._pre_ids
        ]
        if not new:
            return
        import json

        from ag_ui.core import (
            EventType,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )

        for item in _snapshot_items(
            new,
            streamed_ids=self.text_ids,
            tool_names=dict.fromkeys(self.call_ids, ""),
            streamed_result_ids=self.result_ids,
            step_nodes=[],
            current_node="",
        ):
            kind = item["kind"]
            if kind == "content":
                mid = item["id"]
                if mid is None or not item["text"]:
                    continue
                self.text_ids.add(mid)
                yield TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START, message_id=mid, role="assistant")
                yield TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT, message_id=mid, delta=item["text"])
                yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid)
            elif kind == "tool_call":
                tcid = item["id"]
                if tcid is None:
                    continue
                self.call_ids.add(tcid)
                self._synth_call_ids.add(tcid)
                yield ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START, tool_call_id=tcid,
                    tool_call_name=item["name"], parent_message_id=item.get("message_id"))
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS, tool_call_id=tcid,
                    delta=json.dumps(item["args"]))
                yield ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tcid)
            elif kind == "tool_result":
                tcid = item["id"]
                if tcid is None:
                    continue
                self.result_ids.add(tcid)
                self._synth_result_ids.add(tcid)
                if item.get("error"):
                    self.errored_result_ids.add(tcid)
                yield ToolCallResultEvent(
                    type=EventType.TOOL_CALL_RESULT, message_id=item.get("message_id") or tcid,
                    tool_call_id=tcid, content=str(item["raw"]), role="tool")

    async def events(self):
        base = await self._checkpoint_messages()
        if base is not None:
            self._pre_ids = {str(m.id) for m in base if getattr(m, "id", None) is not None}
        try:
            async for ev in self.agent.run(self.run_input):
                t = type(ev).__name__
                if t in ("StepFinishedEvent", "MessagesSnapshotEvent", "RunErrorEvent"):
                    async for synth in self._flush():
                        yield synth
                elif t in ("TextMessageStartEvent", "TextMessageContentEvent",
                           "TextMessageEndEvent", "TextMessageChunkEvent"):
                    mid = getattr(ev, "message_id", None)
                    if mid is not None:
                        self.text_ids.add(mid)
                elif t in ("ToolCallStartEvent", "ToolCallArgsEvent", "ToolCallEndEvent",
                           "ToolCallChunkEvent"):
                    tcid = getattr(ev, "tool_call_id", None)
                    if tcid in self._synth_call_ids:
                        continue  # already announced from the checkpoint; don't repeat it
                    if tcid is not None and t in ("ToolCallStartEvent", "ToolCallChunkEvent"):
                        self.call_ids.add(tcid)
                elif t == "ToolCallResultEvent":
                    tcid = getattr(ev, "tool_call_id", None)
                    if tcid in self._synth_result_ids:
                        continue
                    if tcid is not None:
                        self.result_ids.add(tcid)
                yield ev
        except Exception:
            # A node raised mid-turn: surface what earlier nodes already produced and
            # committed before the error propagates (gh langstage-vscode #105).
            async for synth in self._flush():
                yield synth
            raise


class _ToolTracker:
    """Per-turn tool bookkeeping shared by both ``iter_*`` mappings: which tool calls
    failed, and how long each ran.

    - **Error status** (gh #55, #168): the AG-UI ``ToolCallResultEvent`` drops the
      ToolMessage status, so a failed tool otherwise looks like a success. The adapter's
      ``on_tool_error`` RawEvent carries the ``tool_call_id`` (keyed precisely); a payload
      without it falls back to counting by tool name, as before.
    - **Duration** (gh langstage#160): ``on_tool_start`` then ``on_tool_end`` /
      ``on_tool_error`` RawEvents bracket the actual tool execution (keyed by run id; the
      end event carries the ``tool_call_id``). A tool invoked outside LangChain's tool
      runtime (a hand-written node returning a ``ToolMessage``) has no such events, so its
      ``duration_ms`` stays ``None`` rather than a made-up number.
    """

    def __init__(self):
        self._started: dict = {}
        self._duration_ms: dict = {}
        self._errored_ids: set = set()
        self._errored_names: dict = {}

    def on_raw(self, raw) -> None:
        import time

        if not isinstance(raw, dict):
            return
        kind = raw.get("event")
        if kind not in ("on_tool_start", "on_tool_end", "on_tool_error"):
            return
        run_id = raw.get("run_id")
        if kind == "on_tool_start":
            if run_id is not None:
                self._started[run_id] = time.monotonic()
            return
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        tcid = data.get("tool_call_id")
        output = data.get("output")
        if tcid is None and output is not None:
            tcid = (output.get("tool_call_id") if isinstance(output, dict)
                    else getattr(output, "tool_call_id", None))
        started = self._started.pop(run_id, None) if run_id is not None else None
        if tcid is not None and started is not None:
            self._duration_ms[tcid] = max(0, round((time.monotonic() - started) * 1000))
        status = (output.get("status") if isinstance(output, dict)
                  else getattr(output, "status", None))
        if kind == "on_tool_error" or status == "error":
            if tcid is not None:
                self._errored_ids.add(tcid)
            else:
                name = raw.get("name")
                if name:
                    self._errored_names[name] = self._errored_names.get(name, 0) + 1

    def result(self, tool_call_id, name, known_error: bool = False):
        """Return ``(is_error, duration_ms)`` for one tool result, consuming the bookkeeping."""
        is_error = known_error
        if tool_call_id in self._errored_ids:
            self._errored_ids.discard(tool_call_id)
            is_error = True
        elif not is_error and self._errored_names.get(name, 0) > 0:
            self._errored_names[name] -= 1
            is_error = True
        return is_error, self._duration_ms.pop(tool_call_id, None)


def _unwrap_resume(resume):
    """Return the raw resume payload, accepting either the payload OR a langgraph
    ``Command`` built by :func:`create_resume_input`.

    ``iter_event_frames`` / ``iter_chunk_frames`` wrap ``resume`` into
    ``forwarded_props.command.resume``, which ag-ui-langgraph turns into
    ``Command(resume=...)``. But ``create_resume_input()`` *also* returns a
    ``Command``, so passing it straight through double-wrapped it — the graph's
    ``interrupt()`` then returned the inner ``Command`` instead of the decision and
    a realistic HITL node crashed with ``'Command' object is not subscriptable``
    (gh #82). If the caller already built a ``Command``, use its ``.resume`` value so
    both ``resume=create_resume_input(...)`` and a raw ``resume={"decisions": ...}``
    converge on the same single wrap.
    """
    if resume is None:
        return None
    try:
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - langgraph is always present in practice
        return resume
    return resume.resume if isinstance(resume, Command) else resume


def _terminal_outcome(*, saw_interrupt: bool, saw_error: bool) -> str:
    """The single source of truth for a turn's typed terminal outcome (gh #110).

    Given whether the turn saw an ``error`` frame and/or a pending ``interrupt``
    frame by the time it terminated, return the outcome string the whole family
    agrees on:

    - ``"error"`` if an ``error`` frame fired — it wins even over a pending
      interrupt, because a turn that errors after pausing did **not** pause
      cleanly;
    - else ``"interrupted"`` if a ``complete`` frame arrived while an
      ``interrupt`` was pending (a HITL turn waiting on a decision);
    - else ``"complete"``.

    This is exactly the rule ``SessionAdapter._produce`` used to inline over the
    ``iter_event_frames`` stream (``adapters/session.py``) and that
    ``collect_event_frames`` / ``collect_chunk_frames`` — and every hand-rolled
    accumulator the issue counted (four in one session) — would otherwise
    re-implement and drift on. ``_produce`` and both collectors now call this, so
    the ``complete`` / ``interrupted`` / ``error`` state machine lives in one
    tested place. (``verify()``'s simpler pass/fail ``ok`` derives from it too;
    ``"cancelled"`` is orthogonal — a transport concern ``_produce`` sets on
    ``asyncio.CancelledError``, not part of this frame-driven rule.)
    """
    if saw_error:
        return "error"
    if saw_interrupt:
        return "interrupted"
    return "complete"


def _debug_traceback_extra() -> dict:
    """When ``LANGSTAGE_DEBUG`` is enabled, carry the active exception's traceback in the
    terminal ``error`` frame so a surface's ``--traceback`` / debug mode can show WHERE the
    agent crashed, not just ``Type: message`` (gh langstage-vscode #83). Off by default —
    the frame is byte-identical unless debug is explicitly on, so normal output stays clean.
    """
    import os

    if os.getenv("LANGSTAGE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        import traceback

        tb = traceback.format_exc()
        if tb and tb.strip() != "NoneType: None":
            return {"traceback": tb}
    return {}


async def iter_event_frames(
    agent: Any,
    message: str,
    thread_id: str,
    *,
    resume: Any = None,
    max_result_len: int = 500,
    extractors: Any = (),
    state: Any = None,
):
    """Drive an ``ag-ui-langgraph`` agent in-process and yield ``event_to_dict``-
    shaped frames — the SAME wire vocabulary ``StreamParser`` + ``event_to_dict``
    emit (``content`` / ``tool_start`` / ``tool_end`` / ``interrupt`` /
    ``complete`` / ``error``), sourced from AG-UI events instead.

    This is the retirement path for surfaces on the ``event_to_dict`` wire (the
    vscode sidecar and the web ``SessionAdapter``): swap ``StreamParser`` for the
    in-process AG-UI adapter without changing what the client renders.

    Frame keys beyond the original vocabulary are additive: ``content`` frames carry
    ``message_id`` (a change of id is a message boundary), ``tool_end`` carries a real
    ``duration_ms`` when the tool ran through LangChain's tool runtime, and the terminal
    ``complete`` frame carries ``outcome`` (``"complete"`` / ``"interrupted"``). An
    ``error`` frame is terminal. A finished (non-token-streamed) message is emitted when
    its node finishes, text before its own tool calls (see :class:`_TurnStream`).

    ``agent`` is an already-built ``LangGraphAgent`` (see :func:`build_agent`).
    ``resume`` (a decision answering an interrupt) rides
    ``forwarded_props.command.resume`` -> LangGraph ``Command(resume=...)``.

    ``extractors`` is an optional iterable of :class:`~langstage_core.extractors.base.ToolExtractor`
    (``tool_name`` / ``extracted_type`` / ``extract(content)``). After each tool
    result, the matching extractor (by tool name) runs; a non-None return emits an
    ``extraction`` frame identical to ``event_to_dict(ToolExtractedEvent)`` — the
    AG-UI home for domain callouts (e.g. hermes' skill/memory events). An extractor
    whose ``tool_name`` is the ``"*"`` sentinel (e.g.
    :class:`~langstage_core.GenericToolExtractor`) is used as the *fallback* for any
    tool without a specific extractor, so a generic tool-callout card can render
    without per-tool knowledge (gh #90).
    """
    try:
        from ag_ui.core.types import RunAgentInput, UserMessage
    except ImportError as e:  # pragma: no cover - only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    import json
    import uuid

    allowed_decisions = ["reject", "edit", "respond", "approve"]
    # Dispatch extractors by tool name. An extractor whose tool_name is the "*"
    # sentinel (GenericToolExtractor) is the fallback, applied to any tool without a
    # dedicated extractor — otherwise "*" is just a dict key no real tool matches, so
    # the documented public fallback is dead code on the 1.0 wire (gh #90).
    by_tool = {e.tool_name: e for e in extractors if getattr(e, "tool_name", None) != "*"}
    default_extractor = next(
        (e for e in extractors if getattr(e, "tool_name", None) == "*"), None
    )
    resume = _unwrap_resume(resume)  # accept create_resume_input()'s Command too (gh #82)
    forwarded_props = {"command": {"resume": resume}} if resume is not None else {}
    run_input = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state=dict(state or {}),
        messages=[UserMessage(id=str(uuid.uuid4()), role="user", content=message)],
        tools=[],
        context=[],
        forwarded_props=forwarded_props,
    )

    streamed_text = False
    # Message ids already emitted token-by-token (from TextMessageContentEvent), so
    # the final snapshot can emit the assistant messages it did NOT stream without
    # duplicating the streamed ones. A mixed turn — an earlier node streams, a later
    # node returns a finished AIMessage — otherwise dropped the finished message,
    # because the whole snapshot was suppressed once anything streamed (gh #89).
    streamed_ids: set[str] = set()
    tool_args: dict[str, str] = {}
    tool_names: dict[str, str] = {}
    # Tool results already emitted via the streaming ToolCallResultEvent, so the
    # final snapshot re-emits only the results a non-streaming turn never streamed
    # (gh #91), the tool-frame counterpart of streamed_ids.
    streamed_result_ids: set[str] = set()
    # The langgraph node currently executing, from StepStartedEvent — so a
    # multi-node graph's frames carry the real node instead of a fixed "agent",
    # letting renderers separate one node's output from the next (gh #43).
    current_node = "agent"
    step_nodes: list[str] = []
    # Tool error status (gh #55) + duration (gh langstage#160), from the adapter's
    # on_tool_start / on_tool_end / on_tool_error RawEvents — shared with the chunk wire.
    tools = _ToolTracker()
    # An `interrupt` frame went out: the terminal `complete` then says so (gh #152).
    saw_interrupt = False

    try:
        # Accept a bare compiled graph, not just a prebuilt LangGraphAgent (gh #117):
        # anything that already exposes ``.run`` — a LangGraphAgent, or a test double —
        # is driven directly; a raw ``CompiledStateGraph`` / spec string (no ``.run``) is
        # auto-wrapped via build_agent, so ``iter_event_frames(compiled_graph, ...)``
        # streams — matching the collectors and the README's "Stream any CompiledGraph"
        # claim — instead of surfacing a cryptic ``'CompiledStateGraph' object has no
        # attribute 'run'`` error frame. A genuinely bad input makes build_agent raise a
        # clean TypeError -> a terminal ``error`` frame via the except below, never a
        # leaked AttributeError.
        agent = agent if hasattr(agent, "run") else build_agent(agent)
        # Clone per run: a LangGraphAgent keeps per-run state on the instance
        # (active_run), so two concurrent turns on one shared build_agent() agent —
        # the README's "build once, reuse per session" pattern — corrupted each other
        # (TypeError mid-stream). clone() keeps the graph + checkpointer (thread
        # state) but isolates the run, like build_app / SessionAdapter. (gh #165)
        agent = agent.clone() if hasattr(agent, "clone") else agent
        # _TurnStream surfaces finished (non-token-streamed) messages as real AG-UI
        # events at the step that produced them — in message order, before a later
        # node can error — shared with iter_chunk_frames and the served endpoint
        # (gh #140, langstage-cli #119, langstage-vscode #105/#108).
        stream = _TurnStream(agent, run_input)
        async for ev in stream.events():
            t = type(ev).__name__
            if t == "StepStartedEvent":
                step = getattr(ev, "step_name", None)
                if step:
                    current_node = step
                    step_nodes.append(step)
            elif t == "RawEvent":
                tools.on_raw(getattr(ev, "event", None) or {})
            elif t == "TextMessageContentEvent":
                streamed_text = True
                mid = getattr(ev, "message_id", None)
                if mid is not None:
                    streamed_ids.add(mid)
                # message_id: the AIMessage this delta belongs to. A change of id between
                # two content frames is a message boundary, e.g. two nodes' finished
                # replies, which consumers join with a paragraph break instead of gluing
                # them into one line (gh langstage-vscode #108).
                yield {"type": "content", "content": ev.delta, "role": "assistant",
                       "node": current_node, "message_id": mid}
            elif t in ("ReasoningMessageContentEvent", "ThinkingTextMessageContentEvent"):
                # Reasoning-model chain-of-thought (Anthropic extended thinking, o-series,
                # DeepSeek R1, Qwen, xAI, ...). Surface it as the advertised `reasoning`
                # frame so renderers can show/collapse the thinking separately from the
                # answer, instead of dropping it. Does NOT set streamed_text — reasoning
                # isn't the answer, so a reasoning-only turn still falls back to the final
                # snapshot for the reply. (gh #71)
                delta = getattr(ev, "delta", "")
                if delta:
                    yield {"type": "reasoning", "content": delta, "node": current_node}
            elif t == "ToolCallStartEvent":
                tool_names[ev.tool_call_id] = ev.tool_call_name
                tool_args[ev.tool_call_id] = ""
            elif t == "ToolCallArgsEvent":
                tool_args[ev.tool_call_id] = tool_args.get(ev.tool_call_id, "") + ev.delta
            elif t == "ToolCallEndEvent":
                raw = tool_args.pop(ev.tool_call_id, "")
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw}
                yield {
                    "type": "tool_start",
                    "id": ev.tool_call_id,
                    "name": tool_names.get(ev.tool_call_id, "tool"),
                    "args": args,
                    "node": current_node,
                }
            elif t == "ToolCallResultEvent":
                streamed_result_ids.add(ev.tool_call_id)
                result = _truncate_result(str(getattr(ev, "content", "")), max_result_len)
                name = tool_names.get(ev.tool_call_id, "tool")
                is_error, duration_ms = tools.result(
                    ev.tool_call_id, name, ev.tool_call_id in stream.errored_result_ids
                )
                yield {
                    "type": "tool_end",
                    "id": ev.tool_call_id,
                    "name": name,
                    "result": result,
                    "status": "error" if is_error else "success",
                    "error_message": result if is_error else None,
                    "duration_ms": duration_ms,
                }
                # No extraction for a FAILED tool: the extractor would build a
                # success-shaped card out of the framework's error string, contradicting
                # the tool_end that just said status="error" (gh #177).
                extractor = None if is_error else by_tool.get(name, default_extractor)
                if extractor is not None:
                    # A structured extractor parses raw content, so it must see the
                    # full result. A display-passthrough one (GenericToolExtractor,
                    # caps_content=True) echoes content verbatim, so feed it the
                    # already-truncated `result` — otherwise its extraction frame
                    # carries the full blob next to the capped tool_end (gh #106).
                    ex_content = result if getattr(extractor, "caps_content", False) else getattr(ev, "content", "")
                    data = extractor.extract(ex_content)
                    if data is not None:
                        yield {
                            "type": "extraction",
                            "tool_name": name,
                            "extracted_type": extractor.extracted_type,
                            "data": data,
                        }
            elif t == "CustomEvent" and getattr(ev, "name", None) == "on_interrupt":
                payload = getattr(ev, "value", None)
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        # A non-JSON string is the canonical `interrupt("Approve X?")`
                        # HITL form — keep it as the string so _normalize_interrupt
                        # surfaces it as a single action request, instead of dropping
                        # it to `{}` (which rendered "(no action details provided)" and
                        # asked the human to approve blind). (gh langstage-cli #95)
                        pass
                action_requests, review_configs, decisions = _normalize_interrupt(
                    payload, allowed_decisions
                )
                saw_interrupt = True
                yield {
                    "type": "interrupt",
                    "action_requests": action_requests,
                    "review_configs": review_configs,
                    "allowed_decisions": decisions,
                }
            elif t == "MessagesSnapshotEvent":
                # The final snapshot carries every message the turn produced. Emit the
                # ones NOT already streamed — content, tool calls, AND tool results —
                # via the shared _snapshot_items walk, so a non-streaming turn renders
                # its tool call/result and a tool-call-only AIMessage isn't dropped
                # (gh #91), while a mixed turn still emits a later node's finished
                # AIMessage (gh #89) and a fully-streamed turn double-renders nothing.
                # full_text_by_id restores multi-text-block assistant content that the
                # ag-ui snapshot flattens to its first block (gh langstage-vscode #75).
                full_text_by_id = await _full_text_by_id(agent, thread_id)
                for item in _snapshot_items(
                    ev.messages,
                    streamed_ids=streamed_ids,
                    tool_names=tool_names,
                    streamed_result_ids=streamed_result_ids,
                    step_nodes=step_nodes,
                    current_node=current_node,
                    full_text_by_id=full_text_by_id,
                ):
                    if item["kind"] == "content":
                        yield {"type": "content", "content": item["text"],
                               "role": "assistant", "node": item["node"],
                               "message_id": item["id"]}
                    elif item["kind"] == "tool_call":
                        # A snapshot tool call reconstructs the streaming tool_start;
                        # its result arrives as a separate tool message -> tool_end below.
                        yield {"type": "tool_start", "id": item["id"], "name": item["name"],
                               "args": item["args"], "node": current_node}
                    elif item["kind"] == "tool_result":
                        result = _truncate_result(str(item["raw"]), max_result_len)
                        is_error, duration_ms = tools.result(item["id"], item["name"], item["error"])
                        yield {"type": "tool_end", "id": item["id"], "name": item["name"],
                               "result": result, "status": "error" if is_error else "success",
                               "error_message": result if is_error else None,
                               "duration_ms": duration_ms}
                        extractor = None if is_error else by_tool.get(item["name"], default_extractor)
                        if extractor is not None:
                            ex = result if getattr(extractor, "caps_content", False) else str(item["raw"])
                            data = extractor.extract(ex)
                            if data is not None:
                                yield {"type": "extraction", "tool_name": item["name"],
                                       "extracted_type": extractor.extracted_type, "data": data}
            elif t == "RunErrorEvent":
                yield {"type": "error", "error": getattr(ev, "message", "unknown error")}
                return

    except Exception as exc:  # noqa: BLE001 — a node/graph exception during streaming surfaces
        # as the documented terminal `error` frame instead of propagating out of the iterator
        # and crashing the consumer's `async for` (gh #93). Same treatment build_app.gen() and
        # SessionAdapter already apply; the bare upstream adapter lets node exceptions propagate.
        yield {"type": "error", "error": f"{type(exc).__name__}: {exc}", **_debug_traceback_extra()}
        return

    # The terminal frame names the outcome, so a paused HITL turn is distinguishable
    # from a normal finish without replaying the stream: "interrupted" when an
    # `interrupt` frame went out, else "complete". Additive — `type` is unchanged, and an
    # error turn still ends with its `error` frame and no `complete` (gh #152).
    yield {"type": "complete", "outcome": _terminal_outcome(saw_interrupt=saw_interrupt, saw_error=False)}


async def iter_chunk_frames(
    agent: Any,
    message: str,
    thread_id: str,
    *,
    resume: Any = None,
    max_result_len: int = 500,
    extractors: Any = (),
    state: Any = None,
):
    """Drive an ``ag-ui-langgraph`` agent in-process and yield ``status``-keyed
    chunk-dict frames. A ``streaming`` chunk carries exactly ONE payload key — branch on
    it, never assume ``chunk`` (gh #169):

    - ``{"status": "streaming", "chunk": str, "node", "message_id"}`` — assistant text;
      a change of ``message_id`` is a message boundary;
    - ``{"status": "streaming", "reasoning": str, "node"}`` — reasoning deltas;
    - ``{"status": "streaming", "tool_calls": [{"name", "args", "id"}]}``;
    - ``{"status": "streaming", "tool_result": str, "id", "name", "tool_status",
      "duration_ms"}`` — ``tool_status`` is ``"success"`` / ``"error"``;
    - ``{"status": "streaming", "extraction": {"tool_name", "extracted_type", "data"}}``
      (successful tool results only);

    then ``{"status": "interrupt", "interrupt": {...}}``, and exactly one terminal
    frame: ``{"status": "complete", "outcome": "complete" | "interrupted"}`` or
    ``{"status": "error", "error"}`` (nothing follows an error).

    The chunk-dict counterpart of :func:`iter_event_frames`: the render wire the cli and
    Jupyter loops consume. ``resume`` rides ``forwarded_props.command.resume``; ``state``
    seeds the graph input (for agents whose input carries more than ``messages``).

    ``max_result_len`` caps each ``tool_result`` chunk exactly as it caps
    :func:`iter_event_frames`' ``tool_end`` frame — same default (500), same
    ``…(truncated)`` marker, one shared :func:`_truncate_result`. Before gh #102 this
    mapping had no such parameter and emitted the full result, so a tool returning a
    large blob (a file read, a search dump) flooded the terminal/notebook with no way
    to bound it — on the very wire the README assigns to the CLI and Jupyter surfaces.

    ``extractors`` is the same optional iterable of
    :class:`~langstage_core.extractors.base.ToolExtractor` that :func:`iter_event_frames`
    accepts — the README advertises it for *both* ``iter_*`` mappings, and the CLI/Jupyter
    surfaces are on this wire (gh #92). After each tool result the matching extractor (by
    tool name; an extractor whose ``tool_name`` is the ``"*"`` sentinel — e.g.
    :class:`~langstage_core.GenericToolExtractor` — is the fallback for any tool without a
    dedicated one) runs, and a non-None return emits an ``extraction`` chunk
    ``{"status": "streaming", "extraction": {"tool_name", "extracted_type", "data"}}`` — the
    chunk-wire home for the skill/memory/todo callouts the event wire's ``extraction`` frame
    carries.
    """
    try:
        from ag_ui.core.types import RunAgentInput, UserMessage
    except ImportError as e:  # pragma: no cover - only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    import json
    import uuid

    # Dispatch extractors by tool name, with a "*"-tool_name extractor
    # (GenericToolExtractor) as the fallback — the same scheme iter_event_frames uses,
    # so `extractors=[...]` behaves identically on both `iter_*` mappings (gh #90, #92).
    by_tool = {e.tool_name: e for e in extractors if getattr(e, "tool_name", None) != "*"}
    default_extractor = next(
        (e for e in extractors if getattr(e, "tool_name", None) == "*"), None
    )
    resume = _unwrap_resume(resume)  # accept create_resume_input()'s Command too (gh #82)
    forwarded_props = {"command": {"resume": resume}} if resume is not None else {}
    run_input = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state=dict(state or {}),
        messages=[UserMessage(id=str(uuid.uuid4()), role="user", content=message)],
        tools=[],
        context=[],
        forwarded_props=forwarded_props,
    )

    streamed_text = False
    # Message ids already streamed token-by-token, so the final snapshot can emit the
    # assistant messages it did NOT stream without duplicating them — the mixed-turn
    # fix (gh #89), see iter_event_frames for the full rationale.
    streamed_ids: set[str] = set()
    tool_buf: dict[str, dict[str, str]] = {}
    # tool_call_id -> tool name, retained past ToolCallEndEvent (which pops tool_buf) so
    # the later ToolCallResultEvent can dispatch the right extractor by name (gh #92).
    tool_names: dict[str, str] = {}
    # Tool results already streamed (ToolCallResultEvent), so the final snapshot
    # re-emits only what a non-streaming turn never streamed (gh #91).
    streamed_result_ids: set[str] = set()
    # The langgraph node currently executing (from StepStartedEvent) so a multi-node
    # graph's chunks carry the real node instead of a fixed "agent" — renderers use
    # it to separate one node's output from the next (gh #43).
    current_node = "agent"
    step_nodes: list[str] = []
    # Tool error status + duration, the same bookkeeping the event wire uses. The chunk
    # wire used to have none, so a failed tool rendered exactly like a success on the
    # CLI/Jupyter wire (gh #168).
    tools = _ToolTracker()
    saw_interrupt = False  # the terminal `complete` names the outcome (gh #152)

    try:
        # Auto-wrap a bare compiled graph, exactly as iter_event_frames does (gh #117):
        # drive anything with ``.run`` directly (a LangGraphAgent or a test double), else
        # build it — so ``iter_chunk_frames(compiled_graph, ...)`` streams instead of a
        # cryptic ``'CompiledStateGraph' object has no attribute 'run'`` error frame; a
        # bad input becomes a clean terminal ``error`` frame via the except below.
        agent = agent if hasattr(agent, "run") else build_agent(agent)
        agent = agent.clone() if hasattr(agent, "clone") else agent  # per-run isolation (gh #165)
        # Finished messages surface at the step that produced them (see _TurnStream;
        # gh #140, langstage-cli #119, langstage-vscode #105/#108).
        stream = _TurnStream(agent, run_input)
        async for ev in stream.events():
            t = type(ev).__name__
            if t == "StepStartedEvent":
                step = getattr(ev, "step_name", None)
                if step:
                    current_node = step
                    step_nodes.append(step)
            elif t == "RawEvent":
                tools.on_raw(getattr(ev, "event", None) or {})
            elif t == "TextMessageContentEvent":
                streamed_text = True
                mid = getattr(ev, "message_id", None)
                if mid is not None:
                    streamed_ids.add(mid)
                # message_id marks message boundaries, as on the event wire (vscode #108).
                yield {"status": "streaming", "chunk": ev.delta, "node": current_node,
                       "message_id": mid}
            elif t in ("ReasoningMessageContentEvent", "ThinkingTextMessageContentEvent"):
                # Reasoning-model chain-of-thought on the chunk wire — a distinct
                # `reasoning` key (parallel to `chunk`) so renderers can style/collapse
                # it, rather than dropping it. Not the answer, so no streamed_text. (gh #71)
                delta = getattr(ev, "delta", "")
                if delta:
                    yield {"status": "streaming", "reasoning": delta, "node": current_node}
            elif t == "ToolCallStartEvent":
                tool_buf[ev.tool_call_id] = {"name": ev.tool_call_name, "args": ""}
                tool_names[ev.tool_call_id] = ev.tool_call_name
            elif t == "ToolCallArgsEvent":
                tool_buf.setdefault(ev.tool_call_id, {"name": "tool", "args": ""})["args"] += ev.delta
            elif t == "ToolCallEndEvent":
                tc = tool_buf.pop(ev.tool_call_id, {"name": "tool", "args": ""})
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc["args"]}
                # `id` carries the tool-call id the event wire's tool_start has always
                # carried, so collect_chunk_frames / `--json` no longer report `id: None`
                # and a tool_result can be correlated to its call (gh #149).
                yield {"status": "streaming",
                       "tool_calls": [{"name": tc["name"], "args": args, "id": ev.tool_call_id}]}
            elif t == "ToolCallResultEvent":
                # Cap the result the way the event wire has always capped its `tool_end`
                # frame (gh #102) — the CLI/Jupyter render loops read this chunk straight
                # to the terminal. A structured extractor below still sees the FULL
                # content (it parses a payload; truncation is a display concern); a
                # display-passthrough one is fed the capped result (gh #106).
                streamed_result_ids.add(ev.tool_call_id)
                result = _truncate_result(getattr(ev, "content", ""), max_result_len)
                name = tool_names.get(ev.tool_call_id, "tool")
                is_error, duration_ms = tools.result(
                    ev.tool_call_id, name, ev.tool_call_id in stream.errored_result_ids
                )
                # `tool_result` stays the (capped) result string; the sibling keys are
                # additive: `tool_status` ("success" | "error") mirrors the event wire's
                # tool_end `status` (gh #168), plus the call `id` / `name` / `duration_ms`.
                yield {"status": "streaming", "tool_result": result, "id": ev.tool_call_id,
                       "name": name, "tool_status": "error" if is_error else "success",
                       "duration_ms": duration_ms}
                # Run the matching extractor over the tool result and, on a non-None
                # return, emit an `extraction` chunk — parity with iter_event_frames'
                # `extraction` frame so the CLI/Jupyter surfaces can render the same
                # skill/memory/todo callouts (gh #92). Never for a failed tool (gh #177).
                extractor = None if is_error else by_tool.get(name, default_extractor)
                if extractor is not None:
                    ex_content = result if getattr(extractor, "caps_content", False) else getattr(ev, "content", "")
                    data = extractor.extract(ex_content)
                    if data is not None:
                        yield {
                            "status": "streaming",
                            "extraction": {
                                "tool_name": name,
                                "extracted_type": extractor.extracted_type,
                                "data": data,
                            },
                        }
            elif t == "CustomEvent" and getattr(ev, "name", None) == "on_interrupt":
                payload = getattr(ev, "value", None)
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        # A non-JSON string is the canonical `interrupt("Approve X?")`
                        # HITL form — keep it so _normalize_interrupt surfaces it as a
                        # single action request instead of dropping it to `{}`. (cli #95)
                        pass
                # Normalize to a dict with action_requests so a chunk-wire consumer
                # (cli: interrupt_data.get("action_requests")) doesn't crash on the
                # standard HumanInterrupt *list* shape and gets a populated request. (#40)
                action_requests, review_configs, decisions = _normalize_interrupt(payload)
                saw_interrupt = True
                yield {
                    "status": "interrupt",
                    "interrupt": {
                        "action_requests": action_requests,
                        "review_configs": review_configs,
                        "allowed_decisions": decisions,
                    },
                }
            elif t == "MessagesSnapshotEvent":
                # Emit the not-yet-streamed content, tool calls, AND tool results from
                # the final snapshot via the shared _snapshot_items walk. Before gh #91
                # this branch yielded only text and dropped empty-content messages, so a
                # non-streaming tool agent showed no tool call/result and a tool-call-only
                # AIMessage rendered empty. Still emits a later node's finished message
                # (gh #89) and double-renders nothing on a fully-streamed turn.
                # full_text_by_id restores multi-text-block assistant content that the
                # ag-ui snapshot flattens to its first block (gh langstage-vscode #75).
                full_text_by_id = await _full_text_by_id(agent, thread_id)
                for item in _snapshot_items(
                    ev.messages,
                    streamed_ids=streamed_ids,
                    tool_names=tool_names,
                    streamed_result_ids=streamed_result_ids,
                    step_nodes=step_nodes,
                    current_node=current_node,
                    full_text_by_id=full_text_by_id,
                ):
                    if item["kind"] == "content":
                        yield {"status": "streaming", "chunk": item["text"], "node": item["node"],
                               "message_id": item["id"]}
                    elif item["kind"] == "tool_call":
                        yield {"status": "streaming",
                               "tool_calls": [{"name": item["name"], "args": item["args"],
                                               "id": item["id"]}]}
                    elif item["kind"] == "tool_result":
                        result = _truncate_result(item["raw"], max_result_len)
                        is_error, duration_ms = tools.result(item["id"], item["name"], item["error"])
                        yield {"status": "streaming", "tool_result": result, "id": item["id"],
                               "name": item["name"],
                               "tool_status": "error" if is_error else "success",
                               "duration_ms": duration_ms}
                        extractor = None if is_error else by_tool.get(item["name"], default_extractor)
                        if extractor is not None:
                            ex = result if getattr(extractor, "caps_content", False) else item["raw"]
                            data = extractor.extract(ex)
                            if data is not None:
                                yield {"status": "streaming",
                                       "extraction": {"tool_name": item["name"],
                                                      "extracted_type": extractor.extracted_type,
                                                      "data": data}}
            elif t == "RunErrorEvent":
                yield {"status": "error", "error": getattr(ev, "message", "unknown error")}
                # Terminal, exactly like the event wire and the exception path below:
                # without this return a trailing `complete` followed the error (gh #161).
                return

    except Exception as exc:  # noqa: BLE001 — a node/graph exception during streaming surfaces
        # as the documented terminal `error` frame instead of propagating out of the iterator
        # and crashing the consumer's `async for` (gh #93). Same treatment build_app.gen() and
        # SessionAdapter already apply; the bare upstream adapter lets node exceptions propagate.
        yield {"status": "error", "error": f"{type(exc).__name__}: {exc}", **_debug_traceback_extra()}
        return

    # Names the outcome ("interrupted" after an `interrupt` chunk, else "complete"),
    # mirroring the event wire's terminal frame (gh #152). Additive.
    yield {"status": "complete",
           "outcome": _terminal_outcome(saw_interrupt=saw_interrupt, saw_error=False)}


# Imported at the bottom so verify.py / collect.py can reference build_agent /
# iter_event_frames / _terminal_outcome as already-defined names (avoids a circular
# import). See ADR 0004.
from .collect import (  # noqa: E402,F401
    TurnResult,
    collect_chunk_frames,
    collect_event_frames,
    run_turn,
)
from .verify import VerifyResult, averify, verify  # noqa: E402,F401
