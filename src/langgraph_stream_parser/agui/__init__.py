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

    pip install "langgraph-stream-parser[agui]"

Quick start::

    # Serve any agent spec over AG-UI:
    langstage-agui --agent my_agent.py:graph

    # Or in code:
    from langgraph_stream_parser.agui import build_app
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
    "DEFAULT_AGENT_NAME",
]

DEFAULT_AGENT_NAME = "LangStage Agent"

_IMPORT_HINT = (
    "AG-UI support needs the 'agui' extra: "
    'pip install "langgraph-stream-parser[agui]"'
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
    # AG-UI requires threaded state — the adapter calls graph.aget_state() and
    # supports interrupts/resume, both of which need a checkpointer. Many user
    # graphs are compiled without one (and would otherwise hard-crash with
    # "No checkpointer set"), so attach an in-memory default when absent.
    if getattr(graph, "checkpointer", None) is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            graph.checkpointer = InMemorySaver()
        except Exception:  # pragma: no cover - best-effort; LangGraphAgent will surface real issues
            pass
    return LangGraphAgent(name=name, graph=graph, description=description, config=config)


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
                async for ev in run_agent.run(input_data):
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
    :func:`~langgraph_stream_parser.host.load_agent_spec`) or an already
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

    ``agent`` is an already-built ``LangGraphAgent`` (see :func:`build_agent`).
    ``resume`` (a decision answering an interrupt) rides
    ``forwarded_props.command.resume`` -> LangGraph ``Command(resume=...)``.

    ``extractors`` is an optional iterable of :class:`~langgraph_stream_parser.extractors.base.ToolExtractor`
    (``tool_name`` / ``extracted_type`` / ``extract(content)``). After each tool
    result, the matching extractor (by tool name) runs; a non-None return emits an
    ``extraction`` frame identical to ``event_to_dict(ToolExtractedEvent)`` — the
    AG-UI home for domain callouts (e.g. hermes' skill/memory events).
    """
    try:
        from ag_ui.core.types import RunAgentInput, UserMessage
    except ImportError as e:  # pragma: no cover - only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    import json
    import uuid

    allowed_decisions = ["reject", "edit", "respond", "approve"]
    by_tool = {e.tool_name: e for e in extractors}
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
    tool_args: dict[str, str] = {}
    tool_names: dict[str, str] = {}

    async for ev in agent.run(run_input):
        t = type(ev).__name__
        if t == "TextMessageContentEvent":
            streamed_text = True
            yield {"type": "content", "content": ev.delta, "role": "assistant", "node": "agent"}
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
                "node": "agent",
            }
        elif t == "ToolCallResultEvent":
            result = str(getattr(ev, "content", ""))
            if len(result) > max_result_len:
                result = result[:max_result_len] + "…(truncated)"
            name = tool_names.get(ev.tool_call_id, "tool")
            yield {
                "type": "tool_end",
                "id": ev.tool_call_id,
                "name": name,
                "result": result,
                "status": "success",
                "error_message": None,
                "duration_ms": None,
            }
            extractor = by_tool.get(name)
            if extractor is not None:
                data = extractor.extract(getattr(ev, "content", ""))
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
                    payload = {}
            payload = payload or {}
            yield {
                "type": "interrupt",
                "action_requests": payload.get("action_requests", []),
                "review_configs": payload.get("review_configs", []),
                "allowed_decisions": payload.get("allowed_decisions", allowed_decisions),
            }
        elif t == "MessagesSnapshotEvent" and not streamed_text:
            for m in ev.messages:
                if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                    yield {"type": "content", "content": m.content, "role": "assistant", "node": "agent"}
        elif t == "RunErrorEvent":
            yield {"type": "error", "error": getattr(ev, "message", "unknown error")}
            return

    yield {"type": "complete"}


async def iter_chunk_frames(
    agent: Any,
    message: str,
    thread_id: str,
    *,
    resume: Any = None,
    state: Any = None,
):
    """Drive an ``ag-ui-langgraph`` agent in-process and yield ``stream_graph_updates``
    chunk-dict frames (``{"status": "streaming", "chunk"/"tool_calls"/"tool_result": ...}``,
    ``{"status": "interrupt", ...}``, ``{"status": "complete"}``, ``{"status": "error"}``).

    The chunk-dict counterpart of :func:`iter_event_frames`: the retirement path
    for surfaces on the ``stream_graph_updates`` wire (the cli and Jupyter render
    loops). ``resume`` rides ``forwarded_props.command.resume``; ``state`` seeds the
    graph input (for agents whose input carries more than ``messages``).
    """
    try:
        from ag_ui.core.types import RunAgentInput, UserMessage
    except ImportError as e:  # pragma: no cover - only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    import json
    import uuid

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
    tool_buf: dict[str, dict[str, str]] = {}

    async for ev in agent.run(run_input):
        t = type(ev).__name__
        if t == "TextMessageContentEvent":
            streamed_text = True
            yield {"status": "streaming", "chunk": ev.delta, "node": "agent"}
        elif t == "ToolCallStartEvent":
            tool_buf[ev.tool_call_id] = {"name": ev.tool_call_name, "args": ""}
        elif t == "ToolCallArgsEvent":
            tool_buf.setdefault(ev.tool_call_id, {"name": "tool", "args": ""})["args"] += ev.delta
        elif t == "ToolCallEndEvent":
            tc = tool_buf.pop(ev.tool_call_id, {"name": "tool", "args": ""})
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {"_raw": tc["args"]}
            yield {"status": "streaming", "tool_calls": [{"name": tc["name"], "args": args}]}
        elif t == "ToolCallResultEvent":
            yield {"status": "streaming", "tool_result": getattr(ev, "content", "")}
        elif t == "CustomEvent" and getattr(ev, "name", None) == "on_interrupt":
            payload = getattr(ev, "value", None)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {"action_requests": []}
            yield {"status": "interrupt", "interrupt": payload or {"action_requests": []}}
        elif t == "MessagesSnapshotEvent" and not streamed_text:
            for m in ev.messages:
                if getattr(m, "role", None) == "assistant" and getattr(m, "content", None):
                    yield {"status": "streaming", "chunk": m.content, "node": "agent"}
        elif t == "RunErrorEvent":
            yield {"status": "error", "error": getattr(ev, "message", "unknown error")}

    yield {"status": "complete"}
