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


def _snapshot_items(messages, *, streamed_ids, tool_names, streamed_result_ids, step_nodes, current_node):
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

      ``{"kind": "content", "text", "node"}``          — an assistant text message
      ``{"kind": "tool_call", "name", "args", "id"}``  — a tool call not streamed
      ``{"kind": "tool_result", "name", "raw", "id", "error"}`` — a result not streamed

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
                yield {"kind": "tool_call", "name": name, "args": args, "id": tcid}
            content = getattr(m, "content", None)
            if content:
                if getattr(m, "id", None) not in streamed_ids:
                    idx = ci - offset
                    node = step_nodes[idx] if 0 <= idx < len(step_nodes) else current_node
                    yield {"kind": "content", "text": content, "node": node}
                ci += 1
        elif role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid in streamed_result_ids:
                continue  # already emitted via the streaming ToolCallResultEvent
            yield {
                "kind": "tool_result",
                "name": names.get(tcid, "tool"),
                "raw": getattr(m, "content", "") or "",
                "id": tcid,
                "error": bool(getattr(m, "error", None)),
            }


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
    # Tool names that raised (from on_tool_error RawEvents). The AG-UI
    # ToolCallResultEvent drops the ToolMessage status, so a failed tool otherwise
    # renders as "success"; on_tool_error fires before the result, so we flag it by
    # name and correct the tool_end frame. (gh #55)
    errored_tools: dict[str, int] = {}

    try:
        async for ev in agent.run(run_input):
            t = type(ev).__name__
            if t == "StepStartedEvent":
                step = getattr(ev, "step_name", None)
                if step:
                    current_node = step
                    step_nodes.append(step)
            elif t == "RawEvent":
                raw = getattr(ev, "event", None) or {}
                if raw.get("event") == "on_tool_error":
                    nm = raw.get("name")
                    if nm:
                        errored_tools[nm] = errored_tools.get(nm, 0) + 1
            elif t == "TextMessageContentEvent":
                streamed_text = True
                mid = getattr(ev, "message_id", None)
                if mid is not None:
                    streamed_ids.add(mid)
                yield {"type": "content", "content": ev.delta, "role": "assistant", "node": current_node}
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
                is_error = errored_tools.get(name, 0) > 0
                if is_error:
                    errored_tools[name] -= 1
                yield {
                    "type": "tool_end",
                    "id": ev.tool_call_id,
                    "name": name,
                    "result": result,
                    "status": "error" if is_error else "success",
                    "error_message": result if is_error else None,
                    "duration_ms": None,
                }
                extractor = by_tool.get(name, default_extractor)
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
                        payload = {}
                action_requests, review_configs, decisions = _normalize_interrupt(
                    payload, allowed_decisions
                )
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
                for item in _snapshot_items(
                    ev.messages,
                    streamed_ids=streamed_ids,
                    tool_names=tool_names,
                    streamed_result_ids=streamed_result_ids,
                    step_nodes=step_nodes,
                    current_node=current_node,
                ):
                    if item["kind"] == "content":
                        yield {"type": "content", "content": item["text"],
                               "role": "assistant", "node": item["node"]}
                    elif item["kind"] == "tool_call":
                        # A snapshot tool call reconstructs the streaming tool_start;
                        # its result arrives as a separate tool message -> tool_end below.
                        yield {"type": "tool_start", "id": item["id"], "name": item["name"],
                               "args": item["args"], "node": current_node}
                    elif item["kind"] == "tool_result":
                        result = _truncate_result(str(item["raw"]), max_result_len)
                        yield {"type": "tool_end", "id": item["id"], "name": item["name"],
                               "result": result, "status": "error" if item["error"] else "success",
                               "error_message": result if item["error"] else None, "duration_ms": None}
                        extractor = by_tool.get(item["name"], default_extractor)
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
        yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        return

    yield {"type": "complete"}


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
    """Drive an ``ag-ui-langgraph`` agent in-process and yield ``stream_graph_updates``
    chunk-dict frames (``{"status": "streaming", "chunk"/"tool_calls"/"tool_result"/"extraction": ...}``,
    ``{"status": "interrupt", ...}``, ``{"status": "complete"}``, ``{"status": "error"}``).

    The chunk-dict counterpart of :func:`iter_event_frames`: the retirement path
    for surfaces on the ``stream_graph_updates`` wire (the cli and Jupyter render
    loops). ``resume`` rides ``forwarded_props.command.resume``; ``state`` seeds the
    graph input (for agents whose input carries more than ``messages``).

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

    try:
        async for ev in agent.run(run_input):
            t = type(ev).__name__
            if t == "StepStartedEvent":
                step = getattr(ev, "step_name", None)
                if step:
                    current_node = step
                    step_nodes.append(step)
            elif t == "TextMessageContentEvent":
                streamed_text = True
                mid = getattr(ev, "message_id", None)
                if mid is not None:
                    streamed_ids.add(mid)
                yield {"status": "streaming", "chunk": ev.delta, "node": current_node}
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
                yield {"status": "streaming", "tool_calls": [{"name": tc["name"], "args": args}]}
            elif t == "ToolCallResultEvent":
                # Cap the result the way the event wire has always capped its `tool_end`
                # frame (gh #102) — the CLI/Jupyter render loops read this chunk straight
                # to the terminal. A structured extractor below still sees the FULL
                # content (it parses a payload; truncation is a display concern); a
                # display-passthrough one is fed the capped result (gh #106).
                streamed_result_ids.add(ev.tool_call_id)
                result = _truncate_result(getattr(ev, "content", ""), max_result_len)
                yield {"status": "streaming", "tool_result": result}
                # Run the matching extractor over the tool result and, on a non-None
                # return, emit an `extraction` chunk — parity with iter_event_frames'
                # `extraction` frame so the CLI/Jupyter surfaces can render the same
                # skill/memory/todo callouts (gh #92).
                name = tool_names.get(ev.tool_call_id, "tool")
                extractor = by_tool.get(name, default_extractor)
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
                        payload = {}
                # Normalize to a dict with action_requests so a chunk-wire consumer
                # (cli: interrupt_data.get("action_requests")) doesn't crash on the
                # standard HumanInterrupt *list* shape and gets a populated request. (#40)
                action_requests, review_configs, decisions = _normalize_interrupt(payload)
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
                for item in _snapshot_items(
                    ev.messages,
                    streamed_ids=streamed_ids,
                    tool_names=tool_names,
                    streamed_result_ids=streamed_result_ids,
                    step_nodes=step_nodes,
                    current_node=current_node,
                ):
                    if item["kind"] == "content":
                        yield {"status": "streaming", "chunk": item["text"], "node": item["node"]}
                    elif item["kind"] == "tool_call":
                        yield {"status": "streaming",
                               "tool_calls": [{"name": item["name"], "args": item["args"]}]}
                    elif item["kind"] == "tool_result":
                        result = _truncate_result(item["raw"], max_result_len)
                        yield {"status": "streaming", "tool_result": result}
                        extractor = by_tool.get(item["name"], default_extractor)
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

    except Exception as exc:  # noqa: BLE001 — a node/graph exception during streaming surfaces
        # as the documented terminal `error` frame instead of propagating out of the iterator
        # and crashing the consumer's `async for` (gh #93). Same treatment build_app.gen() and
        # SessionAdapter already apply; the bare upstream adapter lets node exceptions propagate.
        yield {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        return

    yield {"status": "complete"}


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
