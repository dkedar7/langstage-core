"""``langstage-agui`` — serve a LangGraph agent over the AG-UI protocol.

    langstage-agui --agent my_agent.py:graph
    langstage-agui --demo                       # keyless echo agent, no API key
    langstage-agui --demo=tools                 # keyless rich-frame demo (tools/reasoning/interrupt)
    langstage-agui --agent langstage_hermes.agent:graph --port 9000

The agent spec resolves through the shared host config chain, so
``LANGSTAGE_AGENT_SPEC`` / ``langstage.toml`` work too (legacy ``DEEPAGENT_*``
still honoured).
"""
from __future__ import annotations

import sys
from typing import Any

from ..cli import EXIT_FAIL, EXIT_OK, ArgumentParser, exit_code_for_outcome, usage_error

# The keyless built-in demos, keyed by the value of --demo. Bare `--demo` selects
# "echo" (the plain token echo stub, unchanged); `--demo=tools` selects the rich
# demo that exercises every frame type — tool_start/tool_end/extraction/reasoning/
# interrupt (gh #99). DEMO_SPEC stays the echo spec for backward compatibility.
DEMO_SPECS = {
    "echo": "langstage_core.demo.stub:graph",
    "tools": "langstage_core.demo.tools:graph",
}
DEMO_SPEC = DEMO_SPECS["echo"]
DEMO_NAMES = {"echo": "Demo Agent", "tools": "Tool Demo Agent"}


def _exit_code_for(outcome: str) -> int:
    """The `--message` exit code from the turn outcome (gh #120): complete=0,
    error=1, interrupted=2 — the family scheme (ADR 0007)."""
    return exit_code_for_outcome(outcome)


def _run_message(graph: Any, message: str, *, as_json: bool, config: Any = None) -> int:
    """Run ONE turn against ``graph`` with ``message`` and print the reply (gh #120).

    Streams text to stdout over the shipped chunk wire (``iter_chunk_frames``) for the
    human path; ``--json`` prints the typed ``TurnResult`` for scripting. Exit code
    mirrors the turn outcome via :func:`_exit_code_for`. ``config`` is forwarded to
    ``build_agent`` (the ``langstage.toml`` ``[configurable]`` table, gh #170).
    """
    import asyncio

    from ..console import safe_print, safe_write
    from . import _debug_traceback_extra, build_agent, iter_chunk_frames
    from .collect import collect_chunk_frames

    try:
        agent = build_agent(graph, config=config)
    except Exception as exc:  # noqa: BLE001 - reported as a clean error, like --verify
        # A spec that LOADS but isn't a runnable graph (an uncompiled StateGraph, a
        # factory function, any other attribute) makes build_agent raise an actionable
        # TypeError. --verify already reports it as one clean line; --message used to
        # dump a raw traceback, and --json printed no JSON at all. Same one-line
        # `error:` as --verify (exit 1), and a typed error TurnResult under --json so a
        # script still gets parseable output. The traceback only under LANGSTAGE_DEBUG
        # (the gh #132 convention). (gh #180)
        detail = f"{type(exc).__name__}: {exc}"
        tb = _debug_traceback_extra().get("traceback")
        if as_json:
            import json

            safe_print(json.dumps({
                "text": "",
                "outcome": "error",
                "tool_calls": [],
                "extractions": [],
                "reasoning": "",
                "interrupt": None,
                "error": detail,
                "traceback": tb,
            }))
        else:
            safe_print(f"error: agent did not complete a turn: {detail}", file=sys.stderr)
            if tb:
                safe_write(tb if tb.endswith("\n") else tb + "\n", sys.stderr)
        return _exit_code_for("error")

    if as_json:
        import json

        result = asyncio.run(collect_chunk_frames(agent, message, "oneshot"))
        print(json.dumps({
            "text": result.text,
            "outcome": result.outcome,
            "tool_calls": result.tool_calls,
            "extractions": result.extractions,
            "reasoning": result.reasoning,
            "interrupt": result.interrupt,
            "error": result.error,
            "traceback": result.traceback,  # non-null on error under LANGSTAGE_DEBUG (gh #132)
        }))
        return _exit_code_for(result.outcome)

    outcome = "complete"

    async def _stream() -> None:
        nonlocal outcome
        wrote_text = False
        async for chunk in iter_chunk_frames(agent, message, "oneshot"):
            status = chunk.get("status")
            if status == "streaming" and "chunk" in chunk:
                # Model text is arbitrary Unicode: escape what the console can't
                # encode instead of crashing mid-reply on cp1252 (gh #153).
                safe_write(chunk["chunk"], flush=True)
                wrote_text = True
            elif status == "interrupt":
                outcome = "interrupted"
                info = chunk.get("interrupt", {})
                reqs = info.get("action_requests") if isinstance(info, dict) else None
                sys.stderr.write(f"\n[interrupt] agent paused for input: {reqs}\n")
            elif status == "error":
                outcome = "error"
                sys.stderr.write(f"\nerror: {chunk.get('error')}\n")
                # Under LANGSTAGE_DEBUG the error frame carries the crash traceback
                # (gh #132) — print it so `--message` shows WHERE, not just the message.
                tb = chunk.get("traceback")
                if tb:
                    sys.stderr.write(tb if tb.endswith("\n") else tb + "\n")
        if wrote_text:
            sys.stdout.write("\n")

    asyncio.run(_stream())
    return _exit_code_for(outcome)


def main(argv: list[str] | None = None) -> int:
    # Usage errors exit 64, not argparse's 2 (which the family reserves for "paused on
    # a HITL interrupt"). Exit codes: 0 ok / 1 failed / 2 paused / 64 usage (ADR 0007).
    parser = ArgumentParser(
        prog="langstage-agui",
        description="Serve a LangGraph agent over the AG-UI protocol.",
        epilog="Exit codes: 0 ok, 1 failed (no/bad agent, turn error, can't serve), "
        "2 paused on a human-in-the-loop interrupt, 64 usage error.",
    )
    parser.add_argument(
        "--agent",
        "-a",
        dest="agent",
        default=None,
        help="Agent spec (module:attr or path/to/file.py:attr). "
        "Falls back to LANGSTAGE_AGENT_SPEC / langstage.toml.",
    )
    parser.add_argument(
        "--demo",
        nargs="?",
        const="echo",
        choices=["echo", "tools"],
        default=None,
        help="Serve a built-in keyless demo agent (no API key needed). Bare --demo "
        "serves the echo stub; --demo=tools serves the rich-frame demo that exercises "
        "tool_start/tool_end/extraction/reasoning/interrupt.",
    )
    # host/port default to None so the resolved HostConfig (env / langstage.toml /
    # defaults) supplies them and --show-config matches the real bind; an explicit
    # flag overrides. Previously these defaulted to 127.0.0.1:8000 in argparse
    # while --show-config printed HostConfig's localhost:8050 — advertised values
    # disagreed with what the server actually bound (gh #-dogfood).
    parser.add_argument("--host", default=None, help="Bind host (default from host config).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default from host config).")
    parser.add_argument("--path", default="/", help="Endpoint path (default '/').")
    parser.add_argument("--name", default=None, help="Agent display name for AG-UI clients.")
    parser.add_argument(
        "--cors",
        nargs="?",
        const="loopback",
        default=None,
        metavar="ORIGIN[,ORIGIN...]",
        help="Allow a browser frontend on another origin (CORS; off by default). Bare "
        "--cors allows any localhost origin; or list origins, e.g. "
        "--cors http://localhost:5173. '*' only if you pass it explicitly.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved host config and exit.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run one keyless turn against the agent and report whether it works, "
        "then exit (0 ok / 1 failed). The preflight to run right after --agent.",
    )
    parser.add_argument(
        "--message",
        "-m",
        dest="message",
        default=None,
        help="Run ONE turn against the resolved agent with this prompt, print the "
        "reply, and exit (0 complete / 1 error / 2 interrupted). The terminal "
        "smoke-test companion to --verify: --show-config (resolves?) -> --verify "
        "(runs?) -> --message (what does it say?).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="With --message, print the typed TurnResult as JSON (text, tool_calls, "
        "extractions, reasoning, outcome, interrupt, error) for scripting.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the langstage-core version and exit.",
    )
    args = parser.parse_args(argv)

    if args.version:
        from importlib.metadata import PackageNotFoundError, version

        try:
            print(f"langstage-agui (langstage-core {version('langstage-core')})")
        except PackageNotFoundError:  # pragma: no cover
            print("langstage-agui (langstage-core 0.0.0+local)")
        return 0

    from ..host import HostConfig

    if args.demo and args.agent:
        # A usage error on every command (ADR 0007): 64, never 2 (== paused).
        return usage_error(parser, "--demo and --agent are mutually exclusive")

    # CLI flags are overrides on the resolved config so --show-config and the actual
    # bind always agree (and env / langstage.toml host/port/agent work). --agent must
    # be applied HERE, before the --show-config branch — otherwise --show-config
    # resolved without it and reported agent_spec = None while serving used the flag
    # (advertised != honored). (gh #60)
    overrides: dict = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if not args.demo:
        overrides["agent_spec"] = args.agent

    cfg = HostConfig.resolve(overrides=overrides)
    # The [configurable] table is forwarded to the graph on the serve and --message
    # paths below and shown by --show-config, instead of being reserved-but-inert
    # (gh #170). An empty/absent table means no config, exactly as before.
    configurable = cfg.configurable()
    run_config = {"configurable": configurable} if configurable else None

    from ..console import safe_print

    if args.show_config:
        # The AG-UI server consumes agent_spec/host/port, and debug (it gates the
        # traceback on error frames, gh #137). Drop the inherited workspace_root/title
        # rows so --show-config doesn't advertise env vars that have no effect on this
        # surface (same omit_keys treatment the stdio sidecar and JupyterLab launcher
        # already use). (gh #39)
        omit = ["workspace_root", "title"]
        if args.as_json:
            # --show-config --json emits the machine-readable config_dict (the structured
            # twin of describe) instead of silently ignoring --json and printing the human
            # table — so a CI/tooling consumer gets JSON, not scraped brackets. (gh #125)
            import json

            cd = cfg.config_dict(omit_keys=omit, configurable=configurable)
            if args.demo:
                cd["demo"] = {"agent_spec": DEMO_SPECS[args.demo]}
            print(json.dumps(cd, default=str))
            return 0
        described = cfg.describe(omit_keys=omit, configurable=configurable)
        if args.demo:
            described += f"\n  demo: agent_spec resolves to {DEMO_SPECS[args.demo]}"
        # Values are user-controlled (a CJK spec path, ...): never crash on cp1252 (gh #171).
        safe_print(described)
        return 0

    spec: str | None = DEMO_SPECS[args.demo] if args.demo else cfg.agent_spec

    if not spec:
        print(
            "error: no agent spec — pass --agent, --demo, set LANGSTAGE_AGENT_SPEC, "
            "or add [agent].spec to langstage.toml",
            file=sys.stderr,
        )
        # No spec is a "can't run" failure on every command (ADR 0007). A bare 2 here
        # let a CI gate read an unconfigured agent as a benign HITL pause — fail-open
        # (gh #174).
        return EXIT_FAIL

    from . import DEFAULT_AGENT_NAME, ensure_available, serve

    # Fail fast on a missing [agui] extra with a clean hint to stderr — before the
    # "Serving … at <url>" banner, so the user doesn't see a fake success line
    # followed by a traceback (gh #-dogfood).
    try:
        ensure_available()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        # A missing [agui] extra is a "can't run" failure: 1 on every command, never
        # 2 (== paused) (gh #134, ADR 0007).
        return EXIT_FAIL

    # Resolve the spec BEFORE announcing success. serve() loads the spec itself, so
    # an unloadable one (typo'd module, missing attribute, nonexistent file — the
    # most common CLI mistake) used to surface as a raw traceback *after* a banner
    # claiming the server was already up. Loading here gives that case the same
    # clean one-line stderr treatment as its two siblings above — the missing
    # [agui] extra and the no-spec-at-all path. load_agent_spec() already raises
    # descriptive errors, so the message needs no embellishment. (gh #100)
    from ..host import load_agent_spec

    try:
        # --verify / --message print a verdict or reply (--json: a JSON object) on
        # stdout, so the agent's import-time prints go to stderr there and can't
        # corrupt what a script captures (gh langstage-cli #136, langstage #140).
        graph = load_agent_spec(
            spec,
            # A dotted `pkg.mod:attr` from langstage.toml imports relative to that file.
            base_dir=None if args.demo else cfg.toml_dir_for("agent_spec"),
            stdout_to_stderr=bool(args.verify or args.message is not None),
        )
    # ImportError covers ModuleNotFoundError, OSError covers FileNotFoundError,
    # ValueError is the malformed-spec ("no :attr suffix") case, and TypeError is a
    # spec that resolved to a str (gh langstage-cli #149).
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        print(f"error: could not load agent {spec!r}: {exc}", file=sys.stderr)
        # A load failure is a failure (1) on every command, never 2 (== paused)
        # (gh #124, ADR 0007).
        return EXIT_FAIL

    # --verify: the question every adopter asks right after --agent — "did it load
    # AND actually produce a turn?" — which --show-config can't answer (a spec that
    # resolves can still fail to run). Drive the already-shipped keyless verify()
    # over the loaded graph and report, instead of making the user hand-craft an
    # AG-UI POST or write async Python around iter_*. (gh #105)
    if args.verify:
        from . import verify

        result = verify(graph)
        if result.ok:
            print(f"ok: {result.reason} ({result.frames} frames, {result.content_chars} chars)")
            return 0
        detail = result.error_message or result.reason
        print(f"error: agent did not complete a turn: {detail}", file=sys.stderr)
        return 1

    # --message: the companion to --verify. --verify proves the agent *runs* with a
    # canned probe whose output is discarded; --message runs the user's OWN prompt and
    # prints the reply — the terminal smoke-test / quick-chat, with no Python and no
    # server. A thin wrapper over the shipped chunk wire; exit code mirrors the turn
    # outcome (complete=0 / error=1 / interrupted=2), consistent with --verify. (gh #120)
    if args.message is not None:
        return _run_message(graph, args.message, as_json=args.as_json, config=run_config)

    name = args.name or (DEMO_NAMES[args.demo] if args.demo else DEFAULT_AGENT_NAME)
    url = f"http://{cfg.host}:{cfg.port}{args.path}"
    # Bind BEFORE the banner. A port already in use used to print the definitive
    # "Serving … at <url>" success line to stdout and only THEN have uvicorn log the
    # bind error and exit 3 — the false-green banner #100/#134 removed for the other
    # can't-serve cases. Now it's the same clean one-line stderr error and the family
    # can't-start code (1, ADR 0007), and the banner prints only once the port is
    # actually held. (gh #143)
    from . import _bind_socket

    try:
        sock = _bind_socket(cfg.host, cfg.port)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        safe_print(f"error: cannot serve at {url}: {reason}", file=sys.stderr)
        return EXIT_FAIL
    # cfg.host/cfg.port are the resolved values --show-config prints, so the
    # advertised config and the real bind agree.
    safe_print(f"Serving {spec!r} over AG-UI at {url}", flush=True)
    # Pass the loaded graph, not the spec: serve() accepts either, and handing it
    # the graph keeps the module from being imported (and its side effects run) twice.
    try:
        serve(graph, host=cfg.host, port=cfg.port, path=args.path, name=name, config=run_config,
              sock=sock, cors_origins=args.cors)
    finally:
        sock.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
