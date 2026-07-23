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


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="langstage-agui",
        description="Serve a LangGraph agent over the AG-UI protocol.",
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
        print("error: --demo and --agent are mutually exclusive", file=sys.stderr)
        return 2

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

    if args.show_config:
        # The AG-UI server consumes only agent_spec/host/port. Drop the inherited
        # workspace_root/debug/title rows so --show-config doesn't advertise env
        # vars that have no effect on this surface (same omit_keys treatment the
        # stdio sidecar and JupyterLab launcher already use). (gh #39)
        described = cfg.describe(omit_keys=["workspace_root", "debug", "title"])
        if args.demo:
            described += f"\n  demo: agent_spec resolves to {DEMO_SPECS[args.demo]}"
        print(described)
        return 0

    spec: str | None = DEMO_SPECS[args.demo] if args.demo else cfg.agent_spec

    if not spec:
        print(
            "error: no agent spec — pass --agent, --demo, set LANGSTAGE_AGENT_SPEC, "
            "or add [agent].spec to langstage.toml",
            file=sys.stderr,
        )
        return 2

    from . import DEFAULT_AGENT_NAME, ensure_available, serve

    # Fail fast on a missing [agui] extra with a clean hint to stderr — before the
    # "Serving … at <url>" banner, so the user doesn't see a fake success line
    # followed by a traceback (gh #-dogfood).
    try:
        ensure_available()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Resolve the spec BEFORE announcing success. serve() loads the spec itself, so
    # an unloadable one (typo'd module, missing attribute, nonexistent file — the
    # most common CLI mistake) used to surface as a raw traceback *after* a banner
    # claiming the server was already up. Loading here gives that case the same
    # clean one-line stderr treatment as its two siblings above — the missing
    # [agui] extra and the no-spec-at-all path. load_agent_spec() already raises
    # descriptive errors, so the message needs no embellishment. (gh #100)
    from ..host import load_agent_spec

    try:
        graph = load_agent_spec(spec)
    # ImportError covers ModuleNotFoundError, OSError covers FileNotFoundError, and
    # ValueError is the malformed-spec ("no :attr suffix") case load_agent_spec raises.
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        print(f"error: could not load agent {spec!r}: {exc}", file=sys.stderr)
        return 2

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

    name = args.name or (DEMO_NAMES[args.demo] if args.demo else DEFAULT_AGENT_NAME)
    # cfg.host/cfg.port are the resolved values --show-config prints, so the
    # advertised config and the real bind agree.
    print(f"Serving {spec!r} over AG-UI at http://{cfg.host}:{cfg.port}{args.path}")
    # Pass the loaded graph, not the spec: serve() accepts either, and handing it
    # the graph keeps the module from being imported (and its side effects run) twice.
    serve(graph, host=cfg.host, port=cfg.port, path=args.path, name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
