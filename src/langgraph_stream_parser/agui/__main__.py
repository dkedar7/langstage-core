"""``langstage-agui`` — serve a LangGraph agent over the AG-UI protocol.

    langstage-agui --agent my_agent.py:graph
    langstage-agui --demo                       # keyless echo agent, no API key
    langstage-agui --agent langstage_hermes.agent:graph --port 9000

The agent spec resolves through the shared host config chain, so
``LANGSTAGE_AGENT_SPEC`` / ``langstage.toml`` work too (legacy ``DEEPAGENT_*``
still honoured).
"""
from __future__ import annotations

import sys

DEMO_SPEC = "langgraph_stream_parser.demo.stub:graph"


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
        action="store_true",
        help="Serve the built-in keyless demo agent (no API key needed).",
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
        "--version",
        action="store_true",
        help="Print the langgraph-stream-parser version and exit.",
    )
    args = parser.parse_args(argv)

    if args.version:
        from importlib.metadata import PackageNotFoundError, version

        try:
            print(f"langstage-agui (langgraph-stream-parser {version('langgraph-stream-parser')})")
        except PackageNotFoundError:  # pragma: no cover
            print("langstage-agui (langgraph-stream-parser 0.0.0+local)")
        return 0

    from ..host import HostConfig

    # CLI --host/--port are overrides on the resolved config so --show-config and
    # the actual bind always agree (and env / langstage.toml host/port work).
    host_port = {}
    if args.host is not None:
        host_port["host"] = args.host
    if args.port is not None:
        host_port["port"] = args.port

    if args.show_config:
        print(HostConfig.resolve(overrides=host_port).describe())
        return 0

    if args.demo and args.agent:
        print("error: --demo and --agent are mutually exclusive", file=sys.stderr)
        return 2

    overrides = dict(host_port)
    if not args.demo:
        overrides["agent_spec"] = args.agent
    cfg = HostConfig.resolve(overrides=overrides)
    spec: str | None = DEMO_SPEC if args.demo else cfg.agent_spec

    if not spec:
        print(
            "error: no agent spec — pass --agent, --demo, set LANGSTAGE_AGENT_SPEC, "
            "or add [agent].spec to langstage.toml",
            file=sys.stderr,
        )
        return 2

    from . import DEFAULT_AGENT_NAME, serve

    name = args.name or ("Demo Agent" if args.demo else DEFAULT_AGENT_NAME)
    # cfg.host/cfg.port are the resolved values --show-config prints, so the
    # advertised config and the real bind agree.
    print(f"Serving {spec!r} over AG-UI at http://{cfg.host}:{cfg.port}{args.path}")
    serve(spec, host=cfg.host, port=cfg.port, path=args.path, name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
