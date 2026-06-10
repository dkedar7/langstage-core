"""A canonical default deep agent for hosts to fall back on.

Hosts used to each ship their own "default agent" so a user could try the tool
without configuring one. This is the single shared version. It is filesystem
only by design — each host injects its own domain tools (notebook cells,
canvas, etc.) via the ``tools`` argument rather than this module trying to be
everything.

Requires the optional ``deepagents`` dependency:

    pip install langgraph-stream-parser[demo]

This package also ships the keyless **stub agent** behind every surface's
``--demo`` mode (:func:`create_stub_agent` / spec
``langgraph_stream_parser.demo.stub:graph``) — that one needs no extra and no
API key.
"""
from .agent import create_default_agent
from .stub import create_stub_agent

__all__ = ["create_default_agent", "create_stub_agent"]
