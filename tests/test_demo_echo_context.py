"""The demo agents echo only what the user typed, not surface context (gh #192).

Surfaces add host context to the human message before it reaches the agent: the
web app prepends ``[Current time: ...]`` / ``[Working directory: ...]`` /
``[File browser folder: ...]`` lines (via ``prepare_agent_input(context_parts=...)``)
and JupyterLab appends a ``Current directory: ...`` / ``Currently focused: ...``
paragraph. The keyless demo agents used to echo all of it back.
"""
import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import HumanMessage  # noqa: E402

from langstage_core import prepare_agent_input  # noqa: E402
from langstage_core.demo._context import user_text  # noqa: E402
from langstage_core.demo.stub import create_stub_agent  # noqa: E402
from langstage_core.demo.tools import create_tool_demo_agent  # noqa: E402

WEB_PARTS = [
    "[Current time: 2026-09-26 12:00:00 UTC]",
    "[Working directory: /tmp/runner/workspace]",
    "[File browser folder: /tmp/runner/workspace/reports - the folder the user has "
    "open; relative paths still resolve against the working directory]",
]
JUPYTER_SUFFIX = (
    "\n\nCurrent directory: /home/me/work\n"
    "Currently focused file: analysis.ipynb\n"
    "User has selected the following text from cell index 2:\n```\nx = 1\n\ny = 2\n```"
)


def _web(message: str) -> str:
    return prepare_agent_input(message=message, context_parts=WEB_PARTS)["messages"][0][
        "content"
    ]


def _reply(graph, content: str) -> str:
    out = graph.invoke({"messages": [HumanMessage(content=content)]})
    return out["messages"][-1].content


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello", "hello"),
        (_web("hello there"), "hello there"),
        ("hello there" + JUPYTER_SUFFIX, "hello there"),
        (_web("para one\n\npara two"), "para one\n\npara two"),
        ("[not context] but typed inline", "[not context] but typed inline"),
        ("line one\n[bracketed later]", "line one\n[bracketed later]"),
    ],
)
def test_user_text(raw, expected):
    assert user_text(raw) == expected


@pytest.mark.parametrize("wrap", [_web, lambda m: m + JUPYTER_SUFFIX])
def test_stub_echoes_only_user_text(wrap):
    reply = _reply(create_stub_agent(), wrap("hello there"))
    assert reply == "(demo agent) You said: hello there"


@pytest.mark.parametrize("wrap", [_web, lambda m: m + JUPYTER_SUFFIX])
def test_tools_demo_echoes_only_user_text(wrap):
    reply = _reply(create_tool_demo_agent(), wrap("hello there"))
    assert reply == "(demo agent) You said: hello there"


def test_tools_demo_triggers_ignore_context():
    # A workspace path containing a trigger word must not change the route.
    parts = ["[Working directory: /home/me/think-tank]"]
    content = prepare_agent_input(message="hi", context_parts=parts)["messages"][0][
        "content"
    ]
    assert _reply(create_tool_demo_agent(), content) == "(demo agent) You said: hi"


def test_tools_demo_tool_query_is_user_text():
    out = create_tool_demo_agent().invoke(
        {"messages": [HumanMessage(content=_web("please use a tool"))]}
    )
    call = next(m for m in out["messages"] if getattr(m, "tool_calls", None))
    assert call.tool_calls[0]["args"]["query"] == "please use a tool"
