"""A crashing turn is reported once, by core, as before ag-ui-langgraph 0.0.46 (gh #195).

ag-ui-langgraph 0.0.46 catches the graph's exception inside ``run()``, logs it with
``logger.exception("LangGraph run failed")`` and emits a RUN_ERROR with only ``str(exc)``.
That printed a ~50-line traceback to stderr on every surface (debug or not) and stripped
the exception type and the debug traceback from core's ``error`` frames. Core now takes the
exception from that log record while it drives a run and re-raises it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langstage_core.agui import _CaptureRunFailure, _install_run_failure_capture

CRASH_AGENT = textwrap.dedent(
    """
    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    class S(TypedDict):
        messages: Annotated[list, add_messages]

    def boom(state):
        raise RuntimeError("deliberate failure")

    _g = StateGraph(S)
    _g.add_node("boom", boom)
    _g.add_edge(START, "boom")
    _g.add_edge("boom", END)
    graph = _g.compile()
    """
)


def _run(tmp_path, *args, debug=False):
    (tmp_path / "crashagent.py").write_text(CRASH_AGENT)
    env = {k: v for k, v in os.environ.items() if not k.endswith("_DEBUG")}
    env["PYTHONIOENCODING"] = "utf-8"
    if debug:
        env["LANGSTAGE_DEBUG"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "langstage_core.agui", "--agent", "crashagent.py:graph", *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("args", [["--verify"], ["-m", "hi"]])
def test_crashing_turn_prints_one_error_line_without_a_traceback(tmp_path, args):
    r = _run(tmp_path, *args)
    assert r.returncode == 1, r.stderr
    assert "RuntimeError: deliberate failure" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    assert "LangGraph run failed" not in r.stderr, r.stderr


def test_message_json_carries_the_type_and_the_debug_traceback(tmp_path):
    import json

    r = _run(tmp_path, "-m", "hi", "--json", debug=True)
    assert r.returncode == 1, r.stderr
    result = json.loads(r.stdout)
    assert "RuntimeError: deliberate failure" in result["error"]
    assert "Traceback" in (result.get("traceback") or ""), result
    assert "LangGraph run failed" not in r.stderr


def test_records_outside_a_core_run_pass_through(caplog):
    """The filter only acts while core drives a run; anyone else's logs are untouched."""
    _install_run_failure_capture()
    _install_run_failure_capture()  # idempotent
    logger = logging.getLogger("ag_ui_langgraph.agent")
    assert sum(isinstance(f, _CaptureRunFailure) for f in logger.filters) == 1
    with caplog.at_level(logging.ERROR, logger="ag_ui_langgraph.agent"):
        try:
            raise RuntimeError("not ours")
        except RuntimeError:
            logger.exception("LangGraph run failed")
    assert [r.getMessage() for r in caplog.records] == ["LangGraph run failed"]


class _S(TypedDict):
    messages: Annotated[list, add_messages]


def _partial_then_raise_graph():
    def answer(state):
        return {"messages": [AIMessage(content="Here is my partial answer.")]}

    def use_tool(state):
        raise RuntimeError("tool call failed")

    g = StateGraph(_S)
    g.add_node("answer", answer)
    g.add_node("use_tool", use_tool)
    g.add_edge(START, "answer")
    g.add_edge("answer", "use_tool")
    g.add_edge("use_tool", END)
    return g.compile()


async def test_capture_survives_one_task_per_step():
    """A consumer that resumes the stream in a new task per step (the VS Code sidecar
    does, for cancellation) still gets the original exception, after the partial reply."""
    import asyncio

    from langstage_core.agui import iter_event_frames

    agen = iter_event_frames(_partial_then_raise_graph(), "hi", "t-per-task")
    frames = []
    while True:
        try:
            frames.append(await asyncio.ensure_future(agen.__anext__()))
        except StopAsyncIteration:
            break
    contents = [f["content"] for f in frames if f["type"] == "content"]
    assert "".join(contents).strip() == "Here is my partial answer."
    assert frames[-1]["type"] == "error"
    assert frames[-1]["error"] == "RuntimeError: tool call failed"
