"""`langstage-agui -m` ends quietly when the reader closes stdout (gh #238).

`langstage-agui -m ... | head -1` used to crash with a BrokenPipeError traceback and
exit 1, which ADR 0007 reserves for a turn that errored. A closed stdout now ends the
run quietly with exit 0, on POSIX (BrokenPipeError) and on Windows (OSError EINVAL).
"""
from __future__ import annotations

import errno
import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("ag_ui_langgraph")

from langstage_core.agui.__main__ import main  # noqa: E402
from langstage_core.console import is_closed_pipe, safe_print, safe_write  # noqa: E402

AGENT = '''
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
def n(state): return {"messages": [AIMessage(content="\\n".join(f"line {i}" for i in range(500)))]}
g = StateGraph(MessagesState); g.add_node("n", n); g.add_edge(START, "n"); g.add_edge("n", END)
graph = g.compile()
'''


class _ClosingStdout(io.StringIO):
    """A stdout whose reader goes away after ``limit`` characters."""

    def __init__(self, exc: OSError, limit: int = 10) -> None:
        super().__init__()
        self.exc = exc
        self.limit = limit

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    def write(self, text: str) -> int:
        if self.tell() + len(text) > self.limit:
            raise self.exc
        return super().write(text)

    def flush(self) -> None:
        if self.tell() >= self.limit:
            raise self.exc

    def fileno(self) -> int:
        raise io.UnsupportedOperation("no fd under test")


@pytest.fixture
def agent_spec(tmp_path: Path) -> str:
    p = tmp_path / "long_agent.py"
    p.write_text(AGENT, encoding="utf-8")
    return f"{p}:graph"


@pytest.mark.parametrize("as_json", [False, True])
def test_broken_pipe_exits_zero_without_a_traceback(agent_spec, monkeypatch, capsys, as_json):
    monkeypatch.setattr(sys, "stdout", _ClosingStdout(BrokenPipeError(errno.EPIPE, "Broken pipe")))
    argv = ["--agent", agent_spec, "-m", "hi"] + (["--json"] if as_json else [])
    assert main(argv) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_windows_einval_is_a_closed_pipe_too(agent_spec, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", _ClosingStdout(OSError(errno.EINVAL, "Invalid argument")))
    assert main(["--agent", agent_spec, "-m", "hi"]) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_another_os_error_still_raises(agent_spec, monkeypatch):
    monkeypatch.setattr(sys, "stdout", _ClosingStdout(OSError(errno.ENOSPC, "No space left")))
    with pytest.raises(OSError) as info:
        main(["--agent", agent_spec, "-m", "hi"])
    assert info.value.errno == errno.ENOSPC


def test_safe_write_reports_a_closed_pipe_as_broken_pipe_on_every_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(BrokenPipeError):
        safe_write("hello world", _ClosingStdout(OSError(errno.EINVAL, "Invalid argument"), limit=1))
    with pytest.raises(BrokenPipeError):
        safe_print("hello world", file=_ClosingStdout(BrokenPipeError(errno.EPIPE, "x"), limit=1))


def test_is_closed_pipe(monkeypatch):
    assert is_closed_pipe(BrokenPipeError())
    monkeypatch.setattr(sys, "platform", "linux")
    assert not is_closed_pipe(OSError(errno.EINVAL, "Invalid argument"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert is_closed_pipe(OSError(errno.EINVAL, "Invalid argument"))
    assert not is_closed_pipe(OSError(errno.ENOSPC, "No space"))
    assert not is_closed_pipe(ValueError())
