"""Wire-contract golden tests.

``event.to_dict()`` is the single inter-process protocol shared by every host
(FastAPI WebSocket/SSE, Jupyter handler, CLI, the VS Code stdio sidecar) and
the frontends that render those dicts. These snapshots pin the exact shapes so
a change to the schema can't silently drift away from the consumers. If you
intend to change a shape, update this file and every consumer in lockstep.
"""
from langgraph_stream_parser.events import (
    ContentEvent,
    ReasoningEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    DisplayEvent,
    InterruptEvent,
    UsageEvent,
    CompleteEvent,
    ErrorEvent,
    event_to_dict,
)


class TestEventShapes:
    def test_content(self):
        assert ContentEvent(content="hi").to_dict() == {
            "type": "content",
            "content": "hi",
            "role": "assistant",
            "node": None,
        }

    def test_content_with_subagent(self):
        d = ContentEvent(
            content="x", agent_name="researcher", is_subagent=True,
            namespace=("researcher:abc",),
        ).to_dict()
        assert d["type"] == "content"
        assert d["agent_name"] == "researcher"
        assert d["is_subagent"] is True
        assert d["namespace"] == ["researcher:abc"]

    def test_reasoning(self):
        assert ReasoningEvent(content="think").to_dict() == {
            "type": "reasoning",
            "content": "think",
            "source": "content_block",
            "node": None,
        }

    def test_tool_start(self):
        assert ToolCallStartEvent(id="c1", name="search", args={"q": "x"}).to_dict() == {
            "type": "tool_start",
            "id": "c1",
            "name": "search",
            "args": {"q": "x"},
            "node": None,
        }

    def test_tool_end(self):
        assert ToolCallEndEvent(
            id="c1", name="search", result="ok", status="success",
        ).to_dict() == {
            "type": "tool_end",
            "id": "c1",
            "name": "search",
            "result": "ok",
            "status": "success",
            "error_message": None,
            "duration_ms": None,
        }

    def test_extraction(self):
        assert ToolExtractedEvent(
            tool_name="write_todos", extracted_type="todos", data=[{"task": "a"}],
        ).to_dict() == {
            "type": "extraction",
            "tool_name": "write_todos",
            "extracted_type": "todos",
            "data": [{"task": "a"}],
        }

    def test_display(self):
        d = DisplayEvent(display_type="dataframe", data="<table/>", title="T").to_dict()
        assert d["type"] == "display"
        assert d["display_type"] == "dataframe"
        assert d["status"] == "success"
        assert d["title"] == "T"

    def test_interrupt(self):
        d = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"command": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve", "reject"]}],
        ).to_dict()
        assert d["type"] == "interrupt"
        assert d["action_requests"][0]["tool"] == "bash"
        assert set(d["allowed_decisions"]) == {"approve", "reject"}

    def test_usage(self):
        assert UsageEvent(input_tokens=10, output_tokens=5, total_tokens=15).to_dict() == {
            "type": "usage",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "node": None,
        }

    def test_complete(self):
        assert CompleteEvent().to_dict() == {"type": "complete"}

    def test_error(self):
        assert ErrorEvent(error="boom").to_dict() == {"type": "error", "error": "boom"}


class TestEventToDictMaxResultLen:
    def test_default_truncates_long_tool_result(self):
        long = "x" * 1000
        d = event_to_dict(ToolCallEndEvent(id="c", name="t", result=long, status="success"))
        assert d["result"].endswith("...")
        assert len(d["result"]) == 503  # 500 + "..."

    def test_custom_max_result_len(self):
        long = "x" * 100_000
        d = event_to_dict(
            ToolCallEndEvent(id="c", name="t", result=long, status="success"),
            max_result_len=50_000,
        )
        assert len(d["result"]) == 50_003

    def test_max_result_len_ignored_for_other_events(self):
        # Passing max_result_len for a non-tool-end event must not error.
        d = event_to_dict(ContentEvent(content="hi"), max_result_len=10)
        assert d == {"type": "content", "content": "hi", "role": "assistant", "node": None}

    def test_unknown_event_falls_back(self):
        class Weird:
            pass

        d = event_to_dict(Weird())
        assert d["type"] == "unknown"
