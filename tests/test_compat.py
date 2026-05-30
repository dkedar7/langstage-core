"""Tests for backward-compatible convenience functions."""
import pytest
from unittest.mock import MagicMock

from langgraph_stream_parser.compat import (
    stream_graph_updates,
    resume_graph_from_interrupt,
    _event_to_dict,
)
from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    CompleteEvent,
    ErrorEvent,
)

from .fixtures.mocks import (
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    THINK_TOOL_MESSAGE,
    WRITE_TODOS_MESSAGE,
    INTERRUPT_WITH_ACTIONS,
)


class TestEventToDict:
    def test_content_event(self):
        event = ContentEvent(content="Hello", node="agent")
        result = _event_to_dict(event)

        assert result["chunk"] == "Hello"
        assert result["status"] == "streaming"
        assert result["node"] == "agent"

    def test_content_event_no_node(self):
        event = ContentEvent(content="Hello")
        result = _event_to_dict(event)

        assert "node" not in result

    def test_tool_call_start_event(self):
        event = ToolCallStartEvent(
            id="call_1",
            name="search",
            args={"q": "test"},
            node="agent"
        )
        result = _event_to_dict(event)

        assert result["status"] == "streaming"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_tool_call_end_event_returns_none(self):
        event = ToolCallEndEvent(
            id="call_1",
            name="search",
            result="done",
            status="success"
        )
        result = _event_to_dict(event)

        assert result is None

    def test_reflection_extracted(self):
        event = ToolExtractedEvent(
            tool_name="think_tool",
            extracted_type="reflection",
            data="My thoughts"
        )
        result = _event_to_dict(event)

        assert result["chunk"] == "My thoughts"
        assert result["status"] == "streaming"

    def test_todos_extracted(self):
        event = ToolExtractedEvent(
            tool_name="write_todos",
            extracted_type="todos",
            data=[{"task": "Do A"}]
        )
        result = _event_to_dict(event)

        assert result["todo_list"] == [{"task": "Do A"}]

    def test_generic_extracted(self):
        event = ToolExtractedEvent(
            tool_name="canvas",
            extracted_type="canvas_item",
            data={"type": "chart"}
        )
        result = _event_to_dict(event)

        assert result["extracted"]["tool"] == "canvas"
        assert result["extracted"]["type"] == "canvas_item"

    def test_interrupt_event(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash"}],
            review_configs=[{"allowed_decisions": ["approve"]}]
        )
        result = _event_to_dict(event)

        assert result["status"] == "interrupt"
        assert result["interrupt"]["action_requests"] == [{"tool": "bash"}]

    def test_complete_event(self):
        event = CompleteEvent()
        result = _event_to_dict(event)

        assert result == {"status": "complete"}

    def test_error_event(self):
        event = ErrorEvent(error="Something broke")
        result = _event_to_dict(event)

        assert result["status"] == "error"
        assert result["error"] == "Something broke"


class TestStreamGraphUpdates:
    def test_simple_message(self):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([SIMPLE_AI_MESSAGE])

        updates = list(stream_graph_updates(
            mock_agent,
            {"messages": [{"role": "user", "content": "Hi"}]}
        ))

        # Should have content and complete
        content_updates = [u for u in updates if "chunk" in u]
        complete_updates = [u for u in updates if u.get("status") == "complete"]

        assert len(content_updates) >= 1
        assert len(complete_updates) == 1

    def test_tool_call_flow(self):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([
            AI_MESSAGE_WITH_TOOL_CALLS,
            TOOL_MESSAGE_SUCCESS
        ])

        updates = list(stream_graph_updates(mock_agent, {}))

        tool_updates = [u for u in updates if "tool_calls" in u]
        assert len(tool_updates) >= 1

    def test_write_todos_surfaces_todo_list(self):
        """write_todos is in skip_tools (kept out of tool_calls noise) but its
        result must STILL surface as a todo_list update — consumers like the
        deepagent-lab todo sidebar depend on it. Regression guard: skip_tools
        must hide a tool's lifecycle without suppressing its extractor."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([WRITE_TODOS_MESSAGE])

        updates = list(stream_graph_updates(mock_agent, {}))

        todo_updates = [u for u in updates if "todo_list" in u]
        assert len(todo_updates) == 1
        # ...and it must not leak as a tool_calls update.
        assert not [u for u in updates if "tool_calls" in u]

    def test_think_tool_surfaces_as_chunk(self):
        """think_tool is skipped from tool_calls but its reflection must still
        surface (as a chunk update) so the UI shows the agent's thinking."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([THINK_TOOL_MESSAGE])

        updates = list(stream_graph_updates(mock_agent, {}))

        chunk_updates = [u for u in updates if "chunk" in u]
        assert len(chunk_updates) >= 1

    def test_interrupt(self):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([INTERRUPT_WITH_ACTIONS])

        updates = list(stream_graph_updates(mock_agent, {}))

        interrupt_updates = [u for u in updates if u.get("status") == "interrupt"]
        assert len(interrupt_updates) == 1
        assert "action_requests" in interrupt_updates[0]["interrupt"]

    def test_error_handling(self):
        mock_agent = MagicMock()
        mock_agent.stream.side_effect = RuntimeError("Connection failed")

        updates = list(stream_graph_updates(mock_agent, {}))

        error_updates = [u for u in updates if u.get("status") == "error"]
        assert len(error_updates) == 1
        assert "Connection failed" in error_updates[0]["error"]


class TestResumeGraphFromInterrupt:
    def test_resume_with_decisions(self):
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter([SIMPLE_AI_MESSAGE])

        updates = list(resume_graph_from_interrupt(
            mock_agent,
            decisions=[{"type": "approve"}],
            config={"thread_id": "123"}
        ))

        # Should have called agent.stream with a Command object
        mock_agent.stream.assert_called()
        call_args = mock_agent.stream.call_args
        # The first positional arg should be a Command
        resume_input = call_args[0][0]
        assert hasattr(resume_input, 'resume')

    def test_resume_error_handling(self):
        mock_agent = MagicMock()
        mock_agent.stream.side_effect = RuntimeError("Resume failed")

        updates = list(resume_graph_from_interrupt(
            mock_agent,
            decisions=[{"type": "approve"}]
        ))

        error_updates = [u for u in updates if u.get("status") == "error"]
        assert len(error_updates) == 1
