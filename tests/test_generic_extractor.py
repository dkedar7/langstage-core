"""Tests for the opt-in ``GenericToolExtractor`` fallback path.

Without a per-tool extractor a custom tool only emits
``ToolCallStartEvent`` / ``ToolCallEndEvent`` and never
``ToolExtractedEvent``. Hosts that render rich cards by switching on
``ToolExtractedEvent`` have nothing to act on, so the only path for
adding a card is to ship a per-tool extractor in the parser.

The ``default_extractor`` kwarg + bundled ``GenericToolExtractor``
closes that gap: any tool name without a specific extractor surfaces
as a ``ToolExtractedEvent`` of ``extracted_type="tool_call"``, opt-in
so existing hosts that switch on specific extracted types don't
suddenly start receiving new events.
"""

from __future__ import annotations

from langgraph_stream_parser import (
    GenericToolExtractor,
    StreamParser,
    ToolCallEndEvent,
    ToolExtractedEvent,
)
from langgraph_stream_parser.extractors.builtins import ThinkToolExtractor

from .fixtures.mocks import THINK_TOOL_STRING_CONTENT, TOOL_MESSAGE_SUCCESS


def _stream(chunks):
    """Iterate the fixture chunks one at a time — mirrors the parser's input."""
    yield from chunks


class TestGenericToolExtractor:
    def test_extracts_content(self):
        ext = GenericToolExtractor()
        result = ext.extract("hello world")
        assert result == {"content": "hello world"}

    def test_extracts_dict_content(self):
        ext = GenericToolExtractor()
        result = ext.extract({"k": "v"})
        assert result == {"content": {"k": "v"}}

    def test_returns_none_for_none(self):
        # A tool that returns nothing shouldn't fire a spurious card.
        ext = GenericToolExtractor()
        assert ext.extract(None) is None

    def test_empty_string_still_surfaces(self):
        # Empty string ≠ None — host renderer may want to know the tool
        # was called and returned nothing.
        ext = GenericToolExtractor()
        assert ext.extract("") == {"content": ""}

    def test_sentinel_tool_name(self):
        # `tool_name = "*"` is a sentinel; the parser never indexes
        # _extractors by it. Documented so hosts that introspect the
        # parser can identify the default-extractor slot.
        assert GenericToolExtractor().tool_name == "*"
        assert GenericToolExtractor().extracted_type == "tool_call"


class TestParserDefaultExtractor:
    def test_unknown_tool_emits_no_extracted_event_by_default(self):
        """Historical behaviour: no default_extractor → no extracted event for unknown tools."""
        parser = StreamParser()
        events = list(parser.parse(_stream([TOOL_MESSAGE_SUCCESS])))

        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert extracted == []
        # But lifecycle events still fire — the tool call wasn't silently dropped.
        ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(ends) == 1
        assert ends[0].name == "search"

    def test_unknown_tool_emits_generic_event_with_default_extractor(self):
        """Opt-in: passing GenericToolExtractor as default_extractor surfaces unknown tools."""
        parser = StreamParser(default_extractor=GenericToolExtractor())
        events = list(parser.parse(_stream([TOOL_MESSAGE_SUCCESS])))

        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        evt = extracted[0]
        assert evt.tool_name == "search"
        assert evt.extracted_type == "tool_call"
        assert evt.data == {"content": "The weather is sunny and 72F"}
        # Lifecycle events still fire — generic extraction is additive.
        assert any(isinstance(e, ToolCallEndEvent) for e in events)

    def test_known_tool_uses_specific_extractor_not_default(self):
        """Per-tool extractors win over the default fallback."""
        parser = StreamParser(default_extractor=GenericToolExtractor())
        # think_tool has ThinkToolExtractor registered → should NOT fall
        # through to the GenericToolExtractor.
        events = list(parser.parse(_stream([THINK_TOOL_STRING_CONTENT])))

        # ThinkToolExtractor's "reflection" extracted_type routes to a
        # ReasoningEvent, not a generic ToolExtractedEvent of
        # extracted_type="tool_call". So we should see zero generic
        # tool_call events even with the fallback registered.
        generic = [
            e for e in events
            if isinstance(e, ToolExtractedEvent) and e.extracted_type == "tool_call"
        ]
        assert generic == []

    def test_default_extractor_failure_falls_through_cleanly(self):
        """If the default extractor raises, lifecycle events still fire."""
        class BoomExtractor:
            tool_name = "*"
            extracted_type = "tool_call"

            def extract(self, content):  # noqa: ARG002
                raise RuntimeError("simulated failure")

        parser = StreamParser(default_extractor=BoomExtractor())
        events = list(parser.parse(_stream([TOOL_MESSAGE_SUCCESS])))

        # No ToolExtractedEvent (the extractor blew up), but the
        # lifecycle end event still fires — graceful degradation.
        assert not any(isinstance(e, ToolExtractedEvent) for e in events)
        ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(ends) == 1
        assert ends[0].name == "search"

    def test_default_extractor_skipped_when_tool_in_skip_list(self):
        """skip_tools still suppresses lifecycle, but default extractor
        runs (extractor logic is independent of lifecycle skipping —
        this matches the existing per-tool extractor behaviour, where
        extraction is treated as a separate concern from start/end UI
        events).
        """
        parser = StreamParser(
            default_extractor=GenericToolExtractor(),
            skip_tools=["search"],
        )
        events = list(parser.parse(_stream([TOOL_MESSAGE_SUCCESS])))
        # No lifecycle end event for skipped tool.
        assert not any(isinstance(e, ToolCallEndEvent) for e in events)
        # But the extractor still surfaces the content — same contract
        # as a specific extractor would have for a skipped tool.
        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        assert extracted[0].extracted_type == "tool_call"


class TestRegistry:
    def test_generic_extractor_is_top_level_export(self):
        # If this import works, the package surface advertises the new extractor.
        from langgraph_stream_parser import GenericToolExtractor  # noqa: F401

    def test_default_extractor_not_added_to_registry(self):
        """default_extractor lives separately from `_extractors` — it
        shouldn't shadow per-tool registration."""
        parser = StreamParser(default_extractor=GenericToolExtractor())
        # Sentinel tool_name "*" must not appear in the dispatch table.
        assert "*" not in parser._extractors
        # Built-ins are still registered.
        assert "think_tool" in parser._extractors

    def test_specific_extractor_can_be_registered_alongside_default(self):
        class CanvasExtractor:
            tool_name = "add_to_canvas"
            extracted_type = "canvas_item"

            def extract(self, content):
                return {"raw": content}

        parser = StreamParser(default_extractor=GenericToolExtractor())
        parser.register_extractor(CanvasExtractor())
        assert "add_to_canvas" in parser._extractors
        assert parser._default_extractor is not None

    def test_known_extractor_still_wins(self):
        # If you register a per-tool extractor for what used to fall
        # through to the default, the per-tool one takes precedence.
        class SearchSpecificExtractor:
            tool_name = "search"
            extracted_type = "search_result"

            def extract(self, content):
                return {"hit": content}

        parser = StreamParser(default_extractor=GenericToolExtractor())
        parser.register_extractor(SearchSpecificExtractor())

        # Reuse the ThinkToolExtractor instance just so the test reads
        # cleanly — we're really testing the registration table:
        assert ThinkToolExtractor().tool_name == "think_tool"
        events = list(parser.parse(_stream([TOOL_MESSAGE_SUCCESS])))
        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        assert extracted[0].extracted_type == "search_result"
        assert extracted[0].data == {"hit": "The weather is sunny and 72F"}
