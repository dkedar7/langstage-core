"""
Base protocol for tool extractors.

Tool extractors allow users to define custom logic for extracting
meaningful data from tool results. When a ToolMessage is received,
registered extractors are checked and if one matches, a ToolExtractedEvent
is emitted with the extracted data.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolExtractor(Protocol):
    """Protocol for custom tool content extractors.

    Implement this protocol to create extractors for domain-specific tools.
    Register extractors with StreamParser.register_extractor().

    Example:
        class CanvasExtractor:
            tool_name = "add_to_canvas"
            extracted_type = "canvas_item"

            def extract(self, content: Any) -> dict | None:
                if isinstance(content, dict):
                    return content
                if isinstance(content, str):
                    import json
                    try:
                        return json.loads(content)
                    except:
                        return {"type": "markdown", "data": content}
                return None

        parser = StreamParser()
        parser.register_extractor(CanvasExtractor())
    """

    @property
    def tool_name(self) -> str:
        """The name of the tool this extractor handles.

        This should match the 'name' attribute of ToolMessage objects
        that this extractor should process.
        """
        ...

    @property
    def extracted_type(self) -> str:
        """The type name for extracted content.

        This identifies what kind of data was extracted and will be
        set on the ToolExtractedEvent.extracted_type field.

        Examples: "reflection", "todos", "canvas_item"
        """
        ...

    def extract(self, content: Any) -> Any | None:
        """Extract meaningful data from tool content.

        Args:
            content: The raw content from ToolMessage.content.
                This may be a string, dict, list, or other type
                depending on how the tool formats its output.

        Returns:
            Extracted data if extraction succeeds, or None if
            extraction fails or is not applicable. Returning None
            will cause no ToolExtractedEvent to be emitted, but
            a ToolCallEndEvent will still be emitted if tool
            lifecycle tracking is enabled.
        """
        ...
