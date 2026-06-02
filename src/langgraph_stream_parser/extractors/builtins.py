"""
Built-in tool extractors for common LangGraph tools.

These extractors handle tools that are commonly used across
LangGraph applications, such as think_tool for reflections
and write_todos for todo list management.
"""
import ast
import json
import re
from typing import Any


class ThinkToolExtractor:
    """Extractor for think_tool reflections.

    The think_tool is commonly used to give AI agents a scratchpad
    for reasoning. This extractor pulls out the reflection text
    from the tool's output.

    Handles formats:
        - String content (returned as-is)
        - JSON with 'reflection' key
        - Dict with 'reflection' key
    """

    tool_name = "think_tool"
    extracted_type = "reflection"

    def extract(self, content: Any) -> str | None:
        """Extract reflection from think_tool content.

        Args:
            content: The content from the think_tool ToolMessage.

        Returns:
            The reflection string, or None if not found.
        """
        if isinstance(content, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed.get("reflection")
            except (json.JSONDecodeError, TypeError):
                pass
            # Return raw string if not JSON
            return content if content.strip() else None

        if isinstance(content, dict):
            return content.get("reflection")

        return None


class TodoExtractor:
    """Extractor for write_todos tool output.

    The write_todos tool is used for task management within agents.
    This extractor handles the various formats the tool might return
    its todo list in.

    Handles formats:
        - Direct list of todo items
        - JSON string containing array
        - String with embedded array (e.g., "Updated todo list to [...]")
        - Dict with 'todos' key
        - Python literal syntax (single quotes)
    """

    tool_name = "write_todos"
    extracted_type = "todos"

    def extract(self, content: Any) -> list[dict[str, Any]] | None:
        """Extract todo list from write_todos content.

        Args:
            content: The content from the write_todos ToolMessage.

        Returns:
            List of todo items, or None if parsing fails.
        """
        todos = None

        if isinstance(content, str):
            # Look for array pattern first (handles "Updated todo list to [...]" format)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                array_str = match.group(0)

                # Try parsing as Python literal first (handles single quotes)
                try:
                    todos = ast.literal_eval(array_str)
                except (ValueError, SyntaxError):
                    # Fall back to JSON parsing (requires double quotes)
                    try:
                        todos = json.loads(array_str)
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                # No array found, try parsing entire string as JSON
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        todos = parsed.get('todos')
                        # If todos is a string, parse it again
                        if isinstance(todos, str):
                            todos = json.loads(todos)
                    elif isinstance(parsed, list):
                        # Content is directly a list
                        todos = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

        elif isinstance(content, dict):
            todos = content.get('todos')
            if isinstance(todos, str):
                try:
                    todos = json.loads(todos)
                except (json.JSONDecodeError, TypeError):
                    pass

        elif isinstance(content, list):
            # Content is directly a list
            todos = content

        return todos if isinstance(todos, list) else None


class DisplayInlineExtractor:
    """Extractor for display_inline tool output.

    The display_inline tool renders rich content (images, tables, charts,
    HTML, JSON) directly in the chat timeline. The tool returns a JSON
    string with display metadata that this extractor parses.

    Handles formats:
        - JSON string with display_type, data, title, status keys
        - Dict with the same keys (if already parsed)
    """

    tool_name = "display_inline"
    extracted_type = "display_inline"

    def extract(self, content: Any) -> dict[str, Any] | None:
        """Extract display data from display_inline content.

        Args:
            content: The JSON string or dict from display_inline.

        Returns:
            Dict with display_type, data, title, status, etc., or None.
        """
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "display_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return None

        if isinstance(content, dict) and "display_type" in content:
            return content

        return None


def _parse_json_content(content: Any) -> dict[str, Any] | None:
    """Best-effort JSON parse — accepts str or dict, returns None otherwise."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


class SkillManageExtractor:
    """Extractor for skill_manage tool calls (agentskills.io standard).

    Agents that follow the agentskills.io specification expose a tool
    (commonly named ``skill_manage``) for creating, patching, deleting,
    or pinning SKILL.md files in a procedural-memory library. This
    extractor surfaces those mutations as inline events so hosts can
    render "skill created: pdf-merging" alongside the raw tool result.

    Handles formats:
        - JSON string with ``action`` + ``name`` keys
        - Dict with the same keys
        - Plain text containing one of the known action verbs

    Action → extracted_subtype mapping:
        create / write_file → skill_created
        patch               → skill_updated
        pin / unpin         → skill_updated
        delete              → skill_deleted
    """

    tool_name = "skill_manage"
    extracted_type = "skill_event"

    _ACTION_TO_TYPE = {
        "create": "skill_created",
        "patch": "skill_updated",
        "write_file": "skill_updated",
        "delete": "skill_deleted",
        "pin": "skill_updated",
        "unpin": "skill_updated",
    }

    def extract(self, content: Any) -> dict[str, Any] | None:
        """Extract skill action + name from a skill_manage result.

        Args:
            content: JSON string, dict, or plain text from the tool.

        Returns:
            Dict with ``action``, ``name`` (when present), and
            ``extracted_subtype``. ``None`` when the content doesn't
            look like a skill_manage result.
        """
        parsed = _parse_json_content(content)
        if parsed is None:
            if not isinstance(content, str) or not content:
                return None
            text = content.lower()
            for action, etype in self._ACTION_TO_TYPE.items():
                if action in text:
                    return {"action": action, "extracted_subtype": etype}
            return None

        action = parsed.get("action")
        name = parsed.get("name")
        if action is None:
            return None
        etype = self._ACTION_TO_TYPE.get(action, "skill_event")
        out: dict[str, Any] = {"action": action, "extracted_subtype": etype}
        if name is not None:
            out["name"] = name
        return out


class SkillViewExtractor:
    """Extractor for skill_view tool calls (agentskills.io standard).

    Emits a ``skill_loaded`` event when the agent pulls a SKILL.md body
    into its context via the ``skill_view(name)`` tool. Hosts can render
    this differently from creation / update — it's a read, not a
    mutation, and represents the agent activating procedural memory.
    """

    tool_name = "skill_view"
    extracted_type = "skill_loaded"

    def extract(self, content: Any) -> dict[str, Any] | None:
        """Note that a skill body was loaded; carry the size as data."""
        if not content:
            return None
        return {"loaded": True, "body_chars": len(str(content))}


class CompressionExtractor:
    """Extractor for context-compression events.

    When an agent's compression middleware (e.g.
    ``langchain.agents.middleware.SummarizationMiddleware`` or a custom
    implementation) replaces the middle of a long conversation with a
    summary, it can emit a synthetic ``__compression__`` tool message
    so the host UI can show a banner like "context compressed: 47k →
    9k tokens (5x)".

    Expected payload (all fields optional, extractor surfaces what it
    finds):
        - before_tokens: int
        - after_tokens: int
        - ratio: float
        - section_count: int
        - skipped: bool
        - reason: str
    """

    tool_name = "__compression__"
    extracted_type = "compression_summary"

    def extract(self, content: Any) -> dict[str, Any] | None:
        parsed = _parse_json_content(content)
        if parsed is None:
            return None
        keys = {"before_tokens", "after_tokens", "ratio", "section_count", "skipped", "reason"}
        out = {k: parsed[k] for k in keys if k in parsed}
        return out or None


class MemoryExtractor:
    """Extractor for memory tool actions.

    Agents that expose a frozen-snapshot memory tool (commonly named
    ``memory``) for managing persistent MEMORY.md / USER.md files
    surface their action through this extractor. Distinguishes ``target``
    so the two streams render separately.

    Action → extracted_subtype mapping:
        add     → memory_added
        replace → memory_replaced
        remove  → memory_removed
        read    → memory_read
    """

    tool_name = "memory"
    extracted_type = "memory_updated"

    _ACTION_TO_TYPE = {
        "add": "memory_added",
        "replace": "memory_replaced",
        "remove": "memory_removed",
        "read": "memory_read",
    }

    def extract(self, content: Any) -> dict[str, Any] | None:
        parsed = _parse_json_content(content)
        if parsed is None:
            return None
        action = parsed.get("action")
        target = parsed.get("target")
        if action is None or target is None:
            return None
        etype = self._ACTION_TO_TYPE.get(action, "memory_updated")
        out: dict[str, Any] = {
            "action": action,
            "target": target,
            "extracted_subtype": etype,
        }
        if "index" in parsed:
            out["index"] = parsed["index"]
        return out
