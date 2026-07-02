"""
Tool extractors for processing special tool outputs.

This module provides the ToolExtractor protocol and built-in extractors
for common tools like think_tool and write_todos.
"""
from .base import ToolExtractor
from .builtins import (
    CompressionExtractor,
    DisplayInlineExtractor,
    GenericToolExtractor,
    MemoryExtractor,
    SkillManageExtractor,
    SkillViewExtractor,
    ThinkToolExtractor,
    TodoExtractor,
)

__all__ = [
    "CompressionExtractor",
    "DisplayInlineExtractor",
    "GenericToolExtractor",
    "MemoryExtractor",
    "SkillManageExtractor",
    "SkillViewExtractor",
    "ThinkToolExtractor",
    "TodoExtractor",
    "ToolExtractor",
]
