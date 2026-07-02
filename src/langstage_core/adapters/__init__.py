"""Session adapter for driving a LangGraph agent over the in-process AG-UI stream.

The bespoke render adapters (CLI/FastAPI/Jupyter/Print) and the ``StreamParser``
they wrapped were removed in langstage-core 1.0; ``SessionAdapter`` is now
AG-UI-only (see ADR 0003).
"""

from .session import Session, SessionAdapter

__all__ = ["SessionAdapter", "Session"]
