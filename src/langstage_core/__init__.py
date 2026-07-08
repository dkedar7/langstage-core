"""langstage-core — the host + AG-UI bridge for the LangStage agent family.

Since 1.0 (ADR 0003) the bespoke event-translation layer (``StreamParser``, the
``events`` dataclasses, ``event_to_dict``, the render adapters, and the
``stream_graph_updates`` helpers) is gone: surfaces stream through the in-process
AG-UI adapter instead. What remains is the *durable* core —

  - **host**: ``load_agent_spec`` + ``HostConfig`` + ``Workspace`` (write once,
    run on every surface with the same spec string and ``langstage.toml`` config).
  - **agui**: ``build_agent`` + ``iter_event_frames`` / ``iter_chunk_frames``
    (the in-process AG-UI stream, with an ``extractors=`` hook), and ``build_app`` /
    ``serve`` for the wire. (These live under ``langstage_core.agui``, not top-level.)
  - **adapters**: ``SessionAdapter`` / ``Session`` for session-scoped streaming
    (also re-exported at the top level).
  - **tasks**: the durable task-delegation engine.
  - **extractors**: the ``ToolExtractor`` protocol + reusable built-ins.

Quick start::

    from langstage_core.agui import build_agent, iter_event_frames
    agent = build_agent(my_compiled_graph)
    async for frame in iter_event_frames(agent, "hello", "session-1"):
        ...   # {"type": "content" | "tool_start" | "interrupt" | ...}
"""
from importlib.metadata import PackageNotFoundError, version

from .extractors import (
    CompressionExtractor,
    DisplayInlineExtractor,
    GenericToolExtractor,
    MemoryExtractor,
    SkillManageExtractor,
    SkillViewExtractor,
    ThinkToolExtractor,
    TodoExtractor,
    ToolExtractor,
)
from .adapters import Session, SessionAdapter
from .host import (
    HostConfig,
    Workspace,
    apply_workspace,
    load_agent_spec,
    workspace_root,
)
from .resume import create_resume_input, prepare_agent_input
from .tasks import (
    InMemoryTaskStore,
    TASK_TOOLS,
    Task,
    TaskRunner,
    TaskState,
    TaskStore,
    current_task_id,
    get_runner,
    outcome_to_state,
    set_runner,
)

try:
    __version__ = version("langstage-core")
except PackageNotFoundError:  # pragma: no cover - editable/source checkout
    __version__ = "0.0.0+local"

__all__ = [
    # Host conventions
    "load_agent_spec",
    "HostConfig",
    "Workspace",
    "apply_workspace",
    "workspace_root",
    # Session-scoped streaming adapter (also under langstage_core.adapters)
    "SessionAdapter",
    "Session",
    # Input helpers
    "prepare_agent_input",
    "create_resume_input",
    # Extractors (ToolExtractor protocol + reusable built-ins)
    "ToolExtractor",
    "ThinkToolExtractor",
    "TodoExtractor",
    "DisplayInlineExtractor",
    "GenericToolExtractor",
    "SkillManageExtractor",
    "SkillViewExtractor",
    "MemoryExtractor",
    "CompressionExtractor",
    # Task-delegation engine
    "TaskRunner",
    "TaskStore",
    "InMemoryTaskStore",
    "Task",
    "TaskState",
    "TASK_TOOLS",
    "set_runner",
    "get_runner",
    "current_task_id",
    "outcome_to_state",
]
