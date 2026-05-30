"""Host conventions shared across deep-agent surfaces.

Agent-spec loading, the shared ``DEEPAGENT_*`` config schema, and a workspace
wrapper — the plumbing every host (``cowork-dash``, ``deepagent-lab``,
``deepagent-code``, ``deepagent-vscode``) needs but used to reimplement.
"""
from .config import HostConfig
from .loader import load_agent_spec
from .workspace import Workspace

__all__ = [
    "load_agent_spec",
    "HostConfig",
    "Workspace",
]
