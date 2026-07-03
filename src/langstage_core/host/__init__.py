"""Host conventions shared across the LangStage surfaces.

Agent-spec loading, the shared ``LANGSTAGE_*`` config schema (legacy
``DEEPAGENT_*`` still resolves), and a workspace wrapper — the plumbing every
host (``langstage``, ``langstage-jupyter``, ``langstage-cli``,
``langstage-vscode``) needs but used to reimplement.
"""
from .config import HostConfig, load_toml_config
from .loader import load_agent_spec
from .workspace import Workspace, apply_workspace, workspace_root

__all__ = [
    "load_agent_spec",
    "HostConfig",
    "load_toml_config",
    "Workspace",
    "apply_workspace",
    "workspace_root",
]
