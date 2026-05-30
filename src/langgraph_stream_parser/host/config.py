"""Shared ``DEEPAGENT_*`` environment-variable schema for hosts.

``HostConfig`` holds only the keys every host has in common: the agent spec,
the workspace root, and the bind/title basics. Host-specific keys (theme,
auth, canvas/files toggles, model name, Jupyter token, virtual mode, ...)
belong in each host's own subclass — drift below the shared core is fine.

Resolution order is the host's responsibility; the typical chain is
``Python args > CLI args > env vars > defaults``. Use ``from_env`` to seed
from the environment and ``merge`` to layer overrides on top.
"""
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


def _env_bool(value: str | None, default: bool = False) -> bool:
    """Parse an env-var string into a bool."""
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class HostConfig:
    """Shared configuration for deep-agent hosts.

    Subclass to add host-specific fields, and extend ``from_env`` to read
    their env vars:

        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"

            @classmethod
            def from_env(cls):
                base = super().from_env()
                return base.merge(theme=os.getenv("DEEPAGENT_THEME", "auto"))
    """

    agent_spec: str | None = None     # DEEPAGENT_AGENT_SPEC ("path.py:var")
    workspace_root: Path = Path(".")  # DEEPAGENT_WORKSPACE_ROOT
    host: str = "localhost"           # DEEPAGENT_HOST
    port: int = 8050                  # DEEPAGENT_PORT
    debug: bool = False               # DEEPAGENT_DEBUG
    title: str = "Deep Agent"         # DEEPAGENT_TITLE

    @classmethod
    def from_env(cls) -> "HostConfig":
        """Build config from ``DEEPAGENT_*`` environment variables.

        Subclasses that add fields should call ``super().from_env()`` and
        ``merge`` their own keys on top.
        """
        workspace = os.getenv("DEEPAGENT_WORKSPACE_ROOT")
        port = os.getenv("DEEPAGENT_PORT")
        return cls(
            agent_spec=os.getenv("DEEPAGENT_AGENT_SPEC"),
            workspace_root=Path(workspace) if workspace else Path("."),
            host=os.getenv("DEEPAGENT_HOST", "localhost"),
            port=int(port) if port else 8050,
            debug=_env_bool(os.getenv("DEEPAGENT_DEBUG")),
            title=os.getenv("DEEPAGENT_TITLE", "Deep Agent"),
        )

    def merge(self, **overrides: Any) -> "HostConfig":
        """Return a copy with non-``None`` overrides applied.

        ``None`` values are ignored so callers can pass through unset CLI
        flags without clobbering env-derived values.
        """
        valid = {f.name for f in fields(self)}
        applied = {
            k: v for k, v in overrides.items()
            if v is not None and k in valid
        }
        return replace(self, **applied)
