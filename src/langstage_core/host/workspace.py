"""The shared workspace-root wrapper + the one place a host *applies* it.

``Workspace`` removes the repeated ``mkdir`` / safe-join logic. ``apply_workspace``
/ ``workspace_root`` are the single source of truth for "the agent's working
directory" (ADR 0005): a surface calls ``apply_workspace`` once after resolving
config, and everything downstream — the agent's filesystem backend, a file
browser, a BYO tool — reads ``workspace_root`` instead of maintaining a private
global. This replaces the five bespoke "apply" mechanisms (cli ``chdir`` / vscode
env-push / hermes backend-arg / jupyter global-mutate / web hand-sync) that each
drifted into a workspace bug.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Workspace:
    """The directory an agent operates within."""

    root: Path = field(default_factory=lambda: Path("."))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def ensure(self) -> "Workspace":
        """Create the workspace root if it does not exist. Returns self."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def subpath(self, *parts: str) -> Path:
        """Join ``parts`` under the root, refusing to escape it.

        Raises:
            ValueError: If the resolved path would fall outside the root
                (e.g. via ``..`` traversal).
        """
        candidate = self.root.joinpath(*parts).resolve()
        root_resolved = self.root.resolve()
        if root_resolved != candidate and root_resolved not in candidate.parents:
            raise ValueError(f"Path {candidate} escapes workspace root {root_resolved}")
        return candidate

    @property
    def name(self) -> str:
        """The workspace directory name (for display)."""
        return self.root.resolve().name


# The active workspace for this process. Set by apply_workspace(); read by
# workspace_root(). One per process — see ADR 0005 "explicit assumption".
_ACTIVE: "Workspace | None" = None

# The env vars apply_workspace publishes so out-of-process readers (the vscode
# sidecar's subprocess) and legacy tools still see the resolved root. The legacy
# name is written until the ADR 0004 deprecation sunset.
_ENV_CANONICAL = "LANGSTAGE_WORKSPACE_ROOT"
_ENV_LEGACY = "DEEPAGENT_WORKSPACE_ROOT"


def apply_workspace(root, *, chdir: bool = False) -> "Workspace":
    """Make ``root`` the active resolved workspace and publish it as the single
    source of truth (ADR 0005).

    Ensures the directory exists, records it as this process's active workspace,
    and exports it as ``LANGSTAGE_WORKSPACE_ROOT`` (plus the legacy
    ``DEEPAGENT_WORKSPACE_ROOT``) so out-of-process and legacy readers agree.
    ``chdir=True`` also changes the process cwd — taken only by single-process,
    single-agent surfaces (cli, the vscode sidecar); servers root their built
    agent's backend from :func:`workspace_root` instead and never chdir.

    Call ONCE, after ``HostConfig.resolve()`` and before building the agent /
    file browser, so both derive from the same value.

    Returns the :class:`Workspace`.
    """
    global _ACTIVE
    ws = Workspace(root).ensure()
    _ACTIVE = ws
    resolved = str(ws.root.resolve())
    os.environ[_ENV_CANONICAL] = resolved
    os.environ[_ENV_LEGACY] = resolved
    if chdir:
        os.chdir(ws.root)
    return ws


def workspace_root() -> Path:
    """The active resolved workspace root — the ONE accessor tools and surfaces
    read instead of a private global (ADR 0005).

    Prefers the in-process value set by :func:`apply_workspace`; falls back to the
    ``LANGSTAGE_WORKSPACE_ROOT`` / ``DEEPAGENT_WORKSPACE_ROOT`` env (set by a parent
    process), then to the current working directory.
    """
    if _ACTIVE is not None:
        return _ACTIVE.root.resolve()
    env = os.environ.get(_ENV_CANONICAL) or os.environ.get(_ENV_LEGACY)
    return Path(env).resolve() if env else Path.cwd()
