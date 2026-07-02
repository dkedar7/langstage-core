"""A small workspace-root wrapper shared by hosts.

Not load-bearing — just removes the repeated ``mkdir`` / safe-join logic and
gives a single place for the "agent's working directory" concept.
"""
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
            raise ValueError(
                f"Path {candidate} escapes workspace root {root_resolved}"
            )
        return candidate

    @property
    def name(self) -> str:
        """The workspace directory name (for display)."""
        return self.root.resolve().name
