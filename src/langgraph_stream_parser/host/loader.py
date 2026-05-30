"""Load a LangGraph agent from a ``path:object`` spec string.

This is the canonical agent-spec loader for every deep-agent host
(``cowork-dash``, ``deepagent-lab``, ``deepagent-code``, ``deepagent-vscode``).
It replaces the per-host loaders that had drifted apart.

The spec format is **strict**: the ``:variable`` suffix is required. There is
no implicit ``agent``/``graph`` fallback — explicit beats implicit, and the
two hosts that disagreed on the fallback now share one rule.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_agent_spec(spec: str) -> Any:
    """Load a LangGraph agent from a ``"path:object"`` spec string.

    Args:
        spec: ``"path/to/module.py:object_name"`` (file path) or
            ``"package.module:object_name"`` (dotted module path). The
            ``:object_name`` suffix is required.

    Returns:
        The agent object (typically a compiled LangGraph graph).

    Raises:
        ValueError: If the spec has no ``:object`` suffix.
        FileNotFoundError: If a ``.py`` file path does not exist.
        ImportError: If the module cannot be loaded/imported.
        AttributeError: If the object is not found in the module.

    Example:
        agent = load_agent_spec("./my_agent.py:agent")
        agent = load_agent_spec("my_package.agents:research_graph")
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid agent spec {spec!r}. Expected "
            "'path/to/module.py:object_name' or 'package.module:object_name' "
            "(the ':object_name' suffix is required)."
        )

    module_path, _, obj_name = spec.rpartition(":")
    if not module_path or not obj_name:
        raise ValueError(
            f"Invalid agent spec {spec!r}. Both a module/file path and an "
            "object name are required, e.g. 'agent.py:graph'."
        )

    module = _import_module(module_path)

    if not hasattr(module, obj_name):
        raise AttributeError(
            f"Module {module_path!r} has no attribute {obj_name!r}."
        )
    return getattr(module, obj_name)


def _import_module(module_path: str) -> Any:
    """Import a module from a file path or a dotted module path."""
    file_path = Path(module_path)

    # File path: load from source location.
    if file_path.suffix == ".py" or any(sep in module_path for sep in ("/", "\\")):
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")

        # Unique module name so repeated loads of different files don't collide.
        module_name = f"_lsp_agent_{file_path.stem}_{abs(hash(str(file_path)))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, file_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
        return module

    # Dotted module path.
    return importlib.import_module(module_path)
