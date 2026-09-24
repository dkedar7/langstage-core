"""Load a LangGraph agent from a ``path:object`` spec string.

This is the canonical agent-spec loader for every deep-agent host
(``cowork-dash``, ``deepagent-lab``, ``deepagent-code``, ``deepagent-vscode``).
It replaces the per-host loaders that had drifted apart.

The spec format is **strict**: the ``:variable`` suffix is required. There is
no implicit ``agent``/``graph`` fallback — explicit beats implicit, and the
two hosts that disagreed on the fallback now share one rule.

Import semantics match ``python my_agent.py`` / uvicorn, so an agent that runs
under plain Python loads here too:

- ``file.py:attr`` puts the file's own directory on ``sys.path`` first, so the
  agent can import its sibling modules (gh #150).
- ``module:attr`` falls back to the base directory (default: cwd) when the
  top-level package isn't importable otherwise — a project-local package from a
  console-script entry point, whose ``sys.path[0]`` is the venv's ``bin/`` (gh #147).
"""
import contextlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def parse_agent_spec(spec: str) -> tuple[str, str]:
    """Split and validate a spec into ``(module_or_file_path, object_name)``.

    Pure string work — nothing is imported — so a surface can reject a malformed
    spec up front (e.g. at flag-parse time) with the same message the loader
    gives. Surrounding whitespace is stripped (a copy-paste artifact, gh
    langstage-vscode #135); ``~`` is left for the loader to expand.

    Raises:
        ValueError: If the spec is empty, has no ``:object`` suffix, or the part
            after the last ``:`` is not a Python identifier (e.g. the drive colon
            of a colon-less ``C:\\x\\agent.py``).
    """
    spec = (spec or "").strip()
    if ":" not in spec:
        # A colon-less spec is an error, never a silent fallback to some default
        # agent (gh langstage-jupyter #151). Name the fix for the common typo.
        hint = ""
        if spec.endswith(".py"):
            hint = f" Did you mean {spec + ':graph'!r} (or whatever your graph variable is named)?"
        raise ValueError(
            f"Invalid agent spec {spec!r}. Expected "
            "'path/to/module.py:object_name' or 'package.module:object_name' "
            f"(the ':object_name' suffix is required).{hint}"
        )

    module_path, _, obj_name = spec.rpartition(":")
    module_path, obj_name = module_path.strip(), obj_name.strip()
    if not module_path or not obj_name or not obj_name.isidentifier():
        raise ValueError(
            f"Invalid agent spec {spec!r}. Both a module/file path and an "
            "object name are required, e.g. 'agent.py:graph'."
        )
    return module_path, obj_name


def is_file_spec(module_path: str) -> bool:
    """True when the path half of a spec names a ``.py`` file rather than a module."""
    return module_path.endswith(".py") or any(sep in module_path for sep in ("/", "\\"))


def load_agent_spec(
    spec: str,
    *,
    base_dir: str | os.PathLike | None = None,
    stdout_to_stderr: bool = False,
) -> Any:
    """Load a LangGraph agent from a ``"path:object"`` spec string.

    Args:
        spec: ``"path/to/module.py:object_name"`` (file path) or
            ``"package.module:object_name"`` (dotted module path). The
            ``:object_name`` suffix is required. Surrounding whitespace is
            stripped and a leading ``~`` in a file path is expanded.
        base_dir: Directory a relative file path resolves against, and the
            fallback import root for a dotted module path. Defaults to the
            current working directory. Pass
            ``cfg.toml_dir_for("agent_spec")`` to resolve a ``langstage.toml``
            spec against that file's directory.
        stdout_to_stderr: Send anything the agent module ``print``\\ s while it
            is imported to ``sys.stderr``. Use it on machine-readable paths
            (``--json``, single-shot replies) so an import-time banner can't
            corrupt stdout (gh langstage-cli #136, langstage #140). Covers
            Python-level writes to ``sys.stdout``, not raw fd writes from C
            extensions.

    Returns:
        The agent object (typically a compiled LangGraph graph).

    Raises:
        ValueError: If the spec has no ``:object`` suffix (or is otherwise malformed).
        FileNotFoundError: If a ``.py`` file path does not exist.
        ImportError: If the module cannot be loaded/imported.
        AttributeError: If the object is not found in the module.
        TypeError: If the object is a ``str`` — a spec never points at another
            spec (gh langstage-cli #149).

    Example:
        agent = load_agent_spec("./my_agent.py:agent")
        agent = load_agent_spec("my_package.agents:research_graph")
    """
    module_path, obj_name = parse_agent_spec(spec)
    base = Path(base_dir).expanduser() if base_dir is not None else None

    redirect = contextlib.redirect_stdout(sys.stderr) if stdout_to_stderr else contextlib.nullcontext()
    with redirect:
        module = _import_module(module_path, base)

    if not hasattr(module, obj_name):
        raise AttributeError(
            f"Module {module_path!r} has no attribute {obj_name!r}."
        )
    obj = getattr(module, obj_name)
    if isinstance(obj, str):
        # A str attribute used to flow into build_agent(), which treats a str as a spec
        # and imported its VALUE ("openai:gpt-4o" -> "no module 'openai'"), blaming a
        # module the user never named. A spec is never followed a second time.
        raise TypeError(
            f"Agent spec {spec.strip()!r} resolved to a str ({obj!r}), not an agent. "
            f"Point the spec at the compiled graph object itself; {obj_name!r} holds a string."
        )
    return obj


def _ensure_on_sys_path(directory: Path, *, first: bool) -> None:
    """Put ``directory`` on ``sys.path`` once (idempotent across repeated loads).

    Kept for the life of the process, like ``python script.py`` / uvicorn: an agent
    can import a sibling lazily (inside a tool) long after load returns.
    """
    target = os.path.normcase(os.path.abspath(directory))
    for entry in sys.path:
        if os.path.normcase(os.path.abspath(entry or os.curdir)) == target:
            return
    if first:
        sys.path.insert(0, str(directory))
    else:
        sys.path.append(str(directory))
    importlib.invalidate_caches()


def _import_module(module_path: str, base: Path | None) -> Any:
    """Import a module from a file path or a dotted module path."""
    # File path: load from source location.
    if is_file_spec(module_path):
        file_path = Path(module_path).expanduser()
        if base is not None and not file_path.is_absolute():
            file_path = base / file_path
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")

        # The file's own directory goes first on sys.path, exactly as for
        # `python my_agent.py`, so `from tools import ...` finds ./tools.py (gh #150).
        _ensure_on_sys_path(file_path.parent, first=True)

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
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Retry from the base dir (cwd by default) only when the SPEC's own package is
        # what's missing — never mask a missing dependency imported inside it. A
        # console-script entry point has no cwd on sys.path, so a project-local
        # `mypkg.agents:graph` failed where uvicorn resolves it (gh #147). Appended,
        # not prepended: it's a fallback and must not shadow installed packages.
        missing = exc.name or ""
        parts = module_path.split(".")
        own = {".".join(parts[: i + 1]) for i in range(len(parts))}
        if missing not in own:
            raise
        root = (base or Path.cwd()).resolve()
        if not (root / parts[0]).is_dir() and not (root / f"{parts[0]}.py").is_file():
            raise
        _ensure_on_sys_path(root, first=False)
        return importlib.import_module(module_path)
