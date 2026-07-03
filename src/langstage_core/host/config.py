"""Shared ``LANGSTAGE_*`` configuration for hosts.

``HostConfig`` holds the keys every host has in common (agent spec, workspace
root, bind/title basics) and resolves them from one layered chain:

    defaults  <  langstage.toml  <  LANGSTAGE_* env vars  <  explicit overrides

Host-specific keys (theme, auth, model, Jupyter token, ...) belong in each
host's own subclass — drift *below* the shared core is fine — but every host
gets the same resolution order, the same TOML files, and the same env-var
names, so there's one place to look.

Legacy vocabulary (the pre-LangStage names) still works everywhere as a
deprecated fallback: ``DEEPAGENT_*`` env vars, project ``deepagents.toml``,
global ``~/.deepagents/config.toml``, and ``DEEPAGENTS_CONFIG_HOME``. The
canonical names win when both are set; using only the legacy env names emits
a once-per-var ``DeprecationWarning`` *and* a visible one-line stderr notice
(silence it with ``LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1``). Moving the global config out of
``~/.deepagents/`` also exits the schema collision with LangChain's dcode,
which owns that directory now.

Discoverability: ``HostConfig.resolve().describe()`` (or
``python -m langstage_core.host``) prints each value, where it came
from, and the env var / TOML key that sets it — so you never have to remember
the variable names.
"""
import os
import sys
import warnings
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, ClassVar

try:  # tomllib is stdlib on 3.11+; fall back to tomli; else the TOML layer is skipped.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 path
    try:
        import tomli as _tomllib  # type: ignore
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore

GLOBAL_TOML = Path.home() / ".langstage" / "config.toml"
PROJECT_TOML = "langstage.toml"
# Pre-rename locations, still honoured as fallbacks.
LEGACY_GLOBAL_TOML = Path.home() / ".deepagents" / "config.toml"
LEGACY_PROJECT_TOML = "deepagents.toml"

_CANONICAL_ENV_PREFIX = "LANGSTAGE"
_LEGACY_ENV_PREFIX = "DEEPAGENT"

_warned_legacy_env: set[str] = set()


def _env_pair(declared: str) -> tuple[str, str]:
    """Return ``(canonical, legacy)`` env-var names for a declared name.

    Hosts may declare either spelling in their ``_ENV`` maps during the
    rename transition; both resolve, canonical wins.
    """
    if declared.startswith(_CANONICAL_ENV_PREFIX):
        return declared, _LEGACY_ENV_PREFIX + declared[len(_CANONICAL_ENV_PREFIX):]
    if declared.startswith(_LEGACY_ENV_PREFIX):
        return _CANONICAL_ENV_PREFIX + declared[len(_LEGACY_ENV_PREFIX):], declared
    return declared, declared


def _warn_legacy_env(legacy: str, canonical: str) -> None:
    if legacy in _warned_legacy_env:
        return
    _warned_legacy_env.add(legacy)
    # LANGSTAGE_SUPPRESS_LEGACY_NOTICE silences EVERY legacy-env deprecation
    # signal — both the Python DeprecationWarning and the stderr notice — so the
    # "set ... to silence" hint we print is actually honest (a user who sets it
    # shouldn't still see a stray DeprecationWarning leak through, e.g. into a
    # VS Code output channel).
    if _env_bool(os.getenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE")):
        return
    warnings.warn(
        f"{legacy} is deprecated; use {canonical}.",
        DeprecationWarning,
        stacklevel=4,
    )
    _print_legacy_env_notice(legacy, canonical)


def _print_legacy_env_notice(legacy: str, canonical: str) -> None:
    """Print a one-line, user-visible deprecation notice to stderr.

    The ``DeprecationWarning`` raised by the caller is the correct signal for
    programmatic / strict consumers, but Python's *default* warning filter
    silently swallows it — so a real person running any LangStage CLI with a
    legacy ``DEEPAGENT_*`` env var never sees the nudge. Printing here (once per
    var, via the ``_warned_legacy_env`` dedupe in the caller) makes the
    deprecation visible across every surface from the one place they all resolve
    config — no per-surface code needed. ASCII-only so it can't crash a cp1252
    Windows console. Suppressed under pytest (keeps test output clean and can't
    break other repos' suites); the ``LANGSTAGE_SUPPRESS_LEGACY_NOTICE`` opt-out
    is handled by the caller (it gates the warning too).
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    print(
        f"note: {legacy} is deprecated; use {canonical}. "
        "(Legacy DEEPAGENT_* support will be removed in a future release; "
        "set LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1 to silence.)",
        file=sys.stderr,
    )


_warned_legacy_toml: set[str] = set()


def _warn_legacy_toml(path: Path, canonical_name: str) -> None:
    """Visible deprecation notice when a legacy-named TOML file is resolved
    (project ``deepagents.toml`` or global ``~/.deepagents/config.toml``).

    The legacy ``DEEPAGENT_*`` *env vars* already warn on use, but the legacy
    *TOML* files resolved silently — so a user who moved their env to
    ``LANGSTAGE_*`` but kept a ``deepagents.toml`` got no nudge (the same
    advertised-parity gap the env notice closes). Same once-per-file dedupe,
    ``LANGSTAGE_SUPPRESS_LEGACY_NOTICE`` opt-out, and pytest suppression as
    ``_warn_legacy_env``. (gh #25)
    """
    key = str(path)
    if key in _warned_legacy_toml:
        return
    _warned_legacy_toml.add(key)
    if _env_bool(os.getenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE")):
        return
    warnings.warn(
        f"{path} is deprecated; rename it to {canonical_name}.",
        DeprecationWarning,
        stacklevel=4,
    )
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    print(
        f"note: config file {path} uses the legacy name; rename it to "
        f"{canonical_name}. (Legacy deepagents.toml support will be removed in a "
        "future release; set LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1 to silence.)",
        file=sys.stderr,
    )


def _env_bool(value: str | None, default: bool = False) -> bool:
    """Parse an env-var string into a bool."""
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ── TOML layer ───────────────────────────────────────────────────────


def _global_toml_path() -> Path:
    override = os.getenv("LANGSTAGE_CONFIG_HOME") or os.getenv("DEEPAGENTS_CONFIG_HOME")
    if override:
        return Path(override).expanduser() / "config.toml"
    # New home wins when present; otherwise fall back to the legacy location
    # (which load_toml_config skips anyway if the file doesn't exist).
    return GLOBAL_TOML if GLOBAL_TOML.is_file() else LEGACY_GLOBAL_TOML


def _find_project_toml(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for ``langstage.toml``.

    Checks ``langstage.toml`` then legacy ``deepagents.toml`` in each
    directory, so the nearest file wins and the new name wins within a
    directory.
    """
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        for fname in (PROJECT_TOML, LEGACY_PROJECT_TOML):
            candidate = directory / fname
            if candidate.is_file():
                return candidate
    return None


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_toml(path: Path) -> dict:
    if _tomllib is None:  # pragma: no cover
        return {}
    # Decode with utf-8-sig so a leading UTF-8 BOM is stripped. Notepad and
    # PowerShell's `Out-File -Encoding utf8` both write a BOM by default on
    # Windows, and `tomllib.load()` (binary) chokes on it with a cryptic
    # "Invalid statement (at line 1, column 1)" — which, because jupyter's
    # config resolves at import time, bricked the whole extension. (gh #-dogfood)
    try:
        return _tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001 — a broken config must not brick every entrypoint
        # Several surfaces resolve config at import time, so a raw TOMLDecodeError
        # (or an unreadable file) here would kill --version / --help / --demo,
        # `import langstage_jupyter`, and the server extension — not just the command
        # that needs the config. Skip the bad file with a visible one-line notice and
        # fall back to env + defaults. ASCII-only (cp1252-safe). (gh #42)
        print(
            f"note: ignoring malformed config {path} "
            f"({type(exc).__name__}: {exc}); using environment + defaults instead.",
            file=sys.stderr,
        )
        return {}


def load_toml_config(start: Path | None = None) -> tuple[dict, list[Path]]:
    """Load + deep-merge the global and project ``langstage.toml`` files.

    Global is ``~/.langstage/config.toml`` (override the dir with
    ``LANGSTAGE_CONFIG_HOME``; legacy ``~/.deepagents/config.toml`` and
    ``DEEPAGENTS_CONFIG_HOME`` still work as fallbacks); project is the
    nearest ``langstage.toml`` — or legacy ``deepagents.toml`` — at or above
    ``start``/cwd. Project wins on conflicts. Returns
    ``(merged_config, sources_used)``; ``({}, [])`` if no TOML reader is
    available (Python 3.10 without ``tomli``).
    """
    sources: list[Path] = []
    merged: dict = {}
    if _tomllib is None:  # pragma: no cover
        return merged, sources
    gpath = _global_toml_path()
    if gpath.is_file():
        merged = _deep_merge(merged, _read_toml(gpath))
        sources.append(gpath)
        if gpath == LEGACY_GLOBAL_TOML:
            _warn_legacy_toml(gpath, str(GLOBAL_TOML))
    ppath = _find_project_toml(start)
    if ppath is not None:
        merged = _deep_merge(merged, _read_toml(ppath))
        sources.append(ppath)
        if ppath.name == LEGACY_PROJECT_TOML:
            _warn_legacy_toml(ppath, PROJECT_TOML)
    return merged, sources


def _get_dotted(data: dict, dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ── Config dataclass ─────────────────────────────────────────────────


@dataclass
class HostConfig:
    """Shared configuration for deep-agent hosts.

    Subclass to add host-specific fields and extend the ``_ENV`` / ``_TOML``
    maps so they resolve through the same chain:

        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"
            _ENV = {"theme": ("LANGSTAGE_THEME", str)}
            _TOML = {"theme": "ui.theme"}

    ``resolve()`` merges the maps across the MRO, so the subclass inherits all
    of ``HostConfig``'s keys and adds its own.
    """

    agent_spec: str | None = None     # LANGSTAGE_AGENT_SPEC ("path.py:var")
    workspace_root: Path = Path(".")  # LANGSTAGE_WORKSPACE_ROOT
    host: str = "localhost"           # LANGSTAGE_HOST
    port: int = 8050                  # LANGSTAGE_PORT
    debug: bool = False               # LANGSTAGE_DEBUG
    title: str = "LangStage"          # LANGSTAGE_TITLE

    # field -> (env var, caster). Canonical names are LANGSTAGE_*; the
    # matching DEEPAGENT_* legacy names resolve as deprecated fallbacks
    # (see _env_pair).
    _ENV: ClassVar[dict[str, tuple[str, Callable[[str], Any]]]] = {
        "agent_spec": ("LANGSTAGE_AGENT_SPEC", str),
        "workspace_root": ("LANGSTAGE_WORKSPACE_ROOT", Path),
        "host": ("LANGSTAGE_HOST", str),
        "port": ("LANGSTAGE_PORT", int),
        "debug": ("LANGSTAGE_DEBUG", _env_bool),
        "title": ("LANGSTAGE_TITLE", str),
    }
    # field -> dotted key in deepagents.toml
    _TOML: ClassVar[dict[str, str]] = {
        "agent_spec": "agent.spec",
        "workspace_root": "workspace.root",
        "host": "server.host",
        "port": "server.port",
        "debug": "debug",
        "title": "ui.title",
    }

    # ---- map collection across the subclass MRO ----

    @classmethod
    def _env_map(cls) -> dict[str, tuple[str, Callable[[str], Any]]]:
        merged: dict[str, tuple[str, Callable[[str], Any]]] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_ENV", {}))
        return merged

    @classmethod
    def _toml_map(cls) -> dict[str, str]:
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_TOML", {}))
        return merged

    # ---- resolution ----

    @classmethod
    def from_env(cls) -> "HostConfig":
        """Resolve from env vars + defaults only (no TOML, no overrides).

        Kept for back-compat; ``resolve()`` is the fuller entry point.
        """
        return cls.resolve(use_toml=False)

    @classmethod
    def resolve(
        cls,
        *,
        overrides: dict[str, Any] | None = None,
        toml_start: Path | None = None,
        env: dict[str, str] | None = None,
        use_toml: bool = True,
    ) -> "HostConfig":
        """Resolve config through ``defaults < TOML < env < overrides``.

        Each field's origin is recorded for ``describe()`` / ``sources``.

        Args:
            overrides: Highest-precedence values (e.g. CLI flags / Python args).
                ``None`` values are ignored so unset flags don't clobber.
            toml_start: Directory to start the ``deepagents.toml`` search from.
            env: Environment mapping (defaults to ``os.environ``).
            use_toml: Set False to skip the TOML layer entirely.
        """
        overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        env = os.environ if env is None else env
        toml_data, toml_paths = (load_toml_config(toml_start) if use_toml else ({}, []))
        env_map = cls._env_map()
        toml_map = cls._toml_map()

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for f in fields(cls):
            name = f.name
            if f.default is not MISSING:
                val: Any = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                val = f.default_factory()  # type: ignore[misc]
            else:
                val = None
            src = "default"

            tkey = toml_map.get(name)
            if tkey is not None:
                tv = _get_dotted(toml_data, tkey)
                if tv is not None:
                    val = _coerce(f, tv)
                    src = f"toml ({toml_paths[-1].name})" if toml_paths else "toml"

            if name in env_map:
                var, caster = env_map[name]
                canonical, legacy = _env_pair(var)
                ev = env.get(canonical)
                used = canonical
                if ev is None or ev == "":
                    ev = env.get(legacy)
                    used = legacy
                    if ev not in (None, "") and legacy != canonical:
                        _warn_legacy_env(legacy, canonical)
                if ev is not None and ev != "":
                    val = caster(ev)
                    src = f"env:{used}"

            if name in overrides:
                val = overrides[name]
                src = "override"

            values[name] = val
            sources[name] = src

        obj = cls(**values)
        obj._sources = sources           # type: ignore[attr-defined]
        obj._toml_paths = toml_paths     # type: ignore[attr-defined]
        return obj

    def merge(self, **overrides: Any) -> "HostConfig":
        """Return a copy with non-``None`` overrides applied."""
        valid = {f.name for f in fields(self)}
        applied = {k: v for k, v in overrides.items() if v is not None and k in valid}
        return replace(self, **applied)

    # ---- introspection ----

    @property
    def sources(self) -> dict[str, str]:
        """Per-field origin from the last ``resolve()`` (field -> source)."""
        return getattr(self, "_sources", {})

    def describe(self, omit_keys: list[str] | None = None) -> str:
        """Human-readable dump: value, source, and the env var / TOML key.

        This is what ``python -m langstage_core.host`` prints.

        ``omit_keys`` hides inherited keys a particular stage doesn't actually
        honor — e.g. a stdio-only sidecar passes ``omit_keys=["host", "port"]``
        so ``--show-config`` never advertises an env var (with a confident
        source attribution) that has zero effect on that surface.
        """
        omit = set(omit_keys or ())
        env_map = type(self)._env_map()
        toml_map = type(self)._toml_map()
        src = self.sources
        lines = ["Resolved config  (value  [source]):", ""]
        for f in fields(self):
            if f.name in omit:
                continue
            value = getattr(self, f.name)
            origin = src.get(f.name, "default")
            hints = []
            if f.name in env_map:
                canonical, legacy = _env_pair(env_map[f.name][0])
                env_hint = f"env: {canonical}"
                if legacy != canonical:
                    env_hint += f" (legacy {legacy})"
                hints.append(env_hint)
            if f.name in toml_map:
                hints.append(f"toml: {toml_map[f.name]}")
            hint = f"   ({', '.join(hints)})" if hints else ""
            lines.append(f"  {f.name:<16} = {str(value):<26} [{origin}]{hint}")
        toml_paths = getattr(self, "_toml_paths", [])
        lines.append("")
        if toml_paths:
            lines.append("  TOML read from: " + ", ".join(str(p) for p in toml_paths))
        else:
            lines.append("  TOML: no langstage.toml (or legacy deepagents.toml) found")
        return "\n".join(lines)


def _coerce(f: Any, value: Any) -> Any:
    """Coerce a TOML value to the field's expected shape (Path fields only)."""
    if isinstance(getattr(f, "default", None), Path) and not isinstance(value, Path):
        return Path(value)
    return value
