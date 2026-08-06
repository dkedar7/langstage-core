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

# Paths whose TOML parse failed. Callers (loaders, --show-config) consult this so a
# malformed/ignored file is never listed as "read" (gh langstage-hermes #61).
_malformed_toml: set[str] = set()
# Dedupe the "ignoring malformed config" notice — _read_toml is called more than once
# per path (loader + the per-file source-labeling re-read), which double-warned (#61).
_warned_malformed_toml: set[str] = set()
# Same dedupe for a TOML value of the wrong TYPE (below), keyed on (dotted key, value):
# several surfaces each call resolve() in one process (import-time module constants,
# --show-config, the launcher), and one typo shouldn't print the same note three times.
_warned_malformed_toml_value: set[tuple[str, str]] = set()
# Same dedupe for a malformed numeric ENV var (gh #104), keyed on (var, value).
_warned_malformed_env_value: set[tuple[str, str]] = set()
# Same dedupe for a value that coerced fine but failed a semantic validator (gh
# langstage #123: an in-range int that's an out-of-range port), keyed on (field, value).
_warned_invalid_value: set[tuple[str, str]] = set()


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
    """Parse an env-var string into a bool, leniently (never raises).

    Absent/empty -> ``default``; a present but non-truthy value -> ``False``. Kept
    lenient for direct flag reads (``if _env_bool(os.getenv("LANGSTAGE_SUPPRESS_..."))``)
    where a typo should just mean "off", not crash a diagnostic. Config-resolution
    boolean *fields* use :func:`_env_bool_strict` instead, so a malformed value there
    is flagged with a note and falls back to the field default.
    """
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _env_bool_strict(value: str) -> bool:
    """Parse a boolean env value, raising ``ValueError`` on an unrecognized string.

    The caster for boolean *config fields* in ``_ENV`` maps. A malformed boolean
    (``MEMORY_ENABLED=enabled`` — a natural way to try to *enable* something) used to
    coerce silently to ``False`` and flip a ``True`` default off, with ``--show-config``
    even crediting ``[env:...]`` as if honored (gh langstage-hermes #92). Raising here
    lets ``resolve()``'s guard catch it, emit the same one-line ``note:`` a malformed
    *numeric* env already gets (gh #83/#104), and fall back to the field's default —
    so booleans and numbers degrade consistently. Recognized: ``1/true/yes/on`` ->
    True, ``0/false/no/off`` -> False (case-insensitive); anything else raises.
    """
    v = str(value).strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ValueError(
        f"unrecognized boolean {value!r}; expected one of {', '.join(_TRUTHY + _FALSY)}"
    )


def _port_in_range(value: Any) -> int:
    """Validate a resolved port is in the bindable range 1-65535, else raise ``ValueError``.

    ``port`` was type-checked (a non-int env degrades to the default + note) but never
    range-checked, so an in-range *integer* out of the valid *port* range (``70000``)
    sailed through, was advertised by ``--show-config`` and the startup banner, and then
    uvicorn silently masked it to 16 bits (``70000 & 0xFFFF == 4464``) — the server bound
    a *different* port than everything advertised, with no error (gh langstage #123). This
    raises so ``resolve()``'s validator guard degrades it to the default + a note, exactly
    like a malformed numeric value.
    """
    n = int(value)
    if not (1 <= n <= 65535):
        raise ValueError(f"port {n} out of range (must be 1-65535)")
    return n


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
    key = str(path)
    try:
        data = _tomllib.loads(path.read_text(encoding="utf-8-sig"))
        _malformed_toml.discard(key)  # a file that previously failed now parses
        return data
    except Exception as exc:  # noqa: BLE001 — a broken config must not brick every entrypoint
        # Several surfaces resolve config at import time, so a raw TOMLDecodeError
        # (or an unreadable file) here would kill --version / --help / --demo,
        # `import langstage_jupyter`, and the server extension — not just the command
        # that needs the config. Skip the bad file, record it as malformed so it isn't
        # later listed as "read", and warn ONCE (ASCII-only, cp1252-safe). (gh #42, #61)
        _malformed_toml.add(key)
        if key not in _warned_malformed_toml:
            print(
                f"note: ignoring malformed config {path} "
                f"({type(exc).__name__}: {exc}); using environment + defaults instead.",
                file=sys.stderr,
            )
            _warned_malformed_toml.add(key)
        return {}


def _load_toml_files(
    start: Path | None = None,
) -> tuple[dict, list[Path], list[tuple[Path, dict]]]:
    """Like :func:`load_toml_config`, but also return the per-file ``(path, data)`` list
    in precedence order (global then project).

    Keeping each file's own data lets :meth:`HostConfig.resolve` attribute a value to the
    file it actually won from, instead of blindly to the last file read — a global-config
    value must not be mislabeled as coming from the project ``langstage.toml`` (gh
    langstage #119). Each file is read exactly once.
    """
    sources: list[Path] = []
    per_file: list[tuple[Path, dict]] = []
    merged: dict = {}
    if _tomllib is None:  # pragma: no cover
        return merged, sources, per_file
    gpath = _global_toml_path()
    if gpath.is_file():
        data = _read_toml(gpath)
        merged = _deep_merge(merged, data)
        if str(gpath) not in _malformed_toml:  # don't list an ignored file as read (#61)
            sources.append(gpath)
            per_file.append((gpath, data))
            if gpath == LEGACY_GLOBAL_TOML:
                _warn_legacy_toml(gpath, str(GLOBAL_TOML))
    ppath = _find_project_toml(start)
    if ppath is not None:
        data = _read_toml(ppath)
        merged = _deep_merge(merged, data)
        if str(ppath) not in _malformed_toml:
            sources.append(ppath)
            per_file.append((ppath, data))
            if ppath.name == LEGACY_PROJECT_TOML:
                _warn_legacy_toml(ppath, PROJECT_TOML)
    return merged, sources, per_file


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
    merged, sources, _ = _load_toml_files(start)
    return merged, sources


def _get_dotted(data: dict, dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _flatten_toml_keys(data: dict, prefix: str = "") -> list[str]:
    """Dotted leaf keys of a parsed TOML dict: ``{"ui": {"theme": "x"}}`` -> ``["ui.theme"]``.

    Used to spot keys present in a config file that map to no config field — i.e. typo'd
    or misplaced keys the layered config silently ignores (gh langstage #120,
    langstage-vscode #82).
    """
    out: list[str] = []
    for k, v in data.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten_toml_keys(v, f"{dotted}."))
        else:
            out.append(dotted)
    return out


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
        "debug": ("LANGSTAGE_DEBUG", _env_bool_strict),
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
    # Top-level TOML tables whose keys are user-defined passthroughs, NOT config fields,
    # so they must never be flagged as "unknown" (gh langstage #120 / langstage-vscode #82).
    # ``[configurable]`` is forwarded verbatim to the graph's ``config["configurable"]``.
    # A subclass with its own passthrough table overrides this (widen the tuple).
    _TOML_PASSTHROUGH: ClassVar[tuple[str, ...]] = ("configurable",)
    # field -> validator(value) -> value, raising ValueError/TypeError on a value that
    # coerced to the right TYPE but is semantically invalid (e.g. an out-of-range port).
    # resolve() degrades a rejected value to the field default + note, like a malformed
    # numeric value. Merged across the MRO so a subclass can add its own. (gh langstage #123)
    _VALIDATORS: ClassVar[dict[str, Callable[[Any], Any]]] = {
        "port": _port_in_range,
    }

    # ---- map collection across the subclass MRO ----

    @classmethod
    def _env_map(cls) -> dict[str, tuple[str, Callable[[str], Any]]]:
        merged: dict[str, tuple[str, Callable[[str], Any]]] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_ENV", {}))
        return merged

    @classmethod
    def _validators_map(cls) -> dict[str, Callable[[Any], Any]]:
        merged: dict[str, Callable[[Any], Any]] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_VALIDATORS", {}))
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
        toml_data, toml_paths, toml_files = (
            _load_toml_files(toml_start) if use_toml else ({}, [], [])
        )
        env_map = cls._env_map()
        toml_map = cls._toml_map()
        validators = cls._validators_map()

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
            default_val = val  # the built-in default; the degrade target for an invalid resolved value (gh #123)

            tkey = toml_map.get(name)
            if tkey is not None:
                tv = _get_dotted(toml_data, tkey)
                if tv is not None:
                    try:
                        val = _coerce(f, tv)
                    except (ValueError, TypeError) as exc:
                        # An uncoercible value keeps the default AND the "default"
                        # source, so --show-config can never present an unusable
                        # value as a live TOML setting. (gh langstage-jupyter #78)
                        _warn_malformed_toml_value(tkey, tv, exc, val, toml_paths)
                    else:
                        # Attribute the value to the highest-precedence file that actually
                        # defines this key, not blindly to the last file read — a global
                        # (~/.langstage/config.toml) value must not be mislabeled as coming
                        # from the project langstage.toml (gh langstage #119).
                        winner = next(
                            (p for p, d in reversed(toml_files) if _get_dotted(d, tkey) is not None),
                            None,
                        )
                        if winner is not None:
                            src = f"toml ({winner.name})"
                        elif toml_paths:
                            src = f"toml ({toml_paths[-1].name})"
                        else:
                            src = "toml"

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
                    try:
                        val = caster(ev)
                        src = f"env:{used}"
                    except (ValueError, TypeError) as exc:
                        # A malformed numeric env var (LANGSTAGE_PORT=abc, an
                        # unexpanded "$PORT", a stray "8050 x") used to raise an
                        # uncaught error straight out of resolve() and crash every
                        # entrypoint (gh #104). Degrade exactly like the TOML path
                        # above: keep whatever was resolved so far -- crucially the
                        # TOML value if one is set, NOT the built-in default -- so a
                        # bad env var can't clobber a valid langstage.toml value
                        # (gh langstage-jupyter #83), and leave src pointing at that
                        # real source so --show-config never attributes the kept
                        # value to the rejected env var.
                        _warn_malformed_env_value(used, ev, exc, val, src)

            if name in overrides:
                val = overrides[name]
                src = "override"

            validator = validators.get(name)
            if validator is not None and val is not None:
                try:
                    val = validator(val)
                except (ValueError, TypeError) as exc:
                    # Coerced to the right TYPE but semantically invalid (an in-range int
                    # that's an out-of-range PORT, which uvicorn would silently mask to 16
                    # bits). Degrade to the default + note — never a silent misbind — and
                    # reset the source so --show-config can't present the rejected value as
                    # a live setting (gh langstage #123).
                    _warn_invalid_value(name, val, exc, default_val)
                    val, src = default_val, "default"

            values[name] = val
            sources[name] = src

        obj = cls(**values)
        obj._sources = sources           # type: ignore[attr-defined]
        obj._toml_paths = toml_paths     # type: ignore[attr-defined]
        obj._toml_data = toml_data       # type: ignore[attr-defined]  # for unknown_toml_keys()
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

    def unknown_toml_keys(self) -> list[str]:
        """Dotted keys present in the loaded TOML file(s) that map to no config field.

        These are the typo'd / misplaced / unknown keys the layered config **silently
        ignores** — the single most common config mistake, which ``config`` /
        ``--show-config`` couldn't surface, so "edit it, then verify" couldn't catch it
        (gh langstage #120, langstage-vscode #82). Keys under a passthrough table
        (:attr:`_TOML_PASSTHROUGH`, e.g. ``[configurable]``) are user-defined and never
        reported. Sorted; empty when the config is clean or no TOML was read.
        """
        data = getattr(self, "_toml_data", {}) or {}
        valid = set(type(self)._toml_map().values())
        passthrough = type(self)._TOML_PASSTHROUGH
        unknown = []
        for dotted in _flatten_toml_keys(data):
            if dotted in valid:
                continue
            if any(dotted == p or dotted.startswith(p + ".") for p in passthrough):
                continue
            unknown.append(dotted)
        return sorted(unknown)

    def config_dict(self, omit_keys: list[str] | None = None) -> dict:
        """The resolved config as a machine-readable object — the structured twin of
        :meth:`describe` (gh langstage-jupyter #88 / langstage-vscode #71).

        ``--show-config`` is the family's config-diagnostic verb, but it emitted only
        aligned human text, so a CI/tooling consumer had to regex the ``[source]``
        bracket out of formatted output — brittle, and it breaks the moment the human
        layout is tweaked. This returns the same value + provenance every surface's
        ``--show-config --json`` renders, so the config-precedence contract becomes
        assertable instead of scrape-only.

        Same key set and the same source labels as :meth:`describe` (a test pins that
        they agree); ``omit_keys`` hides the same inherited keys a stage doesn't honor.
        Shape::

            {"config": {"<field>": {"value", "source", "env", "legacy_env", "toml"}},
             "toml": {"found": bool, "path": <abs or None>}}

        A host that tracks more (e.g. langstage-jupyter's malformed-TOML flag, gh #86)
        extends the ``toml`` block in its own override.
        """
        omit = set(omit_keys or ())
        env_map = type(self)._env_map()
        toml_map = type(self)._toml_map()
        src = self.sources
        config: dict[str, dict] = {}
        for f in fields(self):
            if f.name in omit:
                continue
            entry: dict = {
                "value": getattr(self, f.name),
                "source": src.get(f.name, "default"),
                "env": None,
                "legacy_env": None,
                "toml": toml_map.get(f.name),
            }
            if f.name in env_map:
                canonical, legacy = _env_pair(env_map[f.name][0])
                entry["env"] = canonical
                entry["legacy_env"] = legacy if legacy != canonical else None
            config[f.name] = entry
        toml_paths = getattr(self, "_toml_paths", [])
        toml_block = {
            "found": bool(toml_paths),
            "path": str(toml_paths[-1]) if toml_paths else None,
            "unknown_keys": self.unknown_toml_keys(),
        }
        return {"config": config, "toml": toml_block}

    def describe(
        self,
        omit_keys: list[str] | None = None,
        configurable: dict | None = None,
    ) -> str:
        """Human-readable dump: value, source, and the env var / TOML key.

        This is what ``python -m langstage_core.host`` prints.

        ``omit_keys`` hides inherited keys a particular stage doesn't actually
        honor — e.g. a stdio-only sidecar passes ``omit_keys=["host", "port"]``
        so ``--show-config`` never advertises an env var (with a confident
        source attribution) that has zero effect on that surface.

        ``configurable`` is the resolved ``[configurable]`` TOML table (the keys
        forwarded to the graph's ``config["configurable"]``). It is rendered here so
        the COMPLETE config diagnostic comes from this one method — every surface's
        ``--show-config`` and interactive ``/config`` render it identically instead of
        each bolting the table on separately and drifting (the recurring gh #55/#57/
        #61/#64/#66 "config-diagnostic drift" class).
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
        unknown = self.unknown_toml_keys()
        if unknown:
            # Surface keys the config silently ignored so a typo/misplacement is visible
            # on the same verb a user runs to check their config (gh langstage #120,
            # langstage-vscode #82). ASCII-only (cp1252-safe).
            lines.append(
                "  unknown TOML keys (ignored - a typo or wrong table?): " + ", ".join(unknown)
            )
        if configurable:
            lines.append("")
            lines.append("  LangGraph configurable:")
            for k, v in configurable.items():
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:50] + "..."
                lines.append(f"    {k}: {v_str}")
        return "\n".join(lines)


_NUMERIC_TYPES: dict[str, type] = {"int": int, "float": float}


def _numeric_field_type(f: Any) -> type | None:
    """Return ``int``/``float`` if the field declares a plain numeric type, else None.

    Prefers the annotation (the declared type is the authority) and falls back to
    the default's type, so it works whether or not the declaring module uses
    ``from __future__ import annotations``.

    ``bool`` is deliberately never returned even though it is a subclass of
    ``int``: a bool *field* must keep accepting TOML ``true``/``false`` untouched,
    and a bool supplied *for* a numeric field is malformed input, not ``1``.
    """
    ann = getattr(f, "type", None)
    if isinstance(ann, type):
        if ann is bool:
            return None
        if ann in (int, float):
            return ann
    elif isinstance(ann, str) and ann in _NUMERIC_TYPES:
        return _NUMERIC_TYPES[ann]
    default = getattr(f, "default", None)
    if isinstance(default, bool):
        return None
    if isinstance(default, (int, float)):
        return type(default)
    return None


def _coerce(f: Any, value: Any) -> Any:
    """Coerce a TOML value to the field's declared shape (Path and numeric fields).

    TOML is typed, but nothing checked that a value's type matched the field it
    landed in — only ``Path`` fields were coerced, and everything else was passed
    through verbatim. So the single most common TOML mistake, quoting a number
    (``execute_timeout = "300"``), yielded the Python ``str`` ``'300'`` for a field
    declared ``float``: ``--show-config`` strips the quotes so it looked correct,
    and the defect surfaced far away as a raw ``TypeError`` the first time
    something did arithmetic on it, with no pointer back to ``langstage.toml``.

    Numeric fields are now cast, so a value that can sensibly be coerced is
    (``"300"`` -> ``300.0``). One that can't (``"warm"``, ``true``, a table)
    raises ``ValueError``/``TypeError``; ``resolve()`` catches it, keeps the
    default, and prints the same one-line ``note: ignoring malformed ...; using
    default ... instead.`` the numeric env casters and malformed-syntax TOML
    (gh #42) already emit. (gh langstage-jupyter #78)
    """
    if isinstance(getattr(f, "default", None), Path) and not isinstance(value, Path):
        return Path(value)

    numeric = _numeric_field_type(f)
    if numeric is not None:
        # Check bool BEFORE the isinstance(value, numeric) fast path: bool subclasses
        # int, so `temperature = true` would otherwise sail through (or coerce to 1.0).
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"expected {numeric.__name__}, got {type(value).__name__}")
        if isinstance(value, numeric):
            return value
        coerced = numeric(value)
        # Reject a fractional value for an int field rather than silently truncating
        # it. `port = 8050.7` -> 8050 would be exactly the defect this fix exists to
        # close: a wrong-typed value quietly accepted as something the user did not
        # write. An integral float (`8050.0`, or a config generator's "8050.0") is
        # unambiguous, so it still coerces. (gh langstage-jupyter #78)
        if numeric is int and float(value) != coerced:
            raise ValueError(f"expected int, got non-integral {value!r}")
        return coerced
    return value


def _warn_malformed_env_value(
    var: str, value: str, exc: Exception, kept: Any, kept_src: str
) -> None:
    """One-line stderr note when a numeric ENV var can't be cast to its field's type.

    The env-side counterpart of :func:`_warn_malformed_toml_value` (gh #104). The env
    layer had gone unguarded — a single bad char in ``LANGSTAGE_PORT`` raised straight
    out of ``resolve()`` and crashed every entrypoint — even though this module's own
    docstrings already advertised that the "numeric env casters" emit exactly this
    note. Now they do.

    Unlike the TOML note, this reports the value that is actually *kept* and where it
    came from (``kept_src``), because the env layer sits above TOML: a rejected env
    var falls back to the ``langstage.toml`` value if one is set, and only otherwise to
    the default. Saying "using default" would be wrong (and would mask gh
    langstage-jupyter #83). ASCII-only so it can't crash a cp1252 Windows console.
    """
    dedupe = (var, value)
    if dedupe in _warned_malformed_env_value:
        return
    _warned_malformed_env_value.add(dedupe)
    # "default X" when nothing else set it; "X (toml (...))" when a real config
    # value is what the rejected env var falls back to.
    kept_desc = f"default {kept!r}" if kept_src == "default" else f"{kept!r} ({kept_src})"
    print(
        f"note: ignoring malformed {var}={value!r} "
        f"({type(exc).__name__}: {exc}); using {kept_desc} instead.",
        file=sys.stderr,
    )


def _warn_invalid_value(field: str, value: Any, exc: Exception, default: Any) -> None:
    """One-line stderr note when a resolved value coerced fine but failed a validator.

    The semantic-validation counterpart of :func:`_warn_malformed_env_value` (gh langstage
    #123: an out-of-range port). Degrades to the field default (validation runs after all
    layers, so there's no lower layer to fall back to). ASCII-only so it can't crash a
    cp1252 Windows console. Deduped so several ``resolve()`` calls in one process don't
    repeat it.
    """
    dedupe = (field, str(value))
    if dedupe in _warned_invalid_value:
        return
    _warned_invalid_value.add(dedupe)
    print(
        f"note: ignoring invalid {field}={value!r} "
        f"({type(exc).__name__}: {exc}); using default {default!r} instead.",
        file=sys.stderr,
    )


def _warn_malformed_toml_value(
    key: str, value: Any, exc: Exception, default: Any, paths: list[Path]
) -> None:
    """One-line stderr note when a TOML value can't be coerced to its field's type.

    Mirrors the numeric env casters' note and #42's malformed-syntax note: name what
    was ignored, why, and what is used instead. It also names the file, because the
    whole point of gh langstage-jupyter #78 is that the user was never pointed back
    at ``langstage.toml``. ASCII-only so it can't crash a cp1252 Windows console.
    """
    dedupe = (key, repr(value))
    if dedupe in _warned_malformed_toml_value:
        return
    _warned_malformed_toml_value.add(dedupe)
    where = f" in {paths[-1]}" if paths else ""
    print(
        f"note: ignoring malformed {key}={value!r}{where} "
        f"({type(exc).__name__}: {exc}); using default {default!r} instead.",
        file=sys.stderr,
    )
