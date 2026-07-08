"""Tests for the layered HostConfig resolver (defaults < TOML < env < overrides)."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from langstage_core.host import HostConfig


@pytest.fixture
def isolated_global(tmp_path, monkeypatch):
    """Point the global deepagents config at an empty dir so the host machine's
    ~/.deepagents/config.toml can't leak into tests."""
    gdir = tmp_path / "global"
    gdir.mkdir()
    monkeypatch.setenv("DEEPAGENTS_CONFIG_HOME", str(gdir))
    return gdir


def _toml(dir_: Path, body: str) -> Path:
    p = dir_ / "deepagents.toml"
    p.write_text(body)
    return p


class TestResolveLayers:
    def test_defaults(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 8050
        assert cfg.agent_spec is None
        assert cfg.workspace_root == Path(".")
        assert set(cfg.sources.values()) == {"default"}

    def test_env_layer(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_AGENT_SPEC": "a.py:g", "DEEPAGENT_PORT": "9000",
                 "DEEPAGENT_DEBUG": "true"},
            toml_start=tmp_path,
        )
        assert cfg.agent_spec == "a.py:g"
        assert cfg.port == 9000
        assert cfg.debug is True
        assert cfg.sources["agent_spec"] == "env:DEEPAGENT_AGENT_SPEC"

    def test_toml_layer(self, isolated_global, tmp_path):
        _toml(tmp_path, '[agent]\nspec = "x.py:graph"\n[server]\nport = 7000\n')
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.agent_spec == "x.py:graph"
        assert cfg.port == 7000
        assert cfg.sources["agent_spec"].startswith("toml")

    def test_precedence_toml_env_override(self, isolated_global, tmp_path):
        _toml(tmp_path, "[server]\nport = 1111\n")
        # toml only
        assert HostConfig.resolve(env={}, toml_start=tmp_path).port == 1111
        # env beats toml
        assert HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, toml_start=tmp_path
        ).port == 2222
        # override beats env
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, overrides={"port": 3333}, toml_start=tmp_path
        )
        assert cfg.port == 3333
        assert cfg.sources["port"] == "override"

    def test_none_override_ignored(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, overrides={"port": None}, toml_start=tmp_path
        )
        assert cfg.port == 2222

    def test_workspace_root_coerced_to_path(self, isolated_global, tmp_path):
        _toml(tmp_path, '[workspace]\nroot = "/tmp/ws"\n')
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.workspace_root == Path("/tmp/ws")


class TestIntrospection:
    def test_describe_lists_var_names_and_keys(self, isolated_global, tmp_path):
        text = HostConfig.resolve(env={}, toml_start=tmp_path).describe()
        assert "DEEPAGENT_AGENT_SPEC" in text   # the var you can never remember
        assert "agent.spec" in text             # its TOML key
        assert "[default]" in text

    def test_describe_marks_source(self, isolated_global, tmp_path):
        text = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "9000"}, toml_start=tmp_path
        ).describe()
        assert "env:DEEPAGENT_PORT" in text

    def test_describe_omit_keys_hides_inert_keys(self, isolated_global, tmp_path):
        # A stdio-only stage hides keys it doesn't honor so --show-config never
        # advertises an env var with no effect on that surface (gh: jupyter #30,
        # vscode #14).
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        full = cfg.describe()
        assert "host" in full and "port" in full
        trimmed = cfg.describe(omit_keys=["host", "port"])
        assert "\n  host " not in trimmed
        assert "\n  port " not in trimmed
        assert "LANGSTAGE_HOST" not in trimmed and "LANGSTAGE_PORT" not in trimmed
        # keys it DOES honor are still shown
        assert "agent_spec" in trimmed and "workspace_root" in trimmed


class TestSubclass:
    def test_subclass_adds_keys_to_same_chain(self, isolated_global, tmp_path):
        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"
            _ENV: ClassVar[dict] = {"theme": ("DEEPAGENT_THEME", str)}
            _TOML: ClassVar[dict] = {"theme": "ui.theme"}

        _toml(tmp_path, '[ui]\ntheme = "solarized"\n')
        # toml provides theme...
        cfg = WebConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.theme == "solarized"
        # ...env overrides it, and base keys still resolve
        cfg = WebConfig.resolve(env={"DEEPAGENT_THEME": "dark"}, toml_start=tmp_path)
        assert cfg.theme == "dark"
        assert cfg.port == 8050
        assert cfg.sources["theme"] == "env:DEEPAGENT_THEME"
        assert "DEEPAGENT_THEME" in cfg.describe()


class TestLangstageVocabulary:
    """Canonical LANGSTAGE_* / langstage.toml with deprecated legacy fallbacks."""

    def test_canonical_env_resolves(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"LANGSTAGE_AGENT_SPEC": "a.py:g", "LANGSTAGE_PORT": "9100"},
            toml_start=tmp_path,
        )
        assert cfg.agent_spec == "a.py:g"
        assert cfg.port == 9100
        assert cfg.sources["agent_spec"] == "env:LANGSTAGE_AGENT_SPEC"

    def test_canonical_beats_legacy(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"LANGSTAGE_PORT": "1111", "DEEPAGENT_PORT": "2222"},
            toml_start=tmp_path,
        )
        assert cfg.port == 1111
        assert cfg.sources["port"] == "env:LANGSTAGE_PORT"

    def test_legacy_env_warns_once(self, isolated_global, tmp_path):
        import langstage_core.host.config as config_mod

        config_mod._warned_legacy_env.discard("DEEPAGENT_TITLE")
        with pytest.warns(DeprecationWarning, match="LANGSTAGE_TITLE"):
            HostConfig.resolve(env={"DEEPAGENT_TITLE": "Old"}, toml_start=tmp_path)

    def test_legacy_env_prints_visible_notice(self, isolated_global, tmp_path, monkeypatch, capsys):
        # The DeprecationWarning is swallowed by Python's default filter, so the
        # resolver ALSO prints a one-line notice to stderr for CLI users. Drop
        # the pytest marker env so the notice isn't suppressed.
        import langstage_core.host.config as config_mod

        config_mod._warned_legacy_env.discard("DEEPAGENT_PORT")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", raising=False)
        HostConfig.resolve(env={"DEEPAGENT_PORT": "9000"}, toml_start=tmp_path)
        err = capsys.readouterr().err
        assert "DEEPAGENT_PORT is deprecated" in err
        assert "LANGSTAGE_PORT" in err
        # ASCII-only — must encode on a cp1252 Windows console.
        err.encode("cp1252")

    def test_legacy_env_notice_silent_under_pytest(self, isolated_global, tmp_path, capsys):
        # PYTEST_CURRENT_TEST is set during this test, so no stderr notice fires
        # (keeps test output clean and can't break captured-output assertions in
        # the surface repos' suites).
        import langstage_core.host.config as config_mod

        config_mod._warned_legacy_env.discard("DEEPAGENT_DEBUG")
        HostConfig.resolve(env={"DEEPAGENT_DEBUG": "true"}, toml_start=tmp_path)
        assert "deprecated" not in capsys.readouterr().err

    def test_legacy_env_notice_suppressed_by_env(self, isolated_global, tmp_path, monkeypatch, capsys):
        import warnings

        import langstage_core.host.config as config_mod

        config_mod._warned_legacy_env.discard("DEEPAGENT_HOST")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", "1")
        # SUPPRESS silences BOTH the stderr notice AND the DeprecationWarning,
        # so the "set ... to silence" hint is honest (no warning leaks through).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HostConfig.resolve(env={"DEEPAGENT_HOST": "0.0.0.0"}, toml_start=tmp_path)
        assert "deprecated" not in capsys.readouterr().err
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_langstage_toml_preferred_in_same_dir(self, isolated_global, tmp_path):
        (tmp_path / "deepagents.toml").write_text("[server]\nport = 1000\n")
        (tmp_path / "langstage.toml").write_text("[server]\nport = 2000\n")
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 2000

    def test_legacy_toml_still_read(self, isolated_global, tmp_path):
        (tmp_path / "deepagents.toml").write_text("[server]\nport = 1234\n")
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 1234

    def test_toml_with_utf8_bom_is_read(self, isolated_global, tmp_path):
        # Notepad / PowerShell `Out-File -Encoding utf8` prepend a UTF-8 BOM on
        # Windows; tomllib.load() (binary) chokes on it. The reader must strip it
        # rather than crash. (gh #-dogfood: a BOM'd langstage.toml bricked jupyter
        # at import time.)
        (tmp_path / "langstage.toml").write_bytes(
            b"\xef\xbb\xbf" + b"[server]\nport = 8123\n"
        )
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 8123

    def test_malformed_toml_does_not_crash(self, isolated_global, tmp_path, capsys):
        # gh #42: a broken langstage.toml must NOT crash resolve(). Config resolves
        # at import time on several surfaces, so a raw TOMLDecodeError bricked
        # --version / --help / --demo and even `import langstage_jupyter`. The bad
        # file is skipped (with a visible notice); env + defaults still resolve.
        (tmp_path / "langstage.toml").write_text("this is not = [valid toml\n")
        cfg = HostConfig.resolve(env={"LANGSTAGE_PORT": "7777"}, toml_start=tmp_path)
        assert cfg.port == 7777  # env layer still applies; the bad TOML was ignored
        err = capsys.readouterr().err
        assert "ignoring malformed config" in err and "langstage.toml" in err
        err.encode("cp1252")  # ASCII-only — must not crash a cp1252 console

    def test_malformed_toml_not_listed_as_read_and_warns_once(
        self, isolated_global, tmp_path, monkeypatch, capsys
    ):
        # gh langstage-hermes #61: a malformed file was appended to `sources`
        # (so --show-config printed "TOML read from: <it>", contradicting the
        # "ignoring malformed" note) and the note was emitted twice (the loader plus
        # the source-labeling re-read each warned).
        import langstage_core.host.config as config_mod

        p = tmp_path / "langstage.toml"
        p.write_text("[ok]\nx = 1\n[oops\n")  # line 3: unterminated table header
        config_mod._malformed_toml.discard(str(p))
        config_mod._warned_malformed_toml.discard(str(p))
        monkeypatch.chdir(tmp_path)

        merged, sources = config_mod.load_toml_config(start=tmp_path)
        config_mod._read_toml(p)  # a second read (as --show-config source-labeling does)

        assert p not in sources, "an ignored/malformed file must not be listed as read"
        assert capsys.readouterr().err.count("ignoring malformed config") == 1

    def test_legacy_toml_warns(self, isolated_global, tmp_path):
        # gh #25: legacy DEEPAGENT_* env warns, but a legacy deepagents.toml used to
        # resolve silently. It now raises a DeprecationWarning too.
        import langstage_core.host.config as config_mod

        p = tmp_path / "deepagents.toml"
        p.write_text("[server]\nport = 1234\n")
        config_mod._warned_legacy_toml.discard(str(p))
        with pytest.warns(DeprecationWarning, match="deepagents.toml"):
            HostConfig.resolve(env={}, toml_start=tmp_path)

    def test_legacy_toml_prints_visible_notice(self, isolated_global, tmp_path, monkeypatch, capsys):
        import langstage_core.host.config as config_mod

        p = tmp_path / "deepagents.toml"
        p.write_text("[server]\nport = 1234\n")
        config_mod._warned_legacy_toml.discard(str(p))
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", raising=False)
        HostConfig.resolve(env={}, toml_start=tmp_path)
        err = capsys.readouterr().err
        assert "deepagents.toml" in err and "legacy name" in err and "langstage.toml" in err
        err.encode("cp1252")

    def test_legacy_toml_notice_silent_under_pytest(self, isolated_global, tmp_path, capsys):
        import langstage_core.host.config as config_mod

        p = tmp_path / "deepagents.toml"
        p.write_text("[server]\nport = 1234\n")
        config_mod._warned_legacy_toml.discard(str(p))
        HostConfig.resolve(env={}, toml_start=tmp_path)  # PYTEST_CURRENT_TEST set
        assert "legacy name" not in capsys.readouterr().err

    def test_nearest_toml_wins_across_dirs(self, isolated_global, tmp_path):
        # langstage.toml in the parent must NOT beat deepagents.toml in cwd.
        (tmp_path / "langstage.toml").write_text("[server]\nport = 1000\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deepagents.toml").write_text("[server]\nport = 2000\n")
        cfg = HostConfig.resolve(env={}, toml_start=sub)
        assert cfg.port == 2000

    def test_langstage_config_home_override(self, tmp_path, monkeypatch):
        gdir = tmp_path / "newhome"
        gdir.mkdir()
        (gdir / "config.toml").write_text("[server]\nport = 4321\n")
        monkeypatch.delenv("DEEPAGENTS_CONFIG_HOME", raising=False)
        monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(gdir))
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 4321

    def test_describe_shows_both_vocabularies(self, isolated_global, tmp_path):
        text = HostConfig.resolve(env={}, toml_start=tmp_path).describe()
        assert "LANGSTAGE_AGENT_SPEC" in text
        assert "legacy DEEPAGENT_AGENT_SPEC" in text

    def test_subclass_legacy_declaration_resolves_canonical_name(
        self, isolated_global, tmp_path
    ):
        """A host still declaring DEEPAGENT_* in its _ENV map picks up the
        LANGSTAGE_* var without any subclass change."""
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass
        class OldHost(HostConfig):
            theme: str = "auto"
            _ENV: ClassVar[dict] = {"theme": ("DEEPAGENT_THEME", str)}

        cfg = OldHost.resolve(env={"LANGSTAGE_THEME": "dark"}, toml_start=tmp_path)
        assert cfg.theme == "dark"
        assert cfg.sources["theme"] == "env:LANGSTAGE_THEME"


class TestFromEnvBackCompat:
    def test_from_env_skips_toml(self, isolated_global, tmp_path, monkeypatch):
        _toml(tmp_path, "[server]\nport = 1234\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DEEPAGENT_PORT", raising=False)
        # from_env ignores TOML even though deepagents.toml is in cwd
        assert HostConfig.from_env().port == 8050
