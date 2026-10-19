"""gh #211: config diagnostics never print a secret's value.
gh #210: an out-of-range --port on the command line is a usage error, not a silent fallback."""
from dataclasses import dataclass
from typing import ClassVar

import pytest

from langstage_core.agui.__main__ import main
from langstage_core.cli import EXIT_USAGE
from langstage_core.host import HostConfig


@dataclass
class _WithSecrets(HostConfig):
    auth_password: str = ""
    api_token: str = ""
    model_max_tokens: int = 100        # named like a credential, but isn't one
    signing_key: str = ""              # a credential not named like one: listed explicitly

    _ENV: ClassVar[dict] = {"auth_password": ("LANGSTAGE_AUTH_PASSWORD", str)}
    _TOML: ClassVar[dict] = {"auth_password": "auth.password"}
    _SECRETS: ClassVar[tuple[str, ...]] = ("signing_key",)


def _cfg(monkeypatch, tmp_path, **over):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSTAGE_AUTH_PASSWORD", "envsecret-123")
    return _WithSecrets.resolve(overrides=over, use_toml=False)


def test_describe_masks_secrets_but_keeps_the_source(monkeypatch, tmp_path):
    text = _cfg(monkeypatch, tmp_path, api_token="tok-456", signing_key="key-789").describe()
    for secret in ("envsecret-123", "tok-456", "key-789"):
        assert secret not in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("auth_password"))
    assert "******** (set)" in line and "[env:LANGSTAGE_AUTH_PASSWORD]" in line
    assert "100" in next(ln for ln in text.splitlines() if "model_max_tokens" in ln)


def test_config_dict_masks_secrets_and_says_whether_set(monkeypatch, tmp_path):
    cd = _cfg(monkeypatch, tmp_path, signing_key="key-789").config_dict()["config"]
    assert cd["auth_password"]["value"] == "********" and cd["auth_password"]["set"] is True
    assert cd["auth_password"]["source"] == "env:LANGSTAGE_AUTH_PASSWORD"
    assert cd["signing_key"]["value"] == "********" and cd["signing_key"]["secret"] is True
    assert cd["api_token"] == {**cd["api_token"], "value": "", "set": False, "secret": True}
    assert cd["model_max_tokens"]["value"] == 100 and "secret" not in cd["model_max_tokens"]
    assert "key-789" not in repr(cd) and "envsecret-123" not in repr(cd)


def test_toml_secret_is_masked(monkeypatch, tmp_path):
    (tmp_path / "langstage.toml").write_text('[auth]\npassword = "tomlsecret"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGSTAGE_AUTH_PASSWORD", raising=False)
    cfg = _WithSecrets.resolve()
    assert cfg.auth_password == "tomlsecret"  # the value itself is untouched
    assert "tomlsecret" not in cfg.describe() and "tomlsecret" not in repr(cfg.config_dict())


@pytest.mark.parametrize("port", ["70000", "0", "-1", "abc"])
def test_bad_port_flag_is_a_usage_error(port, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "--port", port, "--show-config"])
    assert exc.value.code == EXIT_USAGE
    assert "argument --port" in capsys.readouterr().err


def test_bad_port_from_env_still_falls_back(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSTAGE_PORT", "70000")
    assert main(["--demo", "--show-config"]) == 0
    assert "8050" in capsys.readouterr().out
