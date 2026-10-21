"""gh #213: credential-named [configurable] keys are masked in config reports, not in the graph."""

import json
import subprocess
import sys

from langstage_core.host.config import (
    SECRET_MASK,
    HostConfig,
    is_secret_name,
    mask_secrets,
)

FAKE = "NOT-A-REAL-KEY-213"
CONF = {
    "api_key": FAKE,
    "OPENAI_API_KEY": FAKE,
    "db_password": FAKE,
    "model": "openai/gpt-4o-mini",
    "empty_token": "",
    "nested": {"service_secret": FAKE, "region": "us-east"},
    "tools": [{"auth_token": FAKE, "name": "search"}],
}


def test_secret_names():
    for name in (
        "api_key",
        "openai_api_key",
        "OPENAI_API_KEY",
        "db_password",
        "password",
        "x_token",
    ):
        assert is_secret_name(name), name
    for name in ("model", "token_limit", "keyboard", "passwords_file", 5):
        assert not is_secret_name(name), name


def test_mask_secrets_is_a_copy_and_recurses():
    masked = mask_secrets(CONF)
    assert FAKE not in json.dumps(masked)
    assert masked["api_key"] == SECRET_MASK and masked["OPENAI_API_KEY"] == SECRET_MASK
    assert masked["model"] == "openai/gpt-4o-mini"
    assert masked["empty_token"] == ""  # unset stays visibly unset
    assert masked["nested"] == {"service_secret": SECRET_MASK, "region": "us-east"}
    assert masked["tools"] == [{"auth_token": SECRET_MASK, "name": "search"}]
    assert (
        CONF["api_key"] == FAKE and CONF["nested"]["service_secret"] == FAKE
    )  # untouched


def test_describe_and_config_dict_mask_configurable():
    cfg = HostConfig.resolve(use_toml=False)
    text = cfg.describe(configurable=CONF)
    assert FAKE not in text
    assert f"api_key: {SECRET_MASK} (set)" in text
    assert "model: openai/gpt-4o-mini" in text
    out = cfg.config_dict(configurable=CONF)
    assert FAKE not in json.dumps(out, default=str)
    assert out["configurable"]["api_key"] == SECRET_MASK
    assert out["configurable"]["model"] == "openai/gpt-4o-mini"


def test_configurable_forwarded_to_graph_is_unmasked(tmp_path, monkeypatch):
    (tmp_path / "langstage.toml").write_text(
        f'[configurable]\napi_key = "{FAKE}"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    cfg = HostConfig.resolve()
    assert cfg.configurable()["api_key"] == FAKE
    assert FAKE not in cfg.describe(configurable=cfg.configurable())


def test_show_config_commands_never_print_it(tmp_path):
    (tmp_path / "langstage.toml").write_text(
        f'[configurable]\napi_key = "{FAKE}"\nmodel = "openai/gpt-4o-mini"\n',
        encoding="utf-8",
    )
    for args in (
        ["-m", "langstage_core.agui", "--show-config"],
        ["-m", "langstage_core.agui", "--show-config", "--json"],
        ["-m", "langstage_core.host"],
    ):
        r = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        assert FAKE not in r.stdout + r.stderr, args
        assert "api_key" in r.stdout, (args, r.stdout[-500:], r.stderr[-500:])
