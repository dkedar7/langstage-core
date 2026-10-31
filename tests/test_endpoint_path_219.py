"""An endpoint path without a leading slash is normalised, never served unreachable (gh #219)."""

import socket

import pytest

pytest.importorskip("ag_ui")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import langstage_core.agui as agui  # noqa: E402
from langstage_core import load_agent_spec  # noqa: E402
from langstage_core.agui import build_app, normalize_endpoint_path  # noqa: E402
from langstage_core.agui.__main__ import main  # noqa: E402
from langstage_core.cli import EXIT_USAGE  # noqa: E402

BODY = {
    "threadId": "t", "runId": "r", "state": {}, "tools": [], "context": [], "forwardedProps": {},
    "messages": [{"id": "m", "role": "user", "content": "hi"}],
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("agui", "/agui"),
        ("/agui", "/agui"),
        ("/agui/", "/agui"),
        ("//agui", "/agui"),
        (" /agui ", "/agui"),
        ("api/agui", "/api/agui"),
        ("/api//agui/", "/api/agui"),
        ("", "/"),
        ("/", "/"),
        ("//", "/"),
        (None, "/"),
    ],
)
def test_normalize(raw, expected):
    assert normalize_endpoint_path(raw) == expected


@pytest.mark.parametrize("bad", ["a b", "/agui?x=1", "/agui#x", "/{thread}", "/../agui", "/./agui"])
def test_unroutable_paths_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid endpoint path"):
        normalize_endpoint_path(bad)


def test_build_app_without_leading_slash_is_reachable():
    app = build_app(load_agent_spec("langstage_core.demo.stub:graph"), path="nope")
    client = TestClient(app)
    assert client.get("/nope").status_code == 200
    assert client.post("/nope", json=BODY).status_code == 200


def test_build_app_rejects_unroutable_path():
    with pytest.raises(ValueError):
        build_app(load_agent_spec("langstage_core.demo.stub:graph"), path="a b")


def _capture_serve(monkeypatch):
    seen = {}

    def fake_serve(graph, **kw):
        seen.update(kw)

    monkeypatch.setattr(agui, "serve", fake_serve)
    return seen


def test_cli_banner_and_route_agree(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    seen = _capture_serve(monkeypatch)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert main(["--demo", "--host", "127.0.0.1", "--path", "nope", "--port", str(port)]) == 0
    out = capsys.readouterr().out
    assert seen["path"] == "/nope"
    assert f"http://127.0.0.1:{port}/nope" in out


def test_cli_rejects_unroutable_path(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    seen = _capture_serve(monkeypatch)
    assert main(["--demo", "--path", "a b"]) == EXIT_USAGE
    assert "--path" in capsys.readouterr().err
    assert not seen
