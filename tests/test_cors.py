"""Opt-in CORS for the served AG-UI endpoint (gh #162).

``build_app`` / ``serve`` / ``langstage-agui`` install no CORS by default (same-origin
only, never ``*``). ``cors_origins=`` / ``--cors`` opt in: ``"loopback"`` (bare
``--cors``) allows any localhost origin, a list allows exactly those origins, and ``"*"``
only when passed explicitly.
"""
import pytest

pytest.importorskip("ag_ui_langgraph", reason="needs the 'agui' extra")
pytest.importorskip("fastapi", reason="needs fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import langstage_core.agui as agui_pkg  # noqa: E402
from langstage_core.agui import build_app  # noqa: E402
from langstage_core.agui.__main__ import main  # noqa: E402
from langstage_core.demo.stub import graph as stub_graph  # noqa: E402

_PREFLIGHT = {"Access-Control-Request-Method": "POST",
              "Access-Control-Request-Headers": "content-type"}


def _preflight(app, origin):
    with TestClient(app) as c:
        return c.options("/", headers={"Origin": origin, **_PREFLIGHT})


def test_default_installs_no_cors():
    app = build_app(stub_graph)
    assert not [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    r = _preflight(app, "http://localhost:5173")
    assert "access-control-allow-origin" not in r.headers


def test_explicit_allowlist():
    app = build_app(stub_graph, cors_origins=["http://localhost:5173"])
    r = _preflight(app, "http://localhost:5173")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = _preflight(app, "http://evil.example")
    assert "access-control-allow-origin" not in other.headers


def test_comma_separated_string_and_trailing_slash():
    app = build_app(stub_graph, cors_origins="http://a.test/, http://b.test")
    for origin in ("http://a.test", "http://b.test"):
        assert _preflight(app, origin).headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:3000",
                                    "http://[::1]:8080", "https://localhost"])
def test_loopback_allows_any_local_origin(origin):
    app = build_app(stub_graph, cors_origins="loopback")
    assert _preflight(app, origin).headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["http://evil.example", "http://localhost.evil.example",
                                    "http://127.0.0.1.evil.example:80"])
def test_loopback_refuses_non_local_origins(origin):
    app = build_app(stub_graph, cors_origins="loopback")
    assert "access-control-allow-origin" not in _preflight(app, origin).headers


def test_star_only_when_explicit():
    app = build_app(stub_graph, cors_origins=["*"])
    assert _preflight(app, "http://anything.example").headers["access-control-allow-origin"] == "*"


def test_actual_post_carries_the_header():
    app = build_app(stub_graph, cors_origins=["http://localhost:5173"])
    body = {"threadId": "t1", "runId": "r1", "state": {}, "tools": [], "context": [],
            "forwardedProps": {}, "messages": [{"id": "m1", "role": "user", "content": "hi"}]}
    with TestClient(app) as c:
        r = c.post("/", json=body, headers={"Origin": "http://localhost:5173",
                                             "Accept": "text/event-stream"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


class _FakeSock:
    def close(self):
        pass


@pytest.mark.parametrize("argv, expected", [
    ([], None),
    (["--cors"], "loopback"),
    (["--cors", "http://localhost:5173,http://localhost:3000"],
     "http://localhost:5173,http://localhost:3000"),
])
def test_cli_forwards_cors(monkeypatch, argv, expected):
    seen: dict = {}
    monkeypatch.setattr(agui_pkg, "serve", lambda graph, **kw: seen.update(kw))
    monkeypatch.setattr(agui_pkg, "_bind_socket", lambda host, port: _FakeSock())
    assert main(["--demo", *argv]) == 0
    assert seen["cors_origins"] == expected
