"""Demo deployment gate: Basic Auth middleware, noindex header, DEMO_MODE flag."""

import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app

PW = "s3cret-demo"


def _basic(user, pw):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", PW)


# ── Gate active ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path", [
    ("GET", "/"),
    ("GET", "/openapi.json"),
    ("GET", "/api/cfa/tags"),
    ("POST", "/api/paper/trade"),
    ("GET", "/docs"),
    ("GET", "/no-such-route"),
])
def test_401_without_credentials(client, gated, method, path):
    r = client.request(method, path)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Basic ")
    assert r.headers["X-Robots-Tag"] == "noindex, nofollow"


@pytest.mark.parametrize("headers", [
    _basic("demo", "wrong"),
    _basic("admin", PW),
    _basic("demo", PW + "x"),
    {"Authorization": "Bearer " + PW},
    {"Authorization": "Basic not-base64!!"},
    {"Authorization": "Basic " + base64.b64encode(b"demo").decode()},  # no colon
])
def test_401_with_bad_credentials(client, gated, headers):
    r = client.get("/api/cfa/tags", headers=headers)
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_200_with_credentials_frontend(client, gated):
    r = client.get("/", headers=_basic("demo", PW))
    assert r.status_code == 200
    assert "QUANTEX" in r.text
    assert r.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_200_with_credentials_api(client, gated):
    r = client.get("/api/cfa/tags", headers=_basic("demo", PW))
    assert r.status_code == 200
    assert r.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_password_with_colon(client, monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "a:b:c")
    assert client.get("/api/cfa/tags", headers=_basic("demo", "a:b:c")).status_code == 200
    assert client.get("/api/cfa/tags", headers=_basic("demo", "a")).status_code == 401


# ── /health exempt (Render health check) ─────────────────────────────────────

def test_health_200_without_credentials_when_gated(client, gated):
    r = client.get("/health")
    assert r.status_code == 200
    assert "WWW-Authenticate" not in r.headers


def test_health_carries_noindex_when_gated(client, gated):
    assert client.get("/health").headers["X-Robots-Tag"] == "noindex, nofollow"


# ── Gate inactive ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, ""])
def test_inactive_when_password_unset_or_empty(client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("DEMO_PASSWORD", value)
    for path in ("/", "/health", "/api/cfa/tags"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_inactive_logs_startup_line(monkeypatch, caplog):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    with caplog.at_level("INFO"), TestClient(app):
        pass
    assert sum("Demo auth gate INACTIVE" in m for m in caplog.messages) == 1


def test_active_no_inactive_log(monkeypatch, caplog):
    monkeypatch.setenv("DEMO_PASSWORD", PW)
    with caplog.at_level("INFO"), TestClient(app):
        pass
    assert not any("Demo auth gate INACTIVE" in m for m in caplog.messages)


# ── DEMO_MODE flag ───────────────────────────────────────────────────────────

FLAG = "window.QX_DEMO_MODE=true"


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_demo_mode_injects_flag(client, monkeypatch, value):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    monkeypatch.setenv("DEMO_MODE", value)
    html = client.get("/").text
    assert html.count(FLAG) == 1
    assert html.index(FLAG) < html.index("</head>")


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_demo_mode_off_serves_file_unchanged(client, monkeypatch, value):
    from app.main import STATIC_DIR
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    if value is None:
        monkeypatch.delenv("DEMO_MODE", raising=False)
    else:
        monkeypatch.setenv("DEMO_MODE", value)
    html = client.get("/").text
    assert FLAG not in html
    assert html == (STATIC_DIR / "quantex.html").read_text()


def test_frontend_gates_trade_desk_on_flag():
    """Both the tab entry and the panel render are conditioned on the flag."""
    from app.main import STATIC_DIR
    src = (STATIC_DIR / "quantex.html").read_text()
    assert src.count("// DEMO GATE — Trade Desk hidden pending TRADE_DESK_SPEC design decision") == 2
    assert '...(window.QX_DEMO_MODE?[]:[{id:"trades",l:"TRADE DESK",lk:locked}])' in src
    assert '!window.QX_DEMO_MODE&&tab==="trades"&&e(TradeDeck' in src
