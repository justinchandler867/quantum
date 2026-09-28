"""
PORTFOLIO_RISK_SPEC endpoints on a seeded, network-free returns store:
/api/portfolio/health, /api/portfolio/preview (§2 agreement check),
/api/discovery/redundancy (§5, config threshold).
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import HIGH_CORR_THRESHOLD
from app.portfolio_risk import portfolio_health
from app.portfolio_series import regress_beta, weighted_portfolio_returns


@pytest.fixture
def seeded(monkeypatch):
    rng = np.random.default_rng(31)
    n = 320
    idx = pd.bdate_range("2024-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    base = rng.normal(0, 0.008, n)
    R = pd.DataFrame({
        "NVDA": 1.8 * m + rng.normal(0, 0.02, n), "MSFT": m + rng.normal(0, 0.008, n),
        "JNJ": 0.4 * m + rng.normal(0, 0.007, n), "TLT": base, "IEF": base + rng.normal(0, 0.0015, n),
        "SPY": m,
    }, index=idx)
    monkeypatch.setitem(main._store, "returns", R)
    monkeypatch.setattr(main, "_ensure_data", lambda tickers: None)
    monkeypatch.setattr(main, "_j2_yield", lambda t: {"NVDA": 0.0003, "MSFT": 0.007, "JNJ": 0.03}.get(t))
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    return R


def H(d):
    return [{"ticker": t, "weight": w} for t, w in d.items()]


def test_health_endpoint_matches_engine(seeded):
    hold = {"NVDA": 34, "MSFT": 33, "JNJ": 33}
    r = TestClient(main.app).post("/api/portfolio/health", json={"holdings": H(hold)})
    assert r.status_code == 200
    d = r.json()
    eng = portfolio_health(seeded, hold, rf=main.RISK_FREE_RATE)
    assert d["vol"] == pytest.approx(eng["vol"], rel=1e-12)
    assert d["rc_sum"] == pytest.approx(d["vol"], rel=1e-12)
    assert d["yield"]["yield"] == pytest.approx((34 * 0.0003 + 33 * 0.007 + 33 * 0.03) / 100)
    assert TestClient(main.app).post("/api/portfolio/health", json={"holdings": []}).json() == {"status": "empty"}


@pytest.mark.parametrize("cur,prop", [
    ({"NVDA": 50, "MSFT": 50}, {"NVDA": 34, "MSFT": 33, "TLT": 33}),       # add (equal re-split)
    ({"NVDA": 34, "MSFT": 33, "TLT": 33}, {"NVDA": 50, "MSFT": 50}),       # remove
    ({"NVDA": 20, "MSFT": 40, "TLT": 40}, {"NVDA": 35, "MSFT": 40, "TLT": 40}),  # reweight
])
def test_preview_agrees_with_engine_on_post_change_weights(seeded, cur, prop):
    d = TestClient(main.app).post("/api/portfolio/preview", json={"holdings": H(cur), "proposed": H(prop)}).json()
    assert d["status"] == "ok"
    # no shortcut: previewed sigma == engine sigma computed directly on each weight set
    assert d["proposed"]["vol"] == pytest.approx(portfolio_health(seeded, prop)["vol"], rel=1e-12)
    assert d["current"]["vol"] == pytest.approx(portfolio_health(seeded, cur)["vol"], rel=1e-12)
    # line two: beta exactly as /api/portfolio/beta computes it
    chip = TestClient(main.app).post("/api/portfolio/beta", json={"holdings": H(prop)}).json()["beta"]
    assert d["proposed"]["beta"] == chip
    b, _, _ = regress_beta(weighted_portfolio_returns(seeded, prop), seeded["SPY"], 252, min_overlap=60)
    assert chip == b


def test_preview_reports_unloaded_ticker_without_fetching(seeded, monkeypatch):
    def boom(_):
        raise AssertionError("preview must not trigger a data fetch")
    monkeypatch.setattr(main, "_ensure_data", boom)
    d = TestClient(main.app).post("/api/portfolio/preview", json={
        "holdings": H({"NVDA": 100}), "proposed": H({"NVDA": 50, "NEWCO": 50})}).json()
    assert d["proposed"]["excluded"] == ["NEWCO"]
    assert d["proposed"]["vol"] == pytest.approx(d["current"]["vol"])   # NEWCO excluded, not zero-filled


def test_redundancy_endpoint_uses_config_threshold(seeded):
    d = TestClient(main.app).post("/api/discovery/redundancy",
                                  json={"candidates": ["NVDA", "MSFT", "JNJ", "TLT", "IEF"]}).json()
    assert d["threshold"] == HIGH_CORR_THRESHOLD
    flagged = {f["ticker"]: f for f in d["flags"]}
    assert set(flagged) == {"TLT", "IEF"} and flagged["TLT"]["peer"] == "IEF"
    assert flagged["TLT"]["r"] >= HIGH_CORR_THRESHOLD
