"""
Regression test for the stale-store silent ticker drop in /api/screen.

Before the fix, the screen endpoint only called _ensure_data when the store
was completely empty. A second request for a *different* basket reused the
first request's stale return columns, and compute_factor_scores silently
dropped every ticker missing from them. This test seeds the store with
basket A, then screens basket B, and asserts B's tickers survive to results.

The data layer is mocked (no network): fetch_prices returns deterministic
synthetic price walks for any requested tickers (including the QQQ
sub-fetch inside identify_stress_windows), and fetch_fundamentals returns
large-cap rows that clear the hard gates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.screener as screener
from app.config import REFERENCE_HEDGES


def _fake_prices(tickers, include_hedges=True, return_volumes=False):
    cols = list(dict.fromkeys(list(tickers) + (REFERENCE_HEDGES if include_hedges else [])))
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(len(cols))  # deterministic, varies by request shape
    steps = rng.normal(0.0003, 0.012, size=(len(idx), len(cols)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    if return_volumes:
        return prices, pd.DataFrame(2_000_000, index=idx, columns=cols)
    return prices


def _fake_fundamentals(tickers):
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t, "name": t, "sector": "Technology", "industry": "Software",
            "market_cap": 50_000_000_000, "price": 100.0 + i, "avg_volume": 5_000_000,
            "dividend_yield": 1.5, "pe_ratio": 25.0, "forward_pe": 22.0, "pb_ratio": 5.0,
            "earnings_yield": 0.04, "earnings_date": None, "dividend_growth_5y": 5.0,
            "debt_to_equity": 1.0, "revenue_growth": 0.1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "fetch_prices", _fake_prices)
    # identify_stress_windows calls fetch_prices from the data_ingest namespace
    monkeypatch.setattr("app.data_ingest.fetch_prices", _fake_prices)
    monkeypatch.setattr(screener, "fetch_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(main, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(main, "cache_set", lambda *a, **k: None)
    for k in list(main._store.keys()):
        main._store[k] = None
    return TestClient(main.app)


BASKET_A = ["AAA", "BBB", "CCC", "DDD", "EEE"]
BASKET_B = ["FFF", "GGG", "HHH", "III", "JJJ"]


def test_second_basket_tickers_survive_after_first_basket_seeded(client):
    # Seed the store with basket A.
    r1 = client.post("/api/screen", json={
        "goal": "Balanced", "risk_score": 50, "time_horizon_years": 10,
        "max_results": 40, "tickers": BASKET_A,
    })
    assert r1.status_code == 200
    a_tickers = {x["ticker"] for x in r1.json()["shortlist"]}
    assert a_tickers == set(BASKET_A)

    # Now request basket B — none of these were in A's return columns.
    r2 = client.post("/api/screen", json={
        "goal": "Balanced", "risk_score": 50, "time_horizon_years": 10,
        "max_results": 40, "tickers": BASKET_B,
    })
    assert r2.status_code == 200
    b_tickers = {x["ticker"] for x in r2.json()["shortlist"]}
    # Pre-fix, these would be silently dropped (stale store had only A + hedges).
    assert b_tickers == set(BASKET_B), f"missing from results: {set(BASKET_B) - b_tickers}"
