"""
End-to-end test that ETFs carry asset_type="ETF" from yfinance extraction
through the screen response, and that their stored size is net assets
(totalAssets), never $0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
import app.screener as screener
from app.config import REFERENCE_HEDGES


# ── Part 1: extraction — quoteType -> asset_type, totalAssets -> market_cap ──

class _FakeTicker:
    def __init__(self, info):
        self.info = info
        # no `.dividends` attribute -> the div-growth block hits its except


class _FakeTickers:
    def __init__(self, mapping):
        self.tickers = mapping


_INFOS = {
    "SPY": {"regularMarketPrice": 600.0, "quoteType": "ETF", "marketCap": None,
            "totalAssets": 780_000_000_000, "averageVolume": 5_000_000,
            "shortName": "SPDR S&P 500 ETF"},
    "AAPL": {"regularMarketPrice": 225.0, "quoteType": "EQUITY", "marketCap": 3_000_000_000_000,
             "totalAssets": None, "averageVolume": 5_000_000, "shortName": "Apple Inc."},
}


def test_fetch_fundamentals_maps_quotetype_and_size(monkeypatch):
    monkeypatch.setattr(screener.yf, "Tickers",
                        lambda spc: _FakeTickers({s: _FakeTicker(_INFOS[s]) for s in spc.split()}))
    df = screener.fetch_fundamentals(["SPY", "AAPL"]).set_index("ticker")
    assert df.loc["SPY", "asset_type"] == "ETF"
    assert df.loc["AAPL", "asset_type"] == "Stock"
    # ETF size falls back to net assets, never 0.
    assert df.loc["SPY", "market_cap"] == 780_000_000_000
    assert df.loc["AAPL", "market_cap"] == 3_000_000_000_000


# ── Part 2: wiring — asset_type survives to the /api/screen response ─────────

def _fake_prices(tickers, include_hedges=True, return_volumes=False):
    cols = list(dict.fromkeys(list(tickers) + (REFERENCE_HEDGES if include_hedges else [])))
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(len(cols))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (len(idx), len(cols))), axis=0)),
                          index=idx, columns=cols)
    if return_volumes:
        return prices, pd.DataFrame(2_000_000, index=idx, columns=cols)
    return prices


def _fake_fundamentals(tickers):
    rows = []
    for i, t in enumerate(tickers):
        is_etf = t in ("SPY", "BND", "TLT", "GLD")
        rows.append({
            "ticker": t, "name": t, "sector": "Unknown" if is_etf else "Technology",
            "industry": "ETF" if is_etf else "Software",
            "asset_type": "ETF" if is_etf else "Stock",
            "market_cap": 400_000_000_000 if is_etf else 50_000_000_000,
            "total_assets": 400_000_000_000 if is_etf else 0,
            "price": 100.0 + i, "avg_volume": 5_000_000,
            "dividend_yield": 2.0, "pe_ratio": None if is_etf else 25.0,
            "forward_pe": None, "pb_ratio": None, "earnings_yield": None if is_etf else 0.04,
            "earnings_date": None, "dividend_growth_5y": None,
            "debt_to_equity": None, "revenue_growth": None,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "fetch_prices", _fake_prices)
    monkeypatch.setattr("app.data_ingest.fetch_prices", _fake_prices)
    monkeypatch.setattr(screener, "fetch_fundamentals", _fake_fundamentals)
    monkeypatch.setattr(main, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(main, "cache_set", lambda *a, **k: None)
    for k in list(main._store.keys()):
        main._store[k] = None
    return TestClient(main.app)


def test_screen_response_carries_asset_type(client):
    r = client.post("/api/screen", json={
        "goal": "Balanced", "risk_score": 50, "time_horizon_years": 10,
        "max_results": 40, "tickers": ["SPY", "BND", "AAPL", "MSFT", "JPM"],
    })
    assert r.status_code == 200
    by_ticker = {a["ticker"]: a for a in r.json()["shortlist"]}
    assert by_ticker["SPY"]["asset_type"] == "ETF"
    assert by_ticker["BND"]["asset_type"] == "ETF"
    assert by_ticker["AAPL"]["asset_type"] == "Stock"
    # ETF size surfaced as net assets, never $0.
    assert by_ticker["SPY"]["market_cap"] == 400_000_000_000
