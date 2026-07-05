"""
Regression test for the ETF market-cap gate fallback.

yfinance reports marketCap=None/0 for ETFs, so before the fix no ETF could
clear the $2B size gate. apply_hard_gates now falls back to totalAssets
(net assets) when market_cap is 0, so broad-market/bond/commodity ETFs are
eligible on the same dollar-size basis.
"""
from __future__ import annotations

import pandas as pd

from app.screener import apply_hard_gates
from app.config import SCREEN_MIN_MARKET_CAP, SCREEN_MIN_AVG_VOLUME, SCREEN_MIN_PRICE


def _row(ticker, market_cap, total_assets, price=100.0, volume=5_000_000):
    return {
        "ticker": ticker, "market_cap": market_cap, "total_assets": total_assets,
        "avg_volume": volume, "price": price,
    }


def test_etf_with_zero_market_cap_passes_via_total_assets():
    df = pd.DataFrame([
        _row("SPY", 0, 780_000_000_000),   # ETF: no mcap, huge net assets
        _row("TLT", 0, 43_000_000_000),
        _row("AAPL", 3_000_000_000_000, 0),  # stock: normal mcap
    ])
    passed = set(apply_hard_gates(df)["ticker"])
    assert passed == {"SPY", "TLT", "AAPL"}


def test_tiny_etf_still_gated():
    df = pd.DataFrame([
        _row("SMALLETF", 0, 500_000_000),  # net assets below $2B
        _row("SPY", 0, 780_000_000_000),
    ])
    passed = set(apply_hard_gates(df)["ticker"])
    assert passed == {"SPY"}
    assert "SMALLETF" not in passed


def test_stock_gating_unchanged():
    # A stock below the mcap threshold is still dropped (fallback doesn't help it).
    df = pd.DataFrame([
        _row("BIGCO", 10_000_000_000, 0),
        _row("SMALLCO", 1_000_000_000, 0),  # $1B < $2B
    ])
    passed = set(apply_hard_gates(df)["ticker"])
    assert passed == {"BIGCO"}


def test_missing_total_assets_column_does_not_crash():
    # Backward compat: fundamentals without the new column still gate on mcap.
    df = pd.DataFrame([
        {"ticker": "AAPL", "market_cap": 3_000_000_000_000, "avg_volume": 5_000_000, "price": 100.0},
        {"ticker": "SPY", "market_cap": 0, "avg_volume": 5_000_000, "price": 100.0},
    ])
    passed = set(apply_hard_gates(df)["ticker"])
    assert passed == {"AAPL"}  # SPY has no total_assets to fall back to
