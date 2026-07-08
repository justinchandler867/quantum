#!/usr/bin/env python3
"""
AUDIT SCRIPT — MULTI_HORIZON_SPEC.md §4, sealed cases M3 and M5.

M3 and M5 require LIVE data (a real ticker's real adjusted closes, and a real
end-to-end screening run against yfinance) and were therefore moved out of
the offline pytest suite (backend/tests/test_multi_horizon.py), which must
run with no network access. M1, M2, and M4 are pure synthetic-DataFrame unit
tests and stay in pytest.

  M3 — Log-conversion check (F1): for one real ticker, the displayed
       return_1y (post-fix, via run_screening_pipeline) must equal
       P_t/P_(t-252) - 1 computed directly from raw adjusted closes, within
       +/-0.5% (tolerance covers dividend-adjustment timing) — NOT the raw
       sum-of-log-returns value.

  M5 — pct_off_52wk_high must be <= 0 for every asset in a real screen run
       (a positive value is impossible by construction; any positive value
       found here is an engine bug).

Usage:
    python3 scripts/audit_multi_horizon.py
    python3 scripts/audit_multi_horizon.py --ticker AAPL --n 40

Requires network access (yfinance) and the project's normal Python
environment (pandas, numpy, yfinance, plus the app package importable —
run from the backend/ directory or with PYTHONPATH=. set).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.data_ingest import fetch_prices, compute_log_returns  # noqa: E402
from app.screener import run_screening_pipeline  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_m3(ticker: str) -> None:
    hr(f"M3 — log-conversion check (F1) for real ticker {ticker}")
    prices = fetch_prices([ticker], include_hedges=False)
    if ticker not in prices.columns:
        print(f"SKIPPED: {ticker} not returned by fetch_prices (no data / delisted?)")
        return
    px = prices[ticker].dropna()
    if len(px) <= 252:
        print(f"SKIPPED: only {len(px)} days of history for {ticker}, need > 252")
        return

    true_simple_1y = float(px.iloc[-1] / px.iloc[-1 - 252] - 1)

    returns = compute_log_returns(prices)
    result = run_screening_pipeline(returns=returns, tickers=[ticker], prices=prices, max_results=5)
    matches = [a for a in result.shortlist if a.ticker == ticker]
    if not matches:
        print(f"SKIPPED: {ticker} did not survive the screening pipeline's hard "
              f"gates (cap/volume/price) — cannot compare a displayed return_1y.")
        print(f"  (Reference true simple 1y return from raw prices: {true_simple_1y*100:.4f}%)")
        return
    asset = matches[0]
    displayed = asset.return_1y  # already a percent, post F1-fix

    # Also compute the OLD (buggy) log-sum-only display for comparison
    trailing = returns[ticker].iloc[-252:] if len(returns) > 252 else returns[ticker]
    sigma_log = float(trailing.sum())
    old_buggy_display = sigma_log * 100

    print(f"  True simple return (raw px, P_t/P_t-252 - 1): {true_simple_1y*100:.4f}%")
    print(f"  Displayed return_1y (post-F1-fix, from pipeline): {displayed:.4f}%")
    print(f"  OLD buggy display (sum(log)*100, pre-fix):        {old_buggy_display:.4f}%")
    diff = abs(displayed - true_simple_1y * 100)
    print(f"  |displayed - true_simple| = {diff:.4f} pct points (tolerance: 0.5)")
    print(f"  PASS: {diff <= 0.5}")


def run_m5(n: int) -> None:
    hr(f"M5 — pct_off_52wk_high <= 0 for every asset in a real screen run (top {n})")
    from app.screener import load_nasdaq_tickers

    tickers = load_nasdaq_tickers()
    prices, _ = fetch_prices(tickers, include_hedges=True, return_volumes=True)
    returns = compute_log_returns(prices)
    result = run_screening_pipeline(returns=returns, tickers=tickers, prices=prices, max_results=n)

    violations = [
        (a.ticker, a.pct_off_52wk_high)
        for a in result.shortlist
        if a.pct_off_52wk_high is not None and a.pct_off_52wk_high > 0
    ]
    print(f"  Screened {len(result.shortlist)} assets.")
    if violations:
        print(f"  FAIL: {len(violations)} asset(s) with pct_off_52wk_high > 0 (engine bug):")
        for t, v in violations:
            print(f"    {t}: {v}")
    else:
        print("  PASS: every asset has pct_off_52wk_high <= 0 (or None/insufficient data).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL", help="Real ticker for M3 (default: AAPL)")
    parser.add_argument("--n", type=int, default=40, help="max_results for the M5 screen run")
    args = parser.parse_args()

    run_m3(args.ticker)
    run_m5(args.n)
