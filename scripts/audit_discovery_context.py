#!/usr/bin/env python3
"""
AUDIT SCRIPT — CORRELATION_COLUMN_SPEC.md §C, sealed cases S1 and S2.

S1 and S2 require LIVE data (real SPY/VOO/TLT price history) and were
therefore moved out of the offline pytest suite
(backend/tests/test_discovery_context.py), which must run with no network
access. S3-S7 are pure synthetic-DataFrame unit tests and stay in pytest.

  S1 — Portfolio 100% SPY, candidate VOO -> corr_normal >= 0.99,
       band Near-duplicate.
  S2 — Portfolio 100% SPY, candidate TLT -> band Diversifier or Hedge;
       corr_stress differs from corr_normal by a visible margin (sign not
       sealed by the spec — this script reads the sign from data and
       documents it, per the spec's own instruction: "sign not sealed,
       read from data, then document").

Usage:
    python3 scripts/audit_discovery_context.py

Requires network access (yfinance) and the project's normal Python
environment (pandas, numpy, yfinance, plus the app package importable —
run from the backend/ directory or with PYTHONPATH=. set).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_ingest import (  # noqa: E402
    fetch_prices,
    compute_log_returns,
    identify_stress_windows,
)
from app.discovery_context import compute_candidate_context, band_label  # noqa: E402
from app.config import HIGH_CORR_THRESHOLD  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _load_returns():
    tickers = ["SPY", "VOO", "TLT"]
    prices = fetch_prices(tickers, include_hedges=False)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"SKIPPED: missing price data for {missing} — cannot run S1/S2.")
        return None, None
    returns = compute_log_returns(prices)
    stress_mask = identify_stress_windows(returns, benchmark="SPY" if "QQQ" not in returns.columns else "QQQ")
    return returns, stress_mask


def run_s1(returns, stress_mask) -> None:
    hr("S1 — Portfolio 100% SPY, candidate VOO")
    ctx = compute_candidate_context(
        ticker="VOO",
        returns=returns,
        stress_mask=stress_mask,
        holdings={"SPY": 1.0},
        window=252,
    )
    print(f"  status:      {ctx.status}")
    print(f"  corr_normal: {ctx.corr_normal}")
    print(f"  corr_stress: {ctx.corr_stress}")
    print(f"  band:        {ctx.band}")
    print(f"  days_normal: {ctx.days_normal}  days_stress: {ctx.days_stress}")

    ok_corr = ctx.corr_normal is not None and ctx.corr_normal >= 0.99
    ok_band = ctx.band == "Near-duplicate"
    print(f"  PASS corr_normal >= 0.99: {ok_corr}")
    print(f"  PASS band == Near-duplicate: {ok_band}")
    if not (ok_corr and ok_band):
        print("  *** SEALED CASE S1 DIVERGENCE — treat as a finding, do not silently pass. ***")


def run_s2(returns, stress_mask) -> None:
    hr("S2 — Portfolio 100% SPY, candidate TLT")
    ctx = compute_candidate_context(
        ticker="TLT",
        returns=returns,
        stress_mask=stress_mask,
        holdings={"SPY": 1.0},
        window=252,
    )
    print(f"  status:      {ctx.status}")
    print(f"  corr_normal: {ctx.corr_normal}")
    print(f"  corr_stress: {ctx.corr_stress}")
    print(f"  band:        {ctx.band}")
    print(f"  days_normal: {ctx.days_normal}  days_stress: {ctx.days_stress}")

    ok_band = ctx.band in ("Diversifier", "Hedge")
    print(f"  PASS band in (Diversifier, Hedge): {ok_band}")

    if ctx.corr_stress is not None and ctx.corr_normal is not None:
        delta = ctx.corr_stress - ctx.corr_normal
        sign = "positive (stress correlation HIGHER than normal — flip risk)" if delta > 0 \
            else "negative (stress correlation LOWER than normal — hedge strengthens in stress)"
        print(f"  corr_stress - corr_normal = {delta:.4f} -> sign is {sign}")
        print(f"  visible margin (|delta| > 0.05): {abs(delta) > 0.05}")
        print(f"  flip_flag (delta >= 0.20): {ctx.flip_flag}")
    else:
        print("  corr_stress unavailable (no_stress_window) — cannot assess margin/sign.")

    if not ok_band:
        print("  *** SEALED CASE S2 DIVERGENCE — treat as a finding, do not silently pass. ***")


if __name__ == "__main__":
    returns, stress_mask = _load_returns()
    if returns is None:
        sys.exit(1)
    run_s1(returns, stress_mask)
    run_s2(returns, stress_mask)
