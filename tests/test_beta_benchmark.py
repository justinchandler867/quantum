"""
Regression test for the beta self-benchmark bug.

Pre-fix, both beta routines picked the benchmark as
`"QQQ" if "QQQ" in columns else columns[0]`. QQQ was never in a per-request
subset, so the *first ticker* became its own benchmark — AAPL got beta
exactly 1.000 (cov(AAPL,AAPL)/var), and every other ticker's beta was
measured against AAPL instead of the market.

These tests build a synthetic factor model (each stock = b_i * SPY + noise)
and assert compute_return_stats measures beta against SPY explicitly, never
against column[0].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data_ingest import compute_return_stats


def _factor_returns():
    rng = np.random.default_rng(7)
    T = 800
    spy = rng.normal(0.0004, 0.010, T)
    true_beta = {"AAPL": 1.20, "JNJ": 0.50, "XOM": 0.80}
    data = {t: b * spy + rng.normal(0.0, 0.004, T) for t, b in true_beta.items()}
    data["SPY"] = spy
    # Column order puts AAPL first — the exact trigger for the old columns[0] bug.
    df = pd.DataFrame(data, columns=["AAPL", "JNJ", "XOM", "SPY"])
    return df, true_beta


def test_beta_measured_against_spy_not_first_column():
    df, true_beta = _factor_returns()
    stats = compute_return_stats(df, benchmark="SPY")
    # Recovered betas track the true factor loadings (not AAPL-relative).
    assert abs(stats.loc["AAPL", "beta"] - true_beta["AAPL"]) < 0.1
    assert abs(stats.loc["JNJ", "beta"] - true_beta["JNJ"]) < 0.1
    assert abs(stats.loc["XOM", "beta"] - true_beta["XOM"]) < 0.1


def test_aapl_is_not_self_benchmarked_to_exactly_one():
    df, _ = _factor_returns()
    stats = compute_return_stats(df, benchmark="SPY")
    # The old bug forced AAPL (columns[0]) to exactly 1.0. It must not be.
    assert stats.loc["AAPL", "beta"] != 1.0
    assert abs(stats.loc["AAPL", "beta"] - 1.0) > 0.1


def test_benchmark_ticker_gets_beta_one_legitimately():
    df, _ = _factor_returns()
    stats = compute_return_stats(df, benchmark="SPY")
    # SPY vs itself is genuinely 1.0 (cov/var of a series with itself).
    assert abs(stats.loc["SPY", "beta"] - 1.0) < 1e-9


def test_beta_vs_spy_differs_from_beta_vs_aapl():
    df, _ = _factor_returns()
    spy_stats = compute_return_stats(df, benchmark="SPY")
    aapl_stats = compute_return_stats(df, benchmark="AAPL")
    # JNJ's beta is materially different depending on the benchmark chosen —
    # proving the benchmark is actually being used, not ignored.
    assert abs(spy_stats.loc["JNJ", "beta"] - aapl_stats.loc["JNJ", "beta"]) > 0.05
    # And against AAPL, AAPL is the self-benchmarked 1.0 (the old broken value).
    assert abs(aapl_stats.loc["AAPL", "beta"] - 1.0) < 1e-9


def test_missing_benchmark_defaults_to_one_not_crash():
    df, _ = _factor_returns()
    stats = compute_return_stats(df.drop(columns=["SPY"]), benchmark="SPY")
    # No SPY column -> betas default to 1.0 (with a logged warning), never
    # silently re-inferred from column order.
    assert (stats["beta"] == 1.0).all()
