"""
Regression tests for the optimizer feasibility guards (BUG 3a/b/c).

Pre-fix, a per-asset cap that could not sum to 1 across the basket
(n * max_position < 1) produced an infeasible problem; the optimizer
returned normalized equal weights that *violated their own cap*, as a
success. And when n * cap == 1 exactly, it reported a normal solve on a
point where nothing could be optimized.
"""
from __future__ import annotations

import numpy as np

from app.optimizer import optimize_portfolio, Objective, ProfileConstraints


def _inputs(n):
    rng = np.random.default_rng(3)
    mu = np.linspace(0.05, 0.20, n)
    vols = np.linspace(0.10, 0.35, n)
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    cov = np.outer(vols, vols) * corr
    betas = np.linspace(0.6, 1.4, n)
    return mu, cov, betas


def test_two_ticker_tight_cap_relaxes_and_respects_bounds():
    # risk-20 profile cap = 0.15; 2 assets * 0.15 = 0.30 < 1.0 -> infeasible.
    tickers = ["AAPL", "BND"]
    mu, cov, betas = _inputs(2)
    c = ProfileConstraints(max_position_pct=0.15, min_position_pct=0.02, max_beta=0.68)
    r = optimize_portfolio(tickers, mu, cov, betas, Objective.MAX_SHARPE, c)

    # (a) The cap was relaxed, and the response says so honestly.
    assert r.constraints_note is not None
    assert "relaxed" in r.constraints_note

    # (b) Returned weights respect the *effective* bound and sum to 1 — never
    #     the old behavior of 0.5/0.5 silently exceeding a 0.15 cap.
    effective_cap = round((1.0 / 2) * 1.25, 4)  # 0.625
    assert all(w <= effective_cap + 1e-6 for w in r.weights.values())
    assert abs(sum(r.weights.values()) - 1.0) < 1e-6


def test_five_ticker_exact_cap_reports_degenerate_single_point():
    # 5 assets * 0.20 = 1.0 exactly -> the only feasible point is equal weight.
    tickers = list("ABCDE")
    mu, cov, betas = _inputs(5)
    c = ProfileConstraints(max_position_pct=0.20, min_position_pct=0.02, max_beta=1.10)
    r = optimize_portfolio(tickers, mu, cov, betas, Objective.MIN_VOLATILITY, c)

    # (c) Honest message that nothing could be optimized.
    assert "constraints fully determine weights" in r.solver_message
    # No cap relaxation here (n*cap == 1.0, not < 1.0).
    assert r.constraints_note is None
    # Every weight sits at the 0.20 cap.
    assert all(abs(w - 0.20) < 1e-6 for w in r.weights.values())


def test_weights_never_exceed_cap_when_slack_exists():
    # 5 assets, cap 0.30 -> real slack; min-vol should tilt but still obey cap.
    tickers = list("ABCDE")
    mu, cov, betas = _inputs(5)
    c = ProfileConstraints(max_position_pct=0.30, min_position_pct=0.02, max_beta=1.5)
    r = optimize_portfolio(tickers, mu, cov, betas, Objective.MIN_VOLATILITY, c)
    assert all(w <= 0.30 + 1e-6 for w in r.weights.values())
    assert abs(sum(r.weights.values()) - 1.0) < 1e-6
    assert r.constraints_note is None
