"""
Tests for honest solver messaging on determined / infeasible single-point
geometries, and the outlook-inert note.

When n_eligible * max_position == 1 exactly, every asset is forced to the
cap and there is nothing to optimize. The message must distinguish:
  - feasible-but-determined: "constraints fully determine weights"
  - beta-cap-infeasible:     "determined weights violate the beta cap"
And when apply_outlook=True on such a geometry, a constraints_note must warn
that the tilt could not move anything.
"""
from __future__ import annotations

import numpy as np

from app.optimizer import optimize_portfolio, Objective, ProfileConstraints


def _cov(n, vols=None):
    vols = vols if vols is not None else np.linspace(0.10, 0.30, n)
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    return np.outer(vols, vols) * corr


def test_single_point_feasible_reports_fully_determined():
    n = 5
    tickers = list("ABCDE")
    mu = np.linspace(0.05, 0.15, n)
    betas = np.full(n, 0.8)  # port beta 0.8 < cap 1.10 -> feasible
    c = ProfileConstraints(max_position_pct=0.20, min_position_pct=0.02, max_beta=1.10)
    r = optimize_portfolio(tickers, mu, _cov(n), betas, Objective.MIN_VOLATILITY, c)
    assert "constraints fully determine weights" in r.solver_message
    assert "violate the beta cap" not in r.solver_message


def test_single_point_beta_violation_reports_infeasible():
    n = 5
    tickers = list("ABCDE")
    mu = np.linspace(0.05, 0.15, n)
    betas = np.full(n, 1.6)  # port beta 1.6 > cap 1.24 -> infeasible at the forced point
    c = ProfileConstraints(max_position_pct=0.20, min_position_pct=0.02, max_beta=1.24)
    r = optimize_portfolio(tickers, mu, _cov(n), betas, Objective.MAX_SHARPE, c)
    assert "violate the beta cap (infeasible)" in r.solver_message
    assert "1.24" in r.solver_message  # names the cap it breaches


def test_outlook_note_added_on_single_point_geometry():
    n = 5
    tickers = list("ABCDE")
    mu = np.linspace(0.05, 0.15, n)
    betas = np.full(n, 0.8)
    c = ProfileConstraints(max_position_pct=0.20, min_position_pct=0.02, max_beta=1.10)
    r = optimize_portfolio(tickers, mu, _cov(n), betas, Objective.MIN_VOLATILITY, c,
                           apply_outlook=True)
    assert r.constraints_note is not None
    assert "outlook tilt could not alter weights" in r.constraints_note


def test_no_outlook_note_when_geometry_has_slack():
    # 5 assets, cap 0.30 -> slack; outlook could move weights, so no inert note.
    n = 5
    tickers = list("ABCDE")
    mu = np.linspace(0.05, 0.15, n)
    betas = np.full(n, 0.8)
    c = ProfileConstraints(max_position_pct=0.30, min_position_pct=0.02, max_beta=1.5)
    r = optimize_portfolio(tickers, mu, _cov(n), betas, Objective.MIN_VOLATILITY, c,
                           apply_outlook=True)
    assert r.constraints_note is None
    assert "constraints fully determine" not in r.solver_message
