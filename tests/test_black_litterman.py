"""
Sealed-spec acceptance tests for the Black-Litterman-lite return model and the
Ledoit-Wolf constant-correlation shrinkage covariance.

Every numeric target here is hand-derived in the spec (section 4). The tests
either reproduce those numbers to the stated tolerance or fail. Cases A–E are
synthetic and network-free; Case F is a live-marked sanity check.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.return_models import (
    ledoit_wolf_constant_correlation,
    black_litterman,
    equilibrium_prior,
)
from tests import bl_sealed_cases as SC


# ── CASE A — two tilted assets, hand-derived ─────────────────────────────────

def test_A1_equilibrium_prior_high():
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    # Sigma * w = [0.0276, 0.0144]; pi = 2.5 * that = [6.90%, 3.60%].
    assert np.allclose(pi, SC.A1_PI, atol=1e-10), pi


def test_A2_posterior_confidence_1():
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_A, view_confidence=1.0,
                         tau=SC.TAU, delta=SC.DELTA)
    assert np.allclose(mu, SC.A2_C1, atol=1e-10), mu


def test_A3_posterior_confidence_3():
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_A, view_confidence=3.0,
                         tau=SC.TAU, delta=SC.DELTA)
    assert np.allclose(mu, SC.A3_C3, atol=1e-10), mu


def test_A_all_views_closed_form_passthrough():
    # Consequence (1): views on ALL assets -> mu = pi + [c/(1+c)] * t exactly.
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    for c, frac in [(0.5, 1 / 3), (1.0, 0.5), (3.0, 0.75)]:
        mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_A, view_confidence=c,
                             tau=SC.TAU, delta=SC.DELTA)
        assert np.allclose(mu, pi + frac * SC.TILTS_A, atol=1e-12), (c, mu)


# ── CASE B — spillover onto an untilted asset ────────────────────────────────

def test_B1_spillover_beta_scaled():
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_B, view_confidence=1.0,
                         tau=SC.TAU, delta=SC.DELTA)
    delta_pp = (mu - pi) * 100.0
    # Tech moves -0.500pp; Energy moves beta21 * (-0.500pp) = -0.1125pp.
    assert abs(delta_pp[0] - SC.B1_DELTA_PP[0]) < 1e-6, delta_pp
    assert abs(delta_pp[1] - SC.B1_DELTA_PP[1]) < 1e-6, delta_pp
    assert abs(delta_pp[1] - SC.BETA_21 * delta_pp[0]) < 1e-6, delta_pp


def test_B2_tau_invariant():
    mu_005 = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_B, view_confidence=1.0,
                             tau=0.05, delta=SC.DELTA)
    mu_050 = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_B, view_confidence=1.0,
                             tau=0.50, delta=SC.DELTA)
    assert np.allclose(mu_005, mu_050, atol=1e-12), (mu_005, mu_050)


# ── CASE C — limits & nulls ──────────────────────────────────────────────────

def test_C1_confidence_to_zero_collapses_to_prior():
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_C1, view_confidence=1e-6,
                         tau=SC.TAU, delta=SC.DELTA)
    assert np.linalg.norm(mu - pi) < 1e-8, np.linalg.norm(mu - pi)


def test_C2_no_outlook_returns_prior_exactly():
    # apply_outlook=false is expressed as an all-zero tilt vector (no views).
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, np.zeros(2), view_confidence=1.0,
                         tau=SC.TAU, delta=SC.DELTA)
    assert np.array_equal(mu, pi), (mu, pi)


def test_C3_historical_optimizer_output_byte_for_byte():
    # return_model="historical" must not perturb the optimizer's numeric path.
    # Frozen G5 fixture -> hand-captured golden weights/metrics.
    from app.optimizer import optimize_portfolio, Objective, ProfileConstraints

    X = SC.synthetic_G5_returns()
    Xc = X - X.mean(axis=0, keepdims=True)
    sigma = (Xc.T @ Xc) / X.shape[0] * 252
    mu = X.mean(axis=0) * 252
    betas = np.array([1.1, 0.9, 1.2, 0.4, 1.0])
    c = ProfileConstraints(max_position_pct=0.40, min_position_pct=0.02, max_beta=1.5)

    r = optimize_portfolio(SC.G5_TICKERS, mu, sigma, betas, Objective.MAX_SHARPE, c)

    golden = {"AAA": 0.4, "BBB": 0.16, "CCC": 0.4, "DDD": 0.02, "EEE": 0.02}
    assert r.weights == golden, r.weights
    assert r.portfolio_return == 0.28119
    assert r.portfolio_volatility == 0.222495
    assert r.sharpe_ratio == 1.0705
    assert r.beta == 1.092


# ── CASE D — shrinkage exact null (N == 2) ───────────────────────────────────

def test_D1_two_asset_shrinkage_is_noop():
    X = SC.synthetic_D_returns()
    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / X.shape[0] * 252     # annualized MLE sample cov
    sigma, alpha, n_days = ledoit_wolf_constant_correlation(X)
    # N=2 makes the constant-correlation target == the sample matrix: provable no-op.
    assert np.max(np.abs(sigma - sample)) < 1e-12, np.max(np.abs(sigma - sample))
    assert n_days == X.shape[0]


# ── CASE E — shrinkage direction, 5 assets × 40 obs ──────────────────────────

def test_E1_alpha_in_open_unit_interval():
    X = SC.synthetic_E_returns()
    _, alpha, n_days = ledoit_wolf_constant_correlation(X)
    assert 0.05 < alpha <= 1.0, alpha
    assert n_days == 40


def test_E2_shrunk_correlations_between_sample_and_rbar():
    X = SC.synthetic_E_returns()
    n = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / X.shape[0]
    var = np.diag(sample)
    std = np.sqrt(var)
    r_sample = sample / np.outer(std, std)
    rbar = (r_sample.sum() - n) / (n * (n - 1))

    sigma, alpha, _ = ledoit_wolf_constant_correlation(X)
    sigma_daily = sigma / 252
    std_sh = np.sqrt(np.diag(sigma_daily))
    r_shrunk = sigma_daily / np.outer(std_sh, std_sh)

    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = sorted((r_sample[i, j], rbar))
            assert lo - 1e-12 <= r_shrunk[i, j] <= hi + 1e-12, (i, j, r_shrunk[i, j])

    # Variances preserved exactly (diagonal of the target equals the sample).
    assert np.allclose(np.diag(sigma_daily), var, atol=1e-12)


def test_E2_alpha_scale_invariant():
    # alpha must not depend on annualization — same on daily and annualized S.
    X = SC.synthetic_E_returns()
    _, a_annual, _ = ledoit_wolf_constant_correlation(X, annualize=True)
    _, a_daily, _ = ledoit_wolf_constant_correlation(X, annualize=False)
    assert abs(a_annual - a_daily) < 1e-15


# ── CASE F — live equilibrium sanity (network) ───────────────────────────────

@pytest.mark.live
def test_F1_equilibrium_ordering_live():
    """
    G10 basket: equilibrium premia should follow (Sigma * w)_i — equities above
    BND; TLT the lowest (possibly negative for a negative-beta asset). Marked
    `live`; skipped unless yfinance data is reachable.
    """
    pytest.importorskip("yfinance")
    from app.data_ingest import fetch_prices, compute_log_returns

    basket = ["AAPL", "JPM", "JNJ", "XOM", "WMT", "BND", "TLT"]
    try:
        prices, _ = fetch_prices(basket, include_hedges=False, return_volumes=True)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"live price fetch unavailable: {exc}")

    returns = compute_log_returns(prices)
    available = [t for t in basket if t in returns.columns]
    window = returns[available].dropna().tail(5 * 252)
    sigma, _, _ = ledoit_wolf_constant_correlation(window.values)

    # Equal-weight equilibrium ordering check is done in the audit; here we only
    # assert BND does not carry the top premium and the machinery runs.
    caps = np.ones(len(available))
    pi = equilibrium_prior(sigma, caps, delta=2.5)
    order = {t: pi[i] for i, t in enumerate(available)}
    if "BND" in order and "AAPL" in order:
        assert order["AAPL"] > order["BND"]
