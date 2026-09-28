"""
PORTFOLIO_RISK_SPEC.md (+ Amendment A §8) — shared risk engine and Portfolio
Health metrics. Network-free: seeded synthetic returns only.
"""
import math

import numpy as np
import pandas as pd
import pytest

from app.config import HIGH_CORR_THRESHOLD, RISK_FREE_RATE
from app.discovery_context import MIN_OVERLAP_DAYS
from app.portfolio_risk import (
    capture_ratios, effective_bets, historical_var_cvar, max_drawdown, normalize_weights,
    pairwise_correlations, portfolio_health, redundancy_pairs, risk_decomposition,
    select_window, sharpe_1y, trailing_yield, weighted_yield,
)
from app.return_models import ledoit_wolf_constant_correlation


def _frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({
        "HI1": 1.6 * m + rng.normal(0, 0.015, n),   # high vol
        "HI2": 1.4 * m + rng.normal(0, 0.014, n),   # high vol
        "LO1": 0.3 * m + rng.normal(0, 0.003, n),   # low vol
        "LO2": 0.2 * m + rng.normal(0, 0.003, n),   # low vol
        "NEG": -0.9 * m + rng.normal(0, 0.004, n),  # diversifier (negative RC at these weights)
        "SPY": m,
    }, index=idx)


# ── §1 engine identity: sum RC_i == sigma_P ───────────────────────────────────
@pytest.mark.parametrize("seed,holdings", [
    (1, {"HI1": 25, "HI2": 25, "LO1": 25, "LO2": 25}),
    (2, {"HI1": 50, "LO1": 30, "LO2": 20}),
    (3, {"HI1": 40, "HI2": 40, "NEG": 20}),           # contains a negative-RC asset
])
def test_euler_identity_sum_rc_equals_sigma(seed, holdings):
    h = portfolio_health(_frame(seed), holdings)
    assert h["status"] == "ok"
    assert h["rc_sum"] == pytest.approx(h["vol"], rel=1e-12, abs=1e-14)
    if "NEG" in holdings:
        neg = [r for r in h["risk_rows"] if r["ticker"] == "NEG"][0]
        assert neg["rc"] < 0 and h["any_negative_rc"]


def test_engine_uses_shipped_ledoit_wolf_estimator():
    R = _frame(4)
    sub = R.tail(252)[["HI1", "LO1", "LO2"]]
    w = np.array([0.5, 0.3, 0.2])
    cov, _, _ = ledoit_wolf_constant_correlation(sub.values, annualize=True)
    dec = risk_decomposition(sub, {"HI1": 0.5, "LO1": 0.3, "LO2": 0.2})
    assert dec["sigma"] == pytest.approx(math.sqrt(w @ cov @ w), rel=1e-12)


def test_risk_share_differs_from_weight_share():
    h = portfolio_health(_frame(5), {"HI1": 25, "HI2": 25, "LO1": 25, "LO2": 25})
    share = {r["ticker"]: r["risk_share"] for r in h["risk_rows"]}
    assert share["HI1"] > 0.35 and share["LO1"] < 0.10   # 25% weight each, very unequal risk
    assert [r["ticker"] for r in h["risk_rows"]][:2] == sorted(["HI1", "HI2"], key=lambda t: -share[t])


# ── §1 insufficient data: excluded, never imputed ─────────────────────────────
def test_missing_and_short_history_excluded_with_reason():
    R = _frame(6)
    R.loc[R.index[:-40], "LO2"] = np.nan                  # only 40 days in window
    sub, inc, exc = select_window(R, {"HI1": 50, "LO2": 30, "ZZZ": 20})
    reasons = {e["ticker"]: e["reason"] for e in exc}
    assert inc == ["HI1"]
    assert reasons == {"LO2": "insufficient_history", "ZZZ": "not_loaded"}
    h = portfolio_health(R, {"HI1": 50, "LO2": 30, "ZZZ": 20})
    assert {e["ticker"] for e in h["excluded"]} == {"LO2", "ZZZ"}
    assert [r["ticker"] for r in h["risk_rows"]] == ["HI1"]   # never a 0 row for excluded names


def test_empty_and_all_excluded_states():
    assert portfolio_health(_frame(7), {})["status"] == "empty"
    h = portfolio_health(_frame(7), {"ZZZ": 100})
    assert h["status"] == "insufficient_data" and h["excluded"][0]["ticker"] == "ZZZ"


# ── §4 effective bets ─────────────────────────────────────────────────────────
def test_effective_bets_equal_risk_near_n_and_concentrated_far_below():
    rng = np.random.default_rng(8)
    n, idx = 300, pd.bdate_range("2024-01-01", periods=300)
    vols = [0.01, 0.02, 0.03, 0.04]
    R = pd.DataFrame({f"X{i}": rng.normal(0, v, n) for i, v in enumerate(vols)}, index=idx)
    inv = {f"X{i}": 1 / v for i, v in enumerate(vols)}            # inverse-vol ~ equal risk
    eq = portfolio_health(R, inv)["effective_bets"]
    assert eq == pytest.approx(4.0, abs=0.25)
    conc = portfolio_health(R, {"X0": 1, "X1": 1, "X2": 1, "X3": 97})["effective_bets"]
    assert conc < 1.2


def test_effective_bets_abs_convention_matches_spec_without_negatives():
    rc = [0.05, 0.03, 0.02]
    sigma = sum(rc)
    spec = 1 / sum((x / sigma) ** 2 for x in rc)
    assert effective_bets(rc) == pytest.approx(spec)
    # with a negative RC the |RC| shares stay in [0,1]
    assert 1.0 <= effective_bets([0.06, -0.01]) <= 2.0


# ── §8.1 drawdown / VaR / CVaR ────────────────────────────────────────────────
def test_max_drawdown_brute_force_and_dates():
    idx = pd.bdate_range("2024-01-01", periods=6)
    r = pd.Series(np.log([1.10, 0.90, 0.95, 1.20, 0.70, 1.05]), index=idx)
    path = np.exp(np.concatenate([[0], np.cumsum(r.values)]))
    brute = min(path[j] / path[i] - 1 for i in range(len(path)) for j in range(i, len(path)))
    dd = max_drawdown(r)
    assert dd["max_dd"] == pytest.approx(brute)
    assert dd["peak"] == "2024-01-04" and dd["trough"] == "2024-01-05"


def test_cvar_at_least_var_by_construction():
    for seed in range(5):
        h = portfolio_health(_frame(seed), {"HI1": 30, "LO1": 40, "NEG": 30})
        assert h["var"]["cvar"] >= h["var"]["var"] > 0
        assert h["var"]["k"] == math.ceil(0.05 * h["var"]["n"])


# ── §8.3 capture ratios ───────────────────────────────────────────────────────
def test_capture_ratios_seeded_asymmetry():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2024-01-01", periods=252)
    b = pd.Series(rng.normal(0, 0.01, 252), index=idx)
    p = pd.Series(np.where(b > 0, 1.2 * b, 0.5 * b), index=idx)   # up 120%, down 50% by design
    c = capture_ratios(p, b, MIN_OVERLAP_DAYS)
    assert c["up"] == pytest.approx(1.2) and c["down"] == pytest.approx(0.5)


def test_capture_guard_renders_none_below_min_obs():
    idx = pd.bdate_range("2024-01-01", periods=120)
    vals = np.array([0.01] * 100 + [-0.01] * 20)                    # 100 up days, 20 down days
    b = pd.Series(vals, index=idx)
    c = capture_ratios(b * 0.9, b, MIN_OVERLAP_DAYS)
    assert c["up"] == pytest.approx(0.9) and c["down"] is None and c["down_days"] == 20
    # guard boundary: exactly min_obs days is enough, one fewer is not
    k = MIN_OVERLAP_DAYS
    b2 = pd.Series(np.array([0.01] * k + [-0.01] * (k - 1)), index=pd.bdate_range("2024-01-01", periods=2 * k - 1))
    c2 = capture_ratios(b2, b2, k)
    assert c2["up"] == pytest.approx(1.0) and c2["down"] is None


# ── §8.4 Sharpe (shipped per-ticker method, config rf) ────────────────────────
def test_sharpe_matches_add_ticker_formula():
    idx = pd.bdate_range("2024-01-01", periods=252)
    r = pd.Series(np.random.default_rng(10).normal(0.0006, 0.012, 252), index=idx)
    closes = 100 * np.exp(np.concatenate([[0], np.cumsum(r.values)]))
    ret_pct = (closes[-1] / closes[0] - 1) * 100
    vol_pct = r.std() * math.sqrt(252) * 100
    shipped = (ret_pct - RISK_FREE_RATE * 100) / vol_pct                    # app/main.py add_ticker
    assert sharpe_1y(r, RISK_FREE_RATE)["sharpe"] == pytest.approx(shipped, rel=1e-12)


# ── §8.2 pairwise correlation ─────────────────────────────────────────────────
def test_pairwise_avg_equals_mean_of_shrunk_correlations():
    """Constant-correlation shrinkage moves each rho toward rbar, so the mean
    pairwise correlation is identical under the sample and the shrunk matrix."""
    sub = _frame(11).tail(252)[["HI1", "HI2", "LO1", "NEG"]]
    pw = pairwise_correlations(sub)
    cov, _, _ = ledoit_wolf_constant_correlation(sub.values, annualize=False)
    sd = np.sqrt(np.diag(cov))
    rho = cov / np.outer(sd, sd)
    shrunk_mean = rho[np.triu_indices(4, 1)].mean()
    assert pw["avg"] == pytest.approx(shrunk_mean, abs=1e-12)
    assert pw["pairs"] == 6
    assert pairwise_correlations(sub[["HI1"]])["avg"] is None


# ── §8.5 yield (J2) ───────────────────────────────────────────────────────────
def test_trailing_yield_raw_denominator_and_unknowns():
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.Series(np.linspace(40, 50, 300), index=idx)
    divs = pd.Series([0.25, 0.25, 0.25, 0.25, 0.25], index=[idx[10], idx[70], idx[130], idx[190], idx[250]])
    y = trailing_yield(divs, close)
    in_window = divs[divs.index > idx[-1] - pd.Timedelta(days=365)].sum()
    assert y == pytest.approx(in_window / 50.0)
    assert trailing_yield(None, close) is None and trailing_yield(divs, None) is None
    wy = weighted_yield({"A": 0.02, "B": None}, {"A": 0.6, "B": 0.4})
    assert wy["yield"] == pytest.approx(0.02) and wy["unknown"] == ["B"] and wy["coverage"] == pytest.approx(0.6)


# ── §5 redundancy flags ───────────────────────────────────────────────────────
def test_redundancy_flags_pair_above_threshold_only():
    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2024-01-01", periods=300)
    base = rng.normal(0, 0.01, 300)
    R = pd.DataFrame({
        "TLT": base,
        "IEF": base + rng.normal(0, 0.002, 300),     # r ~ 0.98 with TLT
        "AAA": rng.normal(0, 0.01, 300),
        "BBB": rng.normal(0, 0.01, 300),
    }, index=idx)
    out = redundancy_pairs(R, ["TLT", "IEF", "AAA", "BBB", "NOPE"], HIGH_CORR_THRESHOLD)
    flags = {f["ticker"]: f for f in out["flags"]}
    assert set(flags) == {"TLT", "IEF"}
    assert flags["IEF"]["peer"] == "TLT" and flags["IEF"]["r"] >= 0.85
    assert out["threshold"] == HIGH_CORR_THRESHOLD == 0.85
    assert out["not_loaded"] == ["NOPE"] and out["n_pairs"] == 6


def test_normalize_weights_drops_zero_and_renormalizes():
    assert normalize_weights({"A": 30, "B": 10, "C": 0}) == {"A": 0.75, "B": 0.25}
