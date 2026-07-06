"""
Sealed acceptance tests for regime attribution (spec sealed 2026-07-05).

Case A — analytic two-asset, asserted to 1e-4 against hand-derived numbers.
Case B — all-tech null control, correlation nulls to 1e-10.
Case C — frozen G10 fixture, additivity against the endpoint's own totals.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.coc_io_outlook import compute_attribution, apply_regime
from app.config import RISK_FREE_RATE


# ── Case A — analytic two-asset ─────────────────────────────────────────────
# 60% equity (σ=0.20, Technology), 40% bond (σ=0.10, Fixed Income), ρ0=-0.20.
# Regime (default REGIME_2026): both vols ×1.10, ρ→+0.20, tilts Tech -1.0% / FI +0.5%.

def _case_a():
    vols = np.array([0.20, 0.10])
    rho = -0.20
    cov = np.outer(vols, vols) * np.array([[1.0, rho], [rho, 1.0]])
    w = np.array([0.6, 0.4])
    mu0 = np.array([0.08, 0.03])
    tickers = ["EQ", "BND"]
    sectors = {"EQ": "Technology", "BND": "Fixed Income"}
    attr, _, _ = compute_attribution(mu0, cov, w, tickers, sectors, RISK_FREE_RATE)
    return attr


def test_case_a1_return_is_minus_040pp_all_tilts():
    attr = _case_a()
    assert abs(attr["return"]["tilts"] - (-0.400)) < 1e-4


def test_case_a2_vol_shapley_and_correlation_share():
    attr = _case_a()
    assert abs(attr["vol"]["vol_scaling"] - 0.012626) < 1e-4
    assert abs(attr["vol"]["correlation"] - 0.015967) < 1e-4
    # exact additivity: components == Δσ = 0.028593
    assert abs(attr["vol"]["vol_scaling"] + attr["vol"]["correlation"] - 0.028593) < 1e-4
    # correlation share -> headline 56%
    assert "56%" in attr["headline"]


def test_case_a3_bond_rc_jump_is_all_flip():
    attr = _case_a()
    bond = next(p for p in attr["positions"] if p["ticker"] == "BND")
    assert abs(bond["rc_baseline"] - 0.045455) < 1e-4     # 4.55%
    assert abs(bond["rc_regime"] - (1.0 / 7.0)) < 1e-4     # 14.29% exactly
    # uniform scaling -> nothing from scaling; the whole jump is the flip.
    assert abs(bond["drc_from_scaling"]) < 1e-4
    assert abs(bond["drc_from_flip"] - (1.0 / 7.0 - 0.045455)) < 1e-4


# ── Case B — all-tech null control ──────────────────────────────────────────
# 5 Technology names, equal weight. No Fixed-Income pairs -> correlation lever
# is inert; vols all scale ×1.10 (Technology multiplier).

def _case_b():
    tickers = ["NVDA", "AAPL", "MSFT", "AVGO", "AMD"]
    sectors = {t: "Technology" for t in tickers}
    vols = np.array([0.55, 0.28, 0.26, 0.40, 0.60])
    base = np.full((5, 5), 0.45) + 0.55 * np.eye(5)
    cov = np.outer(vols, vols) * base
    w = np.full(5, 0.20)
    mu0 = np.array([0.30, 0.20, 0.18, 0.25, 0.35])
    attr, _, _ = compute_attribution(mu0, cov, w, tickers, sectors, RISK_FREE_RATE)
    return attr, mu0, cov, w, tickers, sectors


def test_case_b1_correlation_component_is_zero_every_metric():
    attr, *_ = _case_b()
    assert abs(attr["vol"]["correlation"]) < 1e-10
    assert abs(attr["sharpe"]["correlation"]) < 1e-10
    assert abs(attr["mdd"]["correlation"]) < 1e-10


def test_case_b2_sigma_ratio_is_exactly_110():
    _, mu0, cov, w, tickers, sectors = _case_b()
    _, regime_cov = apply_regime(mu0, cov, tickers, sectors)
    sigma_0 = float(np.sqrt(w @ cov @ w))
    sigma_R = float(np.sqrt(w @ regime_cov @ w))
    assert abs(sigma_R / sigma_0 - 1.10) < 1e-10


def test_case_b3_return_component_is_minus_1pp():
    attr, *_ = _case_b()
    assert abs(attr["return"]["tilts"] - (-1.000)) < 1e-10


# ── Case C — frozen G10 fixture (live additivity gate) ──────────────────────
G10_TICKERS = ["AAPL", "JPM", "JNJ", "XOM", "WMT", "BND", "TLT"]
G10_SECTORS = {"AAPL": "Technology", "JPM": "Financial Services", "JNJ": "Healthcare",
               "XOM": "Energy", "WMT": "Consumer Defensive",
               "BND": "Fixed Income", "TLT": "Fixed Income"}
G10_WEIGHTS_PCT = {"AAPL": .12, "JPM": .12, "JNJ": .12, "XOM": .12, "WMT": .12,
                   "BND": .25, "TLT": .15}
G10_EXP = np.array([0.15779217, 0.18092236, 0.11686649, 0.20194717, 0.18918858,
                    -0.00165823, -0.07683971])
G10_COV = np.array([
    [0.07867216, 0.02081986, 0.00701573, 0.00320434, 0.0110452, 0.00138299, 0.00114866],
    [0.02081986, 0.06216271, 0.00551497, 0.00524405, 0.00483, 0.00062436, -0.00041092],
    [0.00701573, 0.00551497, 0.03317133, 0.00372954, 0.01208864, 0.00189429, 0.00394376],
    [0.00320434, 0.00524405, 0.00372954, 0.0781819, 0.00645755, -0.00315298, -0.00794585],
    [0.0110452, 0.00483, 0.01208864, 0.00645755, 0.06349007, 0.0011188, 0.00195011],
    [0.00138299, 0.00062436, 0.00189429, -0.00315298, 0.0011188, 0.00260561, 0.00561728],
    [0.00114866, -0.00041092, 0.00394376, -0.00794585, 0.00195011, 0.00561728, 0.0167514],
])


def _case_c():
    w = np.array([G10_WEIGHTS_PCT[t] for t in G10_TICKERS])
    w = w / w.sum()
    attr, mdd_none, mdd_tvc = compute_attribution(
        G10_EXP, G10_COV, w, G10_TICKERS, G10_SECTORS, RISK_FREE_RATE)
    return attr, mdd_none, mdd_tvc, w


def test_case_c1_vol_and_sharpe_additive_to_endpoint_totals():
    attr, _, _, w = _case_c()
    base_vol = float(np.sqrt(w @ G10_COV @ w))
    regime_ret_vec, regime_cov = apply_regime(G10_EXP, G10_COV, G10_TICKERS, G10_SECTORS)
    regime_vol = float(np.sqrt(w @ regime_cov @ w))
    base_sh = (float(w @ G10_EXP) - RISK_FREE_RATE) / base_vol
    regime_sh = (float(w @ regime_ret_vec) - RISK_FREE_RATE) / regime_vol

    vol_sum = attr["vol"]["vol_scaling"] + attr["vol"]["correlation"]
    assert abs(vol_sum - (regime_vol - base_vol)) < 1e-5

    sh_sum = sum(attr["sharpe"][k] for k in ("tilts", "vol_scaling", "correlation"))
    assert abs(sh_sum - (regime_sh - base_sh)) < 1e-5


def test_case_c2_correlation_exceeds_vol_scaling():
    attr, *_ = _case_c()
    assert attr["vol"]["correlation"] > attr["vol"]["vol_scaling"]


def test_case_c3_two_largest_flip_positions_are_bnd_and_tlt():
    attr, *_ = _case_c()
    ranked = sorted(attr["positions"], key=lambda p: p["drc_from_flip"], reverse=True)
    assert {ranked[0]["ticker"], ranked[1]["ticker"]} == {"BND", "TLT"}


def test_case_c4_mdd_components_sum_to_delta_shared_seed():
    attr, mdd_none, mdd_tvc, _ = _case_c()
    mdd_sum = sum(attr["mdd"][k] for k in ("tilts", "vol_scaling", "correlation"))
    # If this fails, the MC states aren't sharing a seed — that's the bug.
    assert abs(mdd_sum - (mdd_tvc - mdd_none)) < 1e-5
