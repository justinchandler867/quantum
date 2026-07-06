"""
Sealed inputs for the Black-Litterman / Ledoit-Wolf audit (spec section 4).

Single source of truth for the synthetic cases so tests/test_black_litterman.py
and audit_black_litterman.py compute against byte-identical inputs. Nothing here
touches the network — Case F (live) sources its own basket in the audit script.
"""
from __future__ import annotations

import numpy as np

# ── Case A/B/C — two-asset hand-derived setup ────────────────────────────────
# sigma1=0.20 (Technology, tilt -1.0%), sigma2=0.15 (Financial Services / Energy),
# rho=0.30, w_mkt=(0.60, 0.40), delta=2.5.
SIGMA_2 = np.array([[0.0400, 0.0090],
                    [0.0090, 0.0225]])
W_MKT_2 = np.array([0.60, 0.40])          # market-cap weights (already normalized)
DELTA = 2.5
TAU = 0.05

# Case A: both assets tilted.
TILTS_A = np.array([-0.010, 0.015])       # Tech -1.0pp, Financial Services +1.5pp

# Case B: asset 2 = Energy, tilt 0 — single view on asset 1.
TILTS_B = np.array([-0.010, 0.0])

# Case C1: single small view, c -> 0 limit collapses posterior to prior.
TILTS_C1 = np.array([-0.005, 0.0])

# Sealed targets (spec section 4).
A1_PI = np.array([0.0690, 0.0360])
A2_C1 = np.array([0.0640, 0.0435])
A3_C3 = np.array([0.0615, 0.04725])
B1_DELTA_PP = np.array([-0.500, -0.1125])  # posterior − prior, in percentage points
BETA_21 = 0.009 / 0.040                     # 0.225


def synthetic_E_returns() -> np.ndarray:
    """
    Case E: synthetic 5 assets × 40 observations with heterogeneous pairwise
    correlations (a lower-triangular factor loading gives each pair a distinct
    correlation). Deterministic — seeded, no Date/random-at-runtime dependence.
    """
    rng = np.random.default_rng(20260706)
    n, t = 5, 40
    loadings = np.array([
        [1.00,  0.00, 0.00, 0.00, 0.00],
        [0.80,  0.60, 0.00, 0.00, 0.00],
        [0.20,  0.10, 0.97, 0.00, 0.00],
        [-0.30, 0.40, 0.10, 0.85, 0.00],
        [0.50, -0.20, 0.30, 0.10, 0.78],
    ])
    z = rng.standard_normal((t, n))
    base = z @ loadings.T
    vols = np.array([0.020, 0.015, 0.030, 0.010, 0.025])
    drift = np.array([0.0005, 0.0003, 0.0008, 0.0002, 0.0006])
    return base * vols + drift


def synthetic_D_returns() -> np.ndarray:
    """Case D: an arbitrary 2-asset return series — the estimator must no-op."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((300, 2)) * np.array([0.010, 0.008]) + 0.0003
    return x


def synthetic_G5_returns() -> "np.ndarray":
    """
    Case C3: a frozen 5-asset daily return panel used to prove that
    return_model='historical' leaves the optimizer path byte-identical.
    Deterministic and network-free.
    """
    rng = np.random.default_rng(1234)
    n, t = 5, 3 * 252
    loadings = np.array([
        [1.00,  0.00, 0.00, 0.00, 0.00],
        [0.70,  0.50, 0.00, 0.00, 0.00],
        [0.30,  0.20, 0.90, 0.00, 0.00],
        [0.10, -0.30, 0.20, 0.80, 0.00],
        [-0.20, 0.10, 0.10, 0.10, 0.85],
    ])
    z = rng.standard_normal((t, n))
    base = z @ loadings.T
    vols = np.array([0.018, 0.014, 0.022, 0.009, 0.026])
    drift = np.array([0.0006, 0.0004, 0.0007, 0.0002, 0.0005])
    return base * vols + drift


G5_TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
