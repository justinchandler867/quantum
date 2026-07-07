"""
Sealed acceptance tests for the View Sandbox (spec sealed 2026-07-06).

S1 — PSD guard clips an impossible correlation structure to the nearest valid one.
S2 — the 2026 preset posted to /api/sandbox/stress is numerically identical to
     /api/stress_test_2026 on the G10 basket.
S3 — a mild valid custom scenario leaves the guard inert (applied=false).
S4 — the empty regime reproduces the baseline exactly; every attribution
     component is zero.
S5 — attribution is additive on an arbitrary custom regime (vol Shapley sums to
     Δvol; MDD waterfall sums to ΔMDD).
S6 — out-of-bounds parameters are rejected at the API with 422 naming the field.
S7 — the 2026 preset passes the PSD guard (applied=false) on the G10 basket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import RISK_FREE_RATE, REFERENCE_HEDGES
from app.coc_io_outlook import (
    _psd_guard, _regime_cov_states, apply_regime, compute_attribution, REGIME_2026,
    SECTOR_TILTS_2026,
)
from tests.test_regime_attribution import G10_COV, G10_TICKERS, G10_SECTORS


# ── Synthetic engine fixture (network-free) ─────────────────────────────────
_VOLS = np.array([0.20, 0.15, 0.10])
_CORR = np.array([[1.0, 0.30, 0.10],
                  [0.30, 1.0, 0.20],
                  [0.10, 0.20, 1.0]])
_COV = np.outer(_VOLS, _VOLS) * _CORR
_MU = np.array([0.08, 0.06, 0.03])
_W = np.array([0.40, 0.35, 0.25])
_TICKERS = ["X", "Y", "Z"]
_SECTORS = {"X": "Technology", "Y": "Financial Services", "Z": "Fixed Income"}


def _empty_regime():
    return {"tilts": {}, "vol_multipliers": {}, "correlation_overrides": []}


# ── S1 — PSD guard clips to the nearest valid correlation structure ──────────

def test_s1_psd_guard_clips_impossible_structure():
    C = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, -0.9],
                  [0.9, -0.9, 1.0]])
    # Eigenvalues {1.9, 1.9, -0.8} -> not PSD.
    assert np.min(np.linalg.eigvalsh(C)) < -1e-10
    out, info = _psd_guard(C, ["A", "B", "C"], C)

    assert info["applied"] is True
    assert abs(out[0, 1] - 0.5) < 1e-6
    assert abs(out[0, 2] - 0.5) < 1e-6
    assert abs(out[1, 2] - (-0.5)) < 1e-6
    assert abs(info["max_correlation_adjustment"] - 0.400) < 1e-3
    # repaired matrix is (numerically) PSD
    assert np.min(np.linalg.eigvalsh(out)) > -1e-10


# ── S3 — a mild valid scenario leaves the guard inert ───────────────────────

def test_s3_mild_valid_scenario_no_adjustment():
    regime = {
        "tilts": {"Technology": -0.005},
        "vol_multipliers": {"Technology": 1.10},
        "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.15}],
    }
    st = _regime_cov_states(_COV, _TICKERS, _SECTORS, regime)
    psd = st["psd_adjustment"]
    assert psd["applied"] is False
    assert psd["max_correlation_adjustment"] == 0.0
    assert psd["affected_pairs"] == []


# ── S4 — empty regime == baseline, all attribution components zero ───────────

def test_s4_empty_regime_equals_baseline():
    mu_R, Sigma_R = apply_regime(_MU, _COV, _TICKERS, _SECTORS, _empty_regime())
    # Returns unchanged; covariance reconstructs to the baseline.
    assert np.allclose(mu_R, _MU, atol=0.0)
    base_vol = float(np.sqrt(_W @ _COV @ _W))
    regime_vol = float(np.sqrt(_W @ Sigma_R @ _W))
    assert abs(regime_vol - base_vol) < 1e-9

    attr, mdd_none, mdd_tvc = compute_attribution(
        _MU, _COV, _W, _TICKERS, _SECTORS, RISK_FREE_RATE, regime=_empty_regime())
    assert attr["return"]["tilts"] == 0.0
    assert abs(attr["vol"]["vol_scaling"]) < 1e-6
    assert abs(attr["vol"]["correlation"]) < 1e-6
    for k in ("tilts", "vol_scaling", "correlation"):
        assert abs(attr["sharpe"][k]) < 1e-6
        assert abs(attr["mdd"][k]) < 1e-6
    assert abs(mdd_tvc - mdd_none) < 1e-9


# ── S5 — attribution additivity on an arbitrary custom regime ───────────────

def test_s5_attribution_additivity_custom_regime():
    regime = {
        "tilts": {"Technology": -0.02, "Financial Services": 0.01, "Fixed Income": 0.005},
        "vol_multipliers": {"Technology": 1.30, "Fixed Income": 1.20},
        "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.45}],
    }
    attr, mdd_none, mdd_tvc = compute_attribution(
        _MU, _COV, _W, _TICKERS, _SECTORS, RISK_FREE_RATE, regime=regime)

    base_vol = float(np.sqrt(_W @ _COV @ _W))
    _, Sigma_R = apply_regime(_MU, _COV, _TICKERS, _SECTORS, regime)
    regime_vol = float(np.sqrt(_W @ Sigma_R @ _W))

    vol_sum = attr["vol"]["vol_scaling"] + attr["vol"]["correlation"]
    assert abs(vol_sum - (regime_vol - base_vol)) < 1e-5

    mdd_sum = sum(attr["mdd"][k] for k in ("tilts", "vol_scaling", "correlation"))
    assert abs(mdd_sum - (mdd_tvc - mdd_none)) < 1e-5


# ── S7 — the 2026 preset passes the guard on G10 (verified, not assumed) ─────

def test_s7_2026_preset_is_psd_on_g10():
    st = _regime_cov_states(np.array(G10_COV), G10_TICKERS, G10_SECTORS, REGIME_2026)
    assert st["psd_adjustment"]["applied"] is False
    assert st["psd_adjustment"]["max_correlation_adjustment"] == 0.0


# ── Endpoint fixture (mocked data, no network) ──────────────────────────────

def _fake_prices(tickers, include_hedges=True, return_volumes=False):
    cols = list(dict.fromkeys(list(tickers) + (REFERENCE_HEDGES if include_hedges else [])))
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(len(cols))
    steps = rng.normal(0.0003, 0.012, size=(len(idx), len(cols)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    if return_volumes:
        return prices, pd.DataFrame(2_000_000, index=idx, columns=cols)
    return prices


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "fetch_prices", _fake_prices)
    monkeypatch.setattr("app.data_ingest.fetch_prices", _fake_prices)
    monkeypatch.setattr(main, "cache_get", lambda *a, **k: None)
    monkeypatch.setattr(main, "cache_set", lambda *a, **k: None)
    for k in list(main._store.keys()):
        main._store[k] = None
    return TestClient(main.app)


G10_WEIGHTS = {"AAPL": .12, "JPM": .12, "JNJ": .12, "XOM": .12, "WMT": .12,
               "BND": .25, "TLT": .15}


# ── S2 — 2026 preset via sandbox == /api/stress_test_2026 on G10 ────────────

def test_s2_sandbox_2026_matches_preset_endpoint(client):
    preset_regime = {
        "name": "2026 Co-CIO Outlook",
        # tilts expressed in percentage points (SECTOR_TILTS_2026 × 100)
        "tilts": {sec: pp * 100.0 for sec, pp in SECTOR_TILTS_2026.items()},
        "vol_multipliers": REGIME_2026["vol_multipliers"],
        "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.20}],
    }
    r_preset = client.post("/api/stress_test_2026",
                           json={"tickers": G10_TICKERS, "weights": G10_WEIGHTS})
    r_sandbox = client.post("/api/sandbox/stress",
                            json={"tickers": G10_TICKERS, "weights": G10_WEIGHTS,
                                  "regime": preset_regime})
    assert r_preset.status_code == 200, r_preset.text
    assert r_sandbox.status_code == 200, r_sandbox.text
    p, s = r_preset.json(), r_sandbox.json()

    assert s["baseline"] == p["baseline"]
    assert s["regime"] == p["regime"]
    assert s["attribution"] == p["attribution"]
    # Guard is a verified no-op on both paths.
    assert p["psd_adjustment"]["applied"] is False
    assert s["psd_adjustment"]["applied"] is False


# ── S6 — out-of-bounds parameters rejected with 422 naming the field ────────

def test_s6_out_of_bounds_rejected_422(client):
    combined = client.post("/api/sandbox/stress", json={
        "tickers": G10_TICKERS, "weights": G10_WEIGHTS,
        "regime": {
            "tilts": {"Technology": 10.0},          # +10pp > +5.0
            "vol_multipliers": {"Energy": 3.0},     # 3.0 > 2.0
            "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.99}],
        },
    })
    assert combined.status_code == 422

    # Each field, named individually.
    r_tilt = client.post("/api/sandbox/stress", json={
        "tickers": G10_TICKERS, "weights": G10_WEIGHTS,
        "regime": {"tilts": {"Technology": 10.0}},
    })
    assert r_tilt.status_code == 422 and "tilts" in r_tilt.json()["detail"]

    r_mult = client.post("/api/sandbox/stress", json={
        "tickers": G10_TICKERS, "weights": G10_WEIGHTS,
        "regime": {"vol_multipliers": {"Energy": 3.0}},
    })
    assert r_mult.status_code == 422 and "vol_multipliers" in r_mult.json()["detail"]

    r_rho = client.post("/api/sandbox/stress", json={
        "tickers": G10_TICKERS, "weights": G10_WEIGHTS,
        "regime": {"correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.99}]},
    })
    assert r_rho.status_code == 422 and "correlation_overrides" in r_rho.json()["detail"]
