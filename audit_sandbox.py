#!/usr/bin/env python3
"""
View Sandbox audit — raw computed values for the sealed cases S1–S7.

Network-free: every case runs off synthetic inputs or the frozen G10 fixture
(tests/test_regime_attribution.py), so the numbers here are the same ones the
acceptance suite (tests/test_sandbox.py) asserts against. Eyeball before commit.

Usage: python3 audit_sandbox.py
"""
from __future__ import annotations

import numpy as np

from app.config import RISK_FREE_RATE
from app.coc_io_outlook import (
    _psd_guard, _regime_cov_states, apply_regime, compute_attribution,
    REGIME_2026, SECTOR_TILTS_2026,
)
from app.main import _validate_regime_bounds, _build_internal_regime
from app.models import RegimeInput
from tests.test_regime_attribution import G10_COV, G10_EXP, G10_TICKERS, G10_SECTORS


def hr(t):
    print("\n" + "═" * 78)
    print(t)
    print("═" * 78)


# Synthetic 3-asset fixture (matches tests/test_sandbox.py).
VOLS = np.array([0.20, 0.15, 0.10])
CORR = np.array([[1.0, 0.30, 0.10], [0.30, 1.0, 0.20], [0.10, 0.20, 1.0]])
COV = np.outer(VOLS, VOLS) * CORR
MU = np.array([0.08, 0.06, 0.03])
W3 = np.array([0.40, 0.35, 0.25])
T3 = ["X", "Y", "Z"]
S3 = {"X": "Technology", "Y": "Financial Services", "Z": "Fixed Income"}

G10_W = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.25, 0.15])
G10_W = G10_W / G10_W.sum()


def _metrics(w, mu, cov):
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    ret = float(w @ mu)
    sh = (ret - RISK_FREE_RATE) / vol if vol > 1e-12 else 0.0
    return ret, vol, sh


def audit_s1():
    hr("S1 — PSD guard clips an impossible correlation structure")
    C = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    print("requested rho (AB, AC, BC) = (0.9, 0.9, -0.9)")
    print("eigenvalues:", np.round(np.linalg.eigvalsh(C), 6))
    out, info = _psd_guard(C, ["A", "B", "C"], C)
    print(f"applied            = {info['applied']}")
    print(f"adjusted rho AB/AC/BC = {out[0,1]:+.6f} / {out[0,2]:+.6f} / {out[1,2]:+.6f}"
          "   (target +0.5 / +0.5 / -0.5)")
    print(f"max_correlation_adjustment = {info['max_correlation_adjustment']:.6f}  (target 0.400)")
    print(f"repaired min eigenvalue    = {np.linalg.eigvalsh(out).min():.3e}")
    print("affected_pairs:")
    for p in info["affected_pairs"]:
        print(f"    {p['tickers']}  requested {p['requested']:+.3f} -> adjusted {p['adjusted']:+.3f}")


def audit_s2():
    hr("S2 — 2026 preset via sandbox-built regime == REGIME_2026 preset (G10 fixture)")
    internal = _build_internal_regime(RegimeInput(
        name="2026",
        tilts={sec: pp * 100.0 for sec, pp in SECTOR_TILTS_2026.items()},   # pp
        vol_multipliers=REGIME_2026["vol_multipliers"],
        correlation_overrides=[{"a": "Fixed Income", "b": "EQUITY", "rho": 0.20}],
    ))
    cov = np.array(G10_COV)
    outs = {}
    for label, regime in (("preset", REGIME_2026), ("sandbox", internal)):
        mu_R, cov_R = apply_regime(G10_EXP, cov, G10_TICKERS, G10_SECTORS, regime)
        b = _metrics(G10_W, G10_EXP, cov)
        r = _metrics(G10_W, mu_R, cov_R)
        attr, _, _ = compute_attribution(
            G10_EXP, cov, G10_W, G10_TICKERS, G10_SECTORS, RISK_FREE_RATE, regime=regime)
        outs[label] = (b, r, attr)
    (bp, rp, ap), (bs, rs, as_) = outs["preset"], outs["sandbox"]
    print(f"preset  regime (ret, vol, sharpe) = ({rp[0]:.8f}, {rp[1]:.8f}, {rp[2]:.6f})")
    print(f"sandbox regime (ret, vol, sharpe) = ({rs[0]:.8f}, {rs[1]:.8f}, {rs[2]:.6f})")
    print(f"max |Δ| metrics = {max(abs(a-b) for a,b in zip(rp+bp, rs+bs)):.2e}")
    dvol = abs(ap['vol']['correlation'] - as_['vol']['correlation'])
    print(f"attribution vol.correlation preset/sandbox = "
          f"{ap['vol']['correlation']:.6f} / {as_['vol']['correlation']:.6f}  |Δ|={dvol:.2e}")


def audit_s3():
    hr("S3 — mild valid custom scenario -> guard inert")
    regime = {"tilts": {"Technology": -0.005},
              "vol_multipliers": {"Technology": 1.10},
              "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.15}]}
    st = _regime_cov_states(COV, T3, S3, regime)
    print("psd_adjustment:", st["psd_adjustment"])


def audit_s4():
    hr("S4 — empty regime reproduces baseline; all attribution components zero")
    empty = {"tilts": {}, "vol_multipliers": {}, "correlation_overrides": []}
    mu_R, cov_R = apply_regime(MU, COV, T3, S3, empty)
    print(f"max |Δmu|        = {np.max(np.abs(mu_R - MU)):.3e}")
    print(f"|Δvol|           = {abs(np.sqrt(W3@cov_R@W3) - np.sqrt(W3@COV@W3)):.3e}")
    attr, mdd_none, mdd_tvc = compute_attribution(
        MU, COV, W3, T3, S3, RISK_FREE_RATE, regime=empty)
    print(f"return.tilts     = {attr['return']['tilts']}")
    print(f"vol   scaling/corr = {attr['vol']['vol_scaling']} / {attr['vol']['correlation']}")
    print(f"sharpe T/V/C     = {attr['sharpe']}")
    print(f"mdd    T/V/C     = {attr['mdd']}")
    print(f"|mdd_tvc-mdd_none| = {abs(mdd_tvc - mdd_none):.3e}")


def audit_s5():
    hr("S5 — attribution additivity on an arbitrary custom regime")
    regime = {"tilts": {"Technology": -0.02, "Financial Services": 0.01, "Fixed Income": 0.005},
              "vol_multipliers": {"Technology": 1.30, "Fixed Income": 1.20},
              "correlation_overrides": [{"a": "Fixed Income", "b": "EQUITY", "rho": 0.45}]}
    attr, mdd_none, mdd_tvc = compute_attribution(
        MU, COV, W3, T3, S3, RISK_FREE_RATE, regime=regime)
    _, cov_R = apply_regime(MU, COV, T3, S3, regime)
    dvol = float(np.sqrt(W3@cov_R@W3) - np.sqrt(W3@COV@W3))
    vol_sum = attr["vol"]["vol_scaling"] + attr["vol"]["correlation"]
    mdd_sum = sum(attr["mdd"][k] for k in ("tilts", "vol_scaling", "correlation"))
    print(f"vol  components sum = {vol_sum:.8f}   Δvol = {dvol:.8f}   |resid|={abs(vol_sum-dvol):.2e}")
    print(f"mdd  components sum = {mdd_sum:.8f}   Δmdd = {mdd_tvc-mdd_none:.8f}   "
          f"|resid|={abs(mdd_sum-(mdd_tvc-mdd_none)):.2e}")
    print(f"psd applied = {_regime_cov_states(COV,T3,S3,regime)['psd_adjustment']['applied']}")


def audit_s6():
    hr("S6 — out-of-bounds parameters rejected (422 naming the field)")
    cases = [
        ("tilt +10pp", RegimeInput(tilts={"Technology": 10.0})),
        ("multiplier 3.0", RegimeInput(vol_multipliers={"Energy": 3.0})),
        ("rho 0.99", RegimeInput(correlation_overrides=[{"a": "Fixed Income", "b": "EQUITY", "rho": 0.99}])),
    ]
    for label, regime in cases:
        try:
            _validate_regime_bounds(regime)
            print(f"  {label:<16} -> NO ERROR (unexpected!)")
        except Exception as exc:
            detail = getattr(exc, "detail", str(exc))
            print(f"  {label:<16} -> 422: {detail}")


def audit_s7():
    hr("S7 — 2026 preset passes the PSD guard on G10 (applied=false)")
    st = _regime_cov_states(np.array(G10_COV), G10_TICKERS, G10_SECTORS, REGIME_2026)
    print("psd_adjustment:", st["psd_adjustment"])


def main():
    print("VIEW SANDBOX AUDIT — raw computed values (S1–S7)")
    audit_s1()
    audit_s2()
    audit_s3()
    audit_s4()
    audit_s5()
    audit_s6()
    audit_s7()
    print()


if __name__ == "__main__":
    main()
