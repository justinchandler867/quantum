#!/usr/bin/env python3.12
"""
AUDIT RUN — Black-Litterman-lite + Ledoit-Wolf shrinkage (sealed-spec build).

Prints the RAW computed values for every sealed case. No interpretation, no
pass/fail — the hand-derived spec numbers are the audit; compare by eye.

  Synthetic (network-free):  A1 A2 A3  B1 B2  C1 C2 C3  D1  E1 E2
  Live (running server):     F1 — G10 basket equilibrium ordering

Usage:
    python3.12 audit_black_litterman.py
    BASE_URL=http://localhost:8000 python3.12 audit_black_litterman.py

Case F needs a running server (uvicorn app.main:app). If unreachable it is
reported as skipped; the local equilibrium computation is still printed.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

import bl_sealed_cases as SC  # noqa: E402
from app.return_models import (  # noqa: E402
    ledoit_wolf_constant_correlation,
    black_litterman,
    equilibrium_prior,
)

np.set_printoptions(precision=8, suppress=True, floatmode="maxprec")


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def vec(x) -> str:
    return "[" + ", ".join(f"{v:+.8f}" for v in np.atleast_1d(x)) + "]"


# ── Synthetic cases ──────────────────────────────────────────────────────────

def audit_A():
    hr("CASE A — two tilted assets (delta=2.5, tau=0.05, w_mkt=[0.60,0.40])")
    print("Sigma        =", SC.SIGMA_2.tolist())
    print("Sigma * w    =", vec(SC.SIGMA_2 @ SC.W_MKT_2))
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    print("A1  pi           =", vec(pi))
    print("A2  posterior c=1=", vec(black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_A, 1.0, SC.TAU, SC.DELTA)))
    print("A3  posterior c=3=", vec(black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_A, 3.0, SC.TAU, SC.DELTA)))


def audit_B():
    hr("CASE B — spillover (asset2 tilt=0, single view on asset1, c=1)")
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    mu = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_B, 1.0, SC.TAU, SC.DELTA)
    print("pi           =", vec(pi))
    print("B1  posterior    =", vec(mu))
    print("B1  (mu-pi) in pp=", vec((mu - pi) * 100.0))
    print("B1  beta21 = Sigma_21/Sigma_11 =", f"{SC.BETA_21:.8f}")
    mu_tau50 = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_B, 1.0, 0.50, SC.DELTA)
    print("B2  posterior tau=0.05 =", vec(mu))
    print("B2  posterior tau=0.50 =", vec(mu_tau50))
    print("B2  max|diff|          =", f"{np.max(np.abs(mu - mu_tau50)):.3e}")


def audit_C():
    hr("CASE C — limits & nulls")
    pi = equilibrium_prior(SC.SIGMA_2, SC.W_MKT_2, delta=SC.DELTA)
    mu_c1 = black_litterman(SC.SIGMA_2, SC.W_MKT_2, SC.TILTS_C1, 1e-6, SC.TAU, SC.DELTA)
    print("C1  ||mu - pi|| (c=1e-6)  =", f"{np.linalg.norm(mu_c1 - pi):.6e}")
    mu_c2 = black_litterman(SC.SIGMA_2, SC.W_MKT_2, np.zeros(2), 1.0, SC.TAU, SC.DELTA)
    print("C2  mu (no views)         =", vec(mu_c2))
    print("C2  pi                    =", vec(pi))
    print("C2  mu == pi exactly      =", bool(np.array_equal(mu_c2, pi)))

    from app.optimizer import optimize_portfolio, Objective, ProfileConstraints
    X = SC.synthetic_G5_returns()
    Xc = X - X.mean(axis=0, keepdims=True)
    sigma = (Xc.T @ Xc) / X.shape[0] * 252
    mu = X.mean(axis=0) * 252
    betas = np.array([1.1, 0.9, 1.2, 0.4, 1.0])
    c = ProfileConstraints(max_position_pct=0.40, min_position_pct=0.02, max_beta=1.5)
    r = optimize_portfolio(SC.G5_TICKERS, mu, sigma, betas, Objective.MAX_SHARPE, c)
    print("C3  historical weights    =", r.weights)
    print("C3  ret/vol/sharpe/beta   =",
          f"{r.portfolio_return}/{r.portfolio_volatility}/{r.sharpe_ratio}/{r.beta}")


def audit_D():
    hr("CASE D — shrinkage exact null (N=2)")
    X = SC.synthetic_D_returns()
    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / X.shape[0] * 252
    sigma, alpha, n = ledoit_wolf_constant_correlation(X)
    print("D1  n_days                =", n)
    print("D1  alpha                 =", f"{alpha:.8f}")
    print("D1  max|Sigma_shrunk - S| =", f"{np.max(np.abs(sigma - sample)):.3e}")


def audit_E():
    hr("CASE E — shrinkage direction (5 assets x 40 obs)")
    X = SC.synthetic_E_returns()
    n = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / X.shape[0]
    std = np.sqrt(np.diag(sample))
    r_sample = sample / np.outer(std, std)
    rbar = (r_sample.sum() - n) / (n * (n - 1))
    sigma, alpha, ndays = ledoit_wolf_constant_correlation(X)
    sigma_daily = sigma / 252
    std_sh = np.sqrt(np.diag(sigma_daily))
    r_shrunk = sigma_daily / np.outer(std_sh, std_sh)
    print("E1  n_days                =", ndays)
    print("E1  alpha                 =", f"{alpha:.8f}")
    print("E1  rbar                  =", f"{rbar:.8f}")
    print("E2  variances preserved max diff =",
          f"{np.max(np.abs(np.diag(sigma_daily) - np.diag(sample))):.3e}")
    print("E2  pairwise  sample_corr  ->  shrunk_corr  (rbar =", f"{rbar:.4f})")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"      ({i},{j})  {r_sample[i, j]:+.6f}  ->  {r_shrunk[i, j]:+.6f}")


# ── Live case F ──────────────────────────────────────────────────────────────

G10 = ["AAPL", "JPM", "JNJ", "XOM", "WMT", "BND", "TLT"]


def audit_F():
    hr("CASE F — live equilibrium sanity, G10 basket " + str(G10))
    base = os.getenv("BASE_URL", "http://localhost:8000")

    # (a) Local equilibrium: fetch prices, shrink, caps, pi, Sigma*w ordering.
    try:
        from app.data_ingest import fetch_prices, compute_log_returns
        from app.main import _get_market_caps
        prices, _ = fetch_prices(G10, include_hedges=False, return_volumes=True)
        returns = compute_log_returns(prices)
        available = [t for t in G10 if t in returns.columns]
        window = returns[available].dropna().tail(5 * 252)
        sigma, alpha, ndays = ledoit_wolf_constant_correlation(window.values)
        caps, note = _get_market_caps(available)
        w_mkt = caps / caps.sum()
        sigw = sigma @ w_mkt
        pi = equilibrium_prior(sigma, caps, delta=2.5)
        print(f"F1  local: n_days={ndays}  alpha={alpha:.6f}  caps_note={note}")
        print(f"    {'ticker':<6}{'w_mkt':>12}{'(Sigma*w)_i':>14}{'pi_i':>14}")
        for i, t in enumerate(available):
            print(f"    {t:<6}{w_mkt[i]:>12.6f}{sigw[i]:>14.8f}{pi[i]:>14.8f}")
    except Exception as exc:
        print(f"F1  local equilibrium unavailable: {exc}")

    # (b) Running server round-trip.
    try:
        import httpx
        with httpx.Client(timeout=180.0) as client:
            client.get(f"{base}/health")
            resp = client.post(f"{base}/api/optimize", json={
                "tickers": G10,
                "objective": "max_sharpe",
                "return_model": "black_litterman",
                "apply_outlook": True,
                "view_confidence": 1.0,
            })
            data = resp.json()
        print(f"F1  server {base} status={resp.status_code}")
        print(f"    return_model_used={data.get('return_model_used')}  "
              f"n_days={data.get('n_days')}  shrinkage_alpha={data.get('shrinkage_alpha')}")
        print(f"    weights={data.get('weights')}")
        print(f"    constraints_note={data.get('constraints_note')}")
    except Exception as exc:
        print(f"F1  server round-trip skipped ({base}): {exc}")


def main():
    print("BLACK-LITTERMAN / LEDOIT-WOLF AUDIT — raw computed values")
    audit_A()
    audit_B()
    audit_C()
    audit_D()
    audit_E()
    audit_F()
    print()


if __name__ == "__main__":
    main()
