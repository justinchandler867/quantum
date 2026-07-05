"""
Verification script for app/coc_io_outlook.py.

Builds a 5-ticker portfolio with synthetic expected returns and runs the
optimizer once without outlook tilts and once with, then prints the
before/after expected-return vectors and the resulting weights so the
direction of the tilt can be eyeballed.

Expected directional shifts when outlook is applied:
  AAPL  (Technology)         → underweight  (-1.0% tilt)
  JPM   (Financial Services) → overweight   (+1.5% tilt)
  JNJ   (Healthcare)         → overweight   (+1.0% tilt)
  XOM   (Energy)             → unchanged    (no tilt)
  BND   (Fixed Income)       → overweight   (+0.5% tilt)
"""
from __future__ import annotations

import numpy as np

from app.coc_io_outlook import apply_outlook_tilts, SECTOR_TILTS_2026
from app.optimizer import optimize_portfolio, Objective, ProfileConstraints


TICKERS = ["AAPL", "JPM", "JNJ", "XOM", "BND"]
SECTORS = {
    "AAPL": "Technology",
    "JPM":  "Financial Services",
    "JNJ":  "Healthcare",
    "XOM":  "Energy",
    "BND":  "Fixed Income",
}

# Synthetic baseline: similar 8% expected returns, so tilts dominate the ranking.
EXP_RET = np.array([0.080, 0.080, 0.080, 0.080, 0.080])

# Synthetic vols (annualized) and a mild correlation structure so the optimizer
# has a non-degenerate problem to solve.
VOLS = np.array([0.25, 0.22, 0.18, 0.30, 0.06])  # Tech/Energy hottest, BND coldest
CORR = np.array([
    [1.00, 0.55, 0.45, 0.40, 0.05],
    [0.55, 1.00, 0.50, 0.35, 0.10],
    [0.45, 0.50, 1.00, 0.30, 0.10],
    [0.40, 0.35, 0.30, 1.00, 0.00],
    [0.05, 0.10, 0.10, 0.00, 1.00],
])
COV = np.outer(VOLS, VOLS) * CORR

# Synthetic betas — equity-like for the first four, ~0 for the bond ETF.
BETAS = np.array([1.20, 1.10, 0.85, 1.00, 0.05])

CONSTRAINTS = ProfileConstraints(
    max_position_pct=0.40,
    min_position_pct=0.02,
    max_beta=1.5,
)


def fmt_pct(x: float, width: int = 7) -> str:
    return f"{x * 100:+{width}.2f}%"


def fmt_weight(x: float, width: int = 6) -> str:
    return f"{x * 100:{width}.2f}%"


def run():
    print("=" * 72)
    print("2026 Co-CIO Outlook tilt verification")
    print("=" * 72)

    tilted = apply_outlook_tilts(EXP_RET, TICKERS, SECTORS)

    print("\nExpected returns (annualized):")
    print(f"  {'Ticker':<6} {'Sector':<20} {'Baseline':>10} {'Tilt':>8} {'Tilted':>10}")
    print(f"  {'-'*6} {'-'*20} {'-'*10} {'-'*8} {'-'*10}")
    for i, t in enumerate(TICKERS):
        sec = SECTORS[t]
        tilt = SECTOR_TILTS_2026.get(sec, 0.0)
        print(f"  {t:<6} {sec:<20} {fmt_pct(EXP_RET[i]):>10} {fmt_pct(tilt):>8} {fmt_pct(tilted[i]):>10}")

    print("\nRunning optimize_portfolio(objective=MAX_SHARPE)…")

    base_res = optimize_portfolio(
        tickers=TICKERS,
        expected_returns=EXP_RET,
        cov_matrix=COV,
        betas=BETAS,
        objective=Objective.MAX_SHARPE,
        constraints=CONSTRAINTS,
    )
    tilt_res = optimize_portfolio(
        tickers=TICKERS,
        expected_returns=tilted,
        cov_matrix=COV,
        betas=BETAS,
        objective=Objective.MAX_SHARPE,
        constraints=CONSTRAINTS,
    )

    print("\nOptimal weights:")
    print(f"  {'Ticker':<6} {'Sector':<20} {'Baseline':>10} {'Outlook':>10} {'Δ':>10}")
    print(f"  {'-'*6} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for t in TICKERS:
        sec = SECTORS[t]
        w_b = base_res.weights[t]
        w_o = tilt_res.weights[t]
        delta = w_o - w_b
        arrow = "↑" if delta > 1e-4 else ("↓" if delta < -1e-4 else "·")
        print(f"  {t:<6} {sec:<20} {fmt_weight(w_b):>10} {fmt_weight(w_o):>10} {fmt_pct(delta):>9} {arrow}")

    print("\nPortfolio metrics:")
    print(f"  {'':<18} {'Baseline':>10} {'Outlook':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10}")
    print(f"  {'Expected return':<18} {fmt_pct(base_res.portfolio_return):>10} {fmt_pct(tilt_res.portfolio_return):>10}")
    print(f"  {'Volatility':<18} {fmt_pct(base_res.portfolio_volatility):>10} {fmt_pct(tilt_res.portfolio_volatility):>10}")
    print(f"  {'Sharpe ratio':<18} {base_res.sharpe_ratio:>10.3f} {tilt_res.sharpe_ratio:>10.3f}")
    print(f"  {'Portfolio beta':<18} {base_res.beta:>10.3f} {tilt_res.beta:>10.3f}")

    # Directional checks
    print("\nDirectional checks:")
    checks = [
        ("AAPL underweighted", tilt_res.weights["AAPL"] < base_res.weights["AAPL"]),
        ("JPM  overweighted ", tilt_res.weights["JPM"]  > base_res.weights["JPM"]),
        ("JNJ  overweighted ", tilt_res.weights["JNJ"]  > base_res.weights["JNJ"]),
        # BND can hit the max_position_pct cap and stay pinned; >= is the
        # correct check (a positive tilt should never reduce its weight).
        ("BND  >= baseline ", tilt_res.weights["BND"]  >= base_res.weights["BND"] - 1e-9),
    ]
    all_ok = True
    for label, ok in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{mark}] {label}")

    # Pure-function check
    original = EXP_RET.copy()
    _ = apply_outlook_tilts(EXP_RET, TICKERS, SECTORS)
    purity_ok = np.allclose(EXP_RET, original)
    print(f"  [{'PASS' if purity_ok else 'FAIL'}] apply_outlook_tilts does not mutate input")

    print("\n" + ("All checks passed." if all_ok and purity_ok else "Some checks FAILED."))


if __name__ == "__main__":
    run()
