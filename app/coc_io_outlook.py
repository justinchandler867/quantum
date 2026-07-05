"""
2026 Co-CIO Outlook — sector-level expected-return tilts for the optimizer.

When the user toggles "2026 Co-CIO Outlook" mode, each ticker's expected
return gets adjusted by its sector's tilt before optimization runs. Same
constraints, same objectives — only the input μ vector changes.

These tilts encode a curated thesis. They are NOT investment advice; the UI
should surface them alongside the source attribution so users treat them as
one view among many. When the thesis changes, update SECTOR_TILTS_2026 and
OUTLOOK_METADATA.last_updated together.
"""
from __future__ import annotations

import numpy as np


# Annualized expected-return adjustment per sector (decimal).
# +0.015 = +1.5 percentage points, -0.010 = -1.0 pp.
# Sectors not listed here get no adjustment.
SECTOR_TILTS_2026: dict[str, float] = {
    "Technology":          -0.010,
    "Financial Services":  +0.015,
    "Healthcare":          +0.010,
    "Industrials":         +0.005,
    "Consumer Cyclical":   -0.005,
    "Consumer Defensive":  +0.005,
    "Fixed Income":        +0.005,
}


OUTLOOK_METADATA = {
    "name": "2026 Co-CIO Outlook",
    "source": (
        "Based on commentary from Fort Washington Co-CIOs "
        "Christopher Shipley & Brendan White, 2025-2026"
    ),
    "thesis_summary": (
        "Late-cycle posture with a defensive rotation. Financials carry "
        "the largest upward tilt as banks benefit from the resettled rate "
        "environment, with healthcare close behind on structural demand. "
        "Technology and consumer cyclical are tilted down on extended "
        "multiples and softening consumer spending. Fixed income gets a "
        "modest upward tilt after the 2022-2024 yield reset made bonds "
        "reasonable again on a real-yield basis."
    ),
    "last_updated": "2026-05",
    "tilts": SECTOR_TILTS_2026,
}


def apply_outlook_tilts(
    expected_returns: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    tilts: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Apply sector-level return tilts to a vector of expected returns.

    Pure function — does not mutate `expected_returns`. Tickers with no
    sector tag (or whose sector has no tilt) receive no adjustment.

    Args:
        expected_returns: Annualized expected returns aligned with `tickers` (n,).
        tickers: Ticker symbols in the same order as `expected_returns`.
        sectors: {ticker: sector_string or None}.
        tilts: Sector → tilt mapping. Defaults to SECTOR_TILTS_2026.

    Returns:
        New ndarray with tilts applied.
    """
    if tilts is None:
        tilts = SECTOR_TILTS_2026

    out = expected_returns.astype(float, copy=True)
    for i, t in enumerate(tickers):
        sec = sectors.get(t)
        if sec is None:
            continue
        delta = tilts.get(sec, 0.0)
        if delta:
            out[i] += delta
    return out


# ── Full regime stress-test parameters ────────────────────────────────────────
#
# The regime is a forward-looking scenario, not a historical fit. Returns get
# the SECTOR_TILTS_2026 nudges; volatilities are scaled per sector; and the
# stock-bond correlation is overridden to reflect Shipley/White's commentary
# that the post-2022 regime has the two asset classes co-moving rather than
# diversifying. When all three combine, diversified portfolios should look
# meaningfully worse — that's the point of the stress test.

# Sectors considered "equity" for the purpose of the stock-bond correlation
# override. Fixed Income paired with any of these takes the regime correlation
# in place of the historical one; pairs that include Commodity / Volatility /
# International / Unknown / Fixed Income on both legs are left alone.
_EQUITY_SECTORS = frozenset({
    "Technology",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Energy",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Utilities",
    "Real Estate",
    "Communication Services",
    "Basic Materials",
    "Broad Market",
})


REGIME_2026 = {
    "vol_multipliers": {
        "Technology":             1.10,
        "Financial Services":     1.05,
        "Healthcare":             0.95,
        "Industrials":            1.00,
        "Energy":                 1.15,
        "Consumer Cyclical":      1.05,
        "Consumer Defensive":     0.95,
        "Utilities":              0.95,
        "Real Estate":            1.05,
        "Communication Services": 1.05,
        "Basic Materials":        1.05,
        "Fixed Income":           1.10,
        "Broad Market":           1.00,
        "International":          1.05,
        "Commodity":              1.10,
        "Volatility":             1.00,
        "Unknown":                1.00,
    },
    "stock_bond_correlation": 0.20,
    "description": (
        "Equities and credit spreads under late-cycle pressure; "
        "stock-bond correlation positive (diversification weakened); "
        "energy vol elevated; defensives slightly damped."
    ),
}


def apply_regime(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str],
    sectors: dict[str, str | None],
    regime: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the full 2026 Co-CIO Outlook regime to (μ, Σ).

    Three transformations:
      1. Sector tilts on expected returns (SECTOR_TILTS_2026).
      2. Sector-specific vol scaling: σ'_i = σ_i × m(sector_i).
      3. Stock-bond correlation override: pairs of (equity, fixed income)
         have their correlation replaced with regime['stock_bond_correlation'].
         All other correlations preserved.

    Reconstruction: Σ' = D' R' D' where D' = diag(σ'), R' is the modified
    correlation matrix.

    Pure function — does not mutate inputs.
    """
    if regime is None:
        regime = REGIME_2026

    new_returns = apply_outlook_tilts(expected_returns, tickers, sectors)

    # Decompose Σ into σ × ρ × σ.
    vols = np.sqrt(np.clip(np.diag(cov_matrix), 0.0, None))
    safe = np.where(vols > 1e-12, vols, 1e-12)
    corr = cov_matrix / np.outer(safe, safe)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Scale vols by sector.
    mults = regime["vol_multipliers"]
    new_vols = vols.copy()
    for i, t in enumerate(tickers):
        sec = sectors.get(t) or "Unknown"
        new_vols[i] *= mults.get(sec, 1.0)

    # Override stock-bond correlations.
    sb_rho = float(regime["stock_bond_correlation"])
    new_corr = corr.copy()
    n = len(tickers)
    for i in range(n):
        sec_i = sectors.get(tickers[i]) or "Unknown"
        for j in range(i + 1, n):
            sec_j = sectors.get(tickers[j]) or "Unknown"
            stock_bond = (
                (sec_i == "Fixed Income" and sec_j in _EQUITY_SECTORS) or
                (sec_j == "Fixed Income" and sec_i in _EQUITY_SECTORS)
            )
            if stock_bond:
                new_corr[i, j] = sb_rho
                new_corr[j, i] = sb_rho

    new_cov = np.outer(new_vols, new_vols) * new_corr
    return new_returns, new_cov
