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
