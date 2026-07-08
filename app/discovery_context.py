"""
Discovery Column: "Corr (calm / stress)"
Implements CORRELATION_COLUMN_SPEC.md §A (Discovery column) ONLY.
§B (rolling stock-bond visual) is explicitly out of scope for this module.

Per §A3: this is 2-series Pearson correlation (candidate vs. value-weighted
portfolio return series). No Ledoit-Wolf shrinkage — shrinkage is for
matrices, not a single pairwise correlation. Do not "improve" this.

Reuses `data_ingest.identify_stress_windows`'s stress-day mask (surfaced by
the caller via `_store["stress_mask"]`) rather than reimplementing
stress-window identification.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import (
    MIN_STRESS_DAYS,
    HIGH_CORR_THRESHOLD,
    HIGH_CORR_NORMAL_THRESHOLD,
    NEGATIVE_CORR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# The correlation-flip tell (§A2): stress cell gets a ⚠ marker when
# corr_stress - corr_normal >= this delta.
FLIP_DELTA = 0.20

# Overlap below this many trading days => insufficient_data (§A4).
MIN_OVERLAP_DAYS = 60


@dataclass
class CandidateContext:
    ticker: str
    corr_normal: float | None
    corr_stress: float | None
    band: str | None
    flip_flag: bool
    days_normal: int
    days_stress: int
    status: str  # "ok" | "held" | "insufficient_data" | "no_stress_window"


def band_label(corr_normal: float) -> str:
    """
    §A2 band labels, anchored to config constants (no duplicated literals).
    """
    if corr_normal >= HIGH_CORR_THRESHOLD:
        return "Near-duplicate"
    if corr_normal >= HIGH_CORR_NORMAL_THRESHOLD:
        return "Very similar"
    if corr_normal >= 0.30:
        return "Related"
    if corr_normal >= NEGATIVE_CORR_THRESHOLD:
        return "Diversifier"
    return "Hedge"


def _portfolio_returns(
    returns: pd.DataFrame,
    holdings: dict[str, float],
    exclude: str | None = None,
) -> pd.Series | None:
    """
    Value-weighted portfolio daily return series: r_P,t = sum(w_i * r_i,t).

    `holdings` maps ticker -> weight (need not already sum to 1; renormalized
    here). If `exclude` is given, that ticker is dropped and the remaining
    weights renormalized (§A1 held-candidate exclusion path).

    Returns None if fewer than 1 holding remains after exclusion/filtering,
    or none of the holdings have return data.
    """
    w = dict(holdings)
    if exclude is not None:
        w.pop(exclude, None)

    tickers = [t for t in w.keys() if t in returns.columns]
    if not tickers:
        return None

    total = sum(w[t] for t in tickers)
    if total <= 0:
        return None

    norm_w = {t: w[t] / total for t in tickers}

    sub = returns[tickers]
    weighted = sub.mul(pd.Series(norm_w))
    port_ret = weighted.sum(axis=1, skipna=False)
    return port_ret


def _aligned_pearson(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    """
    Pearson correlation of two return series over their aligned (inner-join,
    NaN-dropped) overlap. Returns (corr, n_obs). corr is None if n_obs < 2
    or either series has zero variance.
    """
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 2:
        return None, n
    x = joined.iloc[:, 0].values
    y = joined.iloc[:, 1].values
    if np.std(x) == 0 or np.std(y) == 0:
        return None, n
    corr = float(np.corrcoef(x, y)[0, 1])
    corr = max(-1.0, min(1.0, corr))
    return corr, n


def compute_candidate_context(
    ticker: str,
    returns: pd.DataFrame,
    stress_mask: pd.Series,
    holdings: dict[str, float],
    window: int = 252,
) -> CandidateContext:
    """
    Compute the Discovery-column context for one candidate ticker, per §A1.

    `holdings`: ticker -> weight, the user's active portfolio (need not be
    pre-normalized to sum 1 — normalized here and, for the held-candidate
    path, renormalized after excluding the candidate).
    `returns`: full log-returns frame (all available history), used both for
    the trailing 252-day normal window and to slice stress days.
    `stress_mask`: boolean Series aligned to `returns.index`, True = stress
    day, as produced by `data_ingest.identify_stress_windows` — reused
    as-is, not recomputed here.
    """
    is_held = ticker in holdings

    # Single-position portfolio equal to the candidate itself -> both cells "—".
    if is_held and len(holdings) == 1:
        return CandidateContext(
            ticker=ticker, corr_normal=None, corr_stress=None, band=None,
            flip_flag=False, days_normal=0, days_stress=0, status="held",
        )

    exclude = ticker if is_held else None
    port_ret_full = _portfolio_returns(returns, holdings, exclude=exclude)

    if ticker not in returns.columns or port_ret_full is None:
        return CandidateContext(
            ticker=ticker, corr_normal=None, corr_stress=None, band=None,
            flip_flag=False, days_normal=0, days_stress=0,
            status="held" if is_held else "insufficient_data",
        )

    cand_ret_full = returns[ticker]

    # ── Normal regime: trailing `window` aligned days (same convention as
    # compute_normal_correlation) ──────────────────────────────────────────
    cand_tail = cand_ret_full.tail(window)
    port_tail = port_ret_full.tail(window)
    corr_normal, days_normal = _aligned_pearson(cand_tail, port_tail)

    if corr_normal is None or days_normal < MIN_OVERLAP_DAYS:
        return CandidateContext(
            ticker=ticker, corr_normal=None, corr_stress=None, band=None,
            flip_flag=False, days_normal=days_normal, days_stress=0,
            status="held" if is_held else "insufficient_data",
        )

    # ── Stress regime: data_ingest's stress-window mask, restricted to days
    # where the candidate itself has data (its history may be shorter than
    # the full frame) ──────────────────────────────────────────────────────
    mask = stress_mask.reindex(returns.index).fillna(False)
    cand_stress = cand_ret_full[mask.reindex(cand_ret_full.index).fillna(False)]
    port_stress = port_ret_full[mask.reindex(port_ret_full.index).fillna(False)]

    corr_stress, days_stress = _aligned_pearson(cand_stress, port_stress)

    if corr_stress is None or days_stress < MIN_STRESS_DAYS:
        band = band_label(corr_normal)
        return CandidateContext(
            ticker=ticker, corr_normal=round(corr_normal, 4), corr_stress=None,
            band=band, flip_flag=False, days_normal=days_normal, days_stress=days_stress,
            status="no_stress_window",
        )

    band = band_label(corr_normal)
    flip_flag = (corr_stress - corr_normal) >= FLIP_DELTA

    return CandidateContext(
        ticker=ticker,
        corr_normal=round(corr_normal, 4),
        corr_stress=round(corr_stress, 4),
        band=band,
        flip_flag=flip_flag,
        days_normal=days_normal,
        days_stress=days_stress,
        status="held" if is_held else "ok",
    )
