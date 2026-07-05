"""
Acceptance tests for the fit-score redesign: Growth and Income profiles must
produce materially different rankings on the same universe.

Uses a frozen fixture of the real B20 factor values (z-scores, beta, vol,
sharpe, yield) so the test is deterministic and network-free. Each row is
[z_momentum, z_quality, z_value, z_low_vol, z_yield, beta, volatility, sharpe,
dividend_yield], captured from the live pipeline on the B20 basket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.screener import compute_fit_scores

# ticker -> factor row
B20_FIXTURE = {
    "AAPL": [0.7848, 0.9582, -0.1921, 0.0371, -0.9617, 0.8637, 0.2412, 1.6557, 0.35],
    "MSFT": [-1.1571, -1.8499, 0.7088, -0.2757, -0.5616, 0.7925, 0.273, -0.8177, 0.93],
    "NVDA": [1.1372, -0.1441, 0.1816, -1.0538, -0.8513, 1.8616, 0.352, 0.6848, 0.51],
    "AMZN": [0.0282, -0.5701, 0.0754, -0.6406, -1.2031, 1.4284, 0.3101, 0.3096, 0.0],
    "TSLA": [-0.2635, -0.2414, -1.5266, -2.0056, -1.2031, 2.1092, 0.4487, 0.5991, 0.0],
    "JPM": [-0.448, -0.1007, 1.7925, 0.2267, 0.0317, 0.8198, 0.222, 0.7231, 1.79],
    "V": [-1.342, -0.7821, 0.0807, 0.2951, -0.6926, 0.4347, 0.2151, 0.1229, 0.74],
    "JNJ": [0.5001, 2.514, 0.1423, 0.6307, 0.2042, -0.0985, 0.181, 3.0261, 2.04],
    "UNH": [0.2256, -0.1095, 0.0527, -1.608, 0.3008, 0.549, 0.4083, 0.7153, 2.18],
    "XOM": [0.9436, 0.2732, 0.7273, -0.0058, 0.8734, -0.381, 0.2456, 1.0524, 3.01],
    "WMT": [-0.3487, -0.279, -0.27, 0.0121, -0.5892, -0.0404, 0.2438, 0.566, 0.89],
    "KO": [-0.3332, 0.3256, 0.4195, 0.7245, 0.5353, -0.2808, 0.1714, 1.0985, 2.52],
    "PG": [-1.3982, -1.1252, 0.8299, 0.5228, 0.7354, -0.063, 0.1919, -0.1793, 2.81],
    "CAT": [3.0109, 1.8585, -0.5255, -1.2585, -0.734, 1.691, 0.3728, 2.4487, 0.68],
    "HON": [-0.0568, -0.9432, 1.349, -0.093, 1.6529, 0.7528, 0.2545, -0.019, 4.14],
    "NEE": [-0.1702, 0.1553, 0.7979, 0.1449, 0.7423, 0.1271, 0.2303, 0.9486, 2.82],
    "BND": [-0.7099, 0.0719, -1.682, 2.0434, 1.515, 0.0867, 0.0375, 0.875, 3.94],
    "TLT": [-0.7903, -0.7987, -1.682, 1.4716, 1.9358, 0.1303, 0.0956, 0.1082, 4.55],
    "GLD": [0.2029, -0.0841, -1.682, -0.3452, -1.2031, 0.6384, 0.2801, 0.7377, 0.0],
    "SPY": [0.1846, 0.8714, 0.4026, 1.1771, -0.5271, 1.0, 0.1255, 1.5793, 0.98],
}
COLS = ["z_momentum", "z_quality", "z_value", "z_low_vol", "z_yield",
        "beta", "volatility", "sharpe", "dividend_yield"]


def _candidates():
    df = pd.DataFrame.from_dict(B20_FIXTURE, orient="index", columns=COLS)
    df.index.name = "ticker"
    return df.reset_index()


def _top5(goal, risk, horizon):
    fit = compute_fit_scores(_candidates(), goal=goal, risk_score=risk,
                             time_horizon_years=horizon)
    return fit


GROWTH = ("Growth", 95, 30.0)
INCOME = ("Income", 20, 5.0)


def test_growth95_income20_top5_overlap_at_most_2():
    g = list(_top5(*GROWTH).head(5)["ticker"])
    i = list(_top5(*INCOME).head(5)["ticker"])
    overlap = set(g) & set(i)
    assert len(overlap) <= 2, f"top-5s too similar: {overlap} (Growth {g} / Income {i})"


def test_growth95_top5_has_two_high_beta_names():
    g = _top5(*GROWTH).head(5)
    high_beta = g[g["beta"].abs() > 1.1]["ticker"].tolist()
    assert len(high_beta) >= 2, f"Growth-95 top5 lacks high-beta names: {high_beta}"


def test_income20_top5_median_yield_above_universe_median():
    fit = _top5(*INCOME)
    top5_med = fit.head(5)["dividend_yield"].median()
    universe_med = fit["dividend_yield"].median()
    assert top5_med > universe_med, f"Income-20 top5 median yield {top5_med} <= universe {universe_med}"


def test_nvda_rank_flips_between_profiles():
    g = _top5(*GROWTH).reset_index(drop=True)
    i = _top5(*INCOME).reset_index(drop=True)
    nvda_growth_rank = g.index[g["ticker"] == "NVDA"][0]
    nvda_income_rank = i.index[i["ticker"] == "NVDA"][0]
    # NVDA ranks strictly better (lower index) under Growth than under Income.
    assert nvda_growth_rank < nvda_income_rank, (
        f"NVDA did not flip: Growth rank {nvda_growth_rank}, Income rank {nvda_income_rank}"
    )
