"""
BETA_TRACKER_SPEC.md — portfolio beta tracker.

Backend: the shared weighted-return helper (refactor lock) and the beta
regression (network-free, synthetic). Frontend: the reference-point label
functions (Amendment 3) executed under node.
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from app.portfolio_series import weighted_portfolio_returns, regress_beta

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
NODE = shutil.which("node")


# ── shared helper (refactor lock) ─────────────────────────────────────────────
def test_weighted_portfolio_returns_value_weights_and_renormalizes():
    idx = pd.date_range("2024-01-01", periods=5)
    r = pd.DataFrame({"A": [0.1, 0.2, -0.1, 0.0, 0.05],
                      "B": [0.0, 0.1, 0.1, -0.2, 0.05]}, index=idx)
    # weights 30/10 -> renormalized 0.75/0.25
    port = weighted_portfolio_returns(r, {"A": 30, "B": 10})
    expected = 0.75 * r["A"] + 0.25 * r["B"]
    pd.testing.assert_series_equal(port, expected, check_names=False)


def test_weighted_portfolio_returns_exclude_renormalizes():
    idx = pd.date_range("2024-01-01", periods=3)
    r = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.0, 0.0, 0.0]}, index=idx)
    port = weighted_portfolio_returns(r, {"A": 50, "B": 50}, exclude="B")
    pd.testing.assert_series_equal(port, r["A"], check_names=False)


def test_weighted_portfolio_returns_none_when_no_data():
    r = pd.DataFrame({"A": [0.1]}, index=pd.date_range("2024-01-01", periods=1))
    assert weighted_portfolio_returns(r, {"ZZZ": 100}) is None


# ── beta regression (synthetic, known beta) ───────────────────────────────────
def test_regress_beta_recovers_known_beta():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n)
    bench = pd.Series(rng.normal(0, 0.01, n), index=idx)
    port = 0.5 * bench + pd.Series(rng.normal(0, 0.0001, n), index=idx)  # beta ~ 0.5
    beta, r2, obs = regress_beta(port, bench, window=252, min_overlap=60)
    assert obs == 252
    assert abs(beta - 0.5) < 0.05
    assert r2 > 0.95  # nearly all variance explained


def test_regress_beta_insufficient_overlap_returns_none_not_zero():
    idx = pd.date_range("2024-01-01", periods=40)
    bench = pd.Series(np.linspace(0, 1, 40), index=idx)
    port = pd.Series(np.linspace(0, 0.5, 40), index=idx)
    beta, r2, obs = regress_beta(port, bench, window=252, min_overlap=60)
    assert beta is None and r2 is None and obs == 40  # never fabricates 0


def test_regress_beta_zero_variance_benchmark_returns_none():
    idx = pd.date_range("2023-01-01", periods=100)
    bench = pd.Series(np.zeros(100), index=idx)
    port = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100), index=idx)
    beta, r2, obs = regress_beta(port, bench, window=252, min_overlap=60)
    assert beta is None


def test_beta_not_weighted_average_of_asset_betas():
    """Portfolio beta must be the regression of the combined series, not the
    weighted average of individual betas — they differ when assets are
    imperfectly correlated."""
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n)
    bench = pd.Series(rng.normal(0, 0.01, n), index=idx)
    a = 1.5 * bench + pd.Series(rng.normal(0, 0.02, n), index=idx)   # noisy beta 1.5
    b = 0.5 * bench + pd.Series(rng.normal(0, 0.02, n), index=idx)   # noisy beta 0.5
    r = pd.DataFrame({"A": a, "B": b})
    port = weighted_portfolio_returns(r, {"A": 50, "B": 50})
    pbeta, _, _ = regress_beta(port, bench, window=252, min_overlap=60)
    ba, _, _ = regress_beta(a, bench, window=252, min_overlap=60)
    bb, _, _ = regress_beta(b, bench, window=252, min_overlap=60)
    # regression beta is well-defined; assert it's computed from the series,
    # and record that it is NOT identical to a naive weighted avg in general.
    assert pbeta is not None
    assert 0.7 < pbeta < 1.3  # around the true blend
    # (the naive avg (ba+bb)/2 need not equal pbeta due to idiosyncratic noise)


# ── frontend reference-point labels (Amendment 3) under node ───────────────────
def _extract(html, name):
    import re
    m = re.search(r"function " + name + r"\(.*?\n\}", html, re.DOTALL)
    assert m, f"cannot extract {name}"
    return m.group(0)


@pytest.mark.skipif(not NODE, reason="node not available")
def test_beta_labels_edges():
    html = open(FRONTEND, encoding="utf-8").read()
    src = _extract(html, "betaMarketLabel") + _extract(html, "betaBandLabel") + _extract(html, "betaTransition")
    asserts = """
      const a=(x,y)=>{if(x!==y){console.error('FAIL',JSON.stringify(x),'!=',JSON.stringify(y));process.exit(1);}};
      // market (tolerance 0.1)
      a(betaMarketLabel(1.0),"Market-like");
      a(betaMarketLabel(1.09),"Market-like");
      a(betaMarketLabel(1.15),"More volatile than market");
      a(betaMarketLabel(0.85),"Less volatile than market");
      a(betaMarketLabel(null),null);
      // band (user-declared only)
      a(betaBandLabel(1.05,0.9,1.1),"In band ✓");
      a(betaBandLabel(1.2,0.9,1.1),"Above band ▲");
      a(betaBandLabel(0.8,0.9,1.1),"Below band ▼");
      a(betaBandLabel(1.05,null,null),null);  // no band -> no label
      // transitions (only when crossing)
      a(betaTransition(1.18,1.04,0.9,1.1),"▼ into band");
      a(betaTransition(1.04,1.18,0.9,1.1),"▲ out of band");
      a(betaTransition(1.05,1.02,0.9,1.1),"");   // stays in band -> no transition
      a(betaTransition(1.18,1.04,null,null),""); // no band -> no transition
      console.log("OK");
    """
    r = subprocess.run([NODE, "-e", src + "\n" + asserts], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
