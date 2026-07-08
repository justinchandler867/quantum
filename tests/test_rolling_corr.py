"""
CORRELATION_COLUMN_SPEC.md §B — rolling stock-bond correlation visual.

Backend: stress-window run extraction. Frontend: sleeve bucketing (GAP-A rule)
executed under node.
"""
import os
import re
import shutil
import subprocess

import pytest

from app.portfolio_series import contiguous_true_ranges

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
NODE = shutil.which("node")


# ── stress-window ranges ──────────────────────────────────────────────────────
def test_contiguous_true_ranges_basic():
    flags = [False, True, True, False, True]
    labels = ["d0", "d1", "d2", "d3", "d4"]
    assert contiguous_true_ranges(flags, labels) == [["d1", "d2"], ["d4", "d4"]]


def test_contiguous_true_ranges_never_merges_across_gap():
    flags = [True, False, True]
    labels = ["a", "b", "c"]
    assert contiguous_true_ranges(flags, labels) == [["a", "a"], ["c", "c"]]


def test_contiguous_true_ranges_all_and_none():
    assert contiguous_true_ranges([True, True], ["x", "y"]) == [["x", "y"]]
    assert contiguous_true_ranges([False, False], ["x", "y"]) == []


# ── fallback headers (empty-sleeve, GAP-C) ────────────────────────────────────
def test_fallback_headers_distinct_and_correct():
    html = open(FRONTEND, encoding="utf-8").read()
    suffix = " — showing the market-level pair (SPY vs TLT)."
    no_bond = "Your portfolio has no bond sleeve"
    no_equity = "Your portfolio has no equity sleeve"
    assert no_bond in html and no_equity in html      # both rendered
    assert no_bond != no_equity                        # distinct
    assert suffix in html                              # shared labeled-fallback suffix


# ── sleeve bucketing (GAP-A) under node ───────────────────────────────────────
@pytest.mark.skipif(not NODE, reason="node not available")
def test_sleeve_bucketing_rule():
    html = open(FRONTEND, encoding="utf-8").read()
    m = re.search(r"function sleeveOf\(.*?\n\}", html, re.DOTALL)
    assert m, "sleeveOf not found"
    asserts = """
      const a=(x,y)=>{if(x!==y){console.error('FAIL',x,'!=',y);process.exit(1);}};
      a(sleeveOf("Fixed Income"),"bond");
      a(sleeveOf("Technology"),"equity");
      a(sleeveOf("Real Estate"),"equity");      // REITs are equity
      a(sleeveOf("International"),"equity");     // intl stocks are equity
      a(sleeveOf("Broad Market"),"equity");      // broad-market ETFs are equity
      a(sleeveOf("Commodity"),"excluded");       // neither sleeve
      a(sleeveOf("Commodities"),"excluded");     // sample-data spelling tolerated
      a(sleeveOf("Volatility"),"excluded");
      a(sleeveOf(null),"equity");
      console.log("OK");
    """
    r = subprocess.run([NODE, "-e", m.group(0) + "\n" + asserts], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
