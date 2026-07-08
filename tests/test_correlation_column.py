"""
CORRELATION_COLUMN_SPEC.md Amendment 4 + INTERFACE_VERDICTS_SPEC.md Amendment 3.

Covers the discovery correlation picker-column wiring and the SHARPE (1Y)
retrofit to reference-point vocabulary. Frontend is the single-file no-build
app, so structural facts are asserted against the HTML text and the two pure
label functions are executed under node for real edge coverage.
"""
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FRONTEND = os.path.join(ROOT, "static", "quantex.html")
NODE = shutil.which("node")

BANNED = ["diversifier", "diversification benefit", "hedge", "complement", "fits well"]


@pytest.fixture(scope="module")
def html():
    with open(FRONTEND, encoding="utf-8") as fh:
        return fh.read()


def _extract(html, name):
    m = re.search(r"function " + name + r"\(.*?\n\}", html, re.DOTALL)
    assert m, f"could not extract {name}"
    return m.group(0)


# ── chip: default renders no column ───────────────────────────────────────────
def test_chip_default_label_dash(html):
    assert 'Corr vs: "+(corrSel||"—")+" ▾' in html


def test_column_gated_on_selection(html):
    # header and cell only render when corrSel is set
    assert 'corrSel&&e("th"' in html
    assert "corrSel&&(()=>{" in html
    # footnote likewise gated
    assert "corrSel&&e(\"tr\",null,e(\"td\",{colSpan:22" in html


def test_empty_portfolio_option_disabled(html):
    assert "const n=Object.keys(ports[pn]||{}).length;const disabled=n<2;" in html
    assert '(disabled?" (empty)":"")' in html


# ── reference-point vocabulary present (Amendment 3) ──────────────────────────
def test_corr_labels_present(html):
    for lbl in ['"Moves with "+ref', '"Moves against "+ref', '"Unlinked"']:
        assert lbl in html


def test_sharpe_reference_labels_present(html):
    for lbl in ['"Beat market"', '"Trailed market"', '"Lost vs cash"', '"Reference"']:
        assert lbl in html


def test_footnotes_present(html):
    assert ("Correlation of daily returns with the selected portfolio, in calm and "
            "stressed markets. Historical co-movement — not a forecast.") in html
    assert "The label compares each name's Sharpe to SPY's over the same window" in html


# ── prohibited/advice vocabulary absent in the NEW correlation column code ────
def test_no_advice_vocab_in_corr_column(html):
    corr_fn = _extract(html, "corrLabel")
    # the corr cell IIFE
    cell = re.search(r"corrSel&&\(\(\)=>\{.*?\}\)\(\),", html, re.DOTALL)
    assert cell, "corr cell block not found"
    footnote = ("Correlation of daily returns with the selected portfolio, in calm and "
                "stressed markets. Historical co-movement — not a forecast.")
    region = (corr_fn + cell.group(0) + footnote).lower()
    for w in BANNED:
        assert w not in region, f"advice/prohibited word '{w}' in correlation column code"


# ── node edge tests on the actual shipped functions ───────────────────────────
def _run_node(fn_src, asserts):
    script = fn_src + "\nconst a=(x,y)=>{if(x!==y){console.error('FAIL',JSON.stringify(x),'!=',JSON.stringify(y));process.exit(1);}};\n" + asserts + "\nconsole.log('OK');"
    r = subprocess.run([NODE, "-e", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


@pytest.mark.skipif(not NODE, reason="node not available")
def test_corr_label_edges(html):
    _run_node(_extract(html, "corrLabel"), """
        a(corrLabel(0.2,"A"),"Moves with A");
        a(corrLabel(0.199,"A"),"Unlinked");
        a(corrLabel(-0.2,"A"),"Moves against A");
        a(corrLabel(-0.199,"A"),"Unlinked");
        a(corrLabel(0,"A"),"Unlinked");
        a(corrLabel(0.95,"B"),"Moves with B");
        a(corrLabel(null,"A"),null);
    """)


@pytest.mark.skipif(not NODE, reason="node not available")
def test_sharpe_market_label_edges(html):
    _run_node(_extract(html, "sharpeMarketLabel"), """
        a(sharpeMarketLabel(1.0,1.2,true),"Reference");
        a(sharpeMarketLabel(-0.01,1.2,false),"Lost vs cash");
        a(sharpeMarketLabel(1.2,1.2,false),"Beat market");
        a(sharpeMarketLabel(1.3,1.2,false),"Beat market");
        a(sharpeMarketLabel(1.19,1.2,false),"Trailed market");
        a(sharpeMarketLabel(0,1.2,false),"Trailed market");
        a(sharpeMarketLabel(1.1,null,false),null);
        a(sharpeMarketLabel(null,1.2,false),null);
    """)
