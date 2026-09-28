"""
PORTFOLIO_RISK_SPEC §2/§3/§4/§5 + Amendment A §8 — frontend rendering.

Renders the shipped healthPanelBody / volPreviewLines under node (stub React)
with payloads produced by the real engine (app.portfolio_risk) on seeded data,
and asserts the rendered strings, label reuse, color rules and vocabulary.
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from app.config import RISK_FREE_RATE
from app.portfolio_risk import portfolio_health, weighted_yield

HERE = os.path.dirname(__file__)
FRONTEND = os.path.abspath(os.path.join(HERE, "..", "static", "quantex.html"))
NODE = shutil.which("node")
SRC = open(FRONTEND, encoding="utf-8").read()

HARNESS = r"""
const fs=require("fs");const src=fs.readFileSync(process.argv[1],"utf8");
const input=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));
function fn(name){const i=src.indexOf("function "+name+"(");if(i<0)throw new Error("missing "+name);
  const s=src.indexOf("{",src.indexOf(")",i));let d=0,j=s;
  for(;j<src.length;j++){if(src[j]==="{")d++;else if(src[j]==="}"){d--;if(d===0)break;}}return src.slice(i,j+1);}
function constLine(name){const m=src.match(new RegExp("const "+name+"=[^\\n]*"));if(!m)throw new Error("missing const "+name);return m[0].replace(/^const /,"var ");}
var e=(type,props,...children)=>({type,props:props||{},children:children.flat(Infinity)});
var ASSETS=input.assets;
eval(fn("sharpeMarketLabel"));eval(fn("sectorOf"));eval(fn("addWeights"));eval(fn("removeWeights"));
eval(fn("volPreviewLines"));
eval(constLine("pct1"));eval(constLine("pct0"));eval(constLine("signed1"));eval(constLine("AMBER"));
eval(fn("healthPanelBody"));
function text(n){if(n==null||n===false||n===true)return "";if(typeof n!=="object")return String(n);return (n.children||[]).map(text).join("\n");}
function titles(n,acc){acc=acc||[];if(n&&typeof n==="object"){if(n.props&&n.props.title)acc.push(n.props.title);(n.children||[]).forEach(c=>titles(c,acc));}return acc;}
function colors(n,acc){acc=acc||[];if(n&&typeof n==="object"){const st=n.props&&n.props.style;if(st){for(const k of["color","background"])if(st[k])acc.push(st[k]);}(n.children||[]).forEach(c=>colors(c,acc));}return acc;}
const out={};
for(const [k,c] of Object.entries(input.cases)){
  const tree=healthPanelBody(c.d,c.weights,{betaChip:e("span",null,"Portfolio β 1.05 · vs S&P 500"),allCats:true});
  out[k]={text:text(tree).split("\n").map(s=>s.trim()).filter(Boolean),titles:titles(tree),colors:[...new Set(colors(tree))]};
}
out.preview=input.previews.map(p=>volPreviewLines(p.p,p.ctx));
out.add=addWeights({A:34,B:33,C:33},"D");out.rm=removeWeights({A:25,B:25,C:25,D:25},"B");
console.log(JSON.stringify(out));
"""


def _frame():
    rng = np.random.default_rng(21)
    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    m = rng.normal(0.0005, 0.01, n)
    return pd.DataFrame({
        "NVDA": 1.8 * m + rng.normal(0, 0.02, n), "AVGO": 1.7 * m + rng.normal(0, 0.012, n),
        "MSFT": 1.0 * m + rng.normal(0, 0.008, n), "JNJ": 0.4 * m + rng.normal(0, 0.007, n),
        "TLT": -0.5 * m + rng.normal(0, 0.006, n), "SPY": m,
    }, index=idx)


ASSETS = [{"id": "NVDA", "sec": "Technology"}, {"id": "AVGO", "sec": "Technology"},
          {"id": "MSFT", "sec": "Technology"}, {"id": "JNJ", "sec": "Healthcare"},
          {"id": "TLT", "sec": "Fixed Income"}]


def _case(holdings, excluded_extra=None):
    R = _frame()
    if excluded_extra:
        holdings = {**holdings, excluded_extra: 10}
    d = portfolio_health(R, holdings, rf=RISK_FREE_RATE)
    wn = {t: w / sum(holdings.values()) for t, w in holdings.items()}
    d["yield"] = weighted_yield({t: 0.01 for t in wn if t != "NVDA"} | {"NVDA": None}, wn)
    return {"d": json.loads(json.dumps(d)), "weights": holdings}


def render():
    cases = {
        "full": _case({"NVDA": 30, "AVGO": 20, "MSFT": 20, "JNJ": 15, "TLT": 15}, excluded_extra="NEWCO"),
        "all_equity": _case({"NVDA": 45, "AVGO": 30, "MSFT": 25}),
        "single": _case({"MSFT": 100}),
        "empty": {"d": None, "weights": {}},
    }
    previews = [
        {"p": {"current": {"vol": 0.142, "beta": 1.12}, "proposed": {"vol": 0.131, "beta": 1.04}},
         "ctx": {"kind": "add", "id": "TLT", "w": 25, "n": 4}},
        {"p": {"current": {"vol": 0.142, "beta": 1.12}, "proposed": {"vol": 0.155, "beta": 1.2}},
         "ctx": {"kind": "remove", "id": "TLT", "n": 3}},
        {"p": {"current": {"vol": 0.142, "beta": None}, "proposed": {"vol": 0.15, "beta": None}},
         "ctx": {"kind": "reweight", "id": "NVDA", "from": 20, "to": 35}},
    ]
    tmp = os.path.join(HERE, "_health_ui_input.json")
    with open(tmp, "w") as fh:
        json.dump({"assets": ASSETS, "cases": cases, "previews": previews}, fh)
    try:
        r = subprocess.run([NODE, "-e", HARNESS, FRONTEND, tmp], capture_output=True, text=True)
    finally:
        os.remove(tmp)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), cases


@pytest.fixture(scope="module")
def rendered():
    if not NODE:
        pytest.skip("node not available")
    return render()


def test_full_portfolio_strings(rendered):
    out, cases = rendered
    t = out["full"]["text"]
    d = cases["full"]["d"]
    assert "excludes NEWCO: insufficient history" in t
    assert f"Vol {d['vol']*100:.1f}% · annualized, 252d" in t
    assert any(s.startswith("Max drawdown −") and s.endswith("· worst peak-to-trough, past 252d") for s in t)
    assert any(s.startswith("Worst 5% of days: lost ≥") for s in t)
    assert any(re.fullmatch(r"6 holdings · \d+\.\d effective bets", s) for s in t)
    assert "NEWCO  — · Insufficient history" in t
    assert "Risk rows re-base weights to the 5 holdings with history." in t   # vs Composition's all-holdings weights
    assert any("of risk (offsets)" in s for s in t)              # TLT negative RC, factual
    assert any(s.startswith("Avg pairwise correlation ") and "across 10 pairs" in s for s in t)
    assert any(s.startswith("Captured ") and "of market up-days ·" in s for s in t)
    assert any(re.fullmatch(r"Sharpe -?\d+\.\d\d · (Beat market|Trailed market|Lost vs cash)", s) for s in t)
    assert any(s.startswith("Technology ") for s in t)
    assert any(s.startswith("Yield ") and "(excludes NVDA)" in s for s in t)
    assert any(s.startswith("Top holding ") for s in t)
    assert "Portfolio β 1.05 · vs S&P 500" in t                  # relocated chip renders in §8.3


def test_all_equity_single_empty(rendered):
    out, _ = rendered
    eq = out["all_equity"]["text"]
    assert "Technology 100%" in eq
    assert any(s.startswith("Top holding 45% (NVDA)") for s in eq)
    single = out["single"]["text"]
    assert "1 holding · 1.0 effective bets" in single
    assert "Avg pairwise correlation —" in single
    assert any(s.startswith("Top holding 100% (MSFT)") and "Top 3" not in s for s in single)
    assert out["empty"]["text"] == ["Add holdings to see portfolio analytics"]


def test_amber_only_for_listed_extremes_and_no_green_red(rendered):
    out, _ = rendered
    forbidden = {"#1D9E75", "#ef4444", "#34d399", "#f87171", "#22c55e", "#dc2626"}
    for k in ("full", "all_equity", "single", "empty"):
        assert not (set(out[k]["colors"]) & forbidden), k
    # all_equity: top holding 45% > 40% -> amber on the concentration fact
    assert "#f59e0b" in out["all_equity"]["colors"]


def test_preview_lines(rendered):
    out, _ = rendered
    add, rm, rw = out["preview"]
    assert add == ["Portfolio vol 14.2% → 13.1%   (adding TLT at 25% weight · equal re-split of 4 holdings)", "β 1.12 → 1.04"]
    assert rm[0].startswith("Portfolio vol 14.2% → 15.5%   (removing TLT · remaining 3 re-split equally)")
    assert rw == ["Portfolio vol 14.2% → 15.0%   (NVDA 20% → 35% weight)", None]


def test_shared_weight_rules_match_app_semantics(rendered):
    out, _ = rendered
    assert out["add"] == {"A": 25, "B": 25, "C": 25, "D": 25}   # 100/4 exact
    assert out["rm"] == {"A": 34, "C": 33, "D": 33}
    # App's actual add/remove call the shared functions (one implementation).
    assert "setPorts(prev=>{const w=prev[pn];if(w[id]!==undefined)return prev;return{...prev,[pn]:addWeights(w,id)};})" in SRC
    assert "setPorts(prev=>({...prev,[pn]:removeWeights(prev[pn],id)}))" in SRC


def test_sharpe_label_reuse_no_fork():
    assert SRC.count("function sharpeMarketLabel(") == 1
    assert "sharpeMarketLabel(sh.sharpe,spy?spy.sharpe:null,false)" in SRC


def test_sector_bucketing_single_code_path():
    assert SRC.count("function sectorOf(") == 1
    assert "const s=sleeveOf(sectorOf(tk));" in SRC          # §B rolling chart
    assert "const s=sectorOf(t)||\"Unknown\"" in SRC          # §8.5 category weights


PROHIBITED = [r"\boverweight", r"\bunderweight", r"too concentrated", r"well[- ]diversified",
              r"poorly diversified", r"\bshould\b", r"consider adding", r"consider trimming",
              r"\bhealthy\b", r"\bunhealthy\b", r"\bsafe\b", r"\brisky\b"]


def test_prohibited_vocabulary_absent_from_rendered_panel(rendered):
    out, _ = rendered
    blob = "\n".join(s for k in ("full", "all_equity", "single", "empty")
                     for s in out[k]["text"] + out[k]["titles"])
    blob += "\n" + "\n".join(x for p in out["preview"] for x in p if x)
    for pat in PROHIBITED:
        assert not re.search(pat, blob, re.I), pat
