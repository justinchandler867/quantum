"""
Ranker-untouched measurement (PORTFOLIO_RISK_SPEC prime directive).

Dumps screen-match ORDERING, byte-for-byte reproducibly, from both places the
ranker lives, on offline fixtures only (no network):

  1. Frontend: the shipped FALLBACK_ASSETS through fitScore + the Discovery
     row pipeline (factor re-rank when z-scores exist) + cmpRows, executed under
     node, for every goal x risk band x horizon grid point. A second fixture
     gives synthetic z-scored rows so the customScore path is exercised too.
  2. Backend: app.screener.compute_factor_scores -> rank_by_composite ->
     compute_fit_scores on the cached backtest prices (backtest/data/raw/*.pkl,
     local, gitignored), with fundamentals derived deterministically from the
     same cache (earnings_yield = 0; dividend_yield = trailing-12m dividends /
     last raw Close, the J2 raw-price denominator).

Usage:  python scripts/ranker_order_dump.py OUT.json
Run before and after a change; `diff before.json after.json` must be empty.
"""
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.config import FACTOR_WEIGHTS_BY_GOAL  # noqa: E402
from app.data_ingest import compute_log_returns  # noqa: E402
from app.screener import compute_factor_scores, compute_fit_scores, rank_by_composite  # noqa: E402

FRONTEND = os.path.join(ROOT, "static", "quantex.html")
RAW = os.path.join(ROOT, "backtest", "data", "raw")

NODE_SCRIPT = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
function balanced(open, close, startTok){
  const i=src.indexOf(startTok); const s=src.indexOf(open,i); let d=0,j=s;
  for(;j<src.length;j++){ if(src[j]===open)d++; else if(src[j]===close){d--; if(d===0)break;} }
  return src.slice(s,j+1);
}
function fn(name){
  const i=src.indexOf("function "+name+"("); const s=src.indexOf("{",src.indexOf(")",i)); let d=0,j=s;
  for(;j<src.length;j++){ if(src[j]==="{")d++; else if(src[j]==="}"){d--; if(d===0)break;} }
  return src.slice(i,j+1);
}
eval("var FALLBACK_ASSETS = " + balanced("[","]","const FALLBACK_ASSETS = [") + ";");
eval(fn("scoreProfile")); eval(fn("fitScore")); eval(fn("cmpRows"));
// Discovery's default factor weights object, verbatim from the component.
const dw = src.match(/const defaultWeights=(\{Growth:.*?\}\});/);
eval("var defaultWeights=" + dw[1]);
// The Discovery row pipeline, verbatim logic (component-local, so reproduced
// from the source text and asserted against it below).
const rowLine = "fit=Math.max(0,Math.min(100,Math.round(customScore*20+60)));";
if(!src.includes(rowLine)) { console.error("Discovery re-rank line changed"); process.exit(2); }
function rows(assets, profile){
  const factorW = defaultWeights[profile.goal] || defaultWeights.Balanced;
  return assets.map(a=>{
    let fit=a.fit||fitScore(a,profile);
    if(a.zMom!==undefined){
      const customScore=factorW.momentum*(a.zMom||0)+factorW.quality*(a.zQual||0)+factorW.value*(a.zVal||0)+factorW.low_vol*(a.zVol||0)+factorW.yield*(a.zYield||0);
      fit=Math.max(0,Math.min(100,Math.round(customScore*20+60)));
    }
    return {...a,fit};
  });
}
// Synthetic z-scored rows (seeded LCG) to exercise the customScore path.
let seed=12345; const rnd=()=>{seed=(seed*1103515245+12345)%2147483648; return seed/2147483648;};
const SYN=[...Array(40)].map((_,i)=>({id:"S"+String(i).padStart(2,"0"),n:"syn",t:"Stock",sec:"x",
  zMom:+(rnd()*4-2).toFixed(3),zQual:+(rnd()*4-2).toFixed(3),zVal:+(rnd()*4-2).toFixed(3),zVol:+(rnd()*4-2).toFixed(3),zYield:+(rnd()*4-2).toFixed(3)}));
const base={age:"36-45",inc:"$100K-$250K",nw:"$100K-$500K",los:"Hold steady",exp:"Intermediate",liq:"Medium (3-5 yrs)",der:"None"};
const out={};
for(const goal of ["Growth","Income","Preservation","Balanced"])
 for(const hor of ["<1 year","3-7 years","15+ years"])
  for(const der of ["None","Advanced"]){
    const p=scoreProfile({...base,goal,hor,der});
    for(const [name,set] of [["fallback",FALLBACK_ASSETS],["synthetic",SYN]]){
      const r=rows(set,p);
      const key=name+"|"+goal+"|"+hor+"|"+der;
      out[key]={desc:r.slice().sort((a,b)=>cmpRows(a,b,"fit",-1)).map(x=>x.id+":"+x.fit),
                asc:r.slice().sort((a,b)=>cmpRows(a,b,"fit",1)).map(x=>x.id+":"+x.fit)};
    }
  }
console.log(JSON.stringify(out));
"""


def frontend_orders():
    node = shutil.which("node")
    if not node:
        raise SystemExit("node required")
    r = subprocess.run([node, "-e", NODE_SCRIPT, FRONTEND], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("node failed: " + r.stderr)
    return json.loads(r.stdout)


def backend_orders():
    files = sorted(glob.glob(os.path.join(RAW, "*.pkl")))
    if not files:
        raise SystemExit("offline fixture missing: " + RAW)
    adj, fund = {}, []
    for f in files:
        t = os.path.basename(f)[:-4]
        d = pickle.load(open(f, "rb"))
        if d is None or len(d) == 0 or "Adj Close" not in d:
            continue
        d.index = pd.to_datetime(d.index, utc=True).tz_convert(None).normalize()
        adj[t] = d["Adj Close"]
        last_raw = float(d["Close"].dropna().iloc[-1]) if d["Close"].notna().any() else float("nan")
        div12 = float(d["Dividends"].loc[d.index >= d.index[-1] - pd.Timedelta(days=365)].sum())
        dy = round(div12 / last_raw * 100, 4) if last_raw and last_raw == last_raw else 0.0
        fund.append({"ticker": t, "earnings_yield": 0.0, "dividend_yield": dy})
    prices = pd.DataFrame(adj).sort_index()
    prices = prices.loc["2023-07-07":"2026-07-07"]          # fixed 3y window of the cache
    prices = prices.loc[:, prices.notna().mean() > 0.95].ffill(limit=5).dropna()
    returns = compute_log_returns(prices)
    fundamentals = pd.DataFrame(fund)
    fundamentals = fundamentals[fundamentals["ticker"].isin(returns.columns)].reset_index(drop=True)
    scored = compute_factor_scores(fundamentals, returns, benchmark="SPY")
    out = {"_fixture": {"tickers": int(returns.shape[1]), "days": int(returns.shape[0]),
                        "first": str(returns.index[0].date()), "last": str(returns.index[-1].date())}}
    for goal, weights in FACTOR_WEIGHTS_BY_GOAL.items():
        stage2 = rank_by_composite(scored, weights)
        for risk in (20, 50, 88):
            for hor in (0.75, 5.0, 20.0):
                fit = compute_fit_scores(stage2, goal=goal, risk_score=risk, time_horizon_years=hor)
                out[f"{goal}|{risk}|{hor}"] = [f"{t}:{s}" for t, s in zip(fit["ticker"], fit["fit_score"])]
    return out


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    dump = {"frontend": frontend_orders(), "backend": backend_orders()}
    with open(sys.argv[1], "w") as fh:
        json.dump(dump, fh, indent=1, sort_keys=True)
    print(f"frontend keys={len(dump['frontend'])} backend keys={len(dump['backend'])-1} "
          f"fixture={dump['backend']['_fixture']}")
