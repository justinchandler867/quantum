"""
Walk-forward validation runner (VALIDATION_SPEC.md §5-§6). Reads the cached
history, runs 38 quarterly freezes x 3 goals x {6m,12m} x (test + 1000-null),
writes RAW per-freeze results and a mechanically-applied DRAFT verdict.

NOT imported by app/. Deterministic: null seed 20260707 (sealed).
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

import backtest_fit as bf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
DOCS = os.path.abspath(os.path.join(HERE, "..", "docs"))
SEED = 20260707
FREEZE_START, FREEZE_END = "2016Q1", "2025Q2"


def load_matrices():
    adj, rawc, vol, divs = {}, {}, {}, {}
    for path in sorted(glob.glob(os.path.join(RAW, "*.pkl"))):
        tk = os.path.basename(path)[:-4]
        try:
            h = pickle.load(open(path, "rb"))
        except Exception:
            continue
        if h is None or len(h) == 0:
            continue
        idx = pd.to_datetime(h.index).tz_localize(None).normalize()
        h = h.copy(); h.index = idx
        h = h[~h.index.duplicated(keep="last")]
        if "Adj Close" in h:
            adj[tk] = h["Adj Close"]
        elif "Close" in h:
            adj[tk] = h["Close"]
        if "Close" in h:
            rawc[tk] = h["Close"]
        if "Volume" in h:
            vol[tk] = h["Volume"]
        if "Dividends" in h:
            d = h["Dividends"]; divs[tk] = d[d > 0]
    adj = pd.DataFrame(adj).sort_index()
    rawc = pd.DataFrame(rawc).sort_index()
    vol = pd.DataFrame(vol).sort_index()
    return adj, rawc, vol, divs


def freeze_dates(index):
    qs = pd.period_range(FREEZE_START, FREEZE_END, freq="Q")
    out = []
    for q in qs:
        qend = q.to_timestamp(how="end").normalize()
        prior = index[index <= qend]
        if len(prior):
            out.append((str(q), prior[-1]))
    return out


def run():
    print("Loading cache...", flush=True)
    adj, rawc, vol, divs = load_matrices()
    print(f"Loaded {adj.shape[1]} tickers, {adj.shape[0]} dates "
          f"({adj.index.min().date()}..{adj.index.max().date()}); SPY present: {'SPY' in adj.columns}", flush=True)

    freezes = freeze_dates(adj.index)
    rng = np.random.default_rng(SEED)

    rows = []            # per (goal,T,horizon)
    diff_rows = []       # per T: 12m realized betas for the differentiation gate
    coverage = []        # per T eligibility

    for qlabel, T in freezes:
        E = bf.eligible_universe(T, adj, rawc, vol)
        n = len(E)
        thin = n < 150
        cov = {"quarter": qlabel, "T": str(T.date()), "eligible": n, "thin": thin, "built": n >= 20}
        coverage.append(cov)
        print(f"{qlabel} {T.date()}: eligible={n}{' THIN' if thin else ''}"
              f"{'  SKIP(<20)' if n < 20 else ''}", flush=True)
        if n < 20:
            continue

        factors = bf.compute_factors_at(T, E, adj, rawc, divs)
        Earr = np.array(E)
        null_idx = [rng.choice(len(Earr), size=10, replace=False) for _ in range(1000)]
        null_names = [list(Earr[ix]) for ix in null_idx]

        betas_12m = {}
        for hlabel, hd in [("6m", 126), ("12m", 252)]:
            nulls = [bf.forward_metrics(nm, T, hd, adj, rawc, divs) for nm in null_names]
            nulls = [m for m in nulls if m is not None]
            if len(nulls) < 100:
                print(f"    {hlabel}: only {len(nulls)} null portfolios buildable -> skip horizon", flush=True)
                continue
            null_med = {k: float(np.nanmedian([m[k] for m in nulls])) for k in nulls[0]}

            for pname, (g, risk, hor) in bf.PROFILES.items():
                ranked = bf.fit_rank(factors, g, risk, hor)
                top10 = list(ranked.head(10).index)
                tm = bf.forward_metrics(top10, T, hd, adj, rawc, divs)
                if tm is None:
                    continue
                pct = {k: bf.percentile(tm[k], [m[k] for m in nulls]) for k in tm}
                rows.append({
                    "goal": pname, "quarter": qlabel, "T": str(T.date()), "horizon": hlabel,
                    "n_null": len(nulls), "top10": ",".join(top10),
                    **{f"test_{k}": tm[k] for k in tm},
                    **{f"nullmed_{k}": null_med[k] for k in tm},
                    **{f"pct_{k}": pct[k] for k in tm},
                    "vol_below_nullmed": bool(tm["vol"] < null_med["vol"]) if not np.isnan(tm["vol"]) else None,
                })
                if hlabel == "12m":
                    betas_12m[pname] = tm["beta"]
        if {"Growth-95", "Balanced-50", "Income-20"} <= set(betas_12m):
            diff_rows.append({"quarter": qlabel, "T": str(T.date()),
                              "beta_Growth": betas_12m["Growth-95"],
                              "beta_Balanced": betas_12m["Balanced-50"],
                              "beta_Income": betas_12m["Income-20"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "data", "results_raw.csv"), index=False)
    diff = pd.DataFrame(diff_rows)
    json.dump({"coverage": coverage}, open(os.path.join(HERE, "data", "coverage.json"), "w"), indent=2)

    write_raw_md(df, diff, coverage)
    write_verdict_draft(df, diff, coverage)
    print("\nWROTE docs/BACKTEST_RAW_RESULTS.md and docs/BACKTEST_VERDICT_DRAFT.md", flush=True)
    return df, diff, coverage


def _fmt(x, p=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{p}f}"


def write_raw_md(df, diff, coverage):
    built = [c for c in coverage if c["built"]]
    thin = [c for c in coverage if c["thin"] and c["built"]]
    skipped = [c for c in coverage if not c["built"]]
    L = ["# BACKTEST_RAW_RESULTS.md — Fit-Score Walk-Forward (RAW, pre-interpretation)",
         "",
         "Generated by `backtest/run.py` against cached yfinance history. Numbers are raw; "
         "interpretation is in BACKTEST_VERDICT_DRAFT.md. Null seed 20260707.",
         "",
         f"**Coverage:** {len(built)}/{len(coverage)} freezes built; {len(thin)} thin (<150 eligible); "
         f"{len(skipped)} skipped (<20 eligible). Universe cache: see backtest/data/fetch_summary.json.",
         ""]
    for h in ["12m", "6m"]:
        L.append(f"## {h} horizon — per-freeze test portfolio vs 1000-null")
        for goal in ["Growth-95", "Income-20", "Balanced-50"]:
            sub = df[(df.goal == goal) & (df.horizon == h)].sort_values("T")
            if sub.empty:
                continue
            L.append(f"\n### {goal} ({h})")
            L.append("| Freeze | elig-null | test ret | pct ret | test Sharpe | pct Sharpe | test vol | pct vol | test β | pct β | realYld | pct Yld |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                L.append("| {q} | {nn} | {tr} | {pr} | {ts} | {ps} | {tv} | {pv} | {tb} | {pb} | {ty} | {py} |".format(
                    q=r["quarter"], nn=int(r["n_null"]),
                    tr=_fmt(r["test_total_return"]), pr=_fmt(r["pct_total_return"], 0),
                    ts=_fmt(r["test_sharpe"]), ps=_fmt(r["pct_sharpe"], 0),
                    tv=_fmt(r["test_vol"]), pv=_fmt(r["pct_vol"], 0),
                    tb=_fmt(r["test_beta"]), pb=_fmt(r["pct_beta"], 0),
                    ty=_fmt(r["test_realized_yield"], 4), py=_fmt(r["pct_realized_yield"], 0)))
    L.append("\n## Profile differentiation (12m realized β) — per freeze")
    L.append("| Freeze | β Growth | β Balanced | β Income | Growth>Bal>Inc? |")
    L.append("|---|---|---|---|---|")
    for _, r in diff.iterrows():
        ok = (r["beta_Growth"] > r["beta_Balanced"] > r["beta_Income"])
        L.append(f"| {r['quarter']} | {_fmt(r['beta_Growth'])} | {_fmt(r['beta_Balanced'])} | {_fmt(r['beta_Income'])} | {'YES' if ok else 'no'} |")
    open(os.path.join(DOCS, "BACKTEST_RAW_RESULTS.md"), "w").write("\n".join(L) + "\n")


def _pct_series(df, goal, metric, horizon="12m"):
    s = df[(df.goal == goal) & (df.horizon == horizon)][f"pct_{metric}"].dropna()
    return s


def write_verdict_draft(df, diff, coverage):
    L = ["# BACKTEST_VERDICT_DRAFT.md — DRAFT — PENDING AUDIT", "",
         "Mechanical application of the sealed §6 bars to BACKTEST_RAW_RESULTS.md. "
         "No human interpretation added; audit before trusting.", ""]

    # Growth-95: Sharpe percentile
    g = _pct_series(df, "Growth-95", "sharpe")
    g_med = float(g.median()) if len(g) else float("nan")
    g_above = float((g > 50).mean() * 100) if len(g) else float("nan")
    g_verdict = ("PASS" if (g_med >= 60 and g_above >= 55) else "DECORATION" if g_med <= 50 else "MARGINAL")
    L += [f"## Growth-95 — {g_verdict}",
          f"- median Sharpe percentile (12m) = {g_med:.1f} (PASS ≥60, DECORATION ≤50)",
          f"- above 50th pct in {g_above:.0f}% of windows (PASS needs ≥55%)", ""]

    # Income-20: yield pct ≥80 in ≥80%; vol below null median in ≥70%; median Sharpe pct ≥40
    iy = _pct_series(df, "Income-20", "realized_yield")
    iy80 = float((iy >= 80).mean() * 100) if len(iy) else float("nan")
    iy60 = float((iy < 60).mean() * 100) if len(iy) else float("nan")
    ivb = df[(df.goal == "Income-20") & (df.horizon == "12m")]["vol_below_nullmed"].dropna()
    ivb_pct = float((ivb == True).mean() * 100) if len(ivb) else float("nan")
    ish = _pct_series(df, "Income-20", "sharpe")
    ish_med = float(ish.median()) if len(ish) else float("nan")
    inc_pass = (iy80 >= 80 and ivb_pct >= 70 and ish_med >= 40)
    inc_dec = (iy60 >= 50)
    inc_verdict = "PASS" if inc_pass else "DECORATION" if inc_dec else "MARGINAL"
    L += [f"## Income-20 — {inc_verdict}",
          f"- realized-yield pct ≥80 in {iy80:.0f}% of windows (PASS needs ≥80%)",
          f"- volatility below null-median in {ivb_pct:.0f}% of windows (PASS needs ≥70%)",
          f"- median Sharpe percentile = {ish_med:.1f} (PASS needs ≥40)",
          f"- DECORATION check: yield pct <60 in {iy60:.0f}% of windows (DECORATION if ≥50%)", ""]

    # Balanced-50: median Sharpe pct
    b = _pct_series(df, "Balanced-50", "sharpe")
    b_med = float(b.median()) if len(b) else float("nan")
    b_verdict = "PASS" if b_med >= 55 else "DECORATION" if b_med <= 45 else "MARGINAL"
    L += [f"## Balanced-50 — {b_verdict}",
          f"- median Sharpe percentile (12m) = {b_med:.1f} (PASS ≥55, DECORATION ≤45)", ""]

    # Differentiation gate
    if len(diff):
        ok = ((diff.beta_Growth > diff.beta_Balanced) & (diff.beta_Balanced > diff.beta_Income))
        diff_pct = float(ok.mean() * 100)
    else:
        diff_pct = float("nan")
    diff_pass = diff_pct >= 75
    L += [f"## Profile differentiation gate — {'PASS' if diff_pass else 'FAIL'}",
          f"- Growth β > Balanced β > Income β in {diff_pct:.0f}% of windows (needs ≥75%)",
          "- **Overrides everything**: if FAIL, the fit score fails regardless of returns.", ""]

    overall = {"Growth-95": g_verdict, "Income-20": inc_verdict, "Balanced-50": b_verdict,
               "differentiation": "PASS" if diff_pass else "FAIL"}
    L += ["## Draft overall", "```", json.dumps(overall, indent=2), "```",
          "", "Compare against §7 sealed predictions (P1–P5) during audit.", ""]
    open(os.path.join(DOCS, "BACKTEST_VERDICT_DRAFT.md"), "w").write("\n".join(L) + "\n")
    print("\n=== DRAFT VERDICTS ===\n" + json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    run()
