# BACKTEST_RUN_DECISIONS.md — Autonomous Run Judgment Log

Overnight autonomous run of VALIDATION_SPEC.md. Per the amendment, all STOP
conditions are converted to logged judgment calls; where the spec is silent, the
**conservative** option (biases AGAINST the ranker looking good) is chosen.
Nothing here is committed.

## Bars restatement (shared understanding, recorded instead of confirmed)

Primary horizon 12m. **Growth-95** (metric = Sharpe percentile vs the 1000-null):
PASS if median percentile ≥60 AND above 50th in ≥55% of windows; DECORATION if
median ≤50; else MARGINAL. **Income-20**: PASS if realized-yield percentile ≥80
in ≥80% of windows AND vol below null-median in ≥70% AND median Sharpe percentile
≥40; DECORATION if yield percentile <60 in ≥half the windows; else MARGINAL.
**Balanced-50**: PASS if median Sharpe percentile ≥55, DECORATION ≤45, else
MARGINAL. **Differentiation gate (overrides all)**: realized β Growth>Balanced>
Income in ≥75% of windows, else the fit score fails regardless of returns. Regime
slices (2020-Q1, 2022) reported not gated. **Interface actions:** PASS → keep
"Fit score" + add validated-vs-random line; MARGINAL → rename "Fit score"→"Screen
match", language "matches your profile's screen"; DECORATION → remove ranked
presentation from Discovery (or pull tab) + publish the negative result as a
Learn lesson. Momentum-tail + multi-horizon ship regardless (already committed).

## STOP-conditions-turned-judgment-calls

**J1 — Survivorship (L1). Spec NOT silent.** universe.json is today's membership;
delisted losers absent. Options: (a) accept, relative-to-same-universe null; (b)
source PIT delisted universe (paid data, unavailable). Choice: (a). Conservative
handling already in spec — the null draws from the SAME survivor universe, so the
bias is shared; **a pass is weak evidence, a failure is strong**. Logged as the
dominant caveat on any PASS.

**J2 — Corporate actions / price adjustment. Spec NOT silent** (adjusted closes
§5.5; PIT dividends §L2). Fetched with `auto_adjust=False` to keep BOTH: **Adj
Close** (split+dividend adjusted) for all returns/factors/forward metrics, and
raw **Close** for the PIT yield denominator (§L2 "price at freeze date") and the
$5 price gate. Rationale: mixing adjusted price with raw per-share dividends would
overstate historical yield; using raw price for the yield denominator is the
faithful reading and does not flatter the income screen.

**J3 — Delisted/renamed tickers mid-history.** Names failing/empty on yfinance
(symbol change, delist) are dropped, not resurrected (no PIT source). Conservative:
fewer names, and consistent with J1 survivorship. Failure list in
backtest/data/fetch_summary.json.

**J4 — Balanced-50 horizon. Spec SILENT** (§4 gives weights, not horizon). Chose
10.0y (mid between Income 5y and Growth 30y). Affects only the 8/100 horizon
points; immaterial to ranking direction. Not a bias lever either way.

**J5 — Cache format.** Spec says parquet; used per-ticker pickle (no pyarrow
dependency in venv). Functionally identical local cache; zero effect on numbers.

**J6 — Forward-window portfolio construction. Spec: equal-weight buy-and-hold.**
Portfolio value path = mean over names of AdjClose_t / AdjClose_T (true equal-
weight buy-hold, weights drift). Total return = path_end − 1; daily returns =
path.pct_change(); Sharpe uses rf = 4.3%/yr (daily rf/252); max-DD from the path;
realized β = cov(port_daily, SPY_daily)/var(SPY_daily) over the same window. A
halted name drops out of that day's mean (skipna). Standard; no thumb on scale.

**J7 — Realized yield.** Σ dividends received in (T, T+h] ÷ entry raw price,
equal-weight across the 10 names. NOT annualized (6m yield is a half-year figure);
the sealed bar is on 12m, where this is a full-year yield. Logged so the 6m yield
column is not misread as annualized.

**J8 — Percentile convention.** percentile = fraction of null portfolios strictly
below the test value × 100. Strict-below (ties not credited) mildly LOWERS the
test percentile for higher-is-better metrics → conservative (biases against the
ranker). Applied uniformly.

**J9 — Eligibility / thin windows.** At each T: ≥273 trading days of history
(§9), raw price ≥ $5 at T, trailing-252d mean volume ≥ 500K (§L3); market-cap gate
= universe membership. <150 eligible → flagged `thin`; <20 → window skipped
(cannot build a 10-name portfolio + a meaningful 1000-null). Skips are logged and
reported in coverage, never interpolated.

**J10 — Null is goal-independent.** The eligible universe at T does not depend on
goal, so one 1000-portfolio null per T (seed 20260707, fixed draw order over
sorted freezes) is reused across all three goals — matches "same gated universe."

**J11 — Value factor dropped (spec-mandated §L2/§4), goal weights renormalized**
per §4 table. Not a discretionary call; recorded for completeness.

**J12 — Tie-convention sensitivity (Balanced-50, requested post-run).** Balanced-50
is the only verdict near a threshold (median 12m Sharpe percentile 42.8 vs the
DECORATION line at 45), so its robustness to the unsealed percentile tie
convention (J8 = strict-below) was checked by reproducing the exact nulls (seed
20260707, identical freeze order → bit-identical RNG consumption; strict median
reproduced run.py's 42.80 exactly). Result, both conventions over 38 windows:

  - STRICT-BELOW (J8):            median = **42.80**
  - MIDPOINT (strict + 0.5·ties): median = **42.80**

**0 ties in every window** (float portfolio Sharpes never collide; the fit-ranked
top-10 is essentially never redrawn as a random null). The two conventions are
identical here, so Balanced-50 = DECORATION does not hinge on the tie rule (both
< 45). No STOP triggered.

(Additional judgment calls, if any arise at run time, are appended below.)
