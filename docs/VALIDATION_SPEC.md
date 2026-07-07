# Fit-Score Walk-Forward Validation — Sealed Specification

**Status:** SEALED 2026-07-07. No parameter, benchmark, or threshold below may change
after the backtest first runs. Amendments require a dated addendum explaining why,
written BEFORE re-running.
**Author of record:** Justin Chandler (methodology), drafted with Claude Fable 5.
**Implements:** the question a Fort Washington quant asked on 2026-07-06:
*"Has a Growth-95 shortlist ever been evaluated against what it was supposed to beat?"*

---

## 1. Claim under test

The Stage 2→3 screening pipeline (`app/screener.py`) produces, for a given investor
profile, a ranked shortlist whose **top names are better for that profile than random
selection from the same screened universe**. If this is false, the fit score is
decoration and the interface must say so (see §8, pre-registered outcomes).

This tests the RANKING, not the optimizer. `return_model`, BL, and the Outlook are
out of scope.

## 2. What the engine actually computes (code ground truth)

Verified against `app/screener.py` and `app/config.py` on 2026-07-07:

- **Universe:** `app/data/universe.json` (S&P 500 + ETFs + extended), hard-gated by
  market cap ≥ $2B, avg volume ≥ 500K, price ≥ $5 (`config.py:58-60`).
- **Factors** (`compute_factor_scores`, screener.py:329-446):
  - momentum = 0.6 × 12m + 0.4 × 6m log-return sum, skipping most recent 21 days
  - quality = trailing-252d Sharpe
  - low_vol = −annualized vol (trailing 252d)
  - value = earnings yield from **current** yfinance `trailingPE`
  - yield = **current** yfinance `dividendYield`
  - All winsorized at 3σ, z-scored cross-sectionally.
- **Stage 2:** goal-weighted composite, top 120 (`SCREEN_STAGE2_SIZE`).
- **Stage 3 fit score** (`compute_fit_scores`, screener.py:493-574): 55pts goal
  composite (cross-sectionally standardized), 30pts symmetric β-band around
  target_β = 0.20 + risk/100 × 1.30, 8pts horizon, 7pts Sharpe quality floor.

## 3. Honest limitations, stated before results exist

**L1 — Survivorship bias.** `universe.json` is today's membership; delisted losers
are absent. This inflates ABSOLUTE results. Mitigation: the null (§5) draws from the
SAME universe, so relative comparisons are far less biased than comparison to an
index. Consequence for interpretation: **a failure is strong evidence; a pass is
weak evidence** and must be labeled as such in any UI claim.

**L2 — Point-in-time fundamentals are unavailable.** yfinance gives current
`trailingPE` only. Using today's value factor at a 2019 freeze date is lookahead
bias and is REJECTED. Instead:
  - **yield** is reconstructed point-in-time: trailing-12m dividends (from
    `t.dividends` history) ÷ price at freeze date × 100.
  - **value** is EXCLUDED from the backtest composite; remaining goal weights are
    renormalized (see §4). This deviation from production weights is disclosed in
    every output. Upgrade path: FMP point-in-time fundamentals (already a tracked
    workstream) → rerun with true value factor as v2.

**L3 — Gate reconstruction.** Price and volume gates are applied point-in-time from
historical data. The market-cap gate cannot be (no shares-outstanding history);
today's universe membership stands in for it. Shared by test and null portfolios.

**L4 — Overlapping windows.** Quarterly freeze dates with 12m horizons overlap;
window results are not independent. "% of windows" criteria are read as descriptive
coverage, not as if N independent trials.

## 4. Backtest weights (renormalized after dropping value)

| Goal (risk) | momentum | quality | low_vol | yield |
|---|---|---|---|---|
| Growth-95 (prod .50/.20/.20/.05/.05) | .625 | .250 | .0625 | .0625 |
| Income-20 (prod .00/.20/.10/.30/.40) | .000 | .222 | .333 | .444 |
| Balanced-50 (prod .20/.20/.20/.20/.20) | .250 | .250 | .250 | .250 |

Target betas: Growth-95 → 1.435; Income-20 → 0.46; Balanced-50 → 0.85.

## 5. Protocol

1. **Freeze dates:** last trading day of each quarter, 2016-Q1 through 2025-Q2
   inclusive (38 dates).
2. **At each freeze date T, using only data ≤ T:** rebuild gates (L3), factor
   z-scores (§2 definitions, §4 weights), Stage-2 top-120, Stage-3 fit scores —
   a faithful re-implementation of `screener.py` stages with PIT inputs, in a new
   module `backend/backtest/backtest_fit.py`. Production code is not modified.
3. **Test portfolio:** top 10 by fit score, equal-weight, buy-and-hold.
4. **Null distribution:** 1,000 equal-weight portfolios of 10 names drawn uniformly
   without replacement from the same gated, history-qualified universe at T.
   Seed = 20260707 (fixed, sealed).
5. **Horizons:** 6m and 12m forward total returns (dividends included via adjusted
   closes — consistent with `data_ingest.fetch_prices`).
6. **Metrics per portfolio per window:** total return; Sharpe (daily returns,
   rf = 4.3%/yr per `config.py:54`); max drawdown; realized dividend yield
   (Income only); realized portfolio beta vs SPY.
7. **Costs:** zero commission, no slippage, buy-and-hold (no turnover within
   window). Disclosed as favorable-to-strategy; acceptable at $2B+/500K-volume
   universe and retail size.
8. **Score per window:** the test portfolio's percentile within the null
   distribution, per metric.

## 6. Sealed acceptance bar

Primary horizon: **12m**. Primary metric per goal:

**Growth-95 (primary metric: Sharpe percentile vs null):**
- PASS: median-across-windows percentile ≥ 60 AND above 50th percentile in ≥ 55%
  of windows.
- DECORATION: median percentile ≤ 50.
- MARGINAL: anything between.

**Income-20 (income is the promise; return is secondary):**
- PASS: realized-yield percentile ≥ 80 in ≥ 80% of windows AND volatility BELOW
  null median in ≥ 70% of windows AND median Sharpe percentile ≥ 40.
- DECORATION: yield percentile < 60 in ≥ half the windows (the income screen
  doesn't even deliver income).
- MARGINAL: between.

**Balanced-50:** PASS if median Sharpe percentile ≥ 55; DECORATION ≤ 45.

**Profile-differentiation check (all-or-nothing):** Growth-95 realized beta >
Balanced-50 realized beta > Income-20 realized beta in ≥ 75% of windows. Failure
here overrides everything: if personas don't produce behaviorally different
portfolios, the fit score fails at its one job regardless of returns. (This is the
walk-forward version of the shipped regression test Growth-95 ∩ Income-20 ≤ 2.)

**Regime slice (report, not gate):** windows whose 12m horizon includes 2020-Q1 or
calendar 2022 are additionally reported separately — this is where momentum's left
tail should appear, and the result feeds the momentum disclosure copy either way.

## 7. Sealed predictions (hand-derived, to audit the run itself)

Divergence from these = finding (engine bug in the backtest, or model error in my
understanding — both get logged):

- P1: Growth-95 beats the null on RAW RETURN in most 2016–2021 windows and
  underperforms sharply in windows containing 2022 (momentum reversal).
- P2: Growth-95 Sharpe percentiles are weaker than its return percentiles
  (momentum buys vol along with return).
- P3: Income-20 passes the yield and vol legs comfortably (they're near-mechanical:
  the screen selects on trailing yield and low vol, which persist), but its Sharpe
  leg is unremarkable (~40-60th percentile).
- P4: The differentiation check passes in > 85% of windows (β-band is 30/100 points
  and β persists).
- P5: Overall verdicts — Growth-95: MARGINAL. Income-20: PASS. Balanced-50:
  MARGINAL-to-PASS. If Growth-95 PASSes cleanly, suspect the backtest before
  celebrating (L1 says pass-evidence is weak; check the null construction first).

## 8. Pre-registered outcomes → interface actions

Decided now, so results can't negotiate:

- **PASS:** keep fit framing. Add: "Validated against random selection from the
  same universe, 2016–2025, with survivorship-bias caveat" and link the method.
- **MARGINAL:** relabel "Fit score" → "Screen match" everywhere; interface language
  becomes "matches your profile's screen," never "recommended for you." Education
  framing: show the walk-forward chart, teach why marginal is the honest common case.
- **DECORATION:** remove ranked presentation from Discovery (alphabetical or
  size-ordered listing with factor data visible) OR pull the tab pending redesign.
  The result itself becomes a Learn-tab lesson: "we tested our own ranking; here's
  what we found." This outcome, published, is the strongest possible proof of the
  product identity.
- In ALL cases: momentum tail disclosure and multi-horizon columns ship (already
  committed independently of this test).

## 9. Implementation notes (for Claude Code, not Fable)

- New module `backend/backtest/backtest_fit.py` + `backend/backtest/run.py`;
  zero imports FROM it in `app/` (prod path untouched).
- Fetch full price + dividend history once (yfinance `period="max"`), cache to
  parquet in `backend/backtest/data/`; every freeze-date computation reads the
  cache — deterministic, re-runnable, Render-independent.
- Expected data casualties: tickers with < 273 trading days before T are excluded
  at that T (matches `SCREEN_MIN_HISTORY_DAYS` + momentum's 21-day skip).
  Log exclusion counts per T; if the eligible universe at any T falls below 150
  names, flag that window as thin rather than silently proceeding.
- Output: one CSV row per (goal, T, horizon, metric) with test value, null median,
  percentile; plus a summary JSON with verdicts per §6. The audit compares the
  summary against §7 predictions.
- Runtime estimate: dominated by one-time data fetch; computation is trivial
  (38 dates × 3 goals × 1,001 portfolios of vectorized math).

## 10. What this spec deliberately does not do

- No parameter search, no weight tuning, no "try other thresholds" — this is a
  test, not an optimization. If someone (including me) wants different thresholds,
  that's a v2 spec sealed before v2 runs.
- No claim about the optimizer, BL, tilts, or regimes.
- No forecast that passing here predicts future outperformance — the UI language
  in §8 must never imply it does.
