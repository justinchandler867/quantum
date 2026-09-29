# LAUNCH_HONESTY_DECISIONS.md: sealed honesty remediation (unattended run)

**Date:** 2026-09-28. **Branch:** launch-honesty, cut from build3-portfolio-health at 855c8e0. **Baseline:** 224 passed (verified before any change).

Part 1 was written before any code was changed. Part 2 logs the judgment calls made during the remediation.

---

## Part 1: History (read-only investigation)

### 1(a) When each `Math.random` call entered

`git log -S "Math.random" -- static/quantex.html` returns one commit, **496bb73 (2026-04-16), "12 CFA modules, YouTube links, Trade Desk fix, 43 quiz questions".** It is the repository's root commit (the first of 70).

`git blame` marks all nine call sites `^496bb73`, the boundary marker. So every one predates this repo's tracked history: each arrived with the initial import of the single-file frontend, and none was added or changed by any later commit.

| Line (at 855c8e0) | Code | Commit |
|---|---|---|
| 591–594 | `getCorr()`: 4 branches, `base + Math.random()*k` | ^496bb73 |
| 623 | `optimizeLocal()`: random weight draws | ^496bb73 |
| 2769 | `Frontier` local Monte Carlo: random weights | ^496bb73 |
| 3216 | `PaperTrading.getPrice()`: random price when return ≤ 0 | ^496bb73 |
| 3305 | `PaperTrading.executeTrade()`: random price for unknown tickers | ^496bb73 |
| 3335 | `executeTrade()`: transaction id `Math.random().toString(36)` | ^496bb73 |

### 1(b) Classification

**FABRICATES ANALYTICS**

- **`getCorr()` (591–594).** Pairwise correlation made from asset type plus uniform noise, cached per session. It is the covariance input to `calcStats()`. Through `calcStats()` it reaches:
  - the status bar (RETURN / VOL σ / SHARPE / BETA / SORTINO / VAR 95% / DIV YIELD);
  - the Metrics tab (every stat box, risk contribution, factor exposures);
  - the builder stat grid;
  - Compare;
  - Challenges, both the completion checks and "current status";
  - the Ask Quantex context string;
  - Frontier's "Your Portfolio" point;
  - `optimizeLocal`.

  It is also the Correlations tab's matrix whenever the backend call fails.

  Beyond the random correlations, `calcStats()` fabricates further: `maxDD = −2.1·vol`, `Sortino = Sharpe/0.72`, parametric VaR on the fabricated vol, β as a weighted average, and a hard-coded 24.2% benchmark. So every `calcStats()` number is fabricated, not only its volatility.
- **`optimizeLocal()` (623).** A 250-draw random weight search scored by `calcStats()`. It is applied to the user's portfolio when `/api/optimize` fails, under results copy that describes the SLSQP result.
- **Frontier local Monte Carlo (2769).** 150 random portfolios scored by `calcStats()`, shown as the frontier when `/api/frontier` fails, badged "Monte Carlo (250 trials)".
- **`getPrice()` / random fill price (3216, 3305).** Fabricated fill prices. They sit only inside `executeTrade()`, which has no caller (the UI calls `submitOrder` → `/api/paper/order`). It is dead code that would fabricate if revived.

**LEGITIMATE**

- **Transaction id (3335).** A non-analytic identifier, but it lives only in dead `executeTrade()`.

**Net:** there are no LEGITIMATE uses outside dead code.

### 1(c) Does the 761-day truncation hit the local app, and since when?

**Yes.** Since **2025-12-19** (a data date, determined from the price data), Correlations, Frontier, Optimizer and the 2026 Stress Test fail server-side with 422 in any process where the Discovery screen has loaded the universe. That is the normal path, because Discovery requires a screen first. Correlations, Frontier and Optimizer then fall back to the random paths above; the Stress Test shows the 422 text.

**Mechanism** (`app/data_ingest.py::fetch_prices`):

1. A ticker is kept if it has at least 60% of 5 years of rows (≥756).
2. Then `ffill(limit=5).dropna()` drops every row where any kept ticker has no price.
3. So the matrix starts at the listing date of the youngest ticker that passed rule 1. The matrix is therefore floored near 756 rows (about 3 years), not 5 years.

**Stress reference:** `identify_stress_windows` uses QQQ at a −15% drawdown from its rolling 252-day high. `compute_stress_correlation` raises below 20 stress days. `MIN_STRESS_DAYS = 60` only logs a warning.

**Replay** of the fetch rules on the cached prices (`backtest/data/raw`, 593 tickers, to 2026-07-07, current universe):

| As-of date | Rows | Matrix starts | Limited by (listed) | Stress days (QQQ −15%) |
|---|---|---|---|---|
| 2025-03-31 | 795 | 2022-01-28 | CRDO (2022-01-27) | 218 |
| 2025-09-30 | 921 | 2022-01-28 | CRDO | 231 |
| **2025-12-19** | **755** | **2022-12-16** | **GEHC (2022-12-15)** | **13** (flip below the 20 floor) |
| 2026-03-31 | 823 | 2022-12-16 | GEHC | 13 |
| 2026-07-07 | 794 | 2023-05-05 | KVUE (2023-05-04) | 13 |
| 2026-09-28 (live, observed) | 761 | — | — | 13 |

GEHC crossing the 756-row admission rule pulled the start past the end of the 2022 bear market. The only stress days left are the April 2025 drawdown.

**Scope of "since when" for this repo:** `universe.json` (596 names) entered at 75f10e2 on 2026-04-22, after the flip. So every local session in this repo that followed the normal path (a screen first) has used a store where these four features fail.

**Second trigger, independent of the data:** when the first screen exceeds the 180 s frontend timeout (DEMO_READINESS top item 1), `backendOk=false`. Correlations and Frontier then skip the backend and go straight to the random paths, whatever the history length.

**What is not established:** there are no local run logs, so this is reconstructed from the data and the code path, not observed in past sessions. The live observation on 2026-09-28 (761 rows, 13 stress days, 422s) matches the replay.

---

## Part 2: Remediation judgment calls

Format for each entry: ambiguity, options, choice, reasoning. **24 calls.**

### Item 2: removing fabrication

**K1. What counts as a FABRICATES-ANALYTICS fallback.**
- **Choice:** all of `calcStats()`, not only `getCorr()`. Beyond the random correlations it synthesizes Max DD as −2.1·vol, Sortino as Sharpe/0.72, a weighted-average β and a hard-coded 24.2% benchmark return.
- **Consequence:** every consumer was rewired to the backend risk engine (`/api/portfolio/health`) or removed. There is no local computation of any portfolio statistic left in the frontend.

**K2. Metrics tab.**
- **Choice:** replaced with the Portfolio Health panel for the active portfolio, plus one line pointing to the Regime Stress Test.
- **Removed:** the 16 stat boxes, "Return attribution", "Factor exposures" (invented transforms) and the 2008/2020/2022 rows. The crisis rows read fixed per-asset fields that are 0 for every screened name.
- **Also removed:** the CFA stat-box explainers (`CFA_ED`, `StatBox`, `CfaExplainer`). Their only consumer was the removed grid, and their worked examples quoted `calcStats` numbers.

**K3. Compare tab.**
- **Choice:** rebuilt on engine metrics per portfolio: 252-day return, trailing yield, Sharpe, vol, β vs SPY, max drawdown, VaR 95%, effective bets.
- **Dropped:** rows with no engine equivalent (Info Ratio, Sortino, Calmar, Treynor, the three crisis rows).
- **Kept:** the ★ "best value" marker, which is shipped behavior; it now compares real numbers.

**K4. Challenges.**
- **Choice:** every completion check reads engine metrics, and an unavailable metric never satisfies a check. "SPY return of 24.2%" became SPY's return over the same 252-day window.
- **Flag:** completions stored in `localStorage` earlier, possibly earned on fabricated numbers, were **not** cleared. Deleting a user's saved progress is a product decision for you.

**K5. Correlations, Frontier and Optimizer.**
- **Choice:** they always call the backend. Previously they skipped it when `backendOk` was false, which sent them straight to the random paths.
- **On failure:** Correlations and Frontier show the server's cause verbatim. The Optimizer shows "Optimization did not run: … Your weights were not changed." and leaves the weights alone.
- **Frontier's "Your Portfolio" point:** now the backend's `current_portfolio` evaluation on the same inputs.

**K6. Top-bar market strip.**
- **Finding:** it showed hard-coded quotes (SPY 527.84, QQQ 449.21, VIX 18.42, 10Y 4.38%, GLD 302.17). Not `Math.random`, but fabricated market numbers presented as current.
- **Choice:** replaced with last closes of SPY/QQQ/GLD/TLT from the loaded price data plus "CLOSE AS OF {date}", via `/health`. VIX and 10Y are dropped because they are not in the store.
- **Reasoning:** logged because it goes beyond the literal item.

**K7. Dead code with random prices.**
- **Choice:** `executeTrade()` and `getPrice()` (no callers; random fill prices) were deleted.
- **Consequence:** the `Math.random` allowlist is empty, and the test enforces that.

**K8. Stress failure message.**
- **Change:** `compute_stress_correlation` now raises "Not enough market-stress history to compute this: {n} stress days available, 20 required." The old text ended "Consider lowering the drawdown threshold."
- **Unchanged:** the floor (named `MIN_STRESS_OBS = 20`), the −15% threshold and `MIN_STRESS_DAYS = 60`. The server-log warning in `data_ingest` is untouched because it is never user-facing.

### Item 3: history length

**K9. The fix.**
- **Options:** (a) extend `PRICE_HISTORY_YEARS`; (b) stop truncating every ticker to the youngest listing.
- **Choice:** (b). History was already 5 years; the ~3-year frame came from the cross-ticker `dropna`.
- **How:** `fetch_prices` now drops only rows where every ticker is missing, and `compute_log_returns` likewise.
- **Unchanged:** the 60%-of-5-years admission rule, so the screened universe is the same.
- **Consumer audit:** correlation, covariance, stress, discovery context, beta, the portfolio engine and multi-horizon already drop NaNs over the tickers they use. `/api/prices` now keeps only the dates all requested tickers share.

**K10. Measured effects (cached prices, as of 2026-07-07).**
- **Frame:** 794 → 1,253 rows, starting 2021-07-09.
- **Stress days:** 13 → 280, so the stress covariance builds.
- **Screener fit orderings:** identical under the old and new rule in all 12 goal × risk cases, and the last 300 rows are byte-identical. The frontier and ranker dumps are unaffected.

**K11. Side effect, flagged.**
- The optimizer's "Historical averages" expected return (`compute_return_stats` over the full store) now averages each ticker's own available history: about 5 years for most, less for young tickers. Previously it averaged the truncated ~3-year window for everyone. Numbers change; the method text ("recent return, annualized") is still accurate.
- **Also changes:** the 5Y columns. The 5Y return still needs more than 1,260 rows, while a 5-year fetch gives about 1,253, so 5Y return stays "—". Max DD (5Y) now uses the full window.

**K12. Memory from the history change alone.** The returns frame grows from 3.7 to 5.9 MB, and its derived normal and stress copies grow proportionally, for roughly +10 MB total. That is negligible next to the snapshot and refresh figures in K15, so extending history is feasible within 512 MB.

### Item 4: cold start

**K13. Snapshot scope.**
- **Finding:** a cold screen is about 100 s of prices plus about 400 s of fundamentals (measured by the builder: 103 s and 408 s).
- **Choice:** snapshot both, because a prices-only snapshot cannot deliver a first screen in seconds.
- **Format:** `app/data/market_snapshot.json.gz` (4.6 MB): `{asof, built, prices: {dates, columns}, volumes, fundamentals}`.
- **Builder:** committed as `scripts/build_market_snapshot.py`, unlike `caps.json`'s external builder, so the snapshot is reproducible. Rebuilding is a manual step.

**K14. Snapshot data.**
- **As of:** 2026-09-25, the last completed session. Fundamentals (P/E, earnings dates, market cap) are as of the build date.
- **Labeling:** the screen response carries `prices_asof` and `data_source`. Discovery shows "Prices as of 2026-09-25 (snapshot; live refresh runs in the background)".
- **Caching:** the screen cache key includes the as-of date and source, so post-refresh screens are not served from snapshot-era cache entries.

**K15. Live refresh (made opt-in after measuring).**
- **Design:** a daemon thread that fetches prices and fundamentals and swaps them in only when complete (at least 50 fundamentals rows). On failure it logs and keeps the snapshot. It never blocks a request.
- **Measured, macOS RSS, same process:**

  | Stage | RSS |
  |---|---|
  | Libraries imported | 139 MB |
  | App imported | 194 MB |
  | Snapshot loaded | 289 MB |
  | + live price fetch | 391 MB |
  | + live fundamentals fetch | 498 MB (max RSS 499) |

- **End-to-end:** the full cold-start run with the refresh on peaked at 545 MB.
- **Old app for comparison** (855c8e0, same measurement): 204 MB at start → 445 MB after its in-request cold screen (507.6 s).
- **Choice:** the refresh is **off by default**; `QX_LIVE_REFRESH=1` enables it. With the refresh on, a 512 MB instance is at or over its limit; without it, steady state is about 311 MB (start 290, after screen 305, after analytics 311), which is 134 MB below the old app after a screen.
- **Labeling:** the snapshot is labeled "Prices as of {date} (market data snapshot)". Keeping it current means re-running `scripts/build_market_snapshot.py` and committing the file.
- **Tests:** `QX_SNAPSHOT_LOAD=0` disables the startup load; `tests/conftest.py` sets both switches, so tests never touch the network.
- **Caveat:** Linux RSS on Render may differ from macOS. The numbers are relative evidence, not a Render measurement.

**K16. `_ensure_data` behavior.**
- **Before:** a single missing ticker replaced the whole store with a re-fetch of just the requested set. After a custom ticker, the next screen then re-fetched the entire universe.
- **Now:** only the missing tickers are fetched and merged. Tickers a fetch cannot supply are remembered so they are not re-fetched on every request. Universe tickers absent from the snapshot are marked that way at load.

**K17. Date convention.**
- **Found by** the cold-start run: yfinance returns exchange-timezone-aware dates, while the snapshot stores plain dates, and merging failed ("Cannot join tz-naive with tz-aware").
- **Choice:** all frames use tz-naive calendar dates (`_naive_dates`). There is a test with tz-aware fetch output.

**K18. Sample-data banner.**
- **Choice:** shown on every tab whenever the rows are the built-in sample set: while the screen is pending, or after it fails. Text: "SAMPLE DATA: illustrative rows, not live analysis", plus the cause.
- **Removed:** the uvicorn instruction and "may take 30-60s". The status-dot tooltip now reads "Backend connected" or "Backend unreachable".

### Item 5: Sharpe label

**K19. Does screener Sharpe feed the ranker?** **Yes, twice.**
- `compute_factor_scores`: `raw["quality"] = sharpe`, giving `z_quality`, which feeds the goal composite (55 points).
- `compute_fit_scores`: `quality_pts` thresholds on `sharpe` (7 points).
- It was left untouched.

**K20. Display fix.**
- **Backend:** a new display-only field `sharpe_rf` = (exp(Σ 252-day log r) − 1 − rf) ÷ vol, the same formula as the Build 3 panel.
- **Frontend:** `shRf` holds it; `displaySharpe()` is used by the SHARPE column, the SHARPE (1Y) label, the SPY reference and the builder holdings table.
- **Sample and custom rows:** these use `sh`. `/api/ticker/add` already subtracts rf.
- **Sorting:** the SHARPE column sort key moved to `shRf`, so the sort matches the displayed numbers. That is user-initiated sorting only; the default screen-match order is untouched.
- **Still on `sh`:** the builder's SCREEN MATCH `fitScore` and the hidden Trade Desk, which is ranker-consistent.

### Item 6: Ask Quantex

**K21. How the panel knows the AI is off.**
- **Choice:** a new `GET /api/ai/status` → `{enabled: bool}`, revealing nothing else.
- **With no token:** the panel is the header plus one line: "The AI explainer isn't enabled on this demo."
- **Errors:** a 503 mid-session switches to that line. Other errors show "couldn't answer just now (error N)". Raw server text is never displayed.
- **Removed:** the unused Anthropic-key gear and its `localStorage` key.
- **Unchanged:** the backend 503 detail, which is an operator-facing log message.
- **With the token set:** unchanged apart from the gear removal.

### Item 7: β chip

**K22.** All band states ("In band ✓", "Above band ▲", "Below band ▼") render in neutral `#94a3b8`. Build 3's J15 flag is resolved.

### Left in place, flagged

**K23.** Some copy was not touched: `/api/correlation-diagnostics` still says "consider adding bond or gold exposure…" / "Consider adding positions…", Frontier still shows "Near optimal ✓ / Consider optimizing…", and Correlations shows "No high-correlation pairs — good diversification.". These are now computed from real data but still use advice or verdict vocabulary. They are out of this remediation's scope and listed in DEMO_READINESS.

**K24.** The hidden Trade Desk still renders BUY/SELL from the legacy Sharpe verdict and the sample `tgt`/`stop` fields. It is hidden by `DEMO_MODE`, and its redesign is owned by TRADE_DESK_SPEC.

---

## Part 3: Final pass (2026-09-29), judgment calls L1–L12

**L1. The Sharpe sanity check failed; engine fixed (real error, not annualization).**
- **Portfolio:** MU/WDC/DELL/HPE at 25% each, 2025-09-25 → 2026-09-25, 252 daily returns, rf 4.3%.
- **Independent plain numpy** (simple returns, rebalanced to current weights): return +372.33%, volatility 55.84%, **Sharpe 6.5902**.
- **Engine before the fix:** +321.69%, 55.25%, **Sharpe 5.7442**.
- **Cause:** the engine built the portfolio series as a weighted sum of *log* returns and compounded it. That is not a portfolio return. It understates returns by ½·(Σwᵢσᵢ² − σ_P²) per day, which compounds to −50 pp here. Build 3's J3 described this approximation as "0.02 pp on a 2% day". That is true for one day and wrong for a year; the correction is recorded here.
- **Fix:** `select_window` converts the store's log returns to simple returns once. The portfolio series is Σwᵢ·rᵢ (current weights, rebalanced daily), compounded as ∏(1+r), with drawdown on ∏(1+r) and VaR, capture, correlations, the covariance and Sharpe all on simple returns.
- **After:** the engine gives +372.3293%, 55.8449%, **Sharpe 6.590206**, identical to the independent value.
- **Tests:** a regression test (high-volatility synthetic portfolio versus plain price arithmetic) pins it.

**L2. Basis choice: rebalanced to current weights, not buy-and-hold.**
- The panel describes the portfolio at its *current* weights. Rebalanced-to-current-weights is also the convention of the builder's growth chart ("Assumes portfolio rebalanced to current weights").
- Buy-and-hold from the same start weights gives +345.28% here.
- The panel header now states the basis: "current weights, rebalanced daily".

**L3. What stays on log returns.**
- The β chip and β preview (shipped BETA_TRACKER regression on the store's log returns) and the §5 similarity markers (same basis as the Discovery Corr column).
- For β and correlation the log/simple difference is second order, and changing a sealed prior spec's chip was out of scope.

**L4. Sharpe basis labels.**
- **Health panel:** "Sharpe (realized, past 252 trading days)", with the tooltip "Realized: what this portfolio at current weights actually earned over the past 252 trading days, per unit of realized volatility. The Frontier tab's Sharpe is a model expectation instead."
- **Frontier:** "Sharpe (model expectation)" in the info line and table header, with the tooltip "Model expectation: expected return (historical averages over the stored price history) over modeled volatility (blended normal/stress covariance). Portfolio Health's Sharpe is the realized figure for the past 252 trading days instead."
- The status bar and Compare carry the realized label too.

**L5. Wording scope.**
- **Changed:** user-facing app copy about the user's own data or profile (analytics, profile flags, optimizer goals, challenges), plus backend strings shown in the UI and one server-log line.
- **Excluded, and why:**
  - CFA curriculum text (CourseHub readings and quizzes; `paper_trading.py` concept definitions), where "should" and "optimal" are textbook usage.
  - Disclaimers that negate a recommendation.
  - The AI system prompt.
  - The hidden Trade Desk (TRADE_DESK_SPEC).
  - Order-type explainers ("Best for: …"), which are instrument education.
  - Third-party analyst fields in `fundamental.py`, labeled as third-party facts.
- The regression test scans app copy with exactly these exclusions.

**L6. Optimizer goal descriptions.** The "Good if/Good for …" suffixes were deleted rather than reworded; the goal text states only what the objective computes.

**L7. Holdings-count copy.** The backend warning "Only N positions … Consider adding positions to at least 8." is removed. The Correlations tab now always shows "N holdings; the Portfolio Health panel's effective bets shows X.", following your example.

**L8. Frontier's reference point.** "G pp below the frontier's F% at this volatility" interpolates the frontier's return at the portfolio's own volatility. Outside the frontier's volatility range the line says so. Within 0.005 pp it reads "on the frontier at this volatility".

**L9. Challenge reset.**
- **Where:** progress lives in `localStorage` (`qx_challenges`). A versioned migration (`qx_challenges_version = "2"`) runs once at page load, before any component reads progress.
- **What:** it clears everything, including the self-marked "Income Engineer".
- **Notice:** shown once, dismissible, and only if progress actually existed; a first-time user has nothing to reset.
- **Implementation:** storage access goes through small try/catch helpers (`_lsGet`/`_lsSet`/`_lsDel`), per the repo's rule on wrapping `localStorage`.

**L10. `refresh_snapshot.py`.**
- **Design:** wraps `build_market_snapshot.py` (now parameterized by output path). It builds to a temporary file in `app/data/` and validates before an atomic replace: at least 500 tickers, at least 500 fundamentals rows, stress days ≥ `MIN_STRESS_DAYS`, and an as-of date not older than the committed file's. `--check` validates the committed file with no network.
- **Not run.** Only its validator is exercised by a test, against the committed snapshot.

**L11. DEMO_READINESS rewrite.**
- **Replaced** with the post-remediation walkthrough, from a fresh browser pass on this branch.
- **Kept:** the first version's findings, summarized in a "what changed" table (the original stays in git at 06739ea).
- **Listed as open:** items outside every remediation so far (shared paper account and dead trade log, landing proof block, course placeholders, the ETF filings message, the default "Aggressive" profile).

**L12. Merge readiness.** Merge only if the suite is green and the ranker ordering dump is byte-identical against 855c8e0 (re-measured at the end of this pass).
