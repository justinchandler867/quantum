# DEMO_READINESS.md: reviewer walkthrough, DEMO_MODE=1, no keys

**Date:** 2026-09-28. **Branch:** demo-prep. **Scope:** findings only. The only fixes on this branch are the one-worker Dockerfile and the visible SEC Filings control.

## Test configuration

A local instance was configured to behave like the deployed demo:

- Started with `uvicorn app.main:app --workers 1` (the Dockerfile after this branch).
- Environment: `DEMO_MODE=1` and `DEMO_PASSWORD` set.
- No `REPLICATE_API_TOKEN` and no user-entered Anthropic key.
- `REDIS_URL` pointed at a closed port, so there was no Redis.
- `FILINGS_CACHE_DIR` pointed at a fresh empty directory, so the filings cache was ephemeral.
- Live yfinance and EDGAR over the network.

### Evidence tags

Every observation below carries one of these tags:

- **[browser]:** rendered in headless Chrome against the instance, with text captured from the DOM (landing, guest entry, onboarding, first screen). Chrome auto-updated mid-session and would not launch again after that, so the rest of the walk could not be rendered.
- **[API]:** the exact request each tab makes, sent to the same instance, with the status, timing and body recorded.
- **[code]:** the string that `static/quantex.html` renders for that API outcome, cited by line.

When a tab's text is tagged [code], the backend response that drives it was still observed [API].

## Top three items that would embarrass a demo

### 1. Cold start: the first screen times out and the session drops to sample data

**What happened:**

- On a cold instance, Start Discovering ran the 596-ticker screen: a 5-year price fetch, then roughly 6.5 minutes of yfinance fundamentals calls. The server finished the pipeline 8m08s after the request started.
- The frontend aborts at 180 s (`API_TIMEOUT`). At 180.2 s the reviewer saw this [browser]:

  > ⚠ Backend: Request timed out — showing sample data. Start the backend with: uvicorn app.main:app

- The status dot turned from 🟢 to 🔴.
- `setBackendOk(false)` stays in effect for the rest of the session. Correlations and Frontier then skip the backend entirely (`if(... !backendOk) return`), and Paper Trade stops refreshing orders.
- Changing the profile does not retrigger the screen (the retry is guarded by `backendOk!==false`). Only a full page reload does.

**Why it recurs:**

- The screen cache key is (goal, risk score, horizon, max results).
- A second reviewer whose onboarding answers differ from the first reviewer's pays for a fresh cold screen. Measured: 529.4 s for Income / risk 40 / 5y [API], again well past the 180 s abort.

**Why it is worse than a timeout:**

- During those 180 s, and afterwards, the Discovery table shows the 38 hard-coded `FALLBACK_ASSETS` (AAPL 95, MSFT 95, NVDA 95…).
- Those rows carry real-looking labels ("1.31 · Beat market"), and nothing on the table says the rows are sample data [browser].
- The only notice is the warn box, and it tells a reviewer to run a shell command.

### 2. Four builder and analysis features fail server-side; three quietly substitute random-correlation math

**Root cause:**

- `fetch_prices` forward-fills gaps, then drops every row that still has a gap in any ticker. A few recent listings therefore truncate the whole 5-year matrix to 761 days (log: "Price matrix: 761 days × 583 tickers").
- With the current −15% drawdown rule, that window contains only 13 stress days (log: "Only 13 stress days found (minimum 60)").
- Every endpoint that builds a stress covariance therefore returns 422 [API]:

  > Only 13 stress observations — too few for stable estimate. Consider lowering the drawdown threshold.

- Affected endpoints: `/api/correlations`, `/api/correlation-diagnostics`, `/api/frontier`, `/api/optimize`, `/api/stress_test_2026`. `/api/sandbox/stress` uses the same stress path; this is inferred from code and was not called.

**What each tab then shows [code]:**

- **Correlations:** the fetch error is swallowed (`.catch(()=>{})`). The matrix falls back to `getCorr()`, which assigns correlations by asset type plus `Math.random()`. The page then states:
  - `Avg pairwise ρ: …`
  - either `No high-correlation pairs — good diversification.` or `⚠ A / B: ρ=0.78 — limited diversification`

  These are computed from random numbers and presented as analysis, with no notice.
- **Frontier:** falls back to a local Monte Carlo over the same random correlations. The badge reads `⚠ Monte Carlo (250 trials)`, but the loop runs 150 trials (`quantex.html:2532`). The info box says `Near optimal ✓` or `Consider optimizing in Portfolio Builder`.
- **Optimizer** (◈ A/B/C): `optPort` catches the 422 (`console.warn("Backend optimizer failed, using local")`, `quantex.html:4167`) and applies `optimizeLocal()`, a 250-draw random search over random correlations. The results card still says:

  > Found the weights that maximize (return − risk-free) ÷ volatility — the highest Sharpe ratio your constraints allow.

  The weights on screen are not what that sentence describes.
- **Regime Stress Test:** shows the raw 422 detail, including "Consider lowering the drawdown threshold." That is a developer instruction shown to a reviewer.

### 3. Ask Quantex answers every question with a server-configuration message

**What the reviewer sees:**

- The panel is on every tab. The header badge reads `No Key`.
- Any question, including the quick buttons, returns [API: 503]:

  > AI features are currently unavailable (HTTP 503). {"detail":"AI features require a REPLICATE_API_TOKEN environment variable on the server. Set it in your Render dashboard under Environment."}

- The reply is raw JSON, and it names the hosting provider and tells the reviewer to configure it.

**The dead key control:**

- The ⚙ control opens an "Anthropic API Key" field, "(stored locally, never sent to any server except Anthropic)".
- Since the Replicate switch, `askClaude` never uses that key (see the comment at `quantex.html:2711`).
- Entering a key only changes the badge to `Claude ✓`. Every answer is still the 503.

## Tab by tab

### Landing page (before the app) [browser]

- **Works:** the page renders. "Try as guest first" enters the app.
- **Retired vocabulary:** the proof block shows `SHARPE (1Y) 1.42 · Moderate`. "Moderate" is the grade vocabulary Amendment 3 retired; Discovery now renders "Beat market / Trailed market".
- **Cards that do nothing:** "Track Mode" and "Full Pro Mode" look like modes a user can pick, but no such modes exist in the app.
- **Unverifiable claim:** "run through an 18-gate shakedown against sealed predictions. Six findings. All fixed and regression-tested." A reviewer cannot check this from the page.
- **Double login:** after the HTTP Basic Auth prompt, "Sign in" asks for an email and password again. The password is accepted and ignored (`submit()` never reads `pass`), and the account exists only in localStorage.
- **Stale market ticker:** the top bar shows hard-coded values (`MKT` constant): SPY 527.84, QQQ 449.21, VIX 18.42, 10Y 4.38%, GLD 302.17. They are not live and carry no as-of date.

### ① Profile (onboarding) [browser]

- **Works:** the three-step wizard. Default answers produce `88/100 Aggressive`.
- **Advisor-intake wording:**
  - Step copy: "This shapes everything — which assets we recommend, how we optimize, and what risk guardrails apply."
  - Flag text can read "Low loss tolerance — consider more conservative profile" (`scoreProfile`).
  - Eligibility grid: "Options ✗ No / Warrants ✓ Yes".
- **Default drift:** a reviewer who clicks straight through is labeled Aggressive.

### ② Discovery [browser + API]

**Works:**
- Ranked table with the "Ranked by screen match — not a recommendation" framing, the Growth momentum warning, and the Sharpe (1Y) reference labels.
- Search, which returns NASDAQ/NYSE matches instantly [API].
- Adding a mutual fund by symbol (VTSAX, 0.8 s) [API].
- The Corr column, which returns in 0.0 s on a warm store [API].

**Degrades:**
- Every stress correlation is null (`status: "no_stress_window"`, `days_stress: 13`), so the column reads `0.39 / —` for every row [API].
- The corr-flip marker ⚠ can never appear.
- The 3Y, 5Y, MAX DD (5Y), P/E, earnings, dividend-growth, debt/equity and revenue-growth columns are all `—` on sample data (see top item 1).

**Inconsistencies:**
- **Sharpe definitions:** the SHARPE (1Y) tooltip and ⓘ modal say "(1-year return − risk-free rate) ÷ annualized volatility", and "Lost vs cash" is described as "return fell below cash (risk-free 4.3%)". The screener's Sharpe for the whole shortlist is `mean(log r)×252 / vol` with no risk-free subtraction (`app/screener.py:385-386`). "Lost vs cash" therefore fires only when the return is negative, not when it is below 4.3%. By contrast, `/api/ticker/add` does subtract rf (`app/main.py:780-781`), so two rows in the same column can use two formulas.
- **Beta benchmarks:** `/api/ticker/add` computes beta against QQQ (`app/main.py:768-777`); the screener computes it against SPY.
- **Sector labels:** mutual funds added by symbol get `sec: "MUTUALFUND"` (the yfinance quoteType fallback), not a sector.

### Ticker detail modal (row click) [API + code]

**Works:**
- Price chart (`/api/prices` 0.0 s) and news (0.3 s).
- **SEC Filings, new on this branch:** a labeled "SEC Filings ↓" button in the modal header scrolls to the Filings panel. Loading a first-row stock (TWLO) took 3.0 s against an empty cache, and the WHAT CHANGED findings rendered with citation chips [API].

**Browser verification:** the header control could not be exercised in a browser; see Test configuration. It is verified by `tests/test_filings_entry.py`, which renders the real `TickerChart` under node, clicks the button and asserts the scroll target.

**Wrong filings message for ETFs:**
- Loading SPY returns `status: unsupported_foreign`, and the panel says (only SPY was tested; other ETFs likely behave the same):

  > Filings analysis not yet available for foreign private issuers (20-F filers).

- SPY is a US ETF, not a foreign private issuer. The message is false for SPY.

**Third-party headlines:** news can include headlines such as "What Needs To Be True To Buy Twilio Stock Now?" (Trefis). The modal gives no framing that these are third-party.

**Indicator Consensus:** with no indicators toggled, the overlay prompts "Turn on indicators". Family votes use the words bullish, bearish and hold, along with the "Not a trade recommendation" disclaimer.

### Metrics [code]

Everything on this tab comes from `calcStats()`. It does not use backend data.

- **Volatility:** uses `getCorr()` correlations (type-based plus `Math.random()`).
- **VaR / CVaR:** parametric, `1.645σ`.
- **MAX DD:** literally `−2.1 × vol`.
- **Sortino:** `Sharpe / 0.72`.
- **Beta:** a weighted average of per-asset betas. The builder's β chip says "Regressed, not a weighted average of betas", so the same screen family disagrees with itself.
- **BENCH RET:** `β × 24.2` (hard-coded).
- **FACTOR EXPOSURES:** invented transforms. For example, Quality is Sharpe/1.5 capped at 1.
- **STRESS TESTS:** 2008 / 2020 / 2022 read the static `s08/s20/s22` fields. For screened names these are `0`, so a real screened portfolio shows `+0.0%` in every crisis.
- **CFA ⓘ explainers:**
  - SHARPE: "Higher is better. Above 1.0 is good… Below 1.0 — consider optimizing to improve risk-adjusted returns."
  - BETA: formula `βp = Σ(wi × βi)`.

### Correlations [API + code]

- Falls back to random correlations; see top item 2.
- If the backend were healthy, the server's warning copy would still read:
  - "No negative-correlation positions — consider adding bond or gold exposure for crisis protection"
  - "Only N positions … Consider adding positions to at least 8."

  Both are advice vocabulary that survived remediation (`app/main.py:366-375`). The panel headings "— hedging benefit" and "— no diversification in crisis" are verdicts.

### Frontier [API + code]

- Falls back to local Monte Carlo; see top item 2. The mislabeled trial count and "Consider optimizing in Portfolio Builder" apply.

### ⚡ Derivatives [code]

- **Works:** fully client-side (BSM pricer, Greeks, payoff diagrams). No backend needed; the `/api/derivatives/price` 200 was confirmed [API].
- **Strike input ignored in the sandbox:** the strategy sandbox always prices strikes at spot ±5%, whatever the pricer's STRIKE input says.
- **Dead code:** `fetchSpot` does nothing (it has no button, so a reviewer won't notice).

### 📄 Paper Trade [API + code]

- **Works:** a market BUY of 10 AAPL filled at $338.40 [API].

**Shared account:**
- Every visitor shares one server-side account. The frontend never sends `account_id`, so all calls hit `"default"`.
- Reviewer B's positions table fills with reviewer A's trades as soon as B's first order fills (`refreshAccount`).

**Two sources of truth:**
- The ACCOUNT card first renders from the browser's localStorage (`qx_paper`), and only after the first filled order from the server.
- Cash and positions can jump when that happens.

**Metric problems:**
- **Win Rate** is `sells ÷ all trades`, not profitable trades.
- **Sharpe Ratio** is annualized over per-click NAV snapshots, not daily returns.

**Restarts:** all server state is in memory. A restart or free-tier sleep erases it; the localStorage copy does not.

### 📚 Learn: Course [code]

- **Works:** 12 modules with quizzes.
- **Placeholder readings:** six modules (Ethics, Economics, FSA, Equity Valuation, Fixed Income, Alternative Investments) open with:

  > Reading Coming in Next Batch — The full reading for this module is being prepared separately.

- **Placeholder video:** every module's 🎬 Video tab reads "Video content coming soon … curated video lessons from financial education creators."

### 📚 Learn: Challenges [code]

- **Graded on fabricated numbers:** challenges auto-complete based on `calcStats()` values (see Metrics). A reviewer can "earn" Calm Waters or Drawdown Defender on random correlations and the `−2.1×vol` drawdown.
- **Hints name securities:**
  - "Focus on high-Sharpe assets like MSFT, COST, LLY."
  - "Mix low-beta defensive stocks (JNJ, PG, GLD)…"
- **Completion copy overreaches:**
  - "You generated genuine alpha — outperformance beyond what market exposure explains."
  - "…should fall less than the market in downturns."

### ⇄ Compare [code]

- **Works:** needs two non-empty portfolios.
- All values come from `calcStats()` (random correlations).
- "Expected Return" is the trailing 1Y return.
- A ★ plus green highlight marks the "best value per metric", which is a verdict by pigment.

### ◈ A / B / C (builder) [API + code]

- **β chip works** [API: β 1.2445, R² 0.29, 252 obs].
- **Removal preview mismatch:** the chip's remove-hover preview computes β on the remaining weights renormalized. The actual remove (`rmFromPort`) re-splits the remaining holdings equally, so the preview does not describe the result.
- **Rolling stock-bond chart works** [API].
- **Optimizer and stress test broken;** see top item 2.
- **Top stat grid:** RETURN, VOL σ, SHARPE, BETA, SORTINO, TREYNOR, VAR 95%, MAX DD come from `calcStats()` (random correlations), colored green/amber/red.
- **Holdings detail:** Sharpe and screen match are also green/amber/red.

### Trade Desk

- Hidden by `DEMO_MODE=1` as intended [browser: nav shows no TRADE DESK]. Its backend endpoints still exist, but the Trade Desk is client-side only.

### Ask Quantex (right sidebar, all tabs)

- See top item 3.

### Always visible

- The top status bar (PORT, RETURN, VOL σ, SHARPE…) uses `calcStats()`, as on the Metrics tab.
- "Sign Out" returns to the landing page, not to the HTTP auth prompt.

## Other operational notes

- **Render health check:** `/health` is exempt from auth and returns 200. `/docs` and `/openapi.json` are behind auth [API].
- **No Redis:** logs one warning and uses in-memory caching, which is fine.
- **Filings cache:** starts empty on every restart. Measured cost: TWLO filings took 3.0 s cold, acceptable.
- **One worker:** a screen blocks nothing else (it runs in the executor), but each cold screen takes 8 to 9 minutes; see top item 1.
