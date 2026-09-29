# DEMO_READINESS.md: reviewer walkthrough (post-remediation)

**Updated:** 2026-09-29, branch launch-honesty. The first version of this walkthrough (demo-prep, 06739ea) recorded the pre-remediation state. Its top three findings are resolved below; the git history keeps the original.

## Configuration assumed

The deployed demo runs:
- `uvicorn --workers 1` (Dockerfile);
- `DEMO_MODE=1` and `DEMO_PASSWORD` set;
- no `REPLICATE_API_TOKEN`;
- no Redis;
- an ephemeral filings cache;
- `QX_LIVE_REFRESH` unset (off), so the app serves the committed market snapshot.

Everything below was observed in headless Chrome against that configuration on 2026-09-29, unless marked [code].

## What changed since the first walkthrough

| First walkthrough (06739ea) | Now |
|---|---|
| Cold start: first screen took 8 minutes; the frontend gave up at 180 s and showed sample data with "Start the backend with: uvicorn" | The first screen renders in about 0.4 s from the committed snapshot, labelled "Prices as of 2026-09-25 (market data snapshot)". If sample rows are ever shown, a banner says "SAMPLE DATA: illustrative rows, not live analysis". |
| Correlations, Frontier, Optimizer and 2026 Stress Test failed server-side (13 stress days), and three of them quietly used random-number correlations | Price history keeps each ticker's own 5 years (1,253 days, 280 stress days), and all four compute. The random fallbacks are gone; any failure shows its cause. |
| Ask Quantex answered with raw JSON setup instructions | One line: "The AI explainer isn't enabled on this demo." |

## Tab by tab

### Landing page
- **Works:** renders; "Try as guest first" enters the app.
- **Unchanged** (outside the remediation scope):
  - The proof block still shows `SHARPE (1Y) 1.42 · Moderate`, the retired grade vocabulary.
  - The "Track Mode" and "Full Pro Mode" cards are not real modes.
  - The "18-gate shakedown" claim cannot be checked from the page.
  - "Sign in" accepts any email and password (local-only account).

### Top bar and status bar
- **Top bar:** last closes of SPY, QQQ, GLD and TLT from the loaded data, plus "CLOSE AS OF 2026-09-25". The old hard-coded quotes are gone.
- **Status bar:** for the active portfolio it reads RETURN 252D, VOL σ, SHARPE (REALIZED 252D), BETA, MAX DD, VAR 95% 1D and YIELD, all from the backend risk engine. It shows "—" when a value is unavailable, never a guessed number.

### ① Profile
- **Wizard:** three steps, unchanged. Default answers score "88/100 Aggressive", whichever goal is picked.
- **Step copy:** "Your goal sets the factor weights behind screen match, the optimizer's constraints, and the profile limits (max position, max beta)."
- **Profile flags are factual.** Example: `If the portfolio fell 25%: "Sell everything" · profile score N/100 (the score counts this answer at −20)`.

### ② Discovery
- **Header:** "Prices as of 2026-09-25 (market data snapshot)".
- **Table:** ranked by screen match (Balanced: shown unranked, with the walk-forward explanation).
- **Sharpe columns:** SHARPE and SHARPE (1Y) show (1-year return − 4.3% risk-free) ÷ volatility. The labels Beat market / Trailed market / Lost vs cash use that same basis. The ranker's own internal Sharpe input is unchanged.
- **SPY reference:** when SPY isn't in the rows and the SPY reference fetch fails, the label shows "—".
- **Hover preview:** hovering a `+ A` / `✓ A` button shows the vol/β preview card (Build 3).
- **Similarity markers:** candidate rows with a shortlist peer at r ≥ 0.85 show "Similar to X (r)".

### Ticker modal (row click)
- **Contents:** chart, indicators and news, as before.
- **SEC Filings:** the "SEC Filings ↓" header button scrolls to the panel. "Load filings →" fetched the first row's filings (an AES 10-K) in seconds, with WHAT CHANGED findings and citation chips.
- **Known issue:** SPY (and likely other ETFs) returns the message "Filings analysis not yet available for foreign private issuers (20-F filers)", which is wrong for a US ETF [code].

### Metrics
- **Contents:** a one-line basis note, then the Portfolio Health panel: vol, max drawdown, worst-5% day, risk contributions, effective bets, pairwise correlation, capture ratios, Sharpe (realized, past 252 trading days) and composition.
- **Removed:** the old stat boxes, factor bars and crisis rows (random or placeholder inputs).

### Correlations
- **Matrix:** backend matrix for normal, stress or blended regimes.
- **Fact lines:**
  - "N holdings; the Portfolio Health panel's effective bets shows X."
  - "Average pairwise ρ: a normal → b stress".
  - STRESS-CORRELATION FACTS.
  - High-correlation flags ("above the 0.75 flag level" / "stress-window ρ above 0.85").
  - NEGATIVELY CORRELATED PAIRS.
  - Reference hedges.
- **On failure:** "Correlations unavailable for the {regime} regime: {cause} No correlation numbers are shown in place of market data."

### Frontier
- **Chart:** SLSQP frontier from the backend. "Your Portfolio" is the backend's own evaluation of the current weights.
- **Info line:** "Your portfolio (model expectation): return R% · vol V% · Sharpe (model expectation) S · G pp below the frontier's F% at this volatility".
- **Tooltip:** explains that this Sharpe is a model expectation and that the Health panel's is realized.
- **On failure:** an honest failure line, with no local Monte Carlo.

### ⚡ Derivatives
- Client-side pricer and strategy sandbox, unchanged (the strategy sandbox still prices strikes at spot ±5%).

### 📄 Paper Trade
- **Works:** market buys fill at live prices; for example 10 MSFT filled at $508.69.
- **Still open** (outside scope):
  - The account is shared by every viewer (server account "default").
  - "Trades" and the transaction log stay at 0 after fills, because only the removed dead code path wrote them. Order history does record fills.
  - Win Rate and Sharpe here are computed from the (empty) local log.

### 📚 Learn
- **Course:** 13 modules, 174 questions. Six modules still open with "Reading Coming in Next Batch". The video tabs are placeholders.
- **Challenges:**
  - Returning users see once: "Challenge progress was reset: earlier results were computed on placeholder data." It is dismissible.
  - Completion checks read the risk engine's 252-day metrics.
  - Hints and completion text are factual; there are no ticker picks.

### ⇄ Compare
- Engine metrics per portfolio: return and yield, Sharpe (realized, 252d), volatility, beta vs SPY, max drawdown, worst-5% day and effective bets. ★ marks the best value per metric.

### ◈ A / B / C (builder)
- **Holdings:** sliders with the vol/β preview card on remove-hover and on slider drag.
- **Charts:** rolling stock-bond correlation, unchanged.
- **Optimizer:** runs SLSQP. On failure it shows "Optimization did not run: {cause} Your weights were not changed."
- **Stress tools:** the Regime Stress Test and Scenario Sandbox compute.
- **Portfolio Health panel** at the top right, with the β chip (neutral color in every band state). There is no second stat grid.

### Trade Desk
- Hidden by `DEMO_MODE`; its redesign is TRADE_DESK_SPEC's.

### Ask Quantex
- With no token: "The AI explainer isn't enabled on this demo." There is no key prompt.
- With `REPLICATE_API_TOKEN` set: questions go to the server proxy; errors show one plain line.

## Snapshot maintenance

The first screen reads `app/data/market_snapshot.json.gz`: 5-year adjusted closes, volumes and screener fundamentals for the universe. The UI labels it "Prices as of {date}". To refresh it:

```bash
cd backend && source venv/bin/activate
python scripts/refresh_snapshot.py --check   # validate the committed file (no network)
python scripts/refresh_snapshot.py           # rebuild (~8-10 min, network), validate, replace
python -m pytest -q                          # suite must stay green
git add app/data/market_snapshot.json.gz && git commit -m "snapshot: prices as of <asof>"
```

- **Validation:** `refresh_snapshot.py` builds into a temporary file and rejects the result, leaving the committed file untouched, if any check fails:
  - fewer than 500 tickers or 500 fundamentals rows;
  - fewer than 60 stress days;
  - an as-of date older than the committed file's.
- **Live refresh:** `QX_LIVE_REFRESH=1` enables the in-process refresh, which adds about 200 MB. Leave it off on a 512 MB instance (LAUNCH_HONESTY_DECISIONS.md K15).

## Open items for a reviewer

1. Paper Trade: shared account; the trade log and trade count don't update.
2. Landing page proof block uses the retired "Moderate" label; the Track/Pro mode cards are not real.
3. Six course modules have no reading yet; video tabs are placeholders.
4. ETF filings message names the wrong reason.
5. Default onboarding answers produce an "Aggressive" profile.
