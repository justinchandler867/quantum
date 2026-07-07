# Correlation-to-Portfolio Column + Rolling Correlation Visual — Sealed Spec

**Status:** SEALED 2026-07-07. Amendments by dated addendum only, written before rebuild.
**Merges:** the Discovery correlation gap (fit scores are univariate) with roadmap
item 1 (rolling stock-bond correlation with regime overlay).
**Grounded in:** `correlation_engine.py` (regime matrices, LW shrinkage),
`data_ingest.py` (stress-window identification), `config.py` thresholds,
existing endpoints `/api/screen`, `/api/correlations`.

---

## A. Discovery column: "Corr to your portfolio"

### A1. Definition (hand-derived, sealed)

Given the user's active portfolio (client-side holdings, value-weighted,
weights normalized to sum 1) and each shortlisted candidate `c`:

- Portfolio daily return: `r_P,t = Σ w_i · r_i,t` over the aligned trailing
  252-trading-day window (same convention as `compute_normal_correlation`).
- `corr_normal(c) = Pearson(r_c, r_P)` over that window.
- `corr_stress(c) = Pearson(r_c, r_P)` over the stress windows identified by
  `data_ingest` (−15% drawdown regime, ≥ `MIN_STRESS_DAYS` = 60 days).
- If `c` is already held: compute against the portfolio EXCLUDING `c`
  (renormalized), and badge the row `Held`. If the portfolio is a single
  position equal to `c`, both cells render `—`.

### A2. Band labels (anchored to existing config thresholds)

| corr_normal | Label |
|---|---|
| ≥ 0.85 (`HIGH_CORR_THRESHOLD`) | Near-duplicate |
| 0.75 – 0.85 (`HIGH_CORR_NORMAL_THRESHOLD`) | Very similar |
| 0.30 – 0.75 | Related |
| −0.05 – 0.30 | Diversifier |
| < −0.05 (`NEGATIVE_CORR_THRESHOLD`) | Hedge |

Stress cell displays the number plus a ⚠ marker when
`corr_stress − corr_normal ≥ 0.20` (the correlation-flip tell; same 0.20 the
2026 regime forces on stock-bond).

### A3. API

`POST /api/discovery/context`
Request: `{ holdings: [{ticker, weight}], candidates: [tickers], window?: 252 }`
Response per candidate:
`{ ticker, corr_normal, corr_stress, band, flip_flag, days_normal, days_stress, status }`
where `status ∈ {ok, held, insufficient_data, no_stress_window}`.
One batched call per shortlist; reads the cached price frame; no per-candidate
yfinance calls. No LW shrinkage needed (each computation is a 2-series Pearson;
shrinkage is for matrices — do not "improve" this).

### A4. Edge cases (all explicit, none silent)

- Empty portfolio → column hidden; header slot shows: *"Add holdings to see how
  each name relates to what you already own."*
- Overlap < 60 trading days → `insufficient_data`, cell `—`, tooltip explains.
  NEVER default to 0 (0 is a claim, not an absence).
- No qualifying stress window in the candidate's history → `no_stress_window`,
  stress cell `—`.
- ETF candidate vs portfolio holding the same ETF's top names → expected
  Near-duplicate; no special-casing.

### A5. UI copy

Column header: `Corr (calm / stress)`.
Tooltip: *"How this name moves with your current portfolio — in normal markets
and in past stress windows. High correlation means it adds exposure, not
diversification. Stress correlation is the one that matters when it matters."*
Footnote (already in INTERFACE_COPY.md §5) stays; this column is its
structural answer.

## B. Rolling correlation visual (roadmap item 1)

- Chart: rolling 63-day correlation between the user's equity sleeve and bond
  sleeve (sleeves = holdings bucketed by `sectors.json` category; fallback
  when no portfolio: SPY vs TLT, labeled as such).
- Overlay: stress windows from `data_ingest` shaded; horizontal marker at the
  2026-outlook stock-bond assumption (current correlation + 0.20) labeled
  *"2026 outlook assumption"*.
- Endpoint: `POST /api/correlations/rolling`
  `{ series_a: [tickers+weights] | "SPY", series_b: [...] | "TLT", window: 63 }`
  → `{ dates[], corr[], stress_windows[[start,end]] }`.
- Caption (sealed): *"Stock-bond correlation is not a constant. When it flips
  positive, bonds stop cushioning equity losses — this chart shows when that
  happened, and what the 2026 outlook assumes."*

## C. Sealed audit cases (run before any UI work; divergence = finding)

- **S1** Portfolio 100% SPY, candidate VOO → `corr_normal ≥ 0.99`,
  band Near-duplicate.
- **S2** Portfolio 100% SPY, candidate TLT → band Diversifier or Hedge;
  `corr_stress` differs from `corr_normal` by a visible margin (sign not
  sealed — read from data, then document).
- **S3** Synthetic pair with known ρ = 0.50 (Cholesky-constructed, seed
  20260707, n=252) → engine reproduces 0.50 ± 0.02.
- **S4** Candidate = sole holding → `held` path, cells `—`.
- **S5** Candidate with 30 days of history → `insufficient_data`, cell `—`.
- **S6** Empty portfolio → column hidden, placeholder copy renders.
- **S7** 60/40 SPY/TLT portfolio, candidate AAPL → `corr_normal` strictly
  between corr(AAPL,SPY) and corr(AAPL,TLT); sanity-bounds the weighting math.

## D. Non-goals (sealed)

- The fit score stays univariate. This column DISCLOSES the gap; it does not
  patch the ranking. Whether the ranking itself survives is VALIDATION_SPEC.md's
  question, and mixing the two builds would contaminate both.
- No optimizer changes; no auto-exclusion of Near-duplicates from shortlists
  (honesty over paternalism — show, don't hide).
- No new data providers; runs entirely on the cached price frame.

## E. Build order (Claude Code)

1. `/api/discovery/context` + audit cases S1–S7 as pytest (`tests/test_discovery_context.py`).
2. Discovery column UI + edge-case states.
3. `/api/correlations/rolling` + chart + regime shading.
4. Regression: existing suite stays green; `/api/screen` response unchanged
   (context is a separate call — screening latency must not grow).
