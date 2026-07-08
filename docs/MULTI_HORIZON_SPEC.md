# Multi-Horizon Columns + 52-Week Extension — Sealed Spec

**Status:** SEALED 2026-07-07. Amendments by dated addendum, written before rebuild.
**Delivers:** roadmap item 2 (1/3/5Y + max drawdown) + the "% off 52-wk high"
column that INTERFACE_COPY.md §2's held-back sentence references.

## 0. FINDING TO VERIFY FIRST (F1 — potential live display bug)

`data_ingest` computes LOG returns; `screener.py` displays `sum(log returns) × 100`
as `return_1y/6m/3m`. If confirmed, every displayed return is understated for
gains and overstated for losses (e.g., Σlog = 0.842 displays "+84.2%" but the
true simple return is e^0.842 − 1 ≈ +132%). The builder must:
1. Confirm whether `data_ingest` returns are log or simple (read the code, then
   verify numerically for one ticker against raw adjusted closes).
2. If log: fix display conversion everywhere a return is shown to
   `(exp(Σlog) − 1) × 100`, as part of this build, with a regression note.
   FACTOR MATH IS UNTOUCHED — z-scores of momentum on log sums are
   rank-equivalent and stay as they are. Display only.
3. Log the finding either way (bug or my misreading — both are findings).

## 1. Definitions (sealed)

All from the cached adjusted-close price frame (5y depth per
`PRICE_HISTORY_YEARS`); all displayed as SIMPLE total returns:

- `return_1y` = P_t / P_{t−252} − 1 (display %, existing column, fixed per F1)
- `return_3y` = P_t / P_{t−756} − 1 (cumulative, NOT annualized — the tooltip
  in INTERFACE_COPY.md §3 assumes cumulative "total return over each period")
- `return_5y` = P_t / P_{t−1260} − 1 (cumulative)
- `max_dd_5y` = min over available window (≤1260d) of (P/cummax(P) − 1)
- `pct_off_52wk_high` = P_t / max(P over trailing 252d) − 1  (≤ 0 by construction)

Insufficient history: if a ticker lacks the full window for a column, render
`—` with tooltip "listed less than N years" — NEVER a partial-window number in
a column headed "3Y"/"5Y". `max_dd_5y` may use a shorter available window but
must then be flagged (`dd_window_days` in the API response; UI shows an
asterisk under 756 days).

## 2. API

Extend `/api/screen` response fields per asset: `return_3y`, `return_5y`,
`max_dd_5y`, `dd_window_days`, `pct_off_52wk_high`. Computed in the same pass
as existing metrics (no extra data fetches; the price frame already covers 5y).

## 3. UI

- Columns per INTERFACE_COPY.md §3, tooltips verbatim from that doc.
- Rendering rule (sealed there, enforced here): 1Y never renders without
  3Y/5Y/MaxDD alongside.
- Restore §2's held-back final sentence ("The '% off 52-wk high' column shows
  you the extension directly.") in the same commit — per the 2026-07-07
  addendum in INTERFACE_COPY.md.
- Sort: all new columns sortable; default sort unchanged (fit score).

## 4. Sealed audit cases (pytest before UI; divergence = finding)

- **M1** Synthetic series: 1260 days, price doubles smoothly over final 252d
  → return_1y = +100.0% ± 0.1; pct_off_52wk_high = 0.0.
- **M2** Synthetic: price 100 → 150 → 90 → 120 over 5y (linear segments)
  → max_dd_5y = −40.0% ± 0.1 (150→90); return_5y = +20.0% ± 0.1.
- **M3** Log-conversion check (F1): for one real ticker, displayed return_1y
  must equal P_t/P_{t−252} − 1 from raw adjusted closes within ±0.5%
  (tolerance covers dividend-adjustment timing), NOT the log sum.
- **M4** Ticker with 400 days of history → return_1y numeric; return_3y,
  return_5y render `—`; max_dd flagged (dd_window_days = ~400).
- **M5** pct_off_52wk_high is ≤ 0 for every asset in a real screen run
  (> 0 is impossible by construction; any positive value = engine bug).

## 5. Non-goals

- No annualized variants, no CAGR column (v2 if users ask).
- No factor/composite/fit changes — ranks must be byte-identical before and
  after this build (regression: same shortlist order on the same cached data).
- No new data providers.
