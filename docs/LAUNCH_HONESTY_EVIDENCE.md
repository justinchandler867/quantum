# LAUNCH_HONESTY_EVIDENCE.md — raw evidence, launch-honesty remediation

Branch launch-honesty from build3-portfolio-health 855c8e0. venv active for every run. History findings and judgment calls: docs/LAUNCH_HONESTY_DECISIONS.md (Part 1 written before code changed; K1–K24).

## Suite
```
baseline (855c8e0, verified first): 224 passed, 1 deselected
final:                             239 passed, 1 deselected  (+15: tests/test_launch_honesty.py)
node --check on the inline app script: OK
```

## Item 1 — history (summary; full tables in LAUNCH_HONESTY_DECISIONS.md Part 1)
```
496bb73 2026-04-16 12 CFA modules, YouTube links, Trade Desk fix, 43 quiz questions

git blame (855c8e0) lines 591-594, 623, 2769, 3216, 3305, 3335: all ^496bb73 (root commit, 2026-04-16)
Local truncation: YES since data date 2025-12-19 (GEHC crosses the 756-row admission rule -> matrix starts 2022-12-16 -> 13 stress days < 20 floor).
Old app observed (855c8e0, fresh process, today): cold screen 507.6 s; /api/correlations 422; /api/frontier 422.
```

## Item 2 — fabrication removed
```
grep -c 'Math.random' static/quantex.html -> 0
MATH_RANDOM_ALLOWLIST = {} (tests/test_launch_honesty.py)
tests/test_launch_honesty.py::test_no_math_random_outside_allowlist PASSED [  6%]
tests/test_launch_honesty.py::test_fabricating_helpers_removed PASSED    [ 13%]
tests/test_launch_honesty.py::test_old_builder_stat_grid_removed PASSED  [ 20%]
tests/test_launch_honesty.py::test_honest_failure_states_present PASSED  [ 26%]
tests/test_launch_honesty.py::test_stress_failure_message_names_cause_and_numbers PASSED [ 33%]
tests/test_launch_honesty.py::test_health_facts_never_invent_values PASSED [ 40%]
tests/test_launch_honesty.py::test_fetch_prices_keeps_each_tickers_own_history PASSED [ 46%]
tests/test_launch_honesty.py::test_snapshot_loads_and_first_screen_needs_no_network PASSED [ 53%]
tests/test_launch_honesty.py::test_ensure_data_fetches_only_missing_and_keeps_universe PASSED [ 60%]
tests/test_launch_honesty.py::test_display_sharpe_subtracts_rf_and_ranker_sharpe_unchanged PASSED [ 66%]
tests/test_launch_honesty.py::test_frontend_sharpe_cells_use_display_value PASSED [ 73%]
tests/test_launch_honesty.py::test_ai_status_endpoint PASSED             [ 80%]
tests/test_launch_honesty.py::test_ask_quantex_frontend_copy PASSED      [ 86%]
tests/test_launch_honesty.py::test_beta_chip_band_states_neutral PASSED  [ 93%]
tests/test_launch_honesty.py::test_live_refresh_is_opt_in PASSED         [100%]

```
Removed: getCorr, calcStats, optimizeLocal, Frontier local Monte Carlo, dead executeTrade/getPrice, MKT hard-coded quotes, builder stat grid, Metrics stat boxes/factor exposures/crisis rows, CFA stat-box explainers. Rewired to /api/portfolio/health: status bar, Metrics, Compare, Challenges, Ask Quantex context.

Honest failure strings (exact):
```
Correlations unavailable for the
Efficient frontier unavailable:
Optimization did not run:
SAMPLE DATA: illustrative rows, not live analysis
Not enough market-stress history to compute this: {len(subset)} stress days
```

## Item 3 — history length
```
cached prices as of 2026-07-07, fetch rules replayed:
  old rule (cross-ticker dropna): 794 rows, starts 2023-05-05, 13 stress days, build_covariance_matrix -> FAIL
  new rule (dropna how='all'):   1253 rows, starts 2021-07-09, 280 stress days, build_covariance_matrix -> OK
  screener fit orderings old vs new rule, 4 goals x 3 risk: identical (12/12); last 300 rows byte-identical
  returns frame 3.7 MB -> 5.9 MB (normal/stress copies scale with it: ~+10 MB total)
live fresh process (snapshot as of 2026-09-25): 583 tickers, 1252 days, 280 stress days
  correlations HTTP 200  0.0 s
  diagnostics  HTTP 200  0.0 s
  frontier     HTTP 200  0.02 s
  optimize     HTTP 200  0.0 s
  stress_2026  HTTP 200  0.04 s
```

## Item 4 — cold start (empty cache: fresh process, no Redis, in-memory cache empty)
```
shipping defaults (snapshot on, live refresh off):
  process start -> data loaded      1.07 s
  first /api/screen                 0.06 s  (HTTP 200, prices_asof 2026-09-25, source snapshot)
  process start -> first screen     1.13 s
  RSS                               300 MB
  browser: Start Discovering -> table 418 ms; header 'Prices as of 2026-09-25 (market data snapshot)'
old app (855c8e0) same measurement: first screen 507.6 s
with QX_LIVE_REFRESH=1: first screen 0.07 s; second profile 0.06 s; live refresh swapped in after 498.4 s:
  2026-09-28 20:18:43,814 app.main INFO Data installed (snapshot, as of 2026-09-25): 583 tickers, 1252 days, stress days: 280
  2026-09-28 20:18:43,819 app.main INFO Market snapshot loaded: as of 2026-09-25, built 2026-09-28
  2026-09-28 20:26:56,682 app.main INFO Data installed (live, as of 2026-09-25): 583 tickers, 1252 days, stress days: 280
```

### Memory (macOS RSS; relative evidence, not a Render measurement)
```
component breakdown (one process): {'python_baseline': 17, 'after_heavy_imports': 139, 'after_import_app_main': 194, 'after_snapshot_load': 289, 'after_live_fetch_prices': 391, 'after_live_fetch_fundamentals': 498, 'maxrss_MB': 499}
new app, refresh off: start 290 MB, after screen 305 MB, after correlations/frontier/health 311 MB
new app, refresh on: peak 545 MB  -> refresh made opt-in (K15)
old app (855c8e0): start 204 MB, after cold screen 445 MB
```

Snapshot file: app/data/market_snapshot.json.gz 4.6 MB (prices 1253 x 583, volumes, 587 fundamentals rows), as of 2026-09-25; builder scripts/build_market_snapshot.py (fetch_prices 103 s, fundamentals 408 s).

## Item 5 — Sharpe
```
screener Sharpe feeds the ranker: YES — raw['quality']=sharpe -> z_quality -> goal composite; quality_pts thresholds on sharpe. Left untouched.
display-only sharpe_rf = (exp(sum 252d log r) - 1 - rf) / vol; first screen top 5 (ticker, fit, ranker sharpe, displayed sharpe_rf):
  ['TWLO', 83, 1.628, 2.698]
  ['NTAP', 79, 1.172, 1.427]
  ['PANW', 76, 1.327, 1.752]
  ['STT', 76, 1.99, 2.409]
  ['HPE', 75, 1.708, 2.796]

ranker ordering dump (scripts/ranker_order_dump.py):
  before (855c8e0 worktree) sha256 808bb9039f2436a066d793f3c05690c7755d66b0cde362404afa8bbceb2b1296
  after  (final tree)       sha256 808bb9039f2436a066d793f3c05690c7755d66b0cde362404afa8bbceb2b1296
  diff: EMPTY
  frontend ranking inputs hash: 855c8e0 4a3ac5341c2668b7 == final 4a3ac5341c2668b7
  screener.py diff: data-source param + display-only sharpe_rf; no line in compute_factor_scores / rank_by_composite / compute_fit_scores changed
```

## Item 6 — Ask Quantex
```
GET /api/ai/status without token -> {'enabled': False}
browser, no token, sidebar text: "Ask Quantex\nThe AI explainer isn't enabled on this demo."
frontend source: no 'qx_api_key', no 'sk-ant-', no 'proxyResp.text()', no 'REPLICATE_API_TOKEN'
```

## Item 7 — β chip
```
band label color: #94a3b8 for In band / Above band / Below band (test_beta_chip_band_states_neutral)
```

## Browser walkthrough (headless Chrome, DEMO_MODE=1, no token, fresh process)
```
top bar: QUANTEX🟢 SPY 771.35 QQQ 744.50 GLD 393.41 TLT 79.32 CLOSE AS OF 2026-09-25 88/100 · Aggressive · Growth ✎
status bar: PORT A 4 pos. RETURN 252D +321.7% VOL Σ 55.1% SHARPE 5.74 BETA 2.65 MAX DD -25.2% VAR 95% 1D 5.5% YIELD 0.4%
sample-data banner after screen: False
builder still has old stat grid: False
Metrics:
Metrics are computed from the last 252 trading days of market data for this portfolio. Historical crisis scenarios live in the builder's Regime Stress Test.
PORTFOLIO HEALTH
252 trading days · 2025-09-25 → 2026-09-25 · daily log returns
RISK — how much can this hurt
▾
Vol 55.1% · annualized, 252d
Max drawdown −25.2% · worst peak-to-trough, past 252d
Worst 5% of days: lost ≥5.5% (those days averaged −7.0%)
STRUCTURE — is it actually diversified
▾
4 holdings · 3.9 effective bets
MU
Correlations (first lines):
PAIRWISE CORRELATION MATRIX (Ρ)
Normal
Stress
Blended
MU
WDC
DELL
HPE
MU
1.00
0.73
0.43
0.42
WDC
0.73
1.00
0.40
0.41
DELL
0.43
0.40
1.00
0.70
HPE
Frontier:
● Max Sharpe
● Min Volatility
Your portfolio: Return 46.27% · Vol 52.68% · Sharpe 0.797 · Near optimal ✓
KEY FRONTIER POINTS
PORTFOLIO	RETURN	VOLATILITY	SHARPE
Your Portfolio	+46.27%	52.68%	0.797
Max Sharpe	+47.18%	53.09%	0.808
```

## Not verified / flagged
- Render (Linux) RSS not measured; macOS numbers are relative.
- Sample-data banner rendered only via source test (the snapshot screen completes in ~0.4 s, so the banner is never on screen in a healthy run).
- Frontier's 'Your Portfolio' Sharpe (0.80, historical-average expected return over the full store, blended covariance) and the Health panel's Sharpe (5.74, trailing 252-day realized) are both real but use different windows/models on different tabs; the labels differ but a reviewer may compare them.
- Advice/verdict copy still present on real data (K23).
