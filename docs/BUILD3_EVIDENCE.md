# BUILD3_EVIDENCE.md — raw evidence, PORTFOLIO_RISK_SPEC + Amendment A (Build 3)

Branch build3-portfolio-health from demo-prep 06739ea. All commands run with venv/bin/activate. Judgment calls: docs/PORTFOLIO_RISK_RUN_DECISIONS.md (J1–J26).

## Suite before / after
```
before (06739ea, demo-prep):  192 passed, 1 deselected
after  (this branch):          224 passed, 1 deselected   (+32: test_portfolio_risk.py 18, test_portfolio_risk_api.py 6, test_portfolio_health_ui.py 8)
node --check on the inline app script: OK
```

## Item 7 — Ranker untouched (measured)
```
python scripts/ranker_order_dump.py before.json   # at 06739ea, before any Build 3 edit
python scripts/ranker_order_dump.py after.json    # final tree
sha256 before 808bb9039f2436a066d793f3c05690c7755d66b0cde362404afa8bbceb2b1296
sha256 after  808bb9039f2436a066d793f3c05690c7755d66b0cde362404afa8bbceb2b1296
diff before.json after.json   -> empty (exit 0)
coverage: 48 frontend keys (4 goals x 3 horizons x 2 deriv answers x 2 fixtures; each desc+asc), 36 backend orderings; backend fixture {'days': 751, 'first': '2023-07-10', 'last': '2026-07-07', 'tickers': 585}
determinism: two consecutive runs on the same tree byte-identical
files changed vs 06739ea: app/main.py app/models.py static/quantex.html (new: app/portfolio_risk.py, scripts/ranker_order_dump.py, 3 test files, 2 docs)
unchanged: app/screener.py app/config.py app/optimizer.py app/correlation_engine.py app/data_ingest.py app/return_models.py app/portfolio_series.py
frontend ranking inputs (Discovery rows expression + fitScore + cmpRows + scoreProfile + FALLBACK_ASSETS + defaultWeights) sha256[:16]: 06739ea 4a3ac5341c2668b7 == working tree 4a3ac5341c2668b7
```
Sample ordering (identical before/after): backend Growth|88|20.0 top 8 = SATS:90, DELL:79, GOOGL:76, GOOG:75, CAT:73, HPE:71, MU:70, AMD:70

## Item 1 — Engine identity Σ RC_i = σ_P (3 seeded portfolios, one with negative RC)
```
seed=1 {'HI1': 25, 'HI2': 25, 'LO1': 25, 'LO2': 25}
  sigma_P=0.149532097050991  sum(RC)=0.149532097050991  |diff|=2.78e-17  negative_RC=[]
seed=2 {'HI1': 50, 'LO1': 30, 'LO2': 20}
  sigma_P=0.189360249668185  sum(RC)=0.189360249668185  |diff|=2.78e-17  negative_RC=[]
seed=3 {'HI1': 40, 'HI2': 40, 'NEG': 20}
  sigma_P=0.211574094836686  sum(RC)=0.211574094836686  |diff|=2.78e-17  negative_RC=['NEG']
```

## Item 9 — Every §8 metric vs an independent hand computation (seeded 4-asset portfolio)

| metric | engine | hand | abs diff |
|---|---|---|---|
| LW shrinkage α | 0.016822178057 | 0.016822178057 | 6.9e-18 |
| σ_P (annualized) | 0.125784964402 | 0.125784964402 | 0.0e+00 |
| RC HI1 | 0.094096109172 | 0.094096109172 | 1.4e-17 |
| RC HI2 | 0.046691296553 | 0.046691296553 | 6.9e-18 |
| RC LO1 | 0.008916118147 | 0.008916118147 | 0.0e+00 |
| RC NEG | -0.023918559470 | -0.023918559470 | 3.5e-18 |
| effective bets (|RC|) | 2.579605835709 | 2.579605835709 | 8.9e-16 |
| max drawdown | -0.101032526082 | -0.101032526082 | 0.0e+00 |
| VaR 95% 1d | 0.011742628243 | 0.011742628243 | 0.0e+00 |
| CVaR 95% 1d | 0.015463923426 | 0.015463923426 | 0.0e+00 |
| up capture | 0.637651747556 | 0.637651747556 | 1.1e-16 |
| down capture | 0.635846367449 | 0.635846367449 | 4.4e-16 |
| Sharpe (rf 4.3%) | 0.400366185389 | 0.400366185389 | 1.7e-15 |
| avg pairwise corr | -0.068892088346 | -0.068892088346 | 1.8e-16 |
| highest pair r | 0.549241319089 | 0.549241319089 | 1.1e-16 |

Highest pair engine HI1–LO1, hand HI1–LO1. Hand LW is re-derived from Ledoit & Wolf (2004) with explicit loops (not a call into return_models).

Composition metrics (category weights, weighted yield, top holding / top 3) are pure weight arithmetic in the frontend; their rendered values are asserted in tests/test_portfolio_health_ui.py (e.g. all-equity case: 'Technology 100%', 'Top holding 45% (NVDA)').

## Item 10 — CVaR ≥ VaR (seeded + one real portfolio)
```
seeded (item 9 portfolio): VaR 1.174%  CVaR 1.546%  k=13/252  CVaR>=VaR: True
real (cached Adj Close 2025-07-07→2026-07-07, {'AAPL': 20, 'MSFT': 20, 'NVDA': 15, 'JNJ': 15, 'XOM': 10, 'TLT': 20}):
  VaR 0.972%  CVaR 1.436%  k=13/252  CVaR>=VaR: True
  vol 10.68%  maxDD -8.19% (2026-06-01→2026-06-25)  eff bets 3.44  Sharpe 1.35 vs SPY 1.32  capture up 0.61 (138d) down 0.52 (114d)
  Σ RC − σ = 0.00e+00
  MSFT    33.7% of risk ·   20% of weight
  NVDA    31.6% of risk ·   15% of weight
  AAPL    27.4% of risk ·   20% of weight
  JNJ      4.0% of risk ·   15% of weight
  TLT      2.6% of risk ·   20% of weight
  XOM      0.7% of risk ·   10% of weight
```

## Item 11 — Capture ratios: seeded asymmetry + observation guard
```
constructed p = 1.2·b on up days, 0.5·b on down days -> up 1.200000 (137d)  down 0.500000 (115d)
guard (min_obs=60 = MIN_OVERLAP_DAYS): 60 up days -> up 1.0; 59 down days -> down None  (renders '—')
```

## Items 3–4 — Decomposition (2 high-vol + 2 low-vol, equal weight) and effective bets
```
HI1  48% of risk · 25% of weight
HI2  41% of risk · 25% of weight
LO1  6% of risk · 25% of weight
LO2  6% of risk · 25% of weight
negative RC render: NEG −10.2% of risk (offsets) · 20% of weight
(a) equal-risk (inverse-vol, independent) 4 holdings · 3.94 effective bets
(b) concentrated 1/1/1/97             4 holdings · 1.00 effective bets
```

## Item 5 — Redundancy flags
```
TLT: marker 'Similar to IEF (0.98)'
IEF: marker 'Similar to TLT (0.98)'
AAA/BBB r=-0.03 -> no marker; threshold=0.85 from app/config.py HIGH_CORR_THRESHOLD
```


## Item 14 — Rendered panel strings (shipped healthPanelBody under node; payloads from the real engine on seeded data)

### Full portfolio (5 holdings + 1 without history: NEWCO)
```
excludes NEWCO: insufficient history
252 trading days · 2024-03-07 → 2025-02-21 · daily log returns
RISK
— how much can this hurt
▾
Vol 18.3% · annualized, 252d
Max drawdown −20.9% · worst peak-to-trough, past 252d
Worst 5% of days: lost ≥2.0% (those days averaged −2.4%)
STRUCTURE
— is it actually diversified
▾
6 holdings · 2.6 effective bets
NVDA
▲ 59% of risk · 30% of weight
AVGO
27% of risk · 20% of weight
MSFT
14% of risk · 20% of weight
JNJ
5% of risk · 15% of weight
TLT
−3.6% of risk (offsets) · 15% of weight
NEWCO  — · Insufficient history
Risk rows re-base weights to the 5 holdings with history.
Avg pairwise correlation 0.11 · across 10 pairs
Share of portfolio volatility attributable to each holding (weight × marginal contribution). Describes the current portfolio's risk composition — not a recommendation to change it.
SENSITIVITY
— how does it behave vs the market
▾
Portfolio β 1.05 · vs S&P 500
Captured 116% of market up-days · 88% of down-days
EFFICIENCY
— what did it earn for the risk
▾
Sharpe -0.10 · Lost vs cash
COMPOSITION
— what is it, structurally
▾
Technology 64% · Healthcare 14% · Fixed Income 14% · Unknown 9%
fewer
Yield 1.0% · trailing 12m, weighted (excludes NVDA)
Top holding 27% (NVDA) · Top 3: 64%
```
Tooltips (title attributes):
```
σ_P = √(wᵀΣw) on current weights. Σ = Ledoit-Wolf constant-correlation shrinkage (α 0.03) of 252 daily returns, 2024-03-07 → 2025-02-21, annualized ×252.
Peak 2024-06-26 → trough 2024-11-15. Value-weighted portfolio path at current weights.
Historical 95% VaR and CVaR of daily returns, past 252 trading days (13 worst of 252 days). Describes what happened, not what will.
Inverse Herfindahl of risk shares: N_eff = 1 / Σ sᵢ², sᵢ = |RCᵢ| / Σ|RCⱼ|. The |RC| convention keeps a negative contribution (a holding that offsets risk) from producing shares outside 0–1; with no offsets it equals 1/Σ(RCᵢ/σ_P)². Counts how many independent risk sources the portfolio behaves like. 8 holdings concentrated in one sector can be ~2 effective bets.
Risk contribution RC = weight × marginal contribution (Σw)ᵢ/σ_P; shares sum to 100% of σ_P.
Highest: AVGO–MSFT 0.62. Pearson correlation of daily returns, same 252d window.
Mean portfolio return on SPY-up days ÷ mean SPY return on those days (113 up days); same for down days (139 down days). A side with fewer than 60 days shows —.
(return over the window − risk-free 4.3%) ÷ annualized volatility of the portfolio's daily returns (18.4%, realized; the Vol line above uses the shrinkage estimate). SPY same window, same formula: -1.70.
Σ weight × (trailing-12-month cash dividends ÷ latest unadjusted close), over holdings with dividend data.
Share of portfolio weight.
```
Colors used: ['#94a3b8', '#64748b', '#a5b4fc', '#cbd5e1', '#0a0f1c', '#f59e0b', '#6366f1', '#818cf8']

### All-equity portfolio
```
252 trading days · 2024-03-07 → 2025-02-21 · daily log returns
RISK
— how much can this hurt
▾
Vol 27.1% · annualized, 252d
Max drawdown −30.8% · worst peak-to-trough, past 252d
Worst 5% of days: lost ≥3.0% (those days averaged −3.6%)
STRUCTURE
— is it actually diversified
▾
3 holdings · 2.2 effective bets
NVDA
▲ 61% of risk · 45% of weight
AVGO
27% of risk · 30% of weight
MSFT
12% of risk · 25% of weight
Avg pairwise correlation 0.52 · across 3 pairs
Share of portfolio volatility attributable to each holding (weight × marginal contribution). Describes the current portfolio's risk composition — not a recommendation to change it.
SENSITIVITY
— how does it behave vs the market
▾
Portfolio β 1.05 · vs S&P 500
Captured 168% of market up-days · 130% of down-days
EFFICIENCY
— what did it earn for the risk
▾
Sharpe -0.10 · Lost vs cash
COMPOSITION
— what is it, structurally
▾
Technology 100%
Yield 1.0% · trailing 12m, weighted (excludes NVDA)
Top holding 45% (NVDA) · Top 3: 100%
```
Colors used: ['#64748b', '#a5b4fc', '#cbd5e1', '#0a0f1c', '#f59e0b', '#6366f1', '#94a3b8']

### Single-holding portfolio
```
252 trading days · 2024-03-07 → 2025-02-21 · daily log returns
RISK
— how much can this hurt
▾
Vol 18.2% · annualized, 252d
Max drawdown −35.5% · worst peak-to-trough, past 252d
Worst 5% of days: lost ≥1.9% (those days averaged −2.6%)
STRUCTURE
— is it actually diversified
▾
1 holding · 1.0 effective bets
MSFT
▲ 100% of risk · 100% of weight
Avg pairwise correlation —
Share of portfolio volatility attributable to each holding (weight × marginal contribution). Describes the current portfolio's risk composition — not a recommendation to change it.
SENSITIVITY
— how does it behave vs the market
▾
Portfolio β 1.05 · vs S&P 500
Captured 96% of market up-days · 110% of down-days
EFFICIENCY
— what did it earn for the risk
▾
Sharpe -1.88 · Lost vs cash
COMPOSITION
— what is it, structurally
▾
Technology 100%
Yield 1.0% · trailing 12m, weighted
Top holding 100% (MSFT)
```
Colors used: ['#64748b', '#a5b4fc', '#cbd5e1', '#0a0f1c', '#f59e0b']

### Empty portfolio
```
Add holdings to see portfolio analytics
```
Colors used: ['#64748b']

## Item 2 — ΔVol preview rendered strings
```
Portfolio vol 14.2% → 13.1%   (adding TLT at 25% weight · equal re-split of 4 holdings) / β 1.12 → 1.04
Portfolio vol 14.2% → 15.5%   (removing TLT · remaining 3 re-split equally) / β 1.12 → 1.20
Portfolio vol 14.2% → 15.0%   (NVDA 20% → 35% weight)
```
Weight rules shared by the real add/remove and the preview: addWeights({A:34,B:33,C:33},'D') = {'A': 25, 'B': 25, 'C': 25, 'D': 25} ; removeWeights({A..D:25},'B') = {'A': 34, 'C': 33, 'D': 33}

## Item 2 — agreement check (previewed σ_P == engine σ_P on post-change weights)
```
tests/test_portfolio_risk_api.py::test_health_endpoint_matches_engine PASSED [ 16%]
tests/test_portfolio_risk_api.py::test_preview_agrees_with_engine_on_post_change_weights[cur0-prop0] PASSED [ 33%]
tests/test_portfolio_risk_api.py::test_preview_agrees_with_engine_on_post_change_weights[cur1-prop1] PASSED [ 50%]
tests/test_portfolio_risk_api.py::test_preview_agrees_with_engine_on_post_change_weights[cur2-prop2] PASSED [ 66%]
tests/test_portfolio_risk_api.py::test_preview_reports_unloaded_ticker_without_fetching PASSED [ 83%]
tests/test_portfolio_risk_api.py::test_redundancy_endpoint_uses_config_threshold PASSED [100%]

each case asserts pytest.approx(rel=1e-12) between /api/portfolio/preview proposed.vol and portfolio_health(returns, proposed).vol; β line two == /api/portfolio/beta on the same weights
```

## Items 6 & 15 — Prohibited-vocabulary grep (§6 list + healthy/unhealthy/safe/risky) over every added line (git diff 06739ea -- app static, '+' lines) and app/portfolio_risk.py
```
pattern: overweight|underweight|too concentrated|well[- ]diversified|poorly diversified|\bshould\b|consider adding|consider trimming|\bhealthy\b|\bunhealthy\b|\bsafe\b|\brisky\b  (case-insensitive)
result: ZERO HITS (688 lines scanned). Rendered-panel strings + tooltips also asserted clean in tests/test_portfolio_health_ui.py::test_prohibited_vocabulary_absent_from_rendered_panel.
green/red literals in added lines: 3, all in pre-existing shipped lines re-emitted by the diff (verified verbatim at 06739ea): the Corr-column error span (#f87171) and the builder 'Total:' readout (#1D9E75/#ef4444). The relocated β chip also keeps its shipped 'In band ✓' green (J15, flagged).
panel rendered colors (all four cases): no #1D9E75/#ef4444/#34d399/#f87171; amber #f59e0b only on '▲ N% of risk' when a holding >50% of risk and on 'Top holding' when >40% of weight.
```

## Item 12 — sharpeMarketLabel reuse (grep)
```
418:function sharpeMarketLabel(sh, spyRef, isRef){
433:  const label=sharpeMarketLabel(a.sh,spyRef,isRef);
2155:  const shLabel=sh?sharpeMarketLabel(sh.sharpe,spy?spy.sharpe:null,false):null;

one definition (line 418); the panel calls it at the line above containing sh.sharpe — no second implementation.
```

## Item 13 — One bucketing code path (file:line)
```
472:function sleeveOf(sec){
2056:function sectorOf(tk){const a=ASSETS.find(x=>x.id===tk);return a&&a.sec!=null?a.sec:null;}
2162:  const cat={};tks.forEach(t=>{const s=sectorOf(t)||"Unknown";cat[s]=(cat[s]||0)+wn[t];});
2224:      const s=sleeveOf(sectorOf(tk));

static/quantex.html sectorOf() is the single sector lookup; §B RollingCorrChart (sleeveOf(sectorOf(tk))) and §8.5 category weights (sectorOf(t)) both call it.
```

## Live-data run (build3 code, fresh uvicorn, yfinance, no Redis) — 2026-09-28
```
POST /api/portfolio/health  AAPL/MSFT/NVDA/JNJ/TLT 20% each  (2.9 s incl. data load)
  window 2025-09-25 → 2026-09-25, 252 obs, LW α 0.166
  vol 12.90%  Σ RC 12.90% (identity holds)  effective bets 3.31
  max DD −9.19% (2025-10-29 → 2026-03-27)  VaR 1.31%  CVaR 1.64% (13/252)
  capture up 0.75 (135d) down 0.66 (117d)  Sharpe 1.28 (realized vol 12.95%) vs SPY 1.05
  avg pairwise 0.05 across 10 pairs, highest MSFT–NVDA 0.25   yield 1.63% (J2, all 5 known)
  NVDA 40.3% of risk · 20% of weight | MSFT 31.1% | AAPL 20.0% | JNJ 4.5% | TLT 4.1%
POST /api/portfolio/preview  add GLD (equal re-split 17/17/17/17/16/16)
  vol 12.90% → 12.92%   β 0.7117 → 0.7245   (β chip on current weights: 0.7117 — identical)
POST /api/discovery/redundancy  11 candidates (XOM not loaded)
  flags [] (no pair ≥ 0.85), n_pairs 45, not_loaded ['XOM']
```

## Not verified
- Browser rendering/layout of the panel, preview card and Discovery markers (headless Chrome broke mid-session; J26). Strings, tooltips, colors and click semantics are verified under node; visual layout is not.
