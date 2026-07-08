# BACKTEST_VERDICT.md — FINAL (audited)

Mechanical application of the sealed §6 bars to BACKTEST_RAW_RESULTS.md, with the
audit adjudication recorded below. Null seed 20260707; 38/38 freezes built.

## Growth-95 — PASS — UNVALIDATED per P5: survivor-universe result; no interface action until delisting-inclusive re-run
- median Sharpe percentile (12m) = 78.8 (PASS ≥60, DECORATION ≤50)
- above 50th pct in 68% of windows (PASS needs ≥55%)
- §7/P5 flag: a clean Growth-95 PASS is the pre-registered "suspect the backtest
  first" case. The portfolio sits at ~100th-percentile volatility/beta in nearly
  every window; the leading suspect is the survivorship×momentum interaction
  (BACKTEST_RUN_DECISIONS.md J1, DATA_PLUMBING_AUDIT.md §b). Treated as
  **unvalidated**: no §8 PASS action (no "validated" line) until a
  delisting-inclusive (point-in-time universe) re-run.

## Income-20 — MARGINAL
- realized-yield pct ≥80 in 100% of windows (PASS needs ≥80%)
- volatility below null-median in 97% of windows (PASS needs ≥70%)
- median Sharpe percentile = 26.4 (PASS needs ≥40)
- DECORATION check: yield pct <60 in 0% of windows (DECORATION if ≥50%)

## Balanced-50 — DECORATION
- median Sharpe percentile (12m) = 42.8 (PASS ≥55, DECORATION ≤45)

## Profile differentiation gate — PASS
- Growth β > Balanced β > Income β in 100% of windows (needs ≥75%)
- **Overrides everything**: if FAIL, the fit score fails regardless of returns.

## Overall
```
{
  "Growth-95": "PASS (UNVALIDATED — survivor-universe; re-run pending)",
  "Income-20": "MARGINAL",
  "Balanced-50": "DECORATION",
  "differentiation": "PASS"
}
```

## Audit reconciliation

Mixed verdicts on a shared label resolved most-conservative: global rename
Fit score → Screen match (Income MARGINAL + Growth unvalidated); Balanced ranked
presentation removed per DECORATION; Growth validated-line withheld.

- Balanced-50 DECORATION confirmed robust to the unsealed tie convention
  (BACKTEST_RUN_DECISIONS.md J12: strict-below 42.80 == midpoint 42.80).
- §7 prediction divergences (findings): P5 predicted Growth MARGINAL (got
  PASS-unvalidated), Income PASS (got MARGINAL — Sharpe leg 26.4 < predicted
  40–60), Balanced MARGINAL-to-PASS (got DECORATION at 42.8). Differentiation
  100% exceeds P4 (>85%). Growth momentum weakness appeared in 2020Q3–2021Q1
  (Sharpe percentiles 1/0/9), consistent with P1.

