# BETA_TRACKER_SPEC.md — Live Portfolio Beta Tracker

**Status: SPECIFIED — QUEUED AFTER VALIDATION BACKTEST.**

Design captured; not built. Queued behind the VALIDATION_SPEC.md backtest so the
ranking's validated status is known before more analytical surface is added.
Governed by IDENTITY.md §5 (education-native, reference-point vocabulary via
INTERFACE_VERDICTS_SPEC.md Amendment 3).

---

## Feature

A live portfolio beta tracker in the portfolio builder — beta computed and shown
as the user constructs a portfolio, in the same disciplined, disclose-the-method
style as the Sharpe (1Y) and correlation columns.

## UI surfaces

1. **Persistent stat chip** in the portfolio builder:
   `Portfolio β 1.12 · vs S&P 500` — recomputed on every add / remove / reweight.
2. **Marginal preview before committing an add/remove:** `β 1.18 → 1.04` — shows
   the beta the pending change would produce, before it is applied.
3. **Optional user-declared target band** (e.g. `0.9–1.1`):
   - When set, readouts append a state: `Above band ▲` / `In band ✓` / `Below band ▼`.
   - Marginal preview appends band transitions, e.g. `▼ into band` / `▲ out of band`.
   - No band set → no band label rendered.
   - **The tool never suggests or defaults a band.** The band is the user's own
     declared preference; the tool only reports position relative to it.

## Method (sealed)

- Beta = **regression of the value-weighted portfolio daily-return series against
  the benchmark**, over the **same 252-day window as the correlation column**.
- **NOT** the weighted average of individual asset betas. (The weighted-average
  shortcut ignores covariance structure and is what most tools ship; this
  computes the portfolio's own realized beta.)
- Tooltip discloses, factually: **window, benchmark, R², observation count.**
- **Low R² is shown, not hidden** — a low R² means beta explains little of the
  portfolio's movement, and the user is told so rather than shielded from it.

## Vocabulary (INTERFACE_VERDICTS_SPEC.md Amendment 3 — reference-point rule)

- **vs market (benchmark):**
  - `More volatile than market`  (β > 1 + tolerance)
  - `Less volatile than market`  (β < 1 − tolerance)
  - `Market-like`                (|β − 1| < 0.1)
- **vs band:** the user's own declared band only (`Above band ▲` / `In band ✓` /
  `Below band ▼`). No tool-issued band, no verdict, no advice.

## Prerequisite refactor

- Lift `discovery_context.py::_portfolio_returns` (currently module-local) to a
  **shared helper**, so beta and correlation consume the **same** value-weighted
  daily-return series computation. Beta must not fork a second implementation of
  the portfolio-return series. (Flagged as reuse debt during the correlation
  column build — Amendment 4 evidence note.)

## Edge states

- Overlap `< 60` trading days → `—` (never `0`; 0 is a claim, not an absence —
  same convention as CORRELATION_COLUMN_SPEC.md §A4).
- Benchmark **selectable later**; **S&P 500 (SPY) is the v1 default**.

## Non-goals (v1)

- No auto-suggested band, no optimizer coupling, no "adjust to hit band" action.
- No multi-benchmark comparison in v1 (single selectable benchmark, SPY default).
