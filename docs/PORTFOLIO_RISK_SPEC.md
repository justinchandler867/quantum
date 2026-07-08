# PORTFOLIO_RISK_SPEC.md — Portfolio Risk Context Layer (Sealed)

**Status:** SPECIFIED — QUEUED AFTER BETA TRACKER BUILD (hard dependency: the shared portfolio-return-series helper and covariance machinery from BETA_TRACKER_SPEC.md).
**Authority:** IDENTITY.md (education-native capability demo), Amendment 3 (reference-point vocabulary).
**Prime directive:** THE RANKER IS UNTOUCHED. Screen match / fit-score computation, weights, ordering, and every input to them are out of scope. BACKTEST_VERDICT.md adjudicated the ranker as it exists; any change to the score invalidates those verdicts. This layer displays facts BESIDE the ranking. The architecture: the score screens, the context layer informs, the human constructs.
**Build rules:** STOP conditions literal. venv for all test runs. Full suite before/after. Evidence printed, no commit until audit.

---

## §1 — Shared risk engine (backend)

One estimation path feeds every feature below, so all numbers are mutually consistent:

- **Return series:** the shared helper lifted in the beta build (value-weighted portfolio daily returns; per-asset daily returns from Adj Close), trailing 252 trading days — same window as the correlation column and beta tracker.
- **Covariance:** sample covariance of daily returns over that window, annualized (×252). ⚠ CHECK FIRST: the Black-Litterman work shipped a Ledoit-Wolf shrinkage covariance. If that estimator is reusable here, USE IT (one covariance philosophy across the product beats two); if its coupling makes reuse disproportionate, use sample covariance and log the divergence as a recorded decision — but STOP and report which situation holds before choosing.
- **Portfolio vol:** σ_P = √(wᵀΣw), annualized, on current builder weights.
- **Marginal contribution of asset i:** MC_i = (Σw)_i / σ_P. Risk contribution: RC_i = w_i × MC_i, with Σ RC_i = σ_P (verify this identity in tests — it's the correctness check for the whole engine).
- **Insufficient data:** any asset with <60 overlapping days → excluded from the engine with an explicit per-asset "— · Insufficient history" marker wherever its numbers would appear; the portfolio-level stats then carry a visible qualifier "excludes {TICKER}: insufficient history". NEVER silently impute, never default to zero ("0 is a claim, not an absence").

## §2 — ΔVol marginal preview (the crown metric; supersedes β as the primary hover readout)

On hover/tap of any candidate in Discovery, and on any proposed add/remove/reweight in the builder:

```
Portfolio vol 14.2% → 13.1%   (at 10% weight)
β 1.12 → 1.04
```

- ΔVol is line one; the beta preview from BETA_TRACKER_SPEC becomes line two of the same preview card (one card, not two features).
- Proposed weight for Discovery hover: default 10% or the builder's configured default lot — ⚠ if no default-weight concept exists in the builder, STOP and report how adds currently size positions; the preview must use whatever weight an actual add would use, stated visibly ("at 10% weight").
- Direction is shown by the arrow and the numbers only. NO green/red on the delta — a vol reduction is not "good" (it may cost expected return); coloring it would be a verdict by pigment.
- Tooltip: "Annualized volatility of daily returns, trailing 252d, at current weights vs. weights after this change."

## §3 — Risk contribution decomposition (builder panel)

A compact horizontal bar (or equivalent) in the portfolio builder, one row per holding:

```
NVDA   31% of risk · 12% of weight
TLT     4% of risk · 20% of weight
```

- Sorted by risk share, descending. Both percentages shown — the gap between them IS the lesson.
- Negative RC (a true diversifier at these weights) renders factually: "−2% of risk (offsets)".
- Panel footnote: "Share of portfolio volatility attributable to each holding (weight × marginal contribution). Describes the current portfolio's risk composition — not a recommendation to change it."
- NO prescriptive affordances: no "rebalance to fix," no target lines, no highlighting a holding as "too much." The numbers speak; the user decides.

## §4 — Effective bets (summary statistic, builder header)

Next to the beta chip:

```
8 holdings · 2.7 effective bets
```

- Method: inverse Herfindahl of normalized risk contributions, N_eff = 1 / Σ(RC_i/σ_P)². ⚠ Negative RC_i makes this definition ill-behaved; handle by computing on |RC_i| normalized, and disclose the convention in the tooltip. If a cleaner convention exists in the literature you can cite, STOP and propose it rather than silently choosing.
- Tooltip: the formula in words + "Counts how many independent risk sources the portfolio behaves like. 8 holdings concentrated in one sector can be ~2 effective bets."
- This is a fact with a built-in reference point (holdings count vs effective count) — Amendment 3 satisfied by construction. NO grade words ("well diversified"), NO targets.

## §5 — Candidate redundancy flags (Discovery)

When two or more candidates in the current shortlist correlate ≥ 0.85 with EACH OTHER (pairwise, same 252d window):

- Each flagged row gets a small marker: `Similar to TLT (0.91)` — naming the most-correlated shortlist peer and the value. Reference-pointed, factual.
- Threshold 0.85 matches the backend's existing HIGH_CORR_THRESHOLD (config.py) — reuse the constant, do not hardcode a second copy.
- Tooltip: "These candidates' daily returns move nearly identically (r = 0.91 over 252d). Adding several is closer to one position at combined weight than to separate diversifiers." (Educational mechanism statement — permitted; "don't add both" — NOT permitted.)
- Computation: pairwise correlations among visible shortlist candidates only (≤100 rows → ≤4,950 pairs; if the existing /api/discovery/context contract can't serve pairwise candidate-vs-candidate values, this needs a small backend addition — STOP and print the proposed endpoint shape before building it).

## §6 — Vocabulary (Amendment 3, applied)

Every label names its reference: "31% of risk · 12% of weight" (reference = the portfolio), "Similar to TLT (0.91)" (reference = the named peer), "2.7 effective bets" vs "8 holdings" (reference = the holding count), "14.2% → 13.1%" (reference = the current portfolio). PROHIBITED anywhere in this layer: "overweight/underweight" as judgments, "too concentrated," "well/poorly diversified," "should," "consider adding/trimming," and any green/red good/bad color semantics. The flip-marker amber is available for factual extremes only (e.g., a single holding >50% of risk MAY use the amber marker — it is a fact — but with no accompanying imperative).

## §7 — What this spec deliberately does NOT do

- Does not modify the ranker, fit/screen-match score, or its inputs (prime directive).
- Does not make the ranked list portfolio-aware. Candidate ordering never changes based on the user's holdings. (Portfolio-aware ranking = individualized construction advice = a different compliance surface and a reopened validation. If ever wanted, it is a separate IDENTITY-level decision, not a feature.)
- Does not suggest weights, targets, bands, or trades. The optimizer remains the product's formal construction tool; this layer informs manual construction.

## Acceptance (evidence for each, no commit until audit)

- Engine identity test: Σ RC_i = σ_P to numerical tolerance, on ≥3 seeded portfolios including one with a negative-RC asset.
- ΔVol preview: rendered strings for add, remove, and reweight cases + the stated weight; agreement check — the previewed post-change σ_P equals the engine's σ_P computed directly on the post-change weights (no shortcut approximations).
- Risk decomposition: rendered rows for a seeded portfolio where risk share ≠ weight share visibly (construct one: 2 high-vol + 2 low-vol names); negative-RC rendering.
- Effective bets: value + tooltip for (a) an equal-risk portfolio (N_eff ≈ N), (b) a concentrated one (N_eff ≪ N).
- Redundancy flags: a seeded shortlist with a ≥0.85 pair renders the marker with peer + value; a below-threshold pair renders nothing; threshold sourced from config constant (print the reference).
- Prohibited-vocabulary grep over all new code/copy (§6 list): zero hits outside negated disclaimer copy.
- Ranker untouched: no ranking/compute file in the diff beyond the risk engine additions; screen-match ordering byte-identical on the standard offline fixture (measured, before/after dump + empty diff — not inferred).
- Suite green before/after; new tests for items 1–5.

**Commit message:** `portfolio risk context layer: marginal ΔVol preview, risk-contribution decomposition, effective bets, candidate redundancy flags (PORTFOLIO_RISK_SPEC); ranker untouched`

**STOP conditions:** Ledoit-Wolf reuse question (§1); builder default-weight question (§2); negative-RC effective-bets convention (§4); pairwise endpoint shape (§5); anything where the shipped codebase contradicts an assumption here.

---

# PORTFOLIO_RISK_SPEC — Amendment A (2026-07-08): §8 Portfolio Health Assembly

Append to docs/PORTFOLIO_RISK_SPEC.md. Extends Build 3's scope from individual risk features to the assembled Portfolio Health panel. Prime directive unchanged: THE RANKER IS UNTOUCHED. All metrics derive from the §1 shared risk engine — one return series, one covariance, one 252-trading-day window, so every number on the panel is mutually consistent. Amendment 3 vocabulary governs throughout.

## §8 — Portfolio Health panel (builder)

One panel in the portfolio builder — the assembly surface for §§2–5 plus the blocks below. Collapsible sections; the beta chip and §B rolling-correlation chart live adjacent (this corner of the builder is the portfolio-analytics surface). Every metric: value first, reference point named, tooltip carrying the computation facts. No verdicts, no targets, no green/red good/bad semantics; the amber marker permitted only for factual extremes explicitly listed below.

### §8.1 Risk block — "how much can this hurt"

- **Portfolio volatility:** σ_P from §1, annualized, rendered "Vol 14.2% · annualized, 252d".
- **Max drawdown:** worst peak-to-trough of the value-weighted portfolio path over the window: "Max drawdown −18.3% · worst peak-to-trough, past 252d". Tooltip: peak date → trough date.
- **Historical VaR/CVaR (95%, 1-day):** from the empirical daily-return distribution of the portfolio series — no parametric assumption. Rendered as one plain-language pair: "Worst 5% of days: lost ≥1.8% (those days averaged −2.6%)". Tooltip: "Historical 95% VaR and CVaR of daily returns, past 252 trading days. Describes what happened, not what will."
- Amber permitted: none in this block by default. (Drawdown is already alarming enough unstyled.)

### §8.2 Structure block — "is it actually diversified"

- §3 risk-contribution decomposition and §4 effective bets render here (no duplication — this is their home).
- **Average pairwise correlation:** mean of all pairwise 252d correlations among holdings: "Avg pairwise correlation 0.58 · across 28 pairs". Tooltip names the highest pair: "Highest: NVDA–AVGO 0.91". <2 holdings → "—".
- Amber permitted: a single holding >50% of risk contribution (§6's listed extreme).

### §8.3 Sensitivity block — "how does it behave vs the market"

- **Beta chip** (shipped) renders here.
- **Up/down capture (vs SPY, same window):** mean portfolio return on SPY-up days ÷ mean SPY return on those days, and same for down days: "Captured 95% of market up-days · 70% of down-days". Tooltip: day counts for each side. ⚠ If either side has <30 observations in the window, render "—" for that side rather than a noisy ratio.

### §8.4 Efficiency block — "what did it earn for the risk"

- **Portfolio Sharpe vs SPY:** the shipped per-ticker method applied to the portfolio series — same rf (config constant), same window, same Amendment 3 labels: "Sharpe 1.31 · Beat market" / "Trailed market" / "Lost vs cash"; SPY reference value in the tooltip. Reuse the existing sharpeMarketLabel logic — do not fork a second implementation.
- **Sortino (optional, secondary line):** same structure with downside deviation; tooltip states the difference in one sentence ("penalizes only downside volatility"). If including it creates label ambiguity with Sharpe's market comparison, ship Sharpe only and log the deferral.

### §8.5 Composition block — "what is it, structurally"

- **Category weights:** sectors.json bucketing (reuse §B's bucketing code path — one bucketing implementation, not two): "Technology 42% · Fixed Income 20% · …", top categories with a collapsible full list.
- **Portfolio yield:** weighted trailing dividend yield, raw-price denominator per the J2 convention: "Yield 2.1% · trailing 12m, weighted".
- **Concentration facts:** "Top holding 24% (NVDA) · Top 3: 51%". Facts only; no concentration verdict.
- Amber permitted: top holding >40% of weight.

### §8.6 Edge states (uniform)

Any holding <60 overlapping days: excluded per §1 with the visible qualifier; panel-level metrics carry "excludes {TICKER}" once, not per-metric. Empty portfolio: panel renders a single line "Add holdings to see portfolio analytics" — no fabricated placeholder numbers. Metrics whose inputs are partially unavailable render "—" individually; the panel never hides a section to conceal a data gap.

### §8.7 Acceptance additions (extend the spec's list)

9. Each §8 metric verified against an independent hand computation on one seeded 4-asset portfolio (print both values side by side: engine vs hand).
10. VaR/CVaR sanity: CVaR ≥ VaR in magnitude by construction; verified on the seeded portfolio and one real portfolio.
11. Capture ratios: verified on a seeded series with known asymmetry (constructed so up-capture ≠ down-capture by design); <30-observation guard exercised.
12. Sharpe label reuse: grep-level proof the panel calls the existing label function rather than a duplicate.
13. Bucketing reuse: §8.5 and §B share one bucketing code path (cite file:line).
14. Rendered panel strings for: full portfolio, all-equity portfolio, single-holding portfolio, empty portfolio.
15. Prohibited-vocabulary grep extended over the panel (§6 list plus: "healthy", "unhealthy", "safe", "risky" as bare labels — "risk" as a noun in factual copy is fine).

**STOP conditions added:** capture-ratio observation guard threshold if the spec's 30 conflicts with an existing convention in the codebase; any §8 metric whose §1-engine derivation would require touching a shipped compute path rather than consuming its output.
