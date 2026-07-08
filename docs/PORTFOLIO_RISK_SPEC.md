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
