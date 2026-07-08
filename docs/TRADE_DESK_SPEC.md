# TRADE_DESK_SPEC.md — Trade Desk Recommendation Engine Remediation

**Status: UNWRITTEN — requires cold design decision.**

This is a stub. It records the defect and the open question only. It deliberately does **not** design the remediation — that requires a fresh, cold design decision, not an extension tacked onto the SIGNAL-column work (INTERFACE_VERDICTS_SPEC.md).

---

## Defect

The Trade Desk component (`static/quantex.html`, `function TradeDeck`, approx. lines 2176–2200+) is a machine-issued **buy/sell/hold trade recommendation engine**, not a passive readout:

- It maps each held ticker to an action (`buy` / `sell` / `hold`) via `SIG_ACT`.
- It renders imperative action badges (`BUY` / `SELL` / `HOLD`, `ACT` map).
- It emits order types (`Limit` / `Market` / `Stop-Limit` / `Monitor`), a suggested limit entry price (`a.tgt * 0.88`), a stop, a horizon, and a per-trade "rationale" string.
- It is guarded by an inline disclaimer (approx. line 2188): *"Analytical recommendations only — not financial advice. Execute trades manually through your licensed broker."*

That last point is the crux: **this is the disclaimer-plus-advice-affordance pattern that IDENTITY.md §1 and §5 explicitly rule out.** A disclaimer next to an imperative "BUY … Limit $X … Stop $Y" affordance is regulatory camouflage, not education. Under the shipping criterion (§6) and IDENTITY §5, verdict/advice affordances are removed from the UI, not disclaimed. This surface is the most advice-shaped in the app — larger than the SIGNAL column that INTERFACE_VERDICTS_SPEC.md addresses.

## Temp shim currently in place

To decouple this deferred surface from the completed SIGNAL→SHARPE(1Y) work, the Discovery `sig` field now carries a descriptive Sharpe **band** (High/Moderate/Low/Minimal/Negative) instead of a verdict label. The Trade Desk still needs the old verdict string for its `SIG_ACT` lookup and rationale, so a temporary shim reproduces the retired thresholds locally:

- **Location:** `static/quantex.html`, inside `TradeDeck`, `const legacyVerdict = sh => …` — marked `// TEMP SHIM — Trade Desk remediation pending, see TRADE_DESK_SPEC`.
- **Effect:** the desk behaves as before for live/screened data (backend used the identical Sharpe thresholds, so `legacyVerdict(a.sh)` reproduces the old `a.sig` exactly). For the offline 38-asset sample fallback, whose hand-authored verdicts were internally inconsistent with their own Sharpe, the derived action is unchanged (Strong Buy/Buy both collapse to `buy`) but a rationale substring may differ (e.g. "Strong Buy signal" → "Buy signal"). Sample-fallback only.
- The shim must be **removed** as part of this remediation.

## Related deferred item

`app/fundamental.py` `recommendation` field (lines ~110, 386→557, 617→637) is **Quantex-computed** (Claude synthesis + a quant fallback scorer), producing "Strong Buy/Buy/Hold/Sell/Reduce" verdicts — the same defect class. (Distinct from `analyst_recommendation`, line 173, which is third-party yfinance `recommendationKey` — a labeled external fact, out of scope.) Fold the Quantex-computed `recommendation` into this remediation.

## Related copy debt

- Landing proof-block caption "Fundamentals and technicals can disagree" (`static/quantex.html`, proof-block near the MSFT SHARPE(1Y) / Indicator Consensus demo) needs a rewrite when the desk is remediated. It now pairs a Sharpe (1Y) display with a technical tilt, so "Fundamentals" no longer accurately names the left-hand quantity. Deferred with the desk to keep the landing narrative and the desk remediation consistent.

## Vocabulary debt

- **Backend `sig` field emits retired band words, unrendered as of Amendment 3.**
  `app/main.py::_sharpe_band` (and the frontend `sharpeBand`, `mapScreened` line
  ~369) still produce the High/Moderate/Low/Minimal/Negative band vocabulary in
  the `sig` field. As of INTERFACE_VERDICTS_SPEC.md Amendment 3, no UI column
  renders `sig` — the SHARPE (1Y) column uses the frontend reference-point label
  (`sharpeMarketLabel`, vs SPY), and the Trade Desk uses `legacyVerdict(a.sh)`.
  Left in place deliberately: `_sharpe_band` is a pure function of a single
  Sharpe with no SPY reference in scope, so it structurally cannot emit the
  Amendment 3 market-relative labels; changing it would also disturb the sealed
  Amendment-1 closed-set tests for no rendered benefit. Retire the field (or give
  it a reference-aware producer) if/when the backend gains a market reference.

## Open design question (to be answered cold, before any implementation)

**Should a capability demonstration (IDENTITY.md §1) issue machine-generated trade recommendations at all — or should the "trade" surface be reframed as user-directed paper-trading only, where the user states the thesis and the tool executes/records, issuing no buy/sell/hold call of its own?**

Sub-questions that fall out of that:
- If reframed to user-directed only: what happens to order-type suggestions, limit-entry math, stop/target — are they user inputs, educational annotations, or removed?
- If some analytical output stays: how is it made descriptive (what the indicators/metrics *are*) rather than imperative (what to *do*), consistent with the SIGNAL→SHARPE(1Y) remediation?
- Does the "rationale" become an explanation of the metrics rather than a justification for an action?

Do not answer these here. This stub exists to hold the defect and the question until that cold decision is made.
