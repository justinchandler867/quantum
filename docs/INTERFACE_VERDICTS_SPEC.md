# INTERFACE_VERDICTS_SPEC.md — SIGNAL Column & AI ADVISER Remediation (Sealed)

**Status:** Sealed. Build against this verbatim. STOP and report (do not guess) on anything marked ⚠️ UNDERDETERMINED or on any codebase reality that contradicts an assumption here.
**Authority:** IDENTITY.md §5 (pre-registered) and the shipping criterion (§6). Both defects fail the criterion: verdict labels and the regulated word "ADVISER" sit inches from "not a recommendation" copy and create regulatory exposure.
**Scope:** Display/copy layer only. No changes to ranking math, fit scores, factor computation, or any API response semantics beyond the label fields specified below. Ranks must be byte-identical before/after (see Acceptance).

---

## Part A — SIGNAL column

### A1. Defect
The discovery/shortlist table renders per-ticker verdicts ("Strong Buy", "Buy", presumably "Hold"/"Sell" variants) in a SIGNAL column. Verdict vocabulary is advice-shaped and contradicts the "Ranked by screen match — not a recommendation" header.

### A2. Required behavior
Replace verdict labels with **descriptive technical state**. The column shows what the indicators *are*, never what to *do*.

**Rename the column** SIGNAL → **TREND** (header text and any tooltip/aria-label).

**Label vocabulary (closed set — no label outside this list may render):**

Derive from whatever indicator states currently feed the verdict logic. Expected inputs are price-vs-moving-average state, MACD cross state, and/or RSI zone. Map to:

| Underlying state | Label |
|---|---|
| Price above rising intermediate MA, MACD bullish | `Uptrend` |
| Price below falling intermediate MA, MACD bearish | `Downtrend` |
| Mixed / flat MA / no clear directional agreement | `Sideways` |
| RSI ≥ 70 (append to any of the above) | `· RSI 7x` (actual value, e.g. `Uptrend · RSI 74`) |
| RSI ≤ 30 (append) | `· RSI 2x` (actual value) |
| Insufficient data (new listing, gap) | `—` (em dash, with tooltip "Insufficient history") |

⚠️ UNDERDETERMINED: the exact indicator inputs behind the current verdicts. **Locate the function that produces "Strong Buy"/"Buy" strings, print its input signals, and STOP for confirmation of the mapping above before implementing** if the inputs differ materially from price/MA/MACD/RSI (e.g., if the verdict is a composite score with no recoverable state semantics, report that — the fallback is dropping the column entirely rather than inventing a mapping).

**Prohibited vocabulary anywhere in this column, its tooltips, or its legend:** buy, sell, hold, accumulate, avoid, strong, weak (as a verdict), bullish/bearish *as a standalone label* (permitted only inside a tooltip describing MACD state, e.g. "MACD: bearish cross on 6/30"), overbought/oversold as bare labels (permitted as tooltip explanation of an RSI number, since the number itself is the label).

**Tooltip (each cell):** one line stating the facts, e.g. `Above 50-day MA · MACD bullish cross 6/24 · RSI 74`. Facts and dates only; no interpretation beyond the closed-set label.

**Column footnote (once, table footer or legend):**
> Trend describes current technical state only. It is not a forecast and not a recommendation.

### A3. What NOT to do
- Do not soften verdicts into euphemisms ("Favorable", "Attractive", "Positive setup") — these are verdicts wearing a costume and fail the same criterion.
- Do not add color semantics that reintroduce the verdict (e.g., green=good/red=bad on the label itself). Neutral text color for all labels. RSI extreme values may use the existing warning accent if one exists, since an extreme reading is a fact.
- Do not touch sort order, ranking, or fit-score display.

---

## Part B — "AI ADVISER" panel

### B1. Defect
Panel titled with the regulated word "ADVISER" and populated with advice-shaped prompt chips ("What should I hedge?"). Fails IDENTITY.md §1 and §5 regardless of any disclaimer nearby.

### B2. Required behavior

**Rename:** panel title becomes **"Ask Quantex"**. Subtitle line, if the layout has one:
> Explains what the analytics are showing — educational, not advice.

**Sweep:** grep the entire frontend AND backend-served strings for `advis` (case-insensitive, catches adviser/advisor/advisory/advise). Every hit must be removed or reworded. Print the hit list before changing anything. ⚠️ If any hit is in a variable/route/identifier name rather than user-facing copy, report it — identifiers may stay if never rendered, but list them so the decision is explicit.

**Chip rewording — exact replacements.** Pattern: advice-shaped ("what should I do") → mechanism-shaped ("how does this work / what is this showing"). Apply to the current chip set; where a current chip isn't listed below, apply the same pattern and print the proposed rewording for audit before committing.

| Current (advice-shaped) | Replacement (educational) |
|---|---|
| What should I hedge? | How does hedging work for a portfolio like this? |
| What should I buy? *(if present)* | What is this ranking actually measuring? |
| Should I rebalance? *(if present)* | What does rebalancing do, and what are its tradeoffs? |
| Is this portfolio too risky? *(if present)* | How is risk being measured on this screen? |
| What do you recommend? *(if present)* | Walk me through how the fit score is computed |

**System-prompt guardrail (backend):** locate the system prompt for this panel's LLM calls. It must contain an explicit instruction to the effect of: *"You are an educational explainer for Quantex's analytics. You explain what displayed metrics mean and how they are computed. You never recommend buying, selling, holding, or allocating to any security or strategy, never answer 'what should I do' questions with a course of action, and redirect such questions to how the relevant concept works. You are not a financial adviser and must say so if asked."* ⚠️ Print the current system prompt before editing; if there is no system prompt (raw user passthrough), report that as a separate finding — it's a bigger defect than the label.

**Panel footer (persistent, non-dismissable, matching the momentum-tail disclosure treatment):**
> Educational explanations only. Nothing here is investment advice or a recommendation.

### B3. What NOT to do
- Do not rely on the footer disclaimer to keep advice-shaped chips — the chips themselves must change (disclaimer + advice-shaped affordance = camouflage, fails IDENTITY.md).
- Do not rename to anything containing "coach," "guide," "planner," or "strategist" — role-claiming words drift back toward the same problem.

---

## Acceptance (all must pass; print evidence for each)

1. `grep -ri "strong buy\|advis"` across frontend and backend user-facing paths → zero user-facing hits. Print the full grep output.
2. TREND column renders only labels from the closed set in A2, verified across the full current shortlist (print one full column's rendered values).
3. Ranks byte-identical: dump ranked ticker order for the standard test screen before and after; diff is empty. (Same method used for the multi-horizon fix verification.)
4. Test suite: all 85 existing tests still pass; add tests for (a) verdict-vocabulary absence in TREND output, (b) chip text matches replacement table, (c) system prompt contains the guardrail sentence.
5. Screenshot-equivalent check: print rendered strings for panel title, subtitle, footer, and all chips.
6. Stray file `backend/_f1_verify.py`: already deleted from working tree; confirm absent.

**Commit message:** `interface: remove verdict language (SIGNAL→TREND) and rename AI ADVISER→Ask Quantex per IDENTITY.md §5`

**STOP conditions (report, don't improvise):** verdict logic has no recoverable indicator semantics (A2); "advis" hits in identifiers (B2); missing/absent system prompt (B2); any current chip not covered by the replacement table (B2); any test failure traceable to ranking rather than labels (Acceptance 3–4).
