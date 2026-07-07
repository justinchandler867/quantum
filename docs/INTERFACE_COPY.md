# Interface Copy — Relabel + Momentum Reframing (2026-07-07)

Exact strings for Claude Code to apply in `backend/static/quantex.html`.
Rationale for each block is in the comment above it. Strings are final —
edit here first if they must change, then re-apply.

**Addendum 2026-07-07 (applied same day as build):** §2's final sentence
("The '% off 52-wk high' column…") is held back until §3's columns exist —
the disclosure must not reference a column that isn't rendered. Restore the
sentence in the same commit that adds the column. Landing proof block
retitled "Our outlook shows its math." / caption "…every tilt magnitude
disclosed as ours…" and the Outlook modal disclaimer de-attributed
("Quantex's own interpretation of published institutional commentary —
directions inferred, magnitudes chosen by us") — replacing strings that
dangled after §1's attribution removal.

---

## 1. Outlook attribution — REMOVE variant (default; matches compliance email draft)

Applies until compliance approves any attribution. Remove all occurrences of
"Fort Washington", "Co-CIO", "Shipley", "White" from the UI.

- Feature toggle label: `2026 Market Outlook (Quantex interpretation)`
- Modal title: `2026 Market Outlook — how Quantex encodes it`
- Modal body, first paragraph:
  > These tilts are Quantex's own directional interpretation of publicly
  > published institutional commentary. The direction of each tilt follows the
  > commentary; the magnitudes (e.g., +150bp) were chosen by Quantex as
  > order-of-magnitude estimates — no source publishes these numbers. Treat
  > them as a starting hypothesis to weigh, not a forecast to act on.
- Conviction dial relabel: `Tilt strength` (values: `Light / Half / Strong`)
  with caption:
  > This dial sets how much of each tilt the optimizer applies (33% / 50% / 75%).
  > It is your preference, not a measure of anyone's confidence.

## 1b. RELABEL variant (only if compliance approves attribution in writing)

- Modal body, first paragraph:
  > Directions follow Fort Washington Investment Advisors' published 2026
  > Outlook. The specific magnitudes are Quantex's interpretation, not the
  > firm's numbers. Fort Washington is not affiliated with Quantex and has not
  > reviewed or endorsed this feature.

## 2. Momentum tail disclosure (Growth screens)

Placement: directly beneath the Growth shortlist header — table-adjacent,
not in a tooltip, not collapsed. Non-dismissable.

> **What this screen selects on.** Growth screens here are momentum-weighted:
> they surface what has already gone up. The momentum premium is real, and it
> reverses violently in regime shifts — past momentum crashes took back years
> of gains in weeks. The more a name has run, the more of that risk it carries.
> The "% off 52-wk high" column shows you the extension directly.

One-line variant for space-constrained views:
> Momentum screens surface recent winners. That premium reverses hard in
> regime shifts — check the drawdown columns before trusting the rank.

## 3. New columns (Discovery table)

| Column header | Tooltip |
|---|---|
| `% off 52-wk high` | `How far below its 52-week high this trades. Near 0% = extended; deep negative = fallen. Neither is automatically good.` |
| `1Y` / `3Y` / `5Y` | `Total return over each period. One good year is not a track record — compare across horizons.` |
| `Max DD (5Y)` | `Worst peak-to-trough loss over five years. This is what holding it actually felt like.` |

Rule: `1Y` must never render without `3Y`/`5Y` and `Max DD` alongside —
the one-year-window critique is answered structurally, not with a caveat.

## 4. Fit score / rank framing (pending validation verdict)

Until VALIDATION_SPEC.md returns a verdict, the shortlist header reads:

> **Ranked by screen match — not a recommendation.** This ranking reflects how
> closely each name matches your profile's factor screen. Whether that screen
> adds value over random selection from the same universe is currently being
> tested out-of-sample; the interface will show the result either way.

After the verdict, replace per VALIDATION_SPEC.md §8 (pre-registered).

## 5. Univariate-score disclosure (correlation gap)

Placement: footnote row of the Discovery table, always visible.

> Fit scores rate each security on its own — they don't know what you already
> hold. Two top-ranked names can be near-duplicates of each other or of your
> portfolio. Diversification happens in the optimizer, not in this list.

(Structural fix — the correlation-to-portfolio column — is specced separately.)

## 6. Regime stress copy (Finding 5, resolved option a)

> A regime isn't a crash test — an outlook can favor your positioning. Green
> here means this scenario helps you, not that you're safe in all scenarios.

## 7. Universe honesty

Anywhere the UI or marketing describes the pool: the screened universe is
**~600 securities** (S&P 500 + major ETFs + a curated extended list), not
"the Nasdaq" and not 10,000. Also fix the stale docstring in
`screener.py:load_nasdaq_tickers` / module header ("full Nasdaq universe").
