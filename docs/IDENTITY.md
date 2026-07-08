# IDENTITY.md — Product Identity Decision (Sealed)

**Decided:** July 7, 2026, cold, in a fresh session — re-decision of the identity call originally made under adversarial-rehearsal pressure, per the flag planted in that session.
**Status:** In force. Changes to this document require a written amendment with rationale, not silent drift.

---

## 1. The Decision

Quantex is an AI capability demonstration for financial services. It is education-native in presentation: every ranking is a worked example that shows its reasoning, not a recommendation. It is a professional portfolio artifact demonstrating disciplined application of frontier AI to investment analysis.

Quantex is **not**:

- An advisory service, robo-advisor, quant fund, or "AI financial advisor" — in function, in copy, or in how it is described anywhere (portfolio site, LinkedIn, interviews, demos).
- A monetized product. No affiliate revenue, subscriptions, or income of any kind while the author is a registered representative of W&S Brokerage Services. This is foreclosed, not deferred.
- A public tool. No public URL until W&S compliance has reviewed and not objected. Render deployment confirmed suspended as of this date.

The prior "CFA prep wrapper" framing is retired. Education is not camouflage layered over an advice engine; it is the genuine mode of the product. Anything that reads as regulatory camouflage fails the shipping criterion by definition.

## 2. Provenance (stated for the record)

Quantex is the sole personal property of Justin R. Chandler, independently developed:

- Entirely on personally owned hardware, personal accounts, and personal time.
- Using no W&S data, code, systems, internal documents, or client information — including in test fixtures and seeded data.
- With timestamped git history as evidence of independent creation, backed up outside any employer-accessible location.

Disclosure to W&S is voluntary and does not constitute assignment, license, or an invitation to direct further development. Any W&S use of or extension to Quantex requires an explicit written arrangement agreed in advance. Informal feature requests following a demo will be acknowledged and routed to that conversation, not built.

Precondition before disclosure: review all signed employment/RR agreements for invention-assignment or work-product language; obtain legal counsel if that language is broad. The compliance email does not go out before this review is complete.

## 3. What this identity rules in

- The full analytical engine, without restriction: optimizer, Black-Litterman-lite, Shapley regime attribution, View Sandbox, multi-horizon display, correlation context, fit-score ranking. Capability is the point; the engine has no compliance surface — only the pipe to the public does.
- Live demos from localhost or screen-share, narrated by the author.
- The walk-forward validation backtest (VALIDATION_SPEC.md) as the centerpiece artifact — the sealed, out-of-sample answer to "does the ranker add value at all" is the single strongest demonstration of discipline-with-intention, and is promoted accordingly.
- Honest defect disclosure at point of use, per the shipping criterion.

## 4. What this identity rules out (loss inventory, accepted)

- All monetization while a registered rep at W&S. Accepted in writing here so it cannot leak back in through UI copy, affiliate links, or "just testing" experiments.
- External/retail users and the public URL. The retail/institutional advisory ambition is gone, not paused. If a sanctioned version ever exists, that is W&S's decision under W&S's compliance, with the author's role defined by written agreement.
- Verdict language and advice affordances anywhere in the UI (see §5).
- The "AI advisor/quant/fund manager" pitch in all external descriptions of the project.
- Deploy speed. Nothing becomes publicly reachable — including conference demos with audience-accessible URLs — without prior compliance review.

## 5. Pre-registered implications for open UI questions

These are decided now, by this identity, so they are not re-litigated item by item:

- **SIGNAL column:** verdict labels ("Strong Buy"/"Buy") are removed. Replace with descriptive facts or trend states (e.g., RSI level/zone, MACD cross state, "uptrend / consolidating / downtrend"). No imperative or verdict vocabulary anywhere in the table.
- **"AI ADVISER" panel:** renamed (candidates: "AI Tutor," "Ask Quantex"). All chips reworded from advice-shaped ("What should I hedge?") to educational ("How does hedging work in this portfolio's context?"). The regulated word "adviser/advisor" appears nowhere in the UI.
- **Existing honesty work stands:** "Ranked by screen match — not a recommendation" header, momentum-tail disclosure, univariate fit-score footnote, "not a crash test" regime line, "Tilt strength / your preference" dial label, removal of all Fort Washington / Co-CIO attribution. These are identity-consistent, not merely defensive; they do not get softened later.
- **Correlation column picker:** proceed with the picker-chip approach ("Corr vs: Portfolio A ▾") — an explicit user-chosen comparison is the education-native answer; a silent "primary portfolio" default is the advice-shaped one.

## 6. Relationship to the shipping criterion

The standing criterion remains in force: a known defect may be live only if (a) disclosed at point of use in language the target user parses, (b) implicates no one but the author, and (c) cannot cause irreversible user harm or regulatory exposure if severity is misjudged. Identity (c) tightens (c): while the author is a registered rep, any public availability without compliance review is treated as regulatory exposure and fails automatically.

## 7. Standing governance note

The sealed-prediction protocol still has no receptor for premise-level errors (author writes both spec and expectations). This document is itself premise-level. The red-team seat for premise review remains unfilled; until it is, premise decisions like this one get at least one adversarial pass by an outside interlocutor before being sealed — this document received that pass.

*Commit to backend/docs/IDENTITY.md. The compliance email draft remains outside git and is updated to reflect §1–§2 before sending.*
