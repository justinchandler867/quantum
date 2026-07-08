FILINGS_SPEC.md — Filings & Qualitative Intelligence (Sealed)

Status: Sealed. This is the governing document for the qualitative epic. Build in phases, each with its own audit gate. STOP conditions literal.
Authority: IDENTITY.md (education-native capability demo). Amendment 3 extends to qualitative claims: the reference point for every generated statement is a named filing, section, and retrievable passage — "per the FY2025 10-K, Item 1A" is the qualitative form of "vs SPY, same window."
Prime directive (the groundedness rule): every sentence Quantex generates about a company is either (a) anchored to a specific filing passage the user can open with one click, or (b) visibly labeled as synthesis across cited passages. Nothing free-floating. When the filings don't contain an answer, the system says so — it never fills the gap from model memory. This is the invented-attribution defect class applied to language; the entire epic lives or dies on this rule.
What this epic is NOT: no sentiment scoring, no bullish/bearish characterization of filings, no forecasts, no verdicts, no "risk factors suggest…" advice framing. The tool reports what the company disclosed and teaches how to read it.


Phase 0 — fundamental.py remediation (prerequisite, small)

The existing Claude chain (fundamental.py) becomes the epic's engine seed, but its verdict machinery must die first:


Remove the recommendation field entirely: strip the "Strong Buy…Sell" instruction from the synthesis prompt (line ~386) and delete the quant fallback (~617–637). The field does not get renamed or neutralized — it gets removed. (analyst_recommendation, the yfinance passthrough, stays: a labeled third-party fact.)
Rewrite the synthesis prompt's contract: factual description of the business, segments, and financial trends, each claim tied to its data source; explicit instruction that it must not recommend, rate, or characterize attractiveness, and must flag when requested information is unavailable rather than inferring.
The orphaned /api/analyze endpoint stays (it will be consumed by Phase 2) but now serves verdict-free output.
Acceptance: grep proof no verdict vocabulary in prompts or response models; existing suite green.


Phase 1 — EDGAR data spine


Source: SEC EDGAR only. Free, official, primary. Required mechanics: declared User-Agent with contact info per SEC policy; respect the ~10 req/s guidance; local caching of every fetched document (never re-fetch what's cached).
Retrieval path: ticker → CIK (EDGAR company_tickers.json) → submissions JSON → latest 10-K and 10-Q accession numbers → filing documents.
Parsing: extract at minimum Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A) from 10-Ks; MD&A and risk-factor updates from 10-Qs. ⚠ EDGAR HTML is heterogeneous across filers — section-boundary detection WILL fail on some filings. Failure behavior is sealed: a section that can't be confidently bounded is marked UNPARSED for that filing (surfaced, never silently skipped, never approximated). Track and report the parse success rate.
Index: chunk parsed sections (chunk = passage small enough to cite, large enough to read; ~1–3 paragraphs) with metadata {ticker, form, filing_date, accession, item, char_offsets}. Embed and index locally — reuse the sentence-transformers pattern from the RAG demo (all-MiniLM-L6-v2 or better) rather than inventing a new stack.
Storage: on-disk cache + index under backend (gitignored if large). Nothing fetched at import time; lazy per-ticker with explicit refresh.
Acceptance: for 5 diverse tickers (mega-cap, mid-cap, a bank, a REIT, an ADR if supported — print which), show parsed section inventory, chunk counts, UNPARSED flags, and 3 retrieved chunks for a sample query each, with metadata intact.


Phase 2 — Filings tab (per-ticker surface)

A "Filings" tab in the ticker detail view rendering, from the local index + Phase 0 engine:


Business summary: generated, every sentence carrying a citation chip → click opens the source passage (modal or expandable) with the filing/item/date header.
Risk factors, year-over-year diff — the centerpiece: current 10-K Item 1A vs prior year's. Render three lists: ADDED (new risk factors), REMOVED, MATERIALLY CHANGED (with the changed language shown). ⚠ Diffing method: match risk factors by heading/lead-sentence similarity, not raw text diff (filers reorder and reflow). If matching confidence is low for a filing pair, the diff renders "Comparison unavailable — sections could not be reliably aligned" rather than a garbage diff. Every diff entry cites both filings.
MD&A distillation: management's own framing of results and trends, cited per claim; direct quotes marked as quotes.
Financial facts: from the filing (or EDGAR companyfacts), labeled with period and source.
§2a — Findings / "What changed" block (LOAD-BEARING — the tab leads with this, not with the summary): the conclusions layer. A prominent headline block at the top of the tab surfacing the notable facts, each cited:

Diff-derived findings: "3 risk factors added this year — the most since FY2021" / "Customer-concentration risk factor added" / "Going-concern language appears for the first time in available filings."
Materiality extractions (pure extraction, zero judgment, quantified where the filing quantifies): customer concentration ("one customer = 22% of revenue, per Item 1A"), litigation reserves and material legal proceedings, covenant/liquidity language flagged verbatim, segment revenue shifts between filings, auditor changes, material weakness disclosures.
Findings are FACTUAL CONCLUSIONS that land — they name what changed and why it's notable — without attractiveness judgments. "Concentration doubled" is a finding; "concentration is concerning" is a verdict. The first is required; the second is prohibited.
A filing period with nothing notable renders "No material changes detected vs prior filing" — an honest null, itself informative.



Header on the tab (sealed copy): "Summarized from SEC filings. Every claim links to the source passage. Quantex reports what the company disclosed — it does not evaluate or recommend."
Acceptance: rendered tab for 3 tickers including one with a real YoY risk-factor change (print the diff AND the Findings block); citation chips resolve (click-through evidence); one deliberately induced low-confidence diff showing the unavailable-fallback; one honest-null Findings case.


The usefulness test (sealed now, judged at Phase 2 — the owner's verdict)

Separate from the truthfulness harness. When Phase 2 renders, the owner runs the tab on THREE companies he already knows well. Pass bar, pre-committed: (1) the Findings block surfaces things he'd have wanted to know, faster than manual reading, with receipts; (2) the YoY diff catches at least one real, verifiable change; (3) he would genuinely use it himself before a client conversation or study session. Verdict recorded in writing either way. If the verdict is FAIL — the tool reads as an inert fact-lister — the pre-registered consequence executes: the public-facing epic halts, and the project forks to a personal-only build (private branch, restrictions lifted for that fork only, never demoed or disclosed). If PASS, the epic proceeds to Phase 3. The Findings layer's prominence is the spec author's responsibility: a Phase 2 that buries its conclusions fails the SPEC, not the identity.

Phase 3 — Grounded Ask Quantex mode

When a question concerns a specific company, Ask Quantex answers from that company's indexed filings:


Retrieval floor: relevance threshold below which the answer is a decline — "The filings I have for {TICKER} don't address that." (The RAG relevance-floor pattern, enforced at retrieval, verifiable without trusting the generator.)
Answer contract (system prompt, sealed): answer ONLY from provided passages; cite each claim to its passage id; label any cross-passage synthesis as such; if passages are insufficient, say so; never use general knowledge about the company to fill gaps; forecast-shaped questions get redirected to what the filing states plus the standing educational guardrail (Phase 0's no-recommendation contract applies).
Rendering: citations as chips inline, same click-through as Phase 2.
Acceptance: runs against the Phase-3 harness (below) with results printed raw.


The groundedness harness (build alongside Phase 2, gate for Phase 3)

Sealed before Phase 3 ships, in docs/FILINGS_HARNESS.md:


~20 hand-verified Q&A cases across ≥4 tickers: for each, the question, the expected source passage(s), and the key facts a correct answer contains. Answers are graded on citation correctness (does the cited passage actually support the sentence?), not eloquence.
≥5 adversarial cases where the CORRECT output is a decline: questions the filings don't answer, forecast requests, and one question whose plausible-sounding answer exists in model memory but NOT in the filing (the memory-leak trap — the test that matters most).
≥3 cases on UNPARSED/missing filings verifying honest degradation.
Pass bar (sealed now): 100% of adversarial declines decline; ≥90% citation-correctness on answerable cases; ANY instance of an uncited factual claim or a citation that doesn't support its sentence = the run fails, regardless of other scores.
Harness results committed raw (FILINGS_HARNESS_RESULTS.md) before the Phase-3 surface is enabled.


Cross-cutting


News panel: untouched by this epic (verified clean passthrough). A later phase may interleave filings events (8-Ks) into it; out of scope now.
API cost/keys: generation uses the existing Anthropic API path from fundamental.py. Generated artifacts (Phase 2 summaries/diffs) are cached per (ticker, accession) — regenerate only on new filings, never per page-view.
Ranker untouched. Nothing in this epic feeds the screen-match score. Filings data informs the human; it does not re-rank.
STOP conditions: EDGAR access blocked or ToS-relevant surprises; parse success rate below ~70% on the Phase-1 sample (architecture rethink, not brute force); any spec assumption contradicted by fundamental.py's actual structure; harness pass bar unreachable after two iterations (report, don't lower the bar).


Phase commits: one commit per phase, each behind its own audit gate. Phase order is strict: 0 → 1 → (2 + harness) → 3.
