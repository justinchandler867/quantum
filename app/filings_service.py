"""
Filings view orchestrator (FILINGS_SPEC Phase 2).

Assembles the per-ticker Filings tab payload from the EDGAR spine + parser +
findings engine. Enforces the prime directive at the seam: every rendered claim
carries a citation id that resolves to a source passage; nothing free-floating.

Sealed behaviors wired here:
  - Findings / "What changed" block LEADS the payload (§2a, load-bearing).
  - YoY risk-factor diff with honest "comparison unavailable" fallback.
  - Incorporation-by-reference chase: Item 1A/7 that a 10-K carries in its
    Annual Report (Exhibit 13) are pulled from the exhibit and cited to the
    EXHIBIT, not the 10-K. If the chase fails, the section stays UNPARSED.
  - 20-F foreign filers are explicitly out of scope (honest message).
"""
import logging
import re
from dataclasses import dataclass, field

from app.filings_edgar import (
    latest_filings, fetch_document, resolve_cik, find_exhibit, fetch_named_document,
)
from app.filings_parse import (
    parse_filing, locate_section_raw, locate_title_section_raw, html_to_text,
    STATUS_PARSED, STATUS_UNPARSED,
)
from app.filings_findings import (
    split_risk_factors, diff_risk_factors, findings_from_diff, extract_materiality,
    MIN_RF_FOR_DIFF,
)
from app.filings_facts import get_financial_facts

logger = logging.getLogger(__name__)

TAB_HEADER = ("Summarized from SEC filings. Every claim links to the source passage. "
              "Quantex reports what the company disclosed — it does not evaluate or recommend.")

_TITLE_KW = {"1A": ("risk", "factor"), "7": ("management", "discussion")}


# ── Citations ────────────────────────────────────────────────────────────────

class CitationRegistry:
    def __init__(self):
        self.map: dict = {}
        self._n = 0

    def add(self, meta: dict, quote: str = "") -> str:
        cid = f"c{self._n}"
        self._n += 1
        entry = {k: v for k, v in meta.items() if k != "quote"}
        entry["quote"] = re.sub(r"\s+", " ", quote)[:600]
        self.map[cid] = entry
        return cid


# ── Resolved section (with IBR chase) ────────────────────────────────────────

@dataclass
class ResolvedSection:
    item: str
    title: str
    status: str
    method: str
    text: str = ""
    raw_html: str = ""
    source_doc: str = ""     # the document the content actually came from
    form: str = ""
    filing_date: str = ""
    accession: str = ""

    def meta(self) -> dict:
        return {"form": self.form, "filing_date": self.filing_date,
                "accession": self.accession, "item": self.item,
                "source_doc": self.source_doc, "title": self.title}


def _resolve_section(ref, doc, parsed, item: str) -> ResolvedSection:
    """
    Resolve a target section's text + raw HTML, chasing incorporation-by-reference
    into the Annual Report exhibit when needed. Citations name the true source doc.
    """
    sec = parsed.sections.get(item)
    title = sec.title if sec else item
    rs = ResolvedSection(item=item, title=title, status=STATUS_UNPARSED, method="none",
                         form=ref.form, filing_date=ref.filing_date, accession=ref.accession,
                         source_doc=ref.primary_doc)

    if sec and sec.status == STATUS_PARSED:
        rs.status, rs.method, rs.text = STATUS_PARSED, sec.method, sec.text
        rs.raw_html = locate_section_raw(doc, ref.form, item) or ""
        return rs

    # Incorporation-by-reference → chase Exhibit 13 (Annual Report).
    if sec and sec.method == "incorporated-by-ref" and item in _TITLE_KW:
        try:
            ex = find_exhibit(ref, types=("EX-13",))
            if ex:
                ex_doc = fetch_named_document(ref, ex[0])
                raw = locate_title_section_raw(ex_doc, _TITLE_KW[item])
                if raw:
                    text = html_to_text(raw)
                    ok = (len(text) > 4000 and not _looks_like_toc(text))
                    if item == "1A":  # require a splittable risk section
                        ok = ok and sum(
                            not r.is_category for r in split_risk_factors(raw)) >= MIN_RF_FOR_DIFF
                    if ok:
                        rs.status, rs.method = STATUS_PARSED, "exhibit-13"
                        rs.text, rs.raw_html, rs.source_doc = text, raw, ex[0]
                        logger.info(f"{ref.ticker} Item {item}: resolved via {ex[0]} (IBR chase)")
                        return rs
        except Exception as e:  # noqa: BLE001
            logger.warning(f"IBR chase failed for {ref.ticker} Item {item}: {e}")
        rs.method = "incorporated-by-ref-unresolved"
    elif sec:
        rs.method = sec.method
    return rs


def _looks_like_toc(text: str) -> bool:
    """A table-of-contents fragment: many standalone page-number lines up front."""
    head = text[:1500]
    nums = len(re.findall(r"(?m)^\s*\d{1,4}\s*$", head))
    return nums >= 4


# ── Sentence helpers (extractive, grounded) ──────────────────────────────────

def _sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", s).strip()
            for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]


def _lead_sentences(text: str, n: int = 4, minlen: int = 45) -> list[str]:
    out = []
    for s in _sentences(text):
        if len(s) >= minlen and re.search(r"[A-Za-z]", s):
            out.append(s)
        if len(out) >= n:
            break
    return out


def _result_sentences(text: str, n: int = 5) -> list[str]:
    pat = re.compile(r"(?i)(increased|decreased|grew|declined|higher|lower|compared to|"
                     r"year[- ]over[- ]year|primarily due to|driven by)")
    out = []
    for s in _sentences(text):
        if len(s) > 60 and pat.search(s):
            out.append(s)
        if len(out) >= n:
            break
    return out


# ── Main entry ───────────────────────────────────────────────────────────────

def build_filings_view(ticker: str, refresh: bool = False) -> dict:
    ticker = ticker.upper()
    refs = latest_filings(ticker, forms=("10-K",), per_form=2, refresh=refresh)

    if not refs:
        cik = resolve_cik(ticker, refresh=refresh)
        if cik is not None:
            return {"ticker": ticker, "status": "unsupported_foreign", "header": TAB_HEADER,
                    "message": "Filings analysis not yet available for foreign private "
                               "issuers (20-F filers)."}
        return {"ticker": ticker, "status": "no_data", "header": TAB_HEADER,
                "message": f"No SEC 10-K filings found for {ticker}."}

    reg = CitationRegistry()
    cur = refs[0]
    prior = refs[1] if len(refs) > 1 else None
    cur_doc = fetch_document(cur, refresh=refresh)
    cur_parsed = parse_filing(cur_doc, "10-K", ticker=ticker, cik=cur.cik,
                              accession=cur.accession, filing_date=cur.filing_date)

    business = _resolve_section(cur, cur_doc, cur_parsed, "1")
    risk_cur = _resolve_section(cur, cur_doc, cur_parsed, "1A")
    mdna = _resolve_section(cur, cur_doc, cur_parsed, "7")

    risk_prior = None
    prior_parsed = None
    if prior:
        prior_doc = fetch_document(prior, refresh=refresh)
        prior_parsed = parse_filing(prior_doc, "10-K", ticker=ticker, cik=prior.cik,
                                    accession=prior.accession, filing_date=prior.filing_date)
        risk_prior = _resolve_section(prior, prior_doc, prior_parsed, "1A")

    findings = []

    # ── YoY risk-factor diff (centerpiece) ──
    diff_payload = {"available": False,
                    "reason": "Prior-year 10-K not available for comparison.",
                    "added": [], "removed": [], "changed": [], "counts": {}}
    if risk_cur.raw_html and risk_prior and risk_prior.raw_html:
        cur_rfs = split_risk_factors(risk_cur.raw_html)
        pri_rfs = split_risk_factors(risk_prior.raw_html)
        diff = diff_risk_factors(cur_rfs, pri_rfs)
        cc = reg.add(risk_cur.meta(), risk_cur.text[:280])
        pc = reg.add(risk_prior.meta(), risk_prior.text[:280])
        diff_payload = {
            "available": diff["available"], "reason": diff["reason"],
            "counts": diff["counts"], "current_cite": cc, "prior_cite": pc,
            "added": [{"heading": a["heading"], "citation": reg.add(risk_cur.meta(), a["heading"])}
                      for a in diff["added"]],
            "removed": [{"heading": r["heading"], "citation": reg.add(risk_prior.meta(), r["heading"])}
                        for r in diff["removed"]],
            "changed": [{"heading": c["heading"], "prior_heading": c["prior_heading"],
                         "body_similarity": c["body_similarity"],
                         "citation": reg.add(risk_cur.meta(), c["heading"]),
                         "prior_citation": reg.add(risk_prior.meta(), c["prior_heading"])}
                        for c in diff["changed"]],
        }
        if diff["available"]:
            findings += findings_from_diff(diff, risk_cur.meta(), risk_prior.meta())

    # ── Materiality extractions on available sections ──
    for sec in (risk_cur, business, mdna):
        if sec.text:
            findings += extract_materiality(sec.text, sec.meta())

    # ── Findings payload with citations ──
    findings_payload = []
    for f in findings:
        cids = [reg.add(src, src.get("quote", f.detail)) for src in f.sources]
        findings_payload.append({"kind": f.kind, "headline": f.headline,
                                 "detail": f.detail, "citations": cids})

    # ── Business summary (extractive, every sentence cited) ──
    business_summary = []
    if business.text:
        for s in _lead_sentences(business.text, n=4):
            business_summary.append({"text": s, "is_quote": True,
                                     "citation": reg.add(business.meta(), s)})

    # ── MD&A distillation (management's own framing, cited, quotes marked) ──
    mdna_payload = []
    if mdna.text:
        for s in _result_sentences(mdna.text, n=5):
            mdna_payload.append({"text": s, "is_quote": True,
                                 "citation": reg.add(mdna.meta(), s)})

    # ── Financial facts ──
    facts = get_financial_facts(cur.cik, refresh=refresh)

    # ── Filing inventory (surfaces UNPARSED honestly) ──
    inventory = []
    for ref, parsed, resolved_map in [
        (cur, cur_parsed, {"1": business, "1A": risk_cur, "7": mdna}),
        (prior, prior_parsed, {"1A": risk_prior} if risk_prior else {}),
    ]:
        if not parsed:
            continue
        secs = {}
        for item, s in parsed.sections.items():
            r = resolved_map.get(item)
            secs[item] = {"title": s.title,
                          "status": r.status if r else s.status,
                          "method": r.method if r else s.method,
                          "source_doc": r.source_doc if r and r.status == STATUS_PARSED else ref.primary_doc}
        inventory.append({"form": ref.form, "filing_date": ref.filing_date,
                          "accession": ref.accession, "sections": secs})

    findings_null = len(findings_payload) == 0

    return {
        "ticker": ticker, "status": "ok", "header": TAB_HEADER,
        "findings": findings_payload, "findings_null": findings_null,
        "business_summary": business_summary,
        "risk_diff": diff_payload,
        "mdna": mdna_payload,
        "financial_facts": facts,
        "filings": inventory,
        "citations": reg.map,
    }
