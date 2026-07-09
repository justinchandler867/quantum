"""
Findings / "What changed" engine (FILINGS_SPEC Phase 2, §2a — LOAD-BEARING).

Everything here is EXTRACTION, not judgment. A finding names what changed and
why it is notable, quantified where the filing quantifies — never whether that
is good or bad. "Concentration doubled" is a finding; "concentration is
concerning" is a verdict and is prohibited. Every finding carries citations.

Two sources of findings:
  1. YoY risk-factor diff — current 10-K Item 1A vs the prior year's. Risk
     factors are matched by heading similarity (filers reorder/reflow), never
     raw text diff. Low alignment confidence -> the diff is withheld with an
     honest "comparison unavailable", never a garbage diff.
  2. Materiality extractions — customer concentration, litigation/legal
     proceedings, going-concern & covenant/liquidity language (verbatim),
     material-weakness disclosures. Pattern-present -> finding; pattern-absent
     contributes to the honest null.
"""
import re
from dataclasses import dataclass, field, asdict

from app.filings_parse import html_to_text

# ── Risk-factor splitting ────────────────────────────────────────────────────

_BOLD_RUN_RE = re.compile(
    r'(?is)(?:font-weight:\s*(?:700|800|bold)|<b[>\s]|<strong[>\s])'
    r'(?:[^>]*>)?((?:<[^>]+>|[^<])*?)(?:</span>|</b>|</strong>)'
)


@dataclass
class RiskFactor:
    heading: str
    body: str
    is_category: bool  # a section/category label vs an individual risk statement


def _looks_like_risk_statement(text: str) -> bool:
    """An individual risk factor reads as a full statement, not a short label."""
    words = text.split()
    return len(text) >= 45 and len(words) >= 6


def split_risk_factors(section_raw_html: str) -> list[RiskFactor]:
    """
    Split an Item 1A raw-HTML span into risk factors by bold headings (verified
    across AAPL/DKS/O, which each mark risk-factor headings bold). Category
    labels (short noun phrases) are kept but flagged so the diff can focus on
    individual risk statements.
    """
    heads = []
    for m in _BOLD_RUN_RE.finditer(section_raw_html):
        txt = html_to_text(m.group(1)).strip()
        txt = re.sub(r"\s+", " ", txt)
        if 8 <= len(txt) <= 240 and re.search(r"[A-Za-z]", txt) \
                and not re.match(r"(?i)item\s+1a", txt):
            heads.append((m.start(), txt))
    # Dedupe adjacent identical headings (nested bold spans).
    dedup = []
    for pos, txt in heads:
        if dedup and dedup[-1][1] == txt:
            continue
        dedup.append((pos, txt))

    out: list[RiskFactor] = []
    for i, (pos, txt) in enumerate(dedup):
        end = dedup[i + 1][0] if i + 1 < len(dedup) else len(section_raw_html)
        body = html_to_text(section_raw_html[pos:end]).strip()
        out.append(RiskFactor(heading=txt, body=body,
                              is_category=not _looks_like_risk_statement(txt)))
    return out


# ── YoY risk-factor diff ─────────────────────────────────────────────────────

MIN_RF_FOR_DIFF = 5        # too few risk factors => split unreliable => withhold
HEADING_MATCH_THRESHOLD = 0.55   # cosine on headings: matched vs added/removed
BODY_CHANGE_THRESHOLD = 0.82     # matched pair below this similarity => changed


def _cosine_matrix(a_texts: list[str], b_texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    vec.fit(a_texts + b_texts)
    return cosine_similarity(vec.transform(a_texts), vec.transform(b_texts))


def diff_risk_factors(current: list[RiskFactor], prior: list[RiskFactor]) -> dict:
    """
    Align current vs prior risk factors by heading similarity. Returns
    {available, reason, added, removed, changed, counts}. If alignment can't be
    done confidently, available=False with a reason — never a garbage diff.
    """
    cur = [r for r in current if not r.is_category]
    pri = [r for r in prior if not r.is_category]
    if len(cur) < MIN_RF_FOR_DIFF or len(pri) < MIN_RF_FOR_DIFF:
        return {"available": False,
                "reason": "Comparison unavailable — risk factors could not be reliably "
                          "aligned (section structure not resolved for one or both years).",
                "added": [], "removed": [], "changed": [],
                "counts": {"current": len(cur), "prior": len(pri)}}

    sim = _cosine_matrix([r.heading for r in cur], [r.heading for r in pri])
    # One-to-one greedy matching by descending similarity: each current and each
    # prior factor is used at most once (a plain argmax double-assigns one prior
    # to several current factors and inflates "removed").
    pairs = sorted(
        ((float(sim[i][j]), i, j) for i in range(len(cur)) for j in range(len(pri))
         if sim[i][j] >= HEADING_MATCH_THRESHOLD),
        reverse=True,
    )
    cur_to_prior: dict = {}
    used_prior: set = set()
    for _s, i, j in pairs:
        if i in cur_to_prior or j in used_prior:
            continue
        cur_to_prior[i] = j
        used_prior.add(j)

    added, changed = [], []
    for i, rf in enumerate(cur):
        if i not in cur_to_prior:
            added.append({"heading": rf.heading})
            continue
        j = cur_to_prior[i]
        bsim = _cosine_matrix([rf.body], [pri[j].body])[0][0]
        if bsim < BODY_CHANGE_THRESHOLD:
            changed.append({"heading": rf.heading, "prior_heading": pri[j].heading,
                            "body_similarity": round(float(bsim), 3)})
    removed = [{"heading": pri[j].heading}
               for j in range(len(pri)) if j not in used_prior]

    return {"available": True, "reason": "",
            "added": added, "removed": removed, "changed": changed,
            "counts": {"current": len(cur), "prior": len(pri),
                       "added": len(added), "removed": len(removed),
                       "changed": len(changed)}}


# ── Materiality extractions (pure extraction, quantified) ────────────────────

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]


def _find_sentences(text: str, pattern: re.Pattern, limit: int = 2) -> list[str]:
    out = []
    for s in _sentences(text):
        if pattern.search(s):
            out.append(re.sub(r"\s+", " ", s)[:400])
            if len(out) >= limit:
                break
    return out


_CONCENTRATION_RE = re.compile(
    r"(?i)(?:no\s+(?:single|one)\s+(?:customer|client)|one\s+(?:customer|client)|"
    r"(?:largest|significant|single)\s+(?:customer|client)|customer\s+concentration).{0,160}?"
    r"\b\d{1,3}(?:\.\d+)?\s*%|"
    r"\b\d{1,3}(?:\.\d+)?\s*%.{0,60}?(?:of\s+(?:our\s+)?(?:net\s+)?(?:total\s+)?(?:revenue|sales|net\s+revenues))"
)
_GOING_CONCERN_RE = re.compile(r"going\s+concern", re.I)
_MATERIAL_WEAKNESS_RE = re.compile(r"material\s+weakness(?:es)?", re.I)
_COVENANT_RE = re.compile(
    r"\bcovenant(?:s)?\b.{0,120}?(?:comply|compliance|default|breach|require)"
    r"|(?:fail|failure|unable)\s+to\s+.{0,40}?\bcovenant", re.I)
_LITIGATION_RE = re.compile(
    r"(?:legal\s+proceedings|litigation|lawsuit|class\s+action).{0,160}?"
    r"(?:\$\s?\d|reserve|accrual|material|settlement)", re.I)


@dataclass
class Finding:
    kind: str
    headline: str            # the factual conclusion that lands
    detail: str = ""         # supporting extraction, quantified where quantified
    sources: list = field(default_factory=list)  # [{form,filing_date,accession,item,source_doc,quote}]

    def to_dict(self) -> dict:
        return asdict(self)


def _src(sec_meta: dict, quote: str) -> dict:
    d = dict(sec_meta)
    d["quote"] = re.sub(r"\s+", " ", quote)[:400]
    return d


def extract_materiality(section_text: str, sec_meta: dict) -> list[Finding]:
    """
    sec_meta: {form, filing_date, accession, item, source_doc} for citations.
    Returns factual findings for whatever material language is present.
    """
    findings: list[Finding] = []

    for s in _find_sentences(section_text, _CONCENTRATION_RE, limit=1):
        pct = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%", s)
        neg = re.search(r"(?i)no\s+(single|one)\s+(customer|client)", s)
        if neg:
            head = "No single customer exceeds the disclosed revenue-concentration threshold"
        else:
            head = (f"Customer concentration disclosed"
                    + (f": {pct.group(1)}% of revenue tied to a major customer" if pct else ""))
        findings.append(Finding("concentration", head, s, [_src(sec_meta, s)]))

    for s in _find_sentences(section_text, _GOING_CONCERN_RE, limit=1):
        findings.append(Finding("going-concern",
                                "Going-concern language present in the filing", s,
                                [_src(sec_meta, s)]))

    for s in _find_sentences(section_text, _MATERIAL_WEAKNESS_RE, limit=1):
        # Only flag affirmative disclosure, not boilerplate "no material weakness".
        if re.search(r"(?i)no\s+material\s+weakness", s):
            continue
        findings.append(Finding("material-weakness",
                                "Material-weakness language in internal controls disclosed", s,
                                [_src(sec_meta, s)]))

    for s in _find_sentences(section_text, _COVENANT_RE, limit=1):
        findings.append(Finding("covenant",
                                "Debt-covenant / liquidity language disclosed (verbatim)", s,
                                [_src(sec_meta, s)]))

    for s in _find_sentences(section_text, _LITIGATION_RE, limit=1):
        findings.append(Finding("litigation",
                                "Material legal proceedings / litigation disclosed", s,
                                [_src(sec_meta, s)]))

    return findings


def findings_from_diff(diff: dict, cur_meta: dict, prior_meta: dict) -> list[Finding]:
    """Turn the YoY risk-factor diff into headline findings, citing both years."""
    if not diff.get("available"):
        return []
    out: list[Finding] = []
    both = [cur_meta, prior_meta]
    c = diff["counts"]
    if c["added"]:
        out.append(Finding("risk-added",
                           f"{c['added']} risk factor(s) added vs the prior 10-K",
                           "; ".join(a["heading"] for a in diff["added"][:4]), both))
    if c["removed"]:
        out.append(Finding("risk-removed",
                           f"{c['removed']} risk factor(s) removed vs the prior 10-K",
                           "; ".join(r["heading"] for r in diff["removed"][:4]), both))
    if c["changed"]:
        out.append(Finding("risk-changed",
                           f"{c['changed']} risk factor(s) materially reworded vs the prior 10-K",
                           "; ".join(ch["heading"] for ch in diff["changed"][:4]), both))
    # Concentration surfacing from the diff: a newly ADDED concentration factor.
    for a in diff["added"]:
        if re.search(r"(?i)concentrat|single\s+customer|one\s+customer|reliance\s+on", a["heading"]):
            out.append(Finding("risk-added",
                               "Customer-concentration risk factor added this year",
                               a["heading"], both))
            break
    return out
