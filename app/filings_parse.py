"""
Filings section-boundary parser (FILINGS_SPEC Phase 1).

Boundary detection is grounded in a real 10-K (AAPL FY2025, iXBRL/Workiva),
not an assumed format:

  Primary  — TOC anchor links `<a href="#id">Item N.</a>` map each item to a
             unique in-body `id="..."` target, in document order. Section N spans
             [offset(anchor_N), offset(anchor_next)). This is immune to prose
             cross-references: AAPL's "Item 1A" string appears 5x in the text
             (TOC + header + 3 "see Item 1A" mentions), but its anchor is unique.
  Fallback — a heading-context text scan for filers with no usable TOC anchors:
             an item token in heading position followed by its canonical title.
  UNPARSED — a target section that cannot be confidently bounded by EITHER method
             is marked UNPARSED (surfaced, never approximated). SEALED behavior.

Parse success is tracked per filing over its form's target items.
"""
import html as _html
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Target items per form, with a canonical title keyword used to validate a
# candidate boundary (and to disambiguate repeated item numbers across the
# Part I / Part II split in a 10-Q).
#   10-K: Item 1 Business, Item 1A Risk Factors, Item 7 MD&A
#   10-Q: Part I Item 2 MD&A, Part II Item 1A Risk Factors
TARGET_ITEMS = {
    "10-K": {
        "1":  ("Business", ("business",)),
        "1A": ("Risk Factors", ("risk factor",)),
        "7":  ("Management's Discussion and Analysis", ("management", "discussion")),
    },
    "10-Q": {
        "2":  ("Management's Discussion and Analysis", ("management", "discussion")),
        "1A": ("Risk Factors", ("risk factor",)),
    },
}

STATUS_PARSED = "PARSED"
STATUS_UNPARSED = "UNPARSED"


@dataclass
class ParsedSection:
    item: str
    title: str          # canonical title of the target
    status: str         # PARSED | UNPARSED
    method: str         # anchor | heading | none
    text: str = ""
    char_len: int = 0
    detected_header: str = ""   # the header text as it appeared (evidence)


@dataclass
class ParsedFiling:
    ticker: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    sections: dict = field(default_factory=dict)  # item -> ParsedSection

    @property
    def parse_success_rate(self) -> float:
        if not self.sections:
            return 0.0
        parsed = sum(1 for s in self.sections.values() if s.status == STATUS_PARSED)
        return parsed / len(self.sections)

    @property
    def unparsed_items(self) -> list[str]:
        return [i for i, s in self.sections.items() if s.status == STATUS_UNPARSED]


# ── HTML → text ──────────────────────────────────────────────────────────────

def html_to_text(fragment: str) -> str:
    """Strip tags to readable text, preserving paragraph breaks."""
    t = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", fragment)
    t = re.sub(r"(?is)<br\s*/?>", "\n", t)
    t = re.sub(r"(?is)</(p|div|tr|h[1-6]|li|table)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t)
    t = t.replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


_TOKEN_RE = re.compile(r"(?is)<script.*?</script>|<style.*?</style>|<[^>]+>|&[#0-9A-Za-z]+;|.")


def html_to_text_with_map(fragment: str) -> tuple[str, list[int]]:
    """
    Like html_to_text but also returns raw_off: raw_off[k] is the index into
    `fragment` that produced text[k]. Lets a text-space boundary (found by the
    proven detection logic) map precisely back to raw HTML — needed to slice a
    section's raw markup for risk-factor splitting on ANY filer, not just those
    with TOC anchors.
    """
    out: list[str] = []
    src: list[int] = []
    for m in _TOKEN_RE.finditer(fragment):
        tok = m.group(0)
        pos = m.start()
        low = tok.lower()
        if tok.startswith("<"):
            if low.startswith("<script") or low.startswith("<style"):
                ch = " "
            elif re.match(r"<br\s*/?>", low) or re.match(r"</(p|div|tr|h[1-6]|li|table)>", low):
                ch = "\n"
            else:
                ch = " "
            out.append(ch); src.append(pos)
        elif tok.startswith("&"):
            u = _html.unescape(tok)
            u = " " if u == "\xa0" else u
            for c in u:
                out.append(c); src.append(pos)
        else:
            out.append(" " if tok == "\xa0" else tok); src.append(pos)

    # Collapse whitespace, keeping the source index of the first char of each run.
    text_chars: list[str] = []
    off: list[int] = []
    i, n = 0, len(out)
    while i < n:
        c = out[i]
        if c in " \t":
            j = i
            while j < n and out[j] in " \t":
                j += 1
            text_chars.append(" "); off.append(src[i])
            i = j
        elif c == "\n":
            j = i
            while j < n and out[j] in " \t\n":
                j += 1
            text_chars.append("\n"); off.append(src[i])
            i = j
        else:
            text_chars.append(c); off.append(src[i])
            i += 1
    text = "".join(text_chars)
    start = 0
    while start < len(text) and text[start] in " \n":
        start += 1
    end = len(text)
    while end > start and text[end - 1] in " \n":
        end -= 1
    return text[start:end], off[start:end]


# ── Anchor-based detection (primary) ─────────────────────────────────────────

_TOC_LINK_RE = re.compile(
    r'href="#([^"]+)"[^>]*>\s*(?:<[^>]+>\s*)*Item\s+(\d+[A-Za-z]?)\b',
    re.IGNORECASE,
)


def _anchor_sections(doc: str) -> list[dict]:
    """
    Ordered list of {item, anchor, offset} for every TOC item-link whose anchor
    resolves to a unique in-body id target. Sorted by body offset.
    """
    sections: list[dict] = []
    seen_anchor: set[str] = set()
    for m in _TOC_LINK_RE.finditer(doc):
        anchor, item = m.group(1), m.group(2).upper()
        if anchor in seen_anchor:
            continue
        target = re.search(r'id="%s"' % re.escape(anchor), doc)
        if not target:
            continue
        seen_anchor.add(anchor)
        # The id= match lands mid-tag; advance past the tag's closing '>' so the
        # extracted section text starts on real content, not `id="anchor">` junk.
        gt = doc.find(">", target.end())
        offset = gt + 1 if gt != -1 else target.start()
        sections.append({"item": item, "anchor": anchor, "offset": offset})
    sections.sort(key=lambda s: s["offset"])
    return sections


def _extract_between(doc: str, start: int, end: int) -> str:
    return html_to_text(doc[start:end])


def _anchor_candidates(doc: str, targets: dict) -> dict:
    """Best raw candidate {item: {text, header, method}} via TOC anchors."""
    ordered = _anchor_sections(doc)
    out: dict = {}
    for idx, sec in enumerate(ordered):
        item = sec["item"]
        if item not in targets or item in out:
            continue
        start = sec["offset"]
        end = ordered[idx + 1]["offset"] if idx + 1 < len(ordered) else len(doc)
        text = _extract_between(doc, start, end)
        _title, keywords = targets[item]
        header = text[:120].replace("\n", " ").strip()
        # Validate the anchor actually lands on this item's header (not elsewhere)
        # and carries the canonical title keyword — guards a mis-placed anchor.
        starts_with_item = bool(re.match(r"(?i)\s*Item\s+%s\b" % re.escape(item), text))
        has_keyword = all(kw in header.lower() for kw in keywords) or \
            any(kw in text[:400].lower() for kw in keywords)
        if starts_with_item and has_keyword:
            out[item] = {"text": text, "header": header, "method": "anchor"}
    return out


# ── Heading-context fallback ─────────────────────────────────────────────────

def _heading_candidates(doc: str, targets: dict, already: dict) -> dict:
    """
    Fallback for filings without usable TOC anchors. Among all heading-position
    matches for a target item, pick the one whose bounded body is LONGEST — the
    real section body always outweighs a short table-of-contents line pointing
    at it (see DKS: TOC "Item 1A. Risk Factors 15" vs body "ITEM 1A. RISK ...").
    """
    text = html_to_text(doc)
    heads = []
    for m in re.finditer(r"(?im)^\s*Item\s+(\d+[A-Za-z]?)[\.\:\)\s—-]+([A-Z][^\n]{0,80})", text):
        heads.append({"item": m.group(1).upper(), "start": m.start(),
                      "header": (m.group(0)[:120]).strip()})
    heads.sort(key=lambda h: h["start"])
    starts = [h["start"] for h in heads]

    out: dict = {}
    for idx, h in enumerate(heads):
        item = h["item"]
        if item not in targets or item in already:
            continue
        _title, keywords = targets[item]
        if not any(kw in h["header"].lower() for kw in keywords):
            continue  # a bare "Item N" cross-reference, not a real heading
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        body = text[h["start"]:end].strip()
        # Keep the longest bounded body seen for this item (body beats TOC line).
        if item not in out or len(body) > len(out[item]["text"]):
            out[item] = {"text": body, "header": h["header"], "method": "heading"}
    return out


# ── Usability classification (honest degradation) ────────────────────────────

# A section whose entire content just points elsewhere ("incorporated by
# reference") is NOT the substantive section — surfacing it as PARSED would be a
# false parse. Common for banks/insurers that carry Risk Factors / MD&A in the
# Annual Report (Exhibit 13). Confirmed on USB FY2025.
_IBR_RE = re.compile(
    r"(?i)(incorporated\s+(herein\s+)?by\s+reference"
    r"|information\s+(required\s+by|in\s+response\s+to)\s+this\s+item[^.]{0,80}"
    r"(can\s+be\s+found|is\s+(incorporated|included|set\s+forth|contained)|appears))"
)
MIN_SECTION_CHARS = 400   # below this, a Business/Risk/MD&A section is implausible
IBR_MAX_CHARS = 1500      # an IBR pointer is short; long text that merely mentions
                          # "by reference" is still a real section


def _classify(text: str) -> tuple[str, str]:
    """(status, reason). reason doubles as the UNPARSED method for the inventory."""
    if _IBR_RE.search(text[:800]) and len(text) < IBR_MAX_CHARS:
        return STATUS_UNPARSED, "incorporated-by-ref"
    if len(text) < MIN_SECTION_CHARS:
        return STATUS_UNPARSED, "too-short"
    return STATUS_PARSED, "ok"


# ── Public entry point ───────────────────────────────────────────────────────

def parse_filing(doc: str, form: str, ticker: str = "", cik: int = 0,
                 accession: str = "", filing_date: str = "", report_date: str = "") -> ParsedFiling:
    """
    Parse the target sections for `form` out of raw filing HTML.
    Every target item ends up either PARSED (with text) or UNPARSED — never
    silently dropped, never approximated. UNPARSED carries the reason in `method`
    (none | incorporated-by-ref | too-short).
    """
    targets = TARGET_ITEMS.get(form, {})
    pf = ParsedFiling(ticker=ticker, cik=cik, form=form, accession=accession,
                      filing_date=filing_date, report_date=report_date)

    cands = _anchor_candidates(doc, targets)
    remaining = {k: v for k, v in targets.items() if k not in cands}
    if remaining:
        cands.update(_heading_candidates(doc, remaining, cands))

    for item, (title, _kw) in targets.items():
        cand = cands.get(item)
        if not cand:
            pf.sections[item] = ParsedSection(
                item=item, title=title, status=STATUS_UNPARSED, method="none")
            continue
        status, reason = _classify(cand["text"])
        if status == STATUS_PARSED:
            pf.sections[item] = ParsedSection(
                item=item, title=title, status=STATUS_PARSED, method=cand["method"],
                text=cand["text"], char_len=len(cand["text"]),
                detected_header=cand["header"])
        else:
            # Bounded, but not the substantive section — surfaced, not approximated.
            pf.sections[item] = ParsedSection(
                item=item, title=title, status=STATUS_UNPARSED, method=reason,
                detected_header=cand["header"])

    logger.info(
        f"parsed {ticker} {form} {accession}: "
        f"{[f'{i}:{s.status}({s.method})' for i, s in pf.sections.items()]}"
    )
    return pf


# Common annual-report (Exhibit 13) section titles used to bound a target
# section when a 10-K incorporates Item 1A / Item 7 by reference.
AR_SECTION_TITLES = [
    ("management", "discussion"),
    ("quantitative", "qualitative"),
    ("risk", "factor"),
    ("controls", "procedures"),
    ("financial", "statements"),
    ("report", "independent"),
    ("selected", "financial"),
    ("consolidated", "balance"),
]


def locate_title_section_raw(doc: str, target: tuple, min_chars: int = 2000) -> str | None:
    """
    Locate a section in an annual-report exhibit by TITLE heading (no 'Item N'
    prefix — banks carry Risk Factors / MD&A under plain titles). Among heading
    lines matching `target`, choose the one whose body (bounded by the next
    different AR-section title) is longest — the body beats the TOC copy.
    """
    text, off = html_to_text_with_map(doc)
    xref = re.compile(r"(?i)refer to|\bsee\b|page\s+\d|incorporat|beginning on|pursuant to")
    title_pos = []  # (pos, keyset)
    for m in re.finditer(r"(?m)^[^\n]{3,85}$", text):
        line = m.group().strip()
        low = line.lower()
        if xref.search(line):
            continue  # a cross-reference sentence, not a section heading
        for ks in AR_SECTION_TITLES:
            if all(k in low for k in ks):
                title_pos.append((m.start(), ks))
                break
    title_pos.sort()
    cand = [p for p, ks in title_pos if ks == target]
    if not cand:
        return None
    best, best_len = None, 0
    for c in cand:
        nxt = [p for p, ks in title_pos if p > c + min_chars and ks != target]
        end = min(nxt) if nxt else len(text)
        if end - c > best_len:
            best_len, best = end - c, (c, end)
    if best is None or best_len < min_chars:
        return None
    t0, t1 = best
    return doc[off[t0]: off[t1] if t1 < len(off) else len(doc)]


def locate_section_raw(doc: str, form: str, item: str) -> str | None:
    """
    Return the RAW HTML slice for a target section (for risk-factor splitting),
    or None if it can't be confidently located. Anchor spans are raw-space
    already; otherwise map the winning text-space heading boundary back to raw
    via the offset map so heading-only filers (DKS, O) work too.
    """
    targets = TARGET_ITEMS.get(form, {})
    if item not in targets:
        return None
    _title, keywords = targets[item]

    # 1) Anchor: raw offsets directly.
    ordered = _anchor_sections(doc)
    for idx, s in enumerate(ordered):
        if s["item"] == item:
            start = s["offset"]
            end = ordered[idx + 1]["offset"] if idx + 1 < len(ordered) else len(doc)
            head = html_to_text(doc[start:end])[:400].lower()
            if head.lstrip().startswith("item") and any(k in head for k in keywords):
                return doc[start:end]

    # 2) Heading: locate in text space (longest bounded body), map to raw.
    text, off = html_to_text_with_map(doc)
    heads = []
    for m in re.finditer(r"(?im)^\s*Item\s+(\d+[A-Za-z]?)[\.\:\)\s—-]+([A-Z][^\n]{0,80})", text):
        heads.append({"item": m.group(1).upper(), "start": m.start(), "header": m.group(0)})
    heads.sort(key=lambda h: h["start"])
    starts = [h["start"] for h in heads]
    best = None
    for i, h in enumerate(heads):
        if h["item"] != item or not any(k in h["header"].lower() for k in keywords):
            continue
        t0 = h["start"]
        t1 = starts[i + 1] if i + 1 < len(starts) else len(text)
        if best is None or (t1 - t0) > (best[1] - best[0]):
            best = (t0, t1)
    if best is None:
        return None
    t0, t1 = best
    r0 = off[t0]
    r1 = off[t1] if t1 < len(off) else len(doc)
    return doc[r0:r1]
