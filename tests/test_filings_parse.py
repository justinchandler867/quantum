"""
Offline, deterministic tests for the filings parser (FILINGS_SPEC Phase 1).
No network: synthetic HTML fixtures modeled on the real AAPL 10-K structure
(TOC anchor links -> in-body id targets), plus degradation cases.
"""
from app.filings_parse import parse_filing, STATUS_PARSED, STATUS_UNPARSED

_LONG = ("This section contains enough prose to clear the minimum-length "
         "validation gate so the parser treats it as a real bounded section "
         "rather than a stray cross-reference token appearing inline. ") * 3


def _anchored_10k(include_item7=True):
    """A 10-K with a TOC of anchor links and matching in-body id targets."""
    toc = (
        '<table>'
        '<tr><td><a href="#b1">Item 1.</a></td><td>Business</td></tr>'
        '<tr><td><a href="#b1a">Item 1A.</a></td><td>Risk Factors</td></tr>'
    )
    if include_item7:
        toc += '<tr><td><a href="#b7">Item 7.</a></td><td>MD&amp;A</td></tr>'
    toc += '</table>'

    body = (
        f'<div id="b1"><b>Item 1. Business</b>'
        f'<p>The Company designs and sells widgets. {_LONG} '
        f'For risks, see Item 1A of this report. UNIQUEBIZTOKEN.</p></div>'
        f'<div id="b1a"><b>Item 1A. Risk Factors</b>'
        f'<p>The following risk factors could materially affect results. {_LONG} '
        f'UNIQUERISKTOKEN competition and supply concentration.</p></div>'
    )
    if include_item7:
        body += (
            f'<div id="b7"><b>Item 7. Management\'s Discussion and Analysis</b>'
            f'<p>Management discussion of financial condition and results. {_LONG} '
            f'UNIQUEMDATOKEN net sales increased.</p></div>'
        )
    return f'<html><body>{toc}{body}</body></html>'


def test_anchor_detection_parses_all_targets():
    pf = parse_filing(_anchored_10k(), "10-K", ticker="TST")
    assert pf.parse_success_rate == 1.0
    for item in ("1", "1A", "7"):
        s = pf.sections[item]
        assert s.status == STATUS_PARSED
        assert s.method == "anchor"


def test_cross_reference_does_not_break_boundaries():
    """'see Item 1A' inside Item 1 must not leak Item 1A content into Item 1."""
    pf = parse_filing(_anchored_10k(), "10-K", ticker="TST")
    item1 = pf.sections["1"].text
    item1a = pf.sections["1A"].text
    # The cross-reference phrase stays inside Item 1...
    assert "see Item 1A" in item1
    assert "UNIQUEBIZTOKEN" in item1
    # ...but Item 1A's body does NOT bleed into Item 1.
    assert "UNIQUERISKTOKEN" not in item1
    assert "UNIQUERISKTOKEN" in item1a
    assert "UNIQUEBIZTOKEN" not in item1a


def test_missing_section_is_unparsed_not_dropped():
    pf = parse_filing(_anchored_10k(include_item7=False), "10-K", ticker="TST")
    assert pf.sections["1"].status == STATUS_PARSED
    assert pf.sections["1A"].status == STATUS_PARSED
    # Item 7 has no anchor and no heading -> UNPARSED, surfaced, not silently gone.
    assert "7" in pf.sections
    assert pf.sections["7"].status == STATUS_UNPARSED
    assert pf.sections["7"].method == "none"
    assert pf.unparsed_items == ["7"]
    assert abs(pf.parse_success_rate - 2 / 3) < 1e-6


def test_heading_fallback_when_no_toc_anchors():
    """A filer with no TOC anchor links still parses via heading-context scan."""
    html = (
        "<html><body>"
        f"<div>Item 1. Business</div><div>We operate a widget business. {_LONG}</div>"
        f"<div>Item 1A. Risk Factors</div><div>Key risk factors include competition. {_LONG}</div>"
        "</body></html>"
    )
    pf = parse_filing(html, "10-K", ticker="TST")
    assert pf.sections["1"].status == STATUS_PARSED
    assert pf.sections["1"].method == "heading"
    assert pf.sections["1A"].status == STATUS_PARSED
    assert pf.sections["1A"].method == "heading"
    # Item 7 absent entirely -> UNPARSED.
    assert pf.sections["7"].status == STATUS_UNPARSED


def test_unparsed_section_carries_no_text():
    pf = parse_filing(_anchored_10k(include_item7=False), "10-K", ticker="TST")
    assert pf.sections["7"].text == ""
    assert pf.sections["7"].char_len == 0


def test_incorporation_by_reference_is_unparsed_not_false_parsed():
    """
    A section whose whole body just points to the Annual Report (the USB FY2025
    pattern) must be UNPARSED with reason, never a false PARSED of the pointer.
    """
    html = (
        "<html><body>"
        f"<div>Item 1. Business</div><div>We operate a regional bank. {_LONG}</div>"
        "<div>Item 1A. Risk Factors</div>"
        "<div>Information in response to this Item 1A can be found in the 2025 "
        "Annual Report and is incorporated herein by reference.</div>"
        f"<div>Item 7. Management's Discussion and Analysis</div><div>Management "
        f"discussion of results and financial condition. {_LONG}</div>"
        "</body></html>"
    )
    pf = parse_filing(html, "10-K", ticker="BANK")
    assert pf.sections["1"].status == STATUS_PARSED
    assert pf.sections["7"].status == STATUS_PARSED
    s = pf.sections["1A"]
    assert s.status == STATUS_UNPARSED
    assert s.method == "incorporated-by-ref"
    assert s.text == ""                      # no false content
    assert "Risk Factors" in s.detected_header  # but the reason is surfaced


def test_short_toc_fragment_not_false_parsed():
    """A tiny bounded fragment (below the plausibility floor) is UNPARSED."""
    html = (
        "<html><body>"
        f"<div>Item 1. Business</div><div>Real business content here. {_LONG}</div>"
        "<div>Item 1A. Risk Factors 15</div>"          # TOC-style stub w/ page no.
        "<div>Item 1B. Unresolved Staff Comments 26</div>"
        "</body></html>"
    )
    pf = parse_filing(html, "10-K", ticker="TST")
    assert pf.sections["1"].status == STATUS_PARSED
    assert pf.sections["1A"].status == STATUS_UNPARSED
    assert pf.sections["1A"].method == "too-short"


def test_10q_targets_mdna_and_risk_factors():
    """10-Q target set is MD&A (Item 2) + Risk Factors (Item 1A)."""
    html = (
        "<html><body>"
        '<a href="#q2">Item 2.</a><a href="#q1a">Item 1A.</a>'
        f'<div id="q2"><b>Item 2. Management\'s Discussion and Analysis</b>'
        f'<p>Interim management discussion. {_LONG}</p></div>'
        f'<div id="q1a"><b>Item 1A. Risk Factors</b>'
        f'<p>Updated risk factors this quarter. {_LONG}</p></div>'
        "</body></html>"
    )
    pf = parse_filing(html, "10-Q", ticker="TST")
    assert set(pf.sections.keys()) == {"2", "1A"}
    assert pf.sections["2"].status == STATUS_PARSED
    assert pf.sections["1A"].status == STATUS_PARSED
