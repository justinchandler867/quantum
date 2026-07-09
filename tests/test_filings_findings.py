"""
Offline tests for the Findings / YoY-diff / materiality engine (Phase 2, §2a).
No network — synthetic inputs.
"""
from app.filings_findings import (
    RiskFactor, split_risk_factors, diff_risk_factors, extract_materiality,
    findings_from_diff, MIN_RF_FOR_DIFF,
)

META = {"form": "10-K", "filing_date": "2025-01-01", "accession": "a", "item": "1A", "source_doc": "x.htm"}


def _rf(head, body):
    return RiskFactor(heading=head, body=body, is_category=False)


def _common(n=5):
    return [_rf(f"The Company faces risk number {i} affecting operations and financial results.",
               f"Detailed body for risk {i}. " * 12) for i in range(n)]


# ── splitting ────────────────────────────────────────────────────────────────

def test_split_risk_factors_by_bold_headings():
    raw = ('<div><span style="font-weight:700">Macroeconomic Risks</span>'
           '<span style="font-weight:700">The Company depends on a single supplier for key '
           'components which could disrupt production and harm results.</span>'
           '<p>Body about supplier concentration and its consequences for operations.</p>'
           '<b>Intense competition in the industry could reduce the Company margins over time.</b>'
           '<p>Body about competitive pressure.</p></div>')
    rfs = split_risk_factors(raw)
    factors = [r for r in rfs if not r.is_category]
    cats = [r for r in rfs if r.is_category]
    assert len(factors) >= 2
    assert any("single supplier" in f.heading for f in factors)
    assert any("Macroeconomic" in c.heading for c in cats)


# ── diff ─────────────────────────────────────────────────────────────────────

def test_diff_detects_added_and_removed():
    cur = _common() + [_rf("A new cybersecurity breach risk emerged this year exposing customer data.", "new")]
    prior = _common() + [_rf("An old discontinued product line risk no longer applicable to the firm.", "old")]
    diff = diff_risk_factors(cur, prior)
    assert diff["available"] is True
    assert diff["counts"]["added"] >= 1
    assert diff["counts"]["removed"] >= 1
    assert any("cybersecurity" in a["heading"] for a in diff["added"])


def test_diff_detects_material_rewording():
    prior = _common()
    cur = _common()
    cur[0] = _rf(prior[0].heading, "COMPLETELY different body language " * 20)
    diff = diff_risk_factors(cur, prior)
    assert diff["available"] is True
    assert diff["counts"]["changed"] >= 1


def test_identical_filings_yield_no_changes():
    """One-to-one matching: identical risk sets => 0 added / 0 removed / 0 changed
    (a plain argmax would double-assign and inflate 'removed')."""
    same = [_rf(f"Stable risk {i} unchanged year over year in business operations.",
                "identical body " * 10) for i in range(6)]
    diff = diff_risk_factors(same, same)
    assert diff["counts"]["added"] == 0
    assert diff["counts"]["removed"] == 0
    assert diff["counts"]["changed"] == 0
    assert findings_from_diff(diff, {}, {}) == []


def test_diff_withheld_when_too_few_factors():
    cur = _common(2)
    prior = _common(2)
    diff = diff_risk_factors(cur, prior)
    assert diff["available"] is False
    assert "unavailable" in diff["reason"].lower()


def test_findings_from_diff_cites_both_years():
    cur = _common() + [_rf("A new supply concentration risk on one customer emerged this year now.", "n")]
    prior = _common()
    diff = diff_risk_factors(cur, prior)
    cur_meta = dict(META, filing_date="2025-01-01")
    prior_meta = dict(META, filing_date="2024-01-01")
    fs = findings_from_diff(diff, cur_meta, prior_meta)
    assert fs
    added = [f for f in fs if f.kind == "risk-added"][0]
    assert len(added.sources) == 2  # cites current AND prior


# ── materiality (pure extraction, honest null) ───────────────────────────────

def test_concentration_extracted_and_quantified():
    txt = "One customer accounted for 22% of our net revenue during fiscal 2025."
    fs = extract_materiality(txt, META)
    conc = [f for f in fs if f.kind == "concentration"]
    assert conc and "22%" in conc[0].headline


def test_going_concern_flagged():
    txt = "There is substantial doubt about our ability to continue as a going concern."
    assert any(f.kind == "going-concern" for f in extract_materiality(txt, META))


def test_material_weakness_flagged_but_not_negation():
    pos = "We identified a material weakness in our internal control over financial reporting."
    neg = "Management concluded there was no material weakness in internal control."
    assert any(f.kind == "material-weakness" for f in extract_materiality(pos, META))
    assert not any(f.kind == "material-weakness" for f in extract_materiality(neg, META))


def test_litigation_flagged():
    txt = "We are subject to legal proceedings and recorded a reserve of $50 million for settlement."
    assert any(f.kind == "litigation" for f in extract_materiality(txt, META))


def test_honest_null_no_findings_on_benign_text():
    txt = "The Company sells products in many countries and invests in research and development."
    assert extract_materiality(txt, META) == []
