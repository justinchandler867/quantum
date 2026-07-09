"""
Offline tests for filings_facts + filings_service seams (Phase 2). No network.
"""
import app.filings_facts as ff
import app.filings_service as svc


def test_facts_pick_latest_concept_not_stale(monkeypatch):
    """Filers switch revenue concepts; the latest period must win, not the first concept."""
    fake = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2018-09-29", "val": 100, "form": "10-K", "fp": "FY", "fy": 2018, "accn": "old"}]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"end": "2025-09-27", "val": 400, "form": "10-K", "fp": "FY", "fy": 2025, "accn": "new"}]}},
    }}}
    monkeypatch.setattr(ff, "fetch_companyfacts", lambda *a, **k: fake)
    facts = ff.get_financial_facts(1)
    rev = [f for f in facts if f["label"] == "Revenue"]
    assert rev and rev[0]["value"] == 400 and rev[0]["fiscal_year"] == 2025


def test_facts_empty_when_companyfacts_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("404")
    monkeypatch.setattr(ff, "fetch_companyfacts", boom)
    assert ff.get_financial_facts(1) == []


def test_unsupported_foreign_when_cik_but_no_10k(monkeypatch):
    monkeypatch.setattr(svc, "latest_filings", lambda *a, **k: [])
    monkeypatch.setattr(svc, "resolve_cik", lambda *a, **k: 12345)
    v = svc.build_filings_view("TSM")
    assert v["status"] == "unsupported_foreign"
    assert "20-F" in v["message"]


def test_no_data_when_ticker_unknown(monkeypatch):
    monkeypatch.setattr(svc, "latest_filings", lambda *a, **k: [])
    monkeypatch.setattr(svc, "resolve_cik", lambda *a, **k: None)
    v = svc.build_filings_view("ZZZZ")
    assert v["status"] == "no_data"


def test_citation_registry_assigns_and_resolves():
    reg = svc.CitationRegistry()
    cid = reg.add({"form": "10-K", "item": "1A", "source_doc": "x.htm"}, "a quote here")
    assert cid in reg.map
    assert reg.map[cid]["item"] == "1A"
    assert reg.map[cid]["quote"] == "a quote here"


def test_looks_like_toc():
    toc = "Management’s Discussion and Analysis\n22\nOverview\n24\nStatement of Income\n26\nBalance\n28\n"
    body = ("Management’s Discussion and Analysis. The following discussion should be read in "
            "conjunction with the consolidated financial statements and related notes. " * 5)
    assert svc._looks_like_toc(toc) is True
    assert svc._looks_like_toc(body) is False


def test_tab_header_is_sealed_copy():
    assert "does not evaluate or recommend" in svc.TAB_HEADER
    assert "links to the source passage" in svc.TAB_HEADER
