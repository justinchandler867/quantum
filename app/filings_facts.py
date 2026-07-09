"""
Financial facts from EDGAR XBRL companyfacts (FILINGS_SPEC Phase 2).

Pure passthrough of what the company reported — each fact labeled with its
period, unit, and source filing (form + accession). No ratios invented here, no
judgment. A labeled fact is a citation-bearing fact.
"""
import logging

from app.filings_edgar import fetch_companyfacts

logger = logging.getLogger(__name__)

# (concept, label) — first concept present wins per label (filers differ).
_CONCEPTS = [
    ("Revenues", "Revenue"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
    ("NetIncomeLoss", "Net income"),
    ("OperatingIncomeLoss", "Operating income"),
    ("Assets", "Total assets"),
    ("Liabilities", "Total liabilities"),
    ("StockholdersEquity", "Shareholders' equity"),
    ("CashAndCashEquivalentsAtCarryingValue", "Cash & equivalents"),
]


def _latest_annual(units: list[dict]) -> dict | None:
    """Latest annual (10-K, full-year) datapoint by period end."""
    annual = [u for u in units
              if u.get("form") == "10-K" and u.get("fp") == "FY" and u.get("end")]
    if not annual:
        annual = [u for u in units if u.get("form") == "10-K" and u.get("end")]
    if not annual:
        return None
    return max(annual, key=lambda u: u["end"])


def get_financial_facts(cik: int, refresh: bool = False) -> list[dict]:
    """
    Return labeled financial facts (latest annual) with period + source.
    Each: {label, value, unit, period_end, fiscal_year, form, accession}.
    """
    try:
        data = fetch_companyfacts(cik, refresh=refresh)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"companyfacts unavailable for CIK {cik}: {e}")
        return []
    gaap = data.get("facts", {}).get("us-gaap", {})
    # Per label, choose the datapoint with the LATEST period end across all
    # candidate concepts — filers switch concepts over time (e.g. AAPL dropped
    # us-gaap:Revenues in 2019), so first-concept-wins would return stale values.
    label_order: list[str] = []
    best: dict = {}
    for concept, label in _CONCEPTS:
        if label not in label_order:
            label_order.append(label)
        node = gaap.get(concept)
        if not node:
            continue
        pt = _latest_annual(node.get("units", {}).get("USD", []))
        if not pt:
            continue
        prev = best.get(label)
        if prev is None or pt.get("end", "") > prev["period_end"]:
            best[label] = {
                "label": label, "value": pt["val"], "unit": "USD",
                "period_end": pt.get("end", ""), "fiscal_year": pt.get("fy"),
                "form": pt.get("form", "10-K"), "accession": pt.get("accn", ""),
                "source": "EDGAR companyfacts (XBRL)",
            }
    return [best[l] for l in label_order if l in best]
