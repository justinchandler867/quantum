"""
SEC EDGAR client (FILINGS_SPEC Phase 1 — the data spine).

Free, official, primary source. This module:
  - declares a User-Agent with contact info per SEC policy,
  - throttles to stay under the ~10 req/s guidance,
  - caches every fetched document on disk so nothing is re-fetched
    (bandwidth-constrained by design — cache-first, explicit refresh only).

Retrieval path: ticker -> CIK (company_tickers.json) -> submissions JSON ->
latest 10-K / 10-Q accession numbers -> filing documents.

Nothing is fetched at import time; everything is lazy per-ticker.
"""
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import httpx

from app.config import (
    SEC_USER_AGENT,
    SEC_MIN_REQUEST_INTERVAL,
    FILINGS_CACHE_DIR,
)

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# Module-level throttle shared across all callers/threads.
_rate_lock = threading.Lock()
_last_request_ts = [0.0]


@dataclass
class FilingRef:
    """A pointer to one filing's primary document."""
    ticker: str
    cik: int
    form: str          # "10-K", "10-Q"
    accession: str     # dashed, e.g. 0000320193-25-000079
    filing_date: str   # YYYY-MM-DD
    primary_doc: str   # e.g. aapl-20250927.htm
    report_date: str = ""  # period of report (YYYY-MM-DD)

    @property
    def acc_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def doc_url(self) -> str:
        return ARCHIVES_URL.format(cik=self.cik, acc_nodash=self.acc_nodash, doc=self.primary_doc)


# ── HTTP with throttle + retry ───────────────────────────────────────────────

def _throttle() -> None:
    """Block until at least SEC_MIN_REQUEST_INTERVAL has passed since the last request."""
    with _rate_lock:
        now = time.monotonic()
        wait = SEC_MIN_REQUEST_INTERVAL - (now - _last_request_ts[0])
        if wait > 0:
            time.sleep(wait)
        _last_request_ts[0] = time.monotonic()


def _headers(host: str) -> dict:
    # Explicit Host + Accept matter: a bare request to www.sec.gov/Archives is
    # dropped by the server ("disconnected without response"); these fix it.
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


def _http_get(url: str, host: str, retries: int = 3) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            with httpx.Client(timeout=45.0, follow_redirects=True) as c:
                r = c.get(url, headers=_headers(host))
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                raise FileNotFoundError(f"SEC 404: {url}")
            last_exc = ValueError(f"SEC {r.status_code}: {url}")
            logger.warning(f"SEC GET {r.status_code} (attempt {attempt + 1}/{retries}) {url}")
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001 — network layer, retry all
            last_exc = e
            logger.warning(f"SEC GET failed (attempt {attempt + 1}/{retries}) {url}: {e}")
        time.sleep(1.0 * (attempt + 1))
    raise last_exc or RuntimeError(f"SEC GET failed: {url}")


# ── Disk cache (cache-first, never re-fetch unless refresh=True) ──────────────

def _cache_path(*parts: str) -> str:
    p = os.path.join(FILINGS_CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _cached_text(url: str, host: str, cache_parts: tuple, refresh: bool = False) -> str:
    path = _cache_path(*cache_parts)
    if not refresh and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    text = _http_get(url, host)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"cached SEC doc {url} -> {path} ({len(text):,} chars)")
    return text


# ── Public API ───────────────────────────────────────────────────────────────

def resolve_cik(ticker: str, refresh: bool = False) -> int | None:
    """ticker -> CIK via company_tickers.json (cached)."""
    ticker = ticker.upper().strip()
    raw = _cached_text(SEC_TICKERS_URL, "www.sec.gov",
                       ("meta", "company_tickers.json"), refresh=refresh)
    data = json.loads(raw)
    for row in data.values():
        if str(row.get("ticker", "")).upper() == ticker:
            return int(row["cik_str"])
    return None


def get_submissions(cik: int, refresh: bool = False) -> dict:
    """CIK -> submissions JSON (cached)."""
    cik10 = f"{cik:010d}"
    raw = _cached_text(SUBMISSIONS_URL.format(cik10=cik10), "data.sec.gov",
                       ("meta", f"submissions_{cik10}.json"), refresh=refresh)
    return json.loads(raw)


def latest_filings(
    ticker: str,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    per_form: int = 2,
    refresh: bool = False,
) -> list[FilingRef]:
    """
    Most recent `per_form` filings of each requested form, newest first.
    Returns [] if the ticker has no CIK (e.g. a 20-F-only foreign filer).
    """
    cik = resolve_cik(ticker, refresh=refresh)
    if cik is None:
        return []
    sub = get_submissions(cik, refresh=refresh)
    recent = sub.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    acc = recent.get("accessionNumber", [])
    prim = recent.get("primaryDocument", [])
    fdate = recent.get("filingDate", [])
    rdate = recent.get("reportDate", [])

    out: list[FilingRef] = []
    counts = {f: 0 for f in forms}
    for i, f in enumerate(form_list):
        if f in counts and counts[f] < per_form:
            out.append(FilingRef(
                ticker=ticker.upper(), cik=cik, form=f,
                accession=acc[i], filing_date=fdate[i], primary_doc=prim[i],
                report_date=rdate[i] if i < len(rdate) else "",
            ))
            counts[f] += 1
    return out


def fetch_document(ref: FilingRef, refresh: bool = False) -> str:
    """Fetch a filing's primary document HTML (cached on disk, never re-fetched)."""
    return _cached_text(
        ref.doc_url, "www.sec.gov",
        ("raw", str(ref.cik), ref.acc_nodash, ref.primary_doc),
        refresh=refresh,
    )


# ── Exhibit discovery (Phase 2 — incorporation-by-reference chase) ────────────

def _index_htm_url(ref: FilingRef) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{ref.cik}/"
            f"{ref.acc_nodash}/{ref.accession}-index.htm")


def find_exhibit(ref: FilingRef, types: tuple[str, ...] = ("EX-13",),
                 refresh: bool = False) -> tuple[str, str] | None:
    """
    Locate an exhibit document within the SAME accession by SEC document type,
    using the authoritative `-index.htm` type table (index.json's `type` field
    is unreliable — it carries icon names, not EX-13). Returns (doc_name, type)
    or None. Used to chase Item 1A/7 content that a 10-K incorporates by
    reference (banks/insurers → Annual Report / Exhibit 13). Verified on USB.
    """
    raw = _cached_text(_index_htm_url(ref), "www.sec.gov",
                       ("meta", "acc_index", f"{ref.acc_nodash}.htm"), refresh=refresh)
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", raw):
        cells = [re.sub(r"(?s)<[^>]+>", " ", c).strip()
                 for c in re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)]
        joined = " ".join(cells)
        if not any(t.upper() in joined.upper() for t in types):
            continue
        link = re.search(r'href="[^"]*?/([^"/]+\.htm[l]?)"', row, re.I)
        if link:
            matched_type = next((t for t in types if t.upper() in joined.upper()), types[0])
            return link.group(1), matched_type
    return None


def fetch_named_document(ref: FilingRef, doc_name: str, refresh: bool = False) -> str:
    """Fetch an arbitrary document by name from a filing's accession (cached)."""
    url = ARCHIVES_URL.format(cik=ref.cik, acc_nodash=ref.acc_nodash, doc=doc_name)
    return _cached_text(url, "www.sec.gov",
                        ("raw", str(ref.cik), ref.acc_nodash, doc_name), refresh=refresh)


def fetch_companyfacts(cik: int, refresh: bool = False) -> dict:
    """EDGAR XBRL companyfacts (structured financial facts), cached."""
    cik10 = f"{cik:010d}"
    raw = _cached_text(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json",
        "data.sec.gov", ("meta", f"companyfacts_{cik10}.json"), refresh=refresh)
    return json.loads(raw)
