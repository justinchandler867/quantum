"""
Chunking + local retrieval index (FILINGS_SPEC Phase 1, Amendment 1).

- A chunk is a passage small enough to cite, large enough to read (~1-3 paras),
  carrying metadata {ticker, form, filing_date, accession, item, char_offsets}.
- Retrieval is abstraction-first: one `search(query, ticker, k)` contract with a
  scorer-agnostic backend. TF-IDF (scikit-learn, base deps) + cosine is the
  DEFAULT. A sentence-transformers backend is optional and dev/extras-only — it
  is never imported unless FILINGS_RETRIEVAL_BACKEND selects it AND the package
  is installed.

Index storage is on-disk JSON per ticker (chunks + metadata). The TF-IDF matrix
is cheap to rebuild in-memory at query time, so we persist chunks, not vectors —
avoiding version-fragile pickled sklearn objects.
"""
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import numpy as np

from app.config import (
    FILINGS_CACHE_DIR,
    FILINGS_RETRIEVAL_BACKEND,
    FILINGS_ST_MODEL,
    FILINGS_CHUNK_MIN_CHARS,
    FILINGS_CHUNK_MAX_CHARS,
)
from app.filings_edgar import latest_filings, fetch_document
from app.filings_parse import parse_filing, STATUS_PARSED

logger = logging.getLogger(__name__)


# ── Chunk model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    form: str
    filing_date: str
    accession: str
    item: str
    char_start: int   # offset into the section's extracted text
    char_end: int
    text: str

    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "ticker": self.ticker, "form": self.form,
            "filing_date": self.filing_date, "accession": self.accession,
            "item": self.item, "char_offsets": [self.char_start, self.char_end],
        }


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """(start_offset, paragraph_text) for each non-empty paragraph."""
    return [(m.start(), m.group().strip())
            for m in re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, re.S)]


def chunk_section(
    section_text: str, *, ticker: str, form: str, filing_date: str,
    accession: str, item: str,
    min_chars: int = FILINGS_CHUNK_MIN_CHARS,
    max_chars: int = FILINGS_CHUNK_MAX_CHARS,
) -> list[Chunk]:
    """
    Greedily pack paragraphs into ~1-3 paragraph chunks in [min, max] chars.
    A single oversized paragraph is hard-split on whitespace. char_start/end are
    true offsets into `section_text`.
    """
    paras = _split_paragraphs(section_text)
    chunks: list[Chunk] = []

    def emit(start: int, end: int) -> None:
        body = section_text[start:end].strip()
        if not body:
            return
        cid = f"{ticker}:{accession}:{item}:{len(chunks)}"
        chunks.append(Chunk(cid, ticker, form, filing_date, accession, item,
                            start, end, body))

    buf_start: int | None = None
    buf_end = 0
    buf_len = 0
    for pstart, ptext in paras:
        # Oversized single paragraph → hard-split.
        if len(ptext) > max_chars:
            if buf_start is not None:
                emit(buf_start, buf_end)
                buf_start, buf_len = None, 0
            pos = 0
            while pos < len(ptext):
                cut = min(pos + max_chars, len(ptext))
                if cut < len(ptext):
                    sp = ptext.rfind(" ", pos + min_chars, cut)
                    if sp != -1:
                        cut = sp
                emit(pstart + pos, pstart + cut)
                pos = cut
            continue
        if buf_start is None:
            buf_start, buf_end, buf_len = pstart, pstart + len(ptext), len(ptext)
        elif buf_len + len(ptext) + 2 <= max_chars:
            buf_end = pstart + len(ptext)
            buf_len += len(ptext) + 2
        else:
            emit(buf_start, buf_end)
            buf_start, buf_end, buf_len = pstart, pstart + len(ptext), len(ptext)
    if buf_start is not None:
        emit(buf_start, buf_end)
    return chunks


# ── Scorer-agnostic retrieval interface ──────────────────────────────────────

class RetrievalScorer(ABC):
    """Score chunk texts against a query. The one seam behind search()."""
    name = "abstract"

    @abstractmethod
    def rank(self, query: str, texts: list[str], k: int) -> list[tuple[int, float]]:
        """Return [(index_into_texts, score)] for the top-k, score descending."""
        ...


class TfidfScorer(RetrievalScorer):
    """Default backend: TF-IDF vectors + cosine similarity. Base deps only."""
    name = "tfidf"

    def rank(self, query: str, texts: list[str], k: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vec.fit_transform(texts + [query])
        doc_m, q_m = matrix[:-1], matrix[-1]
        sims = cosine_similarity(q_m, doc_m).ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


class SentenceTransformerScorer(RetrievalScorer):
    """
    Optional backend (Amendment 1): dev/extras-only. Never imported unless
    selected by config. Import is deferred so base deploy never needs torch.
    """
    name = "sentence-transformers"

    def __init__(self, model_name: str = FILINGS_ST_MODEL):
        from sentence_transformers import SentenceTransformer  # optional dep
        self._model = SentenceTransformer(model_name)

    def rank(self, query: str, texts: list[str], k: int) -> list[tuple[int, float]]:
        if not texts:
            return []
        from sentence_transformers import util  # optional dep
        emb = self._model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        q = self._model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q, emb).cpu().numpy().ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def get_scorer(backend: str | None = None) -> RetrievalScorer:
    backend = backend or FILINGS_RETRIEVAL_BACKEND
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerScorer()
        except Exception as e:  # noqa: BLE001 — optional dep absent/broken
            logger.warning(f"sentence-transformers backend unavailable ({e}); "
                           f"falling back to TF-IDF")
    return TfidfScorer()


# ── Per-ticker index ─────────────────────────────────────────────────────────

def _index_path(ticker: str) -> str:
    p = os.path.join(FILINGS_CACHE_DIR, "index", f"{ticker.upper()}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


@dataclass
class FilingsIndex:
    ticker: str
    chunks: list[Chunk]
    inventory: list[dict]   # per-filing section status (for the Findings/UI layer)

    def save(self) -> str:
        path = _index_path(self.ticker)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": self.ticker,
                "chunks": [asdict(c) for c in self.chunks],
                "inventory": self.inventory,
            }, f)
        return path

    @classmethod
    def load(cls, ticker: str) -> "FilingsIndex | None":
        path = _index_path(ticker)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(ticker=d["ticker"],
                   chunks=[Chunk(**c) for c in d["chunks"]],
                   inventory=d.get("inventory", []))

    def search(self, query: str, k: int = 3,
               scorer: RetrievalScorer | None = None) -> list[dict]:
        scorer = scorer or get_scorer()
        texts = [c.text for c in self.chunks]
        ranked = scorer.rank(query, texts, k)
        results = []
        for idx, score in ranked:
            c = self.chunks[idx]
            results.append({"score": score, "text": c.text, **c.metadata()})
        return results


# ── Orchestration: lazy per-ticker ingest ────────────────────────────────────

def ingest_ticker(ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q"),
                  per_form: int = 2, refresh: bool = False) -> FilingsIndex:
    """
    Full lazy pipeline for one ticker: fetch latest filings (cached), parse
    target sections, chunk PARSED sections, build + persist the index.
    Returns the index (chunks may be empty if nothing parsed, e.g. a 20-F filer).
    """
    ticker = ticker.upper()
    refs = latest_filings(ticker, forms=forms, per_form=per_form, refresh=refresh)
    chunks: list[Chunk] = []
    inventory: list[dict] = []

    if not refs:
        logger.info(f"{ticker}: no 10-K/10-Q filings (likely a 20-F foreign filer)")

    for ref in refs:
        doc = fetch_document(ref, refresh=refresh)
        pf = parse_filing(doc, ref.form, ticker=ticker, cik=ref.cik,
                          accession=ref.accession, filing_date=ref.filing_date,
                          report_date=ref.report_date)
        sec_status = {}
        for item, sec in pf.sections.items():
            sec_status[item] = {
                "title": sec.title, "status": sec.status, "method": sec.method,
                "char_len": sec.char_len,
            }
            if sec.status == STATUS_PARSED:
                sec_chunks = chunk_section(
                    sec.text, ticker=ticker, form=ref.form,
                    filing_date=ref.filing_date, accession=ref.accession, item=item,
                )
                chunks.extend(sec_chunks)
        inventory.append({
            "form": ref.form, "accession": ref.accession,
            "filing_date": ref.filing_date, "report_date": ref.report_date,
            "primary_doc": ref.primary_doc,
            "parse_success_rate": round(pf.parse_success_rate, 3),
            "sections": sec_status,
        })

    idx = FilingsIndex(ticker=ticker, chunks=chunks, inventory=inventory)
    idx.save()
    logger.info(f"{ticker}: indexed {len(chunks)} chunks across {len(refs)} filings")
    return idx


def search(query: str, ticker: str, k: int = 3,
           scorer: RetrievalScorer | None = None) -> list[dict]:
    """
    The sealed retrieval contract. Loads the ticker's index (ingesting lazily if
    absent) and returns top-k chunks with metadata. Scorer is swappable per
    Amendment 1; default is TF-IDF.
    """
    idx = FilingsIndex.load(ticker)
    if idx is None:
        idx = ingest_ticker(ticker)
    return idx.search(query, k=k, scorer=scorer)
