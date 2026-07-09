"""
Offline, deterministic tests for chunking + the scorer-agnostic retrieval
interface (FILINGS_SPEC Phase 1, Amendment 1). No network.
"""
from app.filings_index import (
    chunk_section, Chunk, FilingsIndex, TfidfScorer, get_scorer,
)

_SECTION = (
    "Company Background. The Company designs and manufactures widgets and gadgets "
    "for a global customer base across several operating segments.\n\n"
    "Products. The widget line generated the majority of net sales during the year, "
    "while the gadget line continued to expand in newer geographies.\n\n"
    "Competition. The markets are highly competitive and characterized by rapid "
    "technological change, aggressive pricing, and frequent new product introductions.\n\n"
    "Human Capital. The Company employs a large workforce and invests in training, "
    "compensation, and retention programs to remain competitive for talent.\n\n"
)


def _chunks():
    return chunk_section(
        _SECTION, ticker="TST", form="10-K", filing_date="2025-01-01",
        accession="0000000000-25-000001", item="1",
        min_chars=80, max_chars=260,
    )


def test_chunk_metadata_complete_and_offsets_valid():
    chunks = _chunks()
    assert chunks, "expected at least one chunk"
    for c in chunks:
        md = c.metadata()
        assert md["ticker"] == "TST"
        assert md["form"] == "10-K"
        assert md["accession"] == "0000000000-25-000001"
        assert md["item"] == "1"
        assert md["filing_date"] == "2025-01-01"
        start, end = md["char_offsets"]
        assert 0 <= start < end <= len(_SECTION)
        # Offsets are TRUE offsets into the section text.
        assert _SECTION[start:end].strip() == c.text


def test_chunks_respect_max_size():
    chunks = chunk_section(
        _SECTION, ticker="TST", form="10-K", filing_date="2025-01-01",
        accession="acc", item="1", min_chars=80, max_chars=260,
    )
    # Allow slight overage only for hard-split single paragraphs; here paragraphs
    # are small so every chunk should be within the cap.
    assert all(len(c.text) <= 260 + 5 for c in chunks)


def test_chunk_ids_unique():
    chunks = _chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_tfidf_retrieval_ranks_relevant_chunk_first():
    chunks = _chunks()
    idx = FilingsIndex(ticker="TST", chunks=chunks, inventory=[])
    results = idx.search("competition and rapid technological change", k=3,
                         scorer=TfidfScorer())
    assert results
    assert "competit" in results[0]["text"].lower()
    # Metadata rides along with every result.
    assert results[0]["ticker"] == "TST"
    assert results[0]["item"] == "1"
    assert "char_offsets" in results[0]


def test_default_scorer_is_tfidf():
    assert get_scorer().name == "tfidf"
    # Unknown/absent optional backend falls back to tfidf, never raises.
    assert get_scorer("sentence-transformers").name in ("tfidf", "sentence-transformers")


def test_index_save_load_roundtrip(tmp_path, monkeypatch):
    import app.filings_index as mod
    monkeypatch.setattr(mod, "_index_path",
                        lambda ticker: str(tmp_path / f"{ticker}.json"))
    chunks = _chunks()
    idx = FilingsIndex(ticker="TST", chunks=chunks, inventory=[{"form": "10-K"}])
    idx.save()
    loaded = FilingsIndex.load("TST")
    assert loaded is not None
    assert len(loaded.chunks) == len(chunks)
    assert loaded.chunks[0].text == chunks[0].text
    assert loaded.inventory == [{"form": "10-K"}]
