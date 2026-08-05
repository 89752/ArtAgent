"""词法检索单测：语言检测、翻译兜底、core FTS5、PDF BM25、混合融合。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import (
    CoreLexicalRetriever,
    PdfBm25Retriever,
    _bm25_scores,
    _tokenize,
    detect_lang,
    translate_query,
)


def _core_csv(path: Path) -> Path:
    pd.DataFrame([
        {
            "artwork_id": "Q1", "title": "Water Lilies", "artist_qid": "Q2",
            "artist_name": "Claude Monet", "collection_name": "", "location": "",
            "inception": "1916", "year": 1916, "year_bucket": "1901-1950",
            "material": "Oil on canvas", "genre": "", "school": "French",
            "movement": "Impressionism", "series": "", "description": "A pond with lilies.",
            "image_url": "https://example.com/lilies.jpg", "license": "",
            "dimensions_raw": "", "width_cm": "", "height_cm": "",
            "source_api": "wikidata", "dedup_key": "monet|lilies|1916",
        },
        {
            "artwork_id": "Q3", "title": "Starry Night", "artist_qid": "Q4",
            "artist_name": "Vincent van Gogh", "collection_name": "", "location": "",
            "inception": "1889", "year": 1889, "year_bucket": "1851-1900",
            "material": "Oil on canvas", "genre": "", "school": "Dutch",
            "movement": "Post-Impressionism", "series": "", "description": "Night sky over a village.",
            "image_url": "", "license": "", "dimensions_raw": "", "width_cm": "",
            "height_cm": "", "source_api": "wikidata", "dedup_key": "gogh|starry|1889",
        },
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_detect_lang():
    assert detect_lang("莫奈的睡莲") == "zh"
    assert detect_lang("Water Lilies by Monet") == "en"
    assert detect_lang("ひまわり") == "ja"
    assert detect_lang("해바라기") == "ko"
    assert detect_lang("tableau de Monet") == "en"  # 拉丁语系归 en 档


def test_translate_query_same_lang_noop():
    assert translate_query("Water Lilies", "en") == "Water Lilies"


def test_translate_query_fallback_on_llm_failure(monkeypatch):
    def boom(prompt):
        raise RuntimeError("no llm")

    monkeypatch.setattr("src.utils.llm.get_deterministic_llm", boom)
    # 测试环境 LEXICAL_TRANSLATE=0：直接返回原文
    assert translate_query("睡莲", "en") == "睡莲"


def test_core_fts_english_query():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _core_csv(Path(tmp) / "artworks_core.csv")
        r = CoreLexicalRetriever(csv_path=csv_path, db_path=Path(tmp) / "lexical.db")
        hits = r.search("Water Lilies", top_k=5)
        assert hits and hits[0].metadata["title"] == "Water Lilies"
        assert hits[0].source == "core"
        assert hits[0].image_refs == ["https://example.com/lilies.jpg"]


def test_core_fts_chinese_query_translated(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.lexical.translate_query",
        lambda q, target: "Water Lilies" if q == "睡莲" else q,
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _core_csv(Path(tmp) / "artworks_core.csv")
        r = CoreLexicalRetriever(csv_path=csv_path, db_path=Path(tmp) / "lexical.db")
        hits = r.search("睡莲", top_k=5)
        assert hits and hits[0].metadata["title"] == "Water Lilies"


def test_core_fts_filters():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _core_csv(Path(tmp) / "artworks_core.csv")
        r = CoreLexicalRetriever(csv_path=csv_path, db_path=Path(tmp) / "lexical.db")
        hits = r.search("night", top_k=10, filters={"school": "Dutch"})
        assert hits and hits[0].metadata["title"] == "Starry Night"


class _FakeChroma:
    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas

    def count(self):
        return len(self._docs)

    def get(self, include=None):
        return {"documents": self._docs, "metadatas": self._metas}


def test_pdf_bm25_chinese_query():
    docs = [
        "莫奈在葛列尔画室认识了布丹",
        "梵高在阿尔勒画了向日葵",
    ]
    metas = [
        {"doc_id": "d1", "page_id": "d1-p1", "kb_id": "k"},
        {"doc_id": "d2", "page_id": "d2-p1", "kb_id": "k"},
    ]
    with patch(
        "src.retrieval.hybrid.get_or_create_chroma_collection",
        return_value=_FakeChroma(docs, metas),
    ):
        r = PdfBm25Retriever()
        hits = r.search("莫奈", top_k=5)
    assert hits and "莫奈" in hits[0].content
    assert hits[0].source == "user_pdf_text"
    assert hits[0].metadata["doc_id"] == "d1"


def test_pdf_bm25_english_query_translated(monkeypatch):
    docs = ["莫奈的睡莲系列创作于吉维尼", "卡拉瓦乔擅长明暗对照"]
    metas = [{"doc_id": "d1", "page_id": "d1-p1"}, {"doc_id": "d2", "page_id": "d2-p1"}]
    monkeypatch.setattr(
        "src.retrieval.lexical.translate_query",
        lambda q, target: "莫奈" if target == "zh" else q,
    )
    with patch(
        "src.retrieval.hybrid.get_or_create_chroma_collection",
        return_value=_FakeChroma(docs, metas),
    ):
        r = PdfBm25Retriever()
        hits = r.search("Monet", top_k=5)
    assert hits and "莫奈" in hits[0].content


def test_bm25_ranks_exact_over_partial():
    docs = [["睡莲"], ["睡", "莲"]]
    scores = _bm25_scores(["睡莲"], docs)
    assert scores[0] > scores[1]


def test_tokenize_mixed():
    toks = _tokenize("莫奈 Water Lilies")
    assert "water" in toks and "lilies" in toks
    assert "莫奈" in toks


def test_hybrid_fuses_lexical_and_vector_and_dedups():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _core_csv(Path(tmp) / "artworks_core.csv")
        lexical = CoreLexicalRetriever(csv_path=csv_path, db_path=Path(tmp) / "lexical.db")

        class _FakeVector:
            source = "core"
            dataset_id = "core"

            def search(self, query, top_k=5, filters=None):
                return [RetrievalResult(
                    content="A pond with lilies.",
                    source="core",
                    score=0.9,
                    metadata={
                        "title": "Water Lilies", "artist": "Claude Monet",
                        "dedup_key": "monet|lilies|1916", "image_url": "https://example.com/lilies.jpg",
                    },
                )]

        h = HybridRetriever()
        h.register("core", _FakeVector())
        h.register("core_lexical", lexical)
        out = h.search("Water Lilies", top_k=5, sources=["core"], rerank=False)
        assert len(out) == 1  # 向量与词法命中同一作品，去重
        assert out[0].metadata["title"] == "Water Lilies"
