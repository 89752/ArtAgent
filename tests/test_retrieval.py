"""检索域统一单测：access / 通道权重 / hybrid / 词法 / 相关性校正 / reranker /
结构化检索 / documents_store。

全程离线：fake 检索器、fake LLM、mock requests，不加载模型、不联网。
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.data import documents_store
from src.data.access import (
    EVIDENCE_SNIPPET_LEN,
    format_evidence_block,
    fuzzy_match,
    row_to_artwork_dict,
)
from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import (
    HybridRetriever,
    RERANK_POOL,
    _channel_weight,
    _dedup,
    _rerank_enabled,
    _rerank_fused,
    _rrf_fuse,
)
from src.retrieval.lexical import (
    CoreLexicalRetriever,
    PdfBm25Retriever,
    _bm25_scores,
    _tokenize,
    detect_lang,
    translate_query,
)
from src.retrieval import reranker as reranker_mod
from src.retrieval import relevance as relevance_mod
from src.retrieval.relevance import _filter_enabled, llm_relevance_filter
from src.retrieval.structured_retriever import (
    CORE_SCHEMA,
    TableSchema,
    StructuredTableRetriever,
    get_structured_retriever,
    register_structured_dataset,
)
from src.agent.state import AgentState
from src.agent.nodes import general as general_mod
import src.utils.governance as gov_mod


@pytest.fixture(autouse=True)
def _jina_key(monkeypatch):
    monkeypatch.setenv("RERANK_API_KEY", "jina_test_key")


def _hit(content, source="semart", score=0.9, **meta) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=score, metadata=meta)


def _patch(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    return old


# ══════════════ access ══════════════
LONG_DESC = "A swirling night sky over a quiet town. " * 20


def _access_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "TITLE": "The Starry Night", "AUTHOR": "Vincent van Gogh",
            "DATE": "1889", "TECHNIQUE": "Oil on canvas", "SCHOOL": "Dutch",
            "TIMEFRAME": "1851-1900", "IMAGE_FILE": "starry.jpg", "DESCRIPTION": LONG_DESC,
        },
        {
            "TITLE": "The Kiss", "AUTHOR": "Gustav Klimt", "DATE": "1908",
            "TECHNIQUE": "Oil and gold leaf on canvas", "SCHOOL": "Austrian",
            "TIMEFRAME": "1901-1950", "IMAGE_FILE": "kiss.jpg", "DESCRIPTION": "short desc",
        },
        {
            "TITLE": "Water Lilies", "AUTHOR": "Claude Monet", "DATE": "1906",
            "TECHNIQUE": "Oil on canvas", "SCHOOL": "French",
            "TIMEFRAME": "1901-1950", "IMAGE_FILE": "lilies.jpg", "DESCRIPTION": "",
        },
    ])


def test_fuzzy_exact_case_insensitive():
    df = _access_df()
    hit = fuzzy_match(df, "TITLE", "the starry night")
    assert len(hit) == 1 and hit.iloc[0]["IMAGE_FILE"] == "starry.jpg"


def test_fuzzy_author_single_token():
    df = _access_df()
    hit = fuzzy_match(df, "AUTHOR", "GOGH")
    assert len(hit) == 1 and hit.iloc[0]["AUTHOR"] == "Vincent van Gogh"


def test_fuzzy_author_longest_token_first():
    df = _access_df()
    hit = fuzzy_match(df, "AUTHOR", "Vincent van Gogh")
    assert len(hit) == 1 and hit.iloc[0]["AUTHOR"] == "Vincent van Gogh"


def test_fuzzy_partial_contains():
    df = _access_df()
    hit = fuzzy_match(df, "TITLE", "Kiss")
    assert len(hit) == 1 and hit.iloc[0]["TITLE"] == "The Kiss"


def test_fuzzy_no_match_returns_empty():
    assert fuzzy_match(_access_df(), "TITLE", "zzz-nonexistent").empty


def test_fuzzy_empty_inputs():
    df = _access_df()
    assert fuzzy_match(df, "TITLE", "").empty
    assert fuzzy_match(df.iloc[0:0], "TITLE", "Kiss").empty


def test_row_to_artwork_dict_from_series():
    df = _access_df()
    d = row_to_artwork_dict(df.iloc[0])
    assert d["title"] == "The Starry Night"
    assert d["author"] == "Vincent van Gogh"
    assert d["image_file"] == "starry.jpg"
    assert len(d["description_snippet"]) == EVIDENCE_SNIPPET_LEN + 3
    assert d["description_snippet"].endswith("...")


def test_row_to_artwork_dict_short_desc_not_truncated():
    d = row_to_artwork_dict(_access_df().iloc[1])
    assert d["description_snippet"] == "short desc"


def test_row_to_artwork_dict_no_truncation_mode():
    d = row_to_artwork_dict(_access_df().iloc[0], snippet_len=None)
    assert d["description_snippet"] == LONG_DESC


def test_row_to_artwork_dict_from_chroma_meta():
    meta = {
        "title": "The Kiss", "author": "Gustav Klimt", "date": "1908",
        "technique": "Oil", "school": "Austrian", "timeframe": "1901-1950",
        "file": "kiss.jpg", "description": "short desc",
    }
    d = row_to_artwork_dict(meta)
    assert d["image_file"] == "kiss.jpg"
    assert d["title"] == "The Kiss"


def test_row_to_artwork_dict_missing_fields():
    d = row_to_artwork_dict(_access_df().iloc[2])
    assert d["description_snippet"] == ""
    assert d["title"] == "Water Lilies"


def test_format_basic():
    docs = [{"title": "The Kiss", "date": "1908", "description_snippet": "short desc"}]
    out = format_evidence_block(docs, "- {title} ({date}): {description_snippet}")
    assert out == "- The Kiss (1908): short desc"


def test_format_missing_date_no_empty_parens():
    docs = [{"title": "The Kiss", "description_snippet": "short desc"}]
    out = format_evidence_block(docs, "- {title} ({date}): {description_snippet}")
    assert "()" not in out
    assert out == "- The Kiss: short desc"


def test_format_missing_desc_no_trailing_colon():
    docs = [{"title": "Water Lilies", "date": "1906", "description_snippet": ""}]
    out = format_evidence_block(docs, "  - {title} ({date}): {description_snippet}")
    assert out == "  - Water Lilies (1906)"


def test_format_pipe_template_missing_author():
    docs = [{"title": "X", "description_snippet": "s"}]
    out = format_evidence_block(docs, "- {author} | {title}: {description_snippet}")
    assert out == "- X: s"


def test_format_pipe_template_full():
    docs = [{"author": "Monet", "title": "Water Lilies", "description_snippet": "s"}]
    out = format_evidence_block(docs, "- {author} | {title}: {description_snippet}")
    assert out == "- Monet | Water Lilies: s"


def test_format_web_results_template():
    docs = [{"title": "t", "snippet": "s", "url": "http://x"}]
    out = format_evidence_block(docs, "- {title}: {snippet} ({url})")
    assert out == "- t: s (http://x)"


# ══════════════ 通道权重 / RRF ══════════════
def test_default_weights_equal_for_main_sources():
    assert _channel_weight("semart") == 1.0
    assert _channel_weight("core") == 1.0
    assert _channel_weight("user_table") == 1.0
    assert _channel_weight("user_pdf_text") == 1.0


def test_noise_channel_down_weighted():
    assert _channel_weight("user_pdf_image") == 0.5
    assert _channel_weight("met_museum") == 0.5
    assert _channel_weight("unknown_future_source") == 1.0


def test_env_override_wins():
    os.environ["CHANNEL_WEIGHT_MET_MUSEUM"] = "0.3"
    try:
        assert _channel_weight("met_museum") == 0.3
    finally:
        os.environ.pop("CHANNEL_WEIGHT_MET_MUSEUM", None)
    assert _channel_weight("met_museum") == 0.5


def test_equal_weight_matches_old_rrf_order():
    a = [_hit("a0"), _hit("a1")]
    b = [_hit("b0", source="user_pdf_text")]
    fused = _rrf_fuse([a, b])
    assert [h.content for h in fused] == ["a0", "b0", "a1"]


def test_single_source_order_preserved_regardless_of_weight():
    hits = [_hit(f"x{i}") for i in range(3)]
    assert [h.content for h in _rrf_fuse([hits])] == ["x0", "x1", "x2"]


def test_lower_weight_source_loses_tie():
    a = [_hit("semart_rank0", source="semart")]
    b = [_hit("image_rank0", source="user_pdf_image")]
    fused = _rrf_fuse([a, b])
    assert fused[0].content == "semart_rank0"


def test_weight_flips_outcome_noise_channel():
    relevant = [_hit("relevant", source="semart")]
    noise = [_hit(f"noise{i}", source="user_pdf_image") for i in range(4)]
    semart_full = [_hit(f"unrelated{i}") for i in range(4)] + relevant
    fused = _rrf_fuse([semart_full, noise])
    assert fused.index(relevant[0]) < fused.index(noise[0])
    os.environ["CHANNEL_WEIGHT_USER_PDF_IMAGE"] = "1.0"
    try:
        fused_equal = _rrf_fuse([semart_full, noise])
        assert fused_equal.index(relevant[0]) > fused_equal.index(noise[0])
    finally:
        os.environ.pop("CHANNEL_WEIGHT_USER_PDF_IMAGE", None)


def test_rrf_head_of_second_source_can_outrank_tail_of_first():
    a = [_hit("a0"), _hit("a1"), _hit("a2")]
    b = [_hit("b0", source="user_pdf_text")]
    assert [h.content for h in _rrf_fuse([a, b])][1] == "b0"


# ══════════════ hybrid 去重 / 搜索 ══════════════
def test_dedup_image_dropped_when_text_same_page():
    hits = [
        _hit("page1-text", source="user_pdf_text", page_id="d1-p3"),
        _hit("page1-image", source="user_pdf_image", page_id="d1-p3"),
        _hit("page2-text", source="user_pdf_text", page_id="d1-p4"),
    ]
    assert [h.content for h in _dedup(hits)] == ["page1-text", "page2-text"]


def test_dedup_image_kept_when_no_text_sibling():
    hits = [
        _hit("page1-image", source="user_pdf_image", page_id="d1-p3"),
        _hit("page2-text", source="user_pdf_text", page_id="d1-p4"),
    ]
    assert len(_dedup(hits)) == 2


def test_dedup_multiple_text_chunks_same_page_kept():
    hits = [
        _hit("chunk-a", source="user_pdf_text", page_id="d1-p3"),
        _hit("chunk-b", source="user_pdf_text", page_id="d1-p3"),
    ]
    assert len(_dedup(hits)) == 2


def test_dedup_ignores_results_without_keys():
    hits = [_hit("s1", title="T1"), _hit("s2", title="T2")]
    assert len(_dedup(hits)) == 2


class _FakeRetriever:
    def __init__(self, source, results=(), dataset_id=None, fail=False):
        self.source = source
        self.dataset_id = dataset_id
        self._results = list(results)
        self.fail = fail

    def search(self, query, top_k=5, filters=None):
        if self.fail:
            raise RuntimeError("boom")
        return self._results[:top_k]


def test_search_single_source_passthrough_order():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("a"), _hit("b"), _hit("c")]))
    assert [x.content for x in h.search("q", top_k=5)] == ["a", "b", "c"]


def test_search_top_k_truncates():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit(f"a{i}") for i in range(5)]))
    assert len(h.search("q", top_k=2)) == 2


def test_search_sources_filter():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("s")]))
    h.register("user_pdf_text", _FakeRetriever("user_pdf_text", [_hit("p", source="user_pdf_text")]))
    assert [x.content for x in h.search("q", top_k=5, sources=["semart"])] == ["s"]


def test_search_dataset_id_filter():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("s")], dataset_id="semart"))
    h.register("user_table", _FakeRetriever("user_table", [_hit("t", source="user_table")], dataset_id="my_table"))
    assert [x.content for x in h.search("q", top_k=5, dataset_id="my_table")] == ["t"]


def test_search_source_failure_tolerated():
    h = HybridRetriever()
    h.register("bad", _FakeRetriever("bad", fail=True))
    h.register("semart", _FakeRetriever("semart", [_hit("ok")]))
    assert [x.content for x in h.search("q", top_k=5)] == ["ok"]


def test_search_empty_when_no_sources():
    assert HybridRetriever().search("q") == []


def test_search_dedup_across_sources():
    h = HybridRetriever()
    h.register("user_pdf_text", _FakeRetriever("user_pdf_text", [_hit("text", source="user_pdf_text", page_id="d1-p3")]))
    h.register("user_pdf_image", _FakeRetriever("user_pdf_image", [_hit("image", source="user_pdf_image", page_id="d1-p3")]))
    assert [x.content for x in h.search("q", top_k=5)] == ["text"]


# ══════════════ 词法检索 ══════════════
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
    assert detect_lang("tableau de Monet") == "en"


def test_translate_query_same_lang_noop():
    assert translate_query("Water Lilies", "en") == "Water Lilies"


def test_translate_query_fallback_on_llm_failure(monkeypatch):
    def boom(prompt):
        raise RuntimeError("no llm")

    monkeypatch.setattr("src.utils.llm.get_deterministic_llm", boom)
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
    docs = ["莫奈在葛列尔画室认识了布丹", "梵高在阿尔勒画了向日葵"]
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
    scores = _bm25_scores(["睡莲"], [["睡莲"], ["睡", "莲"]])
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
                    content="A pond with lilies.", source="core", score=0.9,
                    metadata={
                        "title": "Water Lilies", "artist": "Claude Monet",
                        "dedup_key": "monet|lilies|1916",
                        "image_url": "https://example.com/lilies.jpg",
                    },
                )]

        h = HybridRetriever()
        h.register("core", _FakeVector())
        h.register("core_lexical", lexical)
        out = h.search("Water Lilies", top_k=5, sources=["core"], rerank=False)
        assert len(out) == 1
        assert out[0].metadata["title"] == "Water Lilies"


# ══════════════ 相关性校正 ══════════════
def _items(n=4):
    return [{"title": f"Work {i}", "description_snippet": f"snippet {i}"} for i in range(n)]


class _FakeLLM:
    def __init__(self, content="[0, 1]", error=None):
        self.content = content
        self.error = error
        self.called = False

    def invoke(self, prompt):
        self.called = True
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def test_enabled_override_wins_over_env():
    os.environ["RELEVANCE_FILTER_ENABLED"] = "0"
    try:
        assert _filter_enabled(True) is True
        assert _filter_enabled(None) is False
    finally:
        os.environ.pop("RELEVANCE_FILTER_ENABLED", None)
    assert _filter_enabled(None) is True


def test_disabled_returns_same_object_without_llm():
    llm = _FakeLLM()
    items = _items()
    out = llm_relevance_filter("q", items, llm=llm, enabled=False)
    assert out is items and not llm.called


def test_small_list_passthrough_without_llm():
    llm = _FakeLLM()
    items = _items(2)
    out = llm_relevance_filter("q", items, min_keep=2, llm=llm)
    assert out is items and not llm.called


def test_filters_to_llm_selection_in_original_order():
    items = _items(4)
    out = llm_relevance_filter("q", items, llm=_FakeLLM("[2, 0]"))
    assert [d["title"] for d in out] == ["Work 0", "Work 2"]


def test_markdown_wrapped_json_accepted():
    items = _items(3)
    out = llm_relevance_filter("q", items, llm=_FakeLLM("```json\n[1]\n```"), min_keep=1)
    assert [d["title"] for d in out] == ["Work 1"]


def test_garbage_and_out_of_range_indices_ignored():
    items = _items(4)
    out = llm_relevance_filter("q", items, llm=_FakeLLM('[true, "x", 99, -1, 1]'), min_keep=2)
    assert [d["title"] for d in out] == ["Work 0", "Work 1"]


def test_llm_rejects_all_falls_back_to_min_keep():
    items = _items(4)
    out = llm_relevance_filter("q", items, llm=_FakeLLM("[]"), min_keep=2)
    assert [d["title"] for d in out] == ["Work 0", "Work 1"]


def test_beyond_max_candidates_passes_through():
    items = _items(5)
    out = llm_relevance_filter("q", items, max_candidates=3, min_keep=1, llm=_FakeLLM("[0]"))
    assert [d["title"] for d in out] == ["Work 0", "Work 3", "Work 4"]


def test_user_pdf_image_always_preserved():
    items = [
        {"title": "《手稿》第1页（整页图）", "description_snippet": "[整页图]", "source": "user_pdf_image"},
        {"title": "《手册》第2页（整页图）", "description_snippet": "[整页图]", "source": "user_pdf_image"},
        {"title": "相关画作", "description_snippet": "一幅风景画"},
        {"title": "无关画作", "description_snippet": "另一幅静物画"},
    ]
    out = llm_relevance_filter("手稿内容", items, min_keep=1, llm=_FakeLLM("[3]"))
    sources = [d.get("source") for d in out]
    assert sources.count("user_pdf_image") == 2
    assert out[0]["source"] == "user_pdf_image"
    assert out[1]["source"] == "user_pdf_image"


def test_invalid_json_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM("这不是JSON"))
    assert out is items


def test_non_list_json_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM('{"keep": [0]}'))
    assert out is items


def test_llm_exception_returns_original():
    items = _items()
    out = llm_relevance_filter("q", items, llm=_FakeLLM(error=RuntimeError("boom")))
    assert out is items


def _make_state_with_search_call():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "semantic_search", "args": {"query": "星空 画作"}, "id": "call-1"},
            {"name": "exact_lookup", "args": {"author": "Monet"}, "id": "call-2"},
        ],
    )
    return AgentState(user_query="找星空的画", messages=[HumanMessage(content="找星空的画"), ai])


def _make_tool_output(search_items):
    return {
        "messages": [
            ToolMessage(content=json.dumps(search_items, ensure_ascii=False),
                        name="semantic_search", tool_call_id="call-1", id="m1"),
            ToolMessage(content=json.dumps([{"title": "Monet"}], ensure_ascii=False),
                        name="exact_lookup", tool_call_id="call-2", id="m2"),
        ]
    }


def test_general_tools_filters_semantic_search_message():
    items = _items(4)
    state = _make_state_with_search_call()

    def _fake_governed(tool, args):
        if tool.name == "semantic_search":
            return json.dumps(items, ensure_ascii=False)
        return json.dumps([{"title": "Monet"}], ensure_ascii=False)

    seen = {}

    def _fake_filter(query, got_items, min_keep=2):
        seen["query"] = query
        return [got_items[0], got_items[2]]

    old_gov = _patch(gov_mod, "governed_invoke", _fake_governed)
    old_filter = _patch(general_mod, "llm_relevance_filter", _fake_filter)
    try:
        out = general_mod.general_tools(state)
        msgs = out["messages"]
        kept = json.loads(msgs[0].content)
        assert [d["title"] for d in kept] == ["Work 0", "Work 2"]
        assert seen["query"] == "星空 画作"
        assert msgs[0].tool_call_id == "call-1" and msgs[0].id == "gov:call-1"
        assert json.loads(msgs[1].content) == [{"title": "Monet"}]
    finally:
        _patch(gov_mod, "governed_invoke", old_gov)
        _patch(general_mod, "llm_relevance_filter", old_filter)


def test_general_tools_untouched_when_no_search_call():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "exact_lookup", "args": {"author": "Monet"}, "id": "c9"}],
    )
    state = AgentState(user_query="莫奈的画", messages=[HumanMessage(content="莫奈的画"), ai])

    def _fake_governed(tool, args):
        return "[]"

    def _boom(*a, **kw):
        raise AssertionError("无 semantic_search 调用不应触发过滤")

    old_gov = _patch(gov_mod, "governed_invoke", _fake_governed)
    old_filter = _patch(general_mod, "llm_relevance_filter", _boom)
    try:
        out = general_mod.general_tools(state)
        assert out["messages"][0].content == "[]"
    finally:
        _patch(gov_mod, "governed_invoke", old_gov)
        _patch(general_mod, "llm_relevance_filter", old_filter)


def test_filter_search_message_tolerates_non_json_content():
    msg = ToolMessage(content="tool execution error", name="semantic_search",
                      tool_call_id="c1", id="m1")
    assert general_mod._filter_search_message(msg, "q") is msg


def test_filter_search_message_skips_when_nothing_dropped():
    items = _items(2)
    msg = ToolMessage(content=json.dumps(items), name="semantic_search",
                      tool_call_id="c1", id="m1")
    old_filter = _patch(general_mod, "llm_relevance_filter", lambda q, xs, min_keep=2: xs)
    try:
        assert general_mod._filter_search_message(msg, "q") is msg
    finally:
        _patch(general_mod, "llm_relevance_filter", old_filter)


# ══════════════ reranker ══════════════
class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _PostRecorder:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"results": []}
        self.error = error
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None, **kwargs):
        self.calls.append({
            "url": url, "headers": headers, "json": json, "timeout": timeout,
            "proxies": kwargs.get("proxies"),
        })
        if self.error:
            raise self.error
        return _FakeResp(self.payload)


def test_empty_documents_returns_empty_without_call():
    rec = _PostRecorder()
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        assert reranker_mod.rerank("q", []) == []
        assert rec.calls == []
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    assert reranker_mod.rerank_available() is False
    assert reranker_mod.rerank("q", ["doc"]) is None


def test_jina_payload_and_url(monkeypatch):
    monkeypatch.delenv("RERANK_PROXY", raising=False)
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 0.9}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a"], top_n=1)
        call = rec.calls[0]
        assert call["url"] == reranker_mod.JINA_RERANK_URL
        assert call["headers"]["Authorization"] == "Bearer jina_test_key"
        assert call["json"] == {
            "model": reranker_mod.RERANK_MODEL, "query": "q",
            "documents": ["a"], "top_n": 1,
        }
        assert call["proxies"] is None
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_proxy_applied_when_configured(monkeypatch):
    monkeypatch.setenv("RERANK_PROXY", "http://127.0.0.1:7890")
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 0.9}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a"], top_n=1)
        assert rec.calls[0]["proxies"] == {
            "http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890",
        }
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_proxy_off_when_env_blank(monkeypatch):
    monkeypatch.delenv("RERANK_PROXY", raising=False)
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 0.9}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a"], top_n=1)
        assert rec.calls[0]["proxies"] is None
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_model_can_be_overridden_via_module_config():
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 0.9}]})
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_model = reranker_mod.RERANK_MODEL
    reranker_mod.RERANK_MODEL = "jina-reranker-v3.5-custom"
    try:
        reranker_mod.rerank("q", ["a"])
        assert rec.calls[0]["json"]["model"] == "jina-reranker-v3.5-custom"
    finally:
        reranker_mod.RERANK_MODEL = old_model
        _patch(reranker_mod.requests, "post", old_post)


def test_success_parses_and_sorts_by_score():
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
        ]
    }
    rec = _PostRecorder(payload)
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        assert reranker_mod.rerank("q", ["a", "b", "c"]) == [(0, 0.9), (1, 0.7), (2, 0.5)]
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_top_n_capped_by_document_count():
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 1.0}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a", "b"], top_n=10)
        assert rec.calls[0]["json"]["top_n"] == 2
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_long_documents_truncated_to_char_limit():
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 1.0}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["x" * (reranker_mod.DOC_CHAR_LIMIT + 500)])
        assert len(rec.calls[0]["json"]["documents"][0]) == reranker_mod.DOC_CHAR_LIMIT
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_http_error_retries_then_returns_none():
    rec = _PostRecorder(error=RuntimeError("boom"))
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_sleep = _patch(reranker_mod.time, "sleep", lambda s: None)
    try:
        assert reranker_mod.rerank("q", ["a"]) is None
        assert len(rec.calls) == reranker_mod.MAX_RETRIES + 1
    finally:
        _patch(reranker_mod.requests, "post", old_post)
        _patch(reranker_mod.time, "sleep", old_sleep)


def test_rerank_enabled_override_wins():
    os.environ["RERANK_ENABLED"] = "0"
    try:
        assert _rerank_enabled(True) is True
        assert _rerank_enabled(None) is False
        assert _rerank_enabled(False) is False
    finally:
        os.environ.pop("RERANK_ENABLED", None)


def test_rerank_enabled_default_on():
    os.environ.pop("RERANK_ENABLED", None)
    assert _rerank_enabled(None) is True
    os.environ["RERANK_ENABLED"] = "false"
    try:
        assert _rerank_enabled(None) is False
    finally:
        os.environ.pop("RERANK_ENABLED", None)


def test_pool_not_larger_than_top_k_skips_rerank():
    def _boom(*a, **kw):
        raise AssertionError("候选不足时不应调用 rerank")

    old = _patch(reranker_mod, "rerank", _boom)
    try:
        pool = [_hit("a"), _hit("b")]
        assert [h.content for h in _rerank_fused("q", pool, top_k=5)] == ["a", "b"]
    finally:
        _patch(reranker_mod, "rerank", old)


def test_text_slots_reordered_image_keeps_slot():
    pool = [
        _hit("t0", source="semart"),
        _hit("img", source="user_pdf_image", page_id="d-p1"),
        _hit("t1", source="semart"),
    ]

    def _fake_rerank(query, documents, top_n=None):
        assert documents == ["t0", "t1"]
        return [(1, 0.95), (0, 0.60)]

    old = _patch(reranker_mod, "rerank", _fake_rerank)
    try:
        out = _rerank_fused("q", pool, top_k=1)
        assert [h.content for h in out] == ["t1", "img", "t0"]
        assert out[0].metadata["rerank_score"] == 0.95
        assert "rerank_score" not in out[1].metadata
    finally:
        _patch(reranker_mod, "rerank", old)


def test_rerank_failure_returns_coarse_order():
    old = _patch(reranker_mod, "rerank", lambda *a, **kw: None)
    try:
        pool = [_hit(f"t{i}") for i in range(6)]
        assert [h.content for h in _rerank_fused("q", pool, top_k=2)] == [f"t{i}" for i in range(6)]
    finally:
        _patch(reranker_mod, "rerank", old)


def test_partial_rerank_response_returns_coarse_order():
    old = _patch(reranker_mod, "rerank", lambda *a, **kw: [(2, 0.9), (0, 0.5)])
    try:
        pool = [_hit(f"t{i}") for i in range(4)]
        assert [h.content for h in _rerank_fused("q", pool, top_k=1)] == [f"t{i}" for i in range(4)]
    finally:
        _patch(reranker_mod, "rerank", old)


class _RecordingRetriever:
    def __init__(self, source, n=3):
        self.source = source
        self.dataset_id = None
        self.n = n
        self.seen_top_k = []

    def search(self, query, top_k=5, filters=None):
        self.seen_top_k.append(top_k)
        return [_hit(f"{self.source}{i}", source=self.source) for i in range(min(top_k, self.n))]


def test_search_rerank_off_fetches_top_k_only():
    r = _RecordingRetriever("semart", n=10)
    h = HybridRetriever()
    h.register("semart", r)
    out = h.search("q", top_k=3, rerank=False)
    assert r.seen_top_k == [3]
    assert len(out) == 3


def test_search_rerank_on_fetches_pool():
    r = _RecordingRetriever("semart", n=RERANK_POOL)
    h = HybridRetriever()
    h.register("semart", r)

    def _fake_rerank(query, documents, top_n=None):
        return [(len(documents) - 1, 0.99)] + [(i, 0.5) for i in range(len(documents) - 1)]

    old = _patch(reranker_mod, "rerank", _fake_rerank)
    try:
        out = h.search("q", top_k=2, rerank=True)
        assert r.seen_top_k == [RERANK_POOL]
        assert out[0].content == f"semart{RERANK_POOL - 1}"
        assert len(out) == 2
    finally:
        _patch(reranker_mod, "rerank", old)


# ══════════════ 结构化检索 ══════════════
SR_SCHEMA = TableSchema(
    entity_col="AUTHOR", group_axis_col="TIMEFRAME",
    description_col="DESCRIPTION", image_col="IMAGE_FILE",
)


def _sr_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"TITLE": "Irises", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "1851-1900",
         "DESCRIPTION": "Van Gogh's irises study", "IMAGE_FILE": "irises.jpg"},
        {"TITLE": "The Starry Night", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "1851-1900",
         "DESCRIPTION": "Swirling night sky", "IMAGE_FILE": "starry.jpg"},
        {"TITLE": "Sunflowers", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "",
         "DESCRIPTION": "Yellow still life", "IMAGE_FILE": "sun.jpg"},
        {"TITLE": "Water Lilies", "AUTHOR": "MONET, Claude", "TIMEFRAME": "1901-1950",
         "DESCRIPTION": "Pond reflections", "IMAGE_FILE": "lilies.jpg"},
    ])


def _sr_retriever(**kwargs) -> StructuredTableRetriever:
    return StructuredTableRetriever("test_ds", SR_SCHEMA, df=_sr_df(), source="semart", **kwargs)


def test_core_schema_capabilities_full():
    assert CORE_SCHEMA.supports_timeline is True
    assert CORE_SCHEMA.supports_recommendation is True


def test_schema_no_axis_no_timeline():
    schema = TableSchema(entity_col="NAME", description_col="BIO")
    assert schema.supports_timeline is False
    assert schema.supports_recommendation is True


def test_group_by_axis_sorted_and_unknown_dropped():
    groups = _sr_retriever().group_by_axis("Van Gogh")
    assert list(groups.keys()) == ["1851-1900"]
    assert len(groups["1851-1900"]) == 2


def test_group_by_axis_only_unknown_kept():
    df = _sr_df()
    df["TIMEFRAME"] = ""
    r = StructuredTableRetriever("t_only_unknown", SR_SCHEMA, df=df)
    groups = r.group_by_axis("Van Gogh")
    assert list(groups.keys()) == ["Unknown"]
    assert len(groups["Unknown"]) == 3


def test_group_by_axis_entity_not_found():
    assert _sr_retriever().group_by_axis("zzz-nobody") == {}


def test_group_by_axis_without_axis_col():
    schema = TableSchema(entity_col="AUTHOR", description_col="DESCRIPTION")
    r = StructuredTableRetriever("t_no_axis", schema, df=_sr_df())
    assert r.group_by_axis("Van Gogh") == {}


def test_group_by_axis_sorted_multiple_groups():
    df = _sr_df()
    df.loc[3, "TIMEFRAME"] = "1801-1850"
    df.loc[0, "AUTHOR"] = "MONET, Claude"
    r = StructuredTableRetriever("t_multi", SR_SCHEMA, df=df)
    assert list(r.group_by_axis("Monet").keys()) == ["1801-1850", "1851-1900"]


def test_exclude_by_entity_dataframe():
    out = _sr_retriever().exclude_by_entity(["Vincent van Gogh"])
    assert len(out) == 1 and out.iloc[0]["AUTHOR"] == "MONET, Claude"


def test_exclude_by_entity_empty_names_returns_all():
    assert len(_sr_retriever().exclude_by_entity([])) == 4


def test_exclude_by_entity_short_tokens_ignored():
    assert len(_sr_retriever().exclude_by_entity(["AI"])) == 4


def test_exclude_from_results_dicts():
    results = [
        {"title": "Irises", "author": "GOGH, Vincent van"},
        {"title": "Water Lilies", "author": "MONET, Claude"},
    ]
    out = _sr_retriever().exclude_from_results(results, ["Van Gogh"])
    assert [r["title"] for r in out] == ["Water Lilies"]


def test_exclude_from_results_empty_names():
    results = [{"title": "Irises", "author": "GOGH, Vincent van"}]
    assert _sr_retriever().exclude_from_results(results, []) == results


def test_fuzzy_search_by_entity():
    hits = _sr_retriever().search("Monet", top_k=5)
    assert len(hits) == 1
    h = hits[0]
    assert h.source == "semart"
    assert h.score == 1.0
    assert h.metadata["dataset_id"] == "test_ds"
    assert h.metadata["title"] == "Water Lilies"
    assert h.content == "Pond reflections"
    assert h.image_refs == ["lilies.jpg"]


def test_fuzzy_search_description_fallback():
    hits = _sr_retriever().search("night sky", top_k=5)
    assert len(hits) == 1 and hits[0].metadata["title"] == "The Starry Night"


def test_fuzzy_search_top_k():
    assert len(_sr_retriever().search("Van Gogh", top_k=2)) == 2


def test_fuzzy_search_filters_equality():
    hits = _sr_retriever().search("Van Gogh", top_k=5, filters={"TIMEFRAME": "1851-1900"})
    assert {h.metadata["title"] for h in hits} == {"Irises", "The Starry Night"}


def test_fuzzy_search_no_match():
    assert _sr_retriever().search("zzz-nothing") == []


class _FakeCollection:
    def __init__(self, metadatas, distances, count=None):
        self._metadatas = metadatas
        self._distances = distances
        self._count = count if count is not None else len(metadatas)
        self.last_n_results = None

    def count(self):
        return self._count

    def query(self, query_embeddings, n_results, include):
        self.last_n_results = n_results
        return {
            "metadatas": [self._metadatas[:n_results]],
            "distances": [self._distances[:n_results]],
        }


def _vector_retriever(count=None):
    metas = [
        {"title": "Irises", "author": "GOGH, Vincent van", "file": "irises.jpg",
         "description": "Van Gogh's irises study"},
        {"title": "Water Lilies", "author": "MONET, Claude", "file": "lilies.jpg",
         "description": "Pond reflections"},
    ]
    collection = _FakeCollection(metas, [0.2, 0.35], count=count)
    r = StructuredTableRetriever(
        "t_vec", SR_SCHEMA, df=_sr_df(), source="semart",
        collection_loader=lambda: collection,
        embed_fn_loader=lambda: (lambda text: [0.1, 0.2, 0.3]),
    )
    return r, collection


def test_vector_search_score_and_metadata():
    r, _ = _vector_retriever()
    hits = r.search("anything", top_k=2)
    assert len(hits) == 2
    assert hits[0].metadata["title"] == "Irises"
    assert hits[0].score == 0.8
    assert hits[0].metadata["dataset_id"] == "t_vec"
    assert hits[0].image_refs == ["irises.jpg"]


def test_vector_search_n_results_capped_by_count():
    r, collection = _vector_retriever(count=1)
    hits = r.search("anything", top_k=5)
    assert collection.last_n_results == 1
    assert len(hits) == 1


def test_lazy_df_loader():
    calls = []
    r = StructuredTableRetriever("t_lazy", SR_SCHEMA, df_loader=lambda: calls.append(1) or _sr_df())
    assert calls == []
    _ = r.df
    _ = r.df
    assert calls == [1]


def test_registry_register_and_get():
    register_structured_dataset("t_reg", SR_SCHEMA, df=_sr_df())
    assert get_structured_retriever("t_reg").dataset_id == "t_reg"


def test_registry_unknown_dataset_raises():
    try:
        get_structured_retriever("zzz-unregistered")
    except KeyError:
        return
    raise AssertionError("未注册的数据源应抛 KeyError")


# ══════════════ documents_store ══════════════
def _reset_docs():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    documents_store._reset_for_tests(path)
    old_legacy = documents_store._LEGACY_STATUS_FILE
    documents_store._LEGACY_STATUS_FILE = Path("/nonexistent/doc_status.json")
    try:
        documents_store.init_db()
    finally:
        documents_store._LEGACY_STATUS_FILE = old_legacy


def test_init_creates_table():
    _reset_docs()
    assert documents_store.DB_PATH.exists()


def test_add_and_get_document():
    _reset_docs()
    documents_store.add_document(
        doc_id="pdf-1", kind="pdf", doc_name="test.pdf",
        status="processing", file_path="uploads/default/pdf-1/document.pdf",
        file_size=1024,
    )
    doc = documents_store.get_document("pdf-1")
    assert doc["doc_id"] == "pdf-1"
    assert doc["kind"] == "pdf"
    assert doc["doc_name"] == "test.pdf"
    assert doc["file_size"] == 1024
    assert doc["status"] == "processing"


def test_upsert_document_updates_status_on_existing():
    _reset_docs()
    documents_store.add_document(doc_id="pdf-1", kind="pdf", status="processing")
    documents_store.upsert_document("pdf-1", status="done", text_chunks=3)
    doc = documents_store.get_document("pdf-1")
    assert doc["status"] == "done"
    assert doc["text_chunks"] == 3


def test_update_document():
    _reset_docs()
    documents_store.add_document(doc_id="pdf-2", kind="pdf")
    documents_store.update_document(
        "pdf-2", status="done", pages=10, text_chunks=5,
        metadata={"route_distribution": {"text": 8, "multimodal": 2}},
    )
    doc = documents_store.get_document("pdf-2")
    assert doc["status"] == "done"
    assert doc["pages"] == 10
    assert doc["text_chunks"] == 5
    assert doc["route_distribution"] == {"text": 8, "multimodal": 2}


def test_metadata_merge():
    _reset_docs()
    documents_store.add_document(
        doc_id="tab-1", kind="table",
        metadata={"rows": 12, "dataset_id": "table_tab-1"},
    )
    documents_store.update_document(
        "tab-1", status="active",
        metadata={"confirmed_schema": {"entity_col": "name"}},
    )
    doc = documents_store.get_document("tab-1")
    assert doc["rows"] == 12
    assert doc["dataset_id"] == "table_tab-1"
    assert doc["confirmed_schema"]["entity_col"] == "name"


def test_list_documents_order():
    _reset_docs()
    documents_store.add_document(doc_id="b", kind="pdf", started_at="2026-08-01 10:00:00")
    documents_store.add_document(doc_id="a", kind="pdf", started_at="2026-08-01 12:00:00")
    assert [d["doc_id"] for d in documents_store.list_documents()] == ["a", "b"]


def test_delete_document():
    _reset_docs()
    documents_store.add_document(doc_id="del", kind="pdf")
    assert documents_store.delete_document("del")
    assert documents_store.get_document("del") is None


def test_migrate_legacy_json():
    _reset_docs()
    legacy = {
        "pdf-doc": {
            "doc_name": "legacy.pdf", "kb_id": "default", "status": "done",
            "started_at": "2026-08-01 10:00:00", "pages": 5,
            "route_distribution": {"text": 4, "dual": 1},
            "text_chunks": 10, "image_pages": 1, "elapsed_sec": 12.3,
        },
        "table-doc": {
            "doc_name": "legacy.csv", "kb_id": "default", "kind": "table",
            "status": "active", "started_at": "2026-08-01 11:00:00",
            "table_path": "uploads/default/table-doc/table.csv",
            "dataset_id": "table_table-doc", "rows": 20, "cols": 4,
            "columns": ["a", "b", "c", "d"],
            "confirmed_schema": {"entity_col": "a"},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        legacy_path = Path(tmp) / "doc_status.json"
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        old_legacy = documents_store._LEGACY_STATUS_FILE
        documents_store._LEGACY_STATUS_FILE = legacy_path
        try:
            documents_store._migrate_legacy_json()
        finally:
            documents_store._LEGACY_STATUS_FILE = old_legacy
    pdf_doc = documents_store.get_document("pdf-doc")
    assert pdf_doc["kind"] == "pdf"
    assert pdf_doc["pages"] == 5
    assert pdf_doc["route_distribution"]["text"] == 4
    table_doc = documents_store.get_document("table-doc")
    assert table_doc["kind"] == "table"
    assert table_doc["rows"] == 20
    assert table_doc["dataset_id"] == "table_table-doc"
    assert table_doc["confirmed_schema"]["entity_col"] == "a"


def test_status_dict_shape():
    _reset_docs()
    documents_store.add_document(
        doc_id="shape", kind="table",
        metadata={
            "dataset_id": "table_shape", "rows": 7,
            "supports_timeline": True, "supports_recommendation": False,
        },
    )
    doc = documents_store.get_document("shape")
    assert doc["dataset_id"] == "table_shape"
    assert doc["rows"] == 7
    assert doc["supports_timeline"] is True
    assert doc["supports_recommendation"] is False
