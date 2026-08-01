# tests/test_hybrid.py
"""
HybridRetriever（src/retrieval/hybrid.py）纯单测：
fake 检索器验证 RRF 融合、page_id/doc_id 去重、sources/dataset_id 过滤，
不加载 SemArt、不调 LLM、不联网，秒级完成。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import HybridRetriever, _dedup, _rrf_fuse


def _hit(content, source="semart", score=0.9, **meta) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=score, metadata=meta)


class _FakeRetriever:
    """模拟 BaseRetriever 的最小实现。"""

    def __init__(self, source, results=(), dataset_id=None, fail=False):
        self.source = source
        self.dataset_id = dataset_id
        self._results = list(results)
        self.fail = fail

    def search(self, query, top_k=5, filters=None):
        if self.fail:
            raise RuntimeError("boom")
        return self._results[:top_k]


# ── RRF 融合 ─────────────────────────────────────────────────────
def test_rrf_single_source_preserves_order():
    a = [_hit(f"a{i}", score=0.9 - i * 0.1) for i in range(3)]
    fused = _rrf_fuse([a])
    assert [h.content for h in fused] == ["a0", "a1", "a2"]


def test_rrf_two_sources_interleave_by_rank():
    # A 源 [a0, a1]，B 源 [b0]：a0 与 b0 同 rank 0（同分，a0 先出现排前），a1 垫底
    a = [_hit("a0"), _hit("a1")]
    b = [_hit("b0", source="user_pdf_text")]
    fused = _rrf_fuse([a, b])
    assert [h.content for h in fused] == ["a0", "b0", "a1"]


def test_rrf_head_of_second_source_can_outrank_tail_of_first():
    # A 源只有一个低排名长尾结果 b0 应排在 a1 前（rank 0 > rank 1）
    a = [_hit("a0"), _hit("a1"), _hit("a2")]
    b = [_hit("b0", source="user_pdf_text")]
    fused = _rrf_fuse([a, b])
    assert [h.content for h in fused][1] == "b0"


# ── page_id/doc_id 去重 ──────────────────────────────────────────
def test_dedup_image_dropped_when_text_same_page():
    hits = [
        _hit("page1-text", source="user_pdf_text", page_id="d1-p3"),
        _hit("page1-image", source="user_pdf_image", page_id="d1-p3"),
        _hit("page2-text", source="user_pdf_text", page_id="d1-p4"),
    ]
    out = _dedup(hits)
    assert [h.content for h in out] == ["page1-text", "page2-text"]


def test_dedup_image_kept_when_no_text_sibling():
    # 无文字 chunk 的同页，整页图作为兜底证据保留
    hits = [
        _hit("page1-image", source="user_pdf_image", page_id="d1-p3"),
        _hit("page2-text", source="user_pdf_text", page_id="d1-p4"),
    ]
    assert len(_dedup(hits)) == 2


def test_dedup_multiple_text_chunks_same_page_kept():
    # 同页多个文字 chunk 内容不同、互不冲突，全部保留
    hits = [
        _hit("chunk-a", source="user_pdf_text", page_id="d1-p3"),
        _hit("chunk-b", source="user_pdf_text", page_id="d1-p3"),
    ]
    assert len(_dedup(hits)) == 2


def test_dedup_ignores_results_without_keys():
    # SemArt 行无 page_id：不参与去重，全部保留
    hits = [_hit("s1", title="T1"), _hit("s2", title="T2")]
    assert len(_dedup(hits)) == 2


# ── HybridRetriever.search ───────────────────────────────────────
def test_search_single_source_passthrough_order():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("a"), _hit("b"), _hit("c")]))
    out = h.search("q", top_k=5)
    assert [x.content for x in out] == ["a", "b", "c"]


def test_search_top_k_truncates():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit(f"a{i}") for i in range(5)]))
    assert len(h.search("q", top_k=2)) == 2


def test_search_sources_filter():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("s")]))
    h.register("user_pdf_text", _FakeRetriever("user_pdf_text", [_hit("p", source="user_pdf_text")]))
    out = h.search("q", top_k=5, sources=["semart"])
    assert [x.content for x in out] == ["s"]


def test_search_dataset_id_filter():
    h = HybridRetriever()
    h.register("semart", _FakeRetriever("semart", [_hit("s")], dataset_id="semart"))
    h.register("user_table", _FakeRetriever("user_table", [_hit("t", source="user_table")], dataset_id="my_table"))
    out = h.search("q", top_k=5, dataset_id="my_table")
    assert [x.content for x in out] == ["t"]


def test_search_source_failure_tolerated():
    h = HybridRetriever()
    h.register("bad", _FakeRetriever("bad", fail=True))
    h.register("semart", _FakeRetriever("semart", [_hit("ok")]))
    out = h.search("q", top_k=5)
    assert [x.content for x in out] == ["ok"]


def test_search_empty_when_no_sources():
    assert HybridRetriever().search("q") == []


def test_search_dedup_across_sources():
    h = HybridRetriever()
    h.register("user_pdf_text", _FakeRetriever("user_pdf_text", [_hit("text", source="user_pdf_text", page_id="d1-p3")]))
    h.register("user_pdf_image", _FakeRetriever("user_pdf_image", [_hit("image", source="user_pdf_image", page_id="d1-p3")]))
    out = h.search("q", top_k=5)
    # 双路线同页命中：整页图被丢弃，只留文字 chunk
    assert [x.content for x in out] == ["text"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 hybrid 全部 {len(fns)} 个单测通过！")
