# tests/test_reranker.py
"""
qwen3-rerank 精排（src/retrieval/reranker.py + hybrid.py 精排段）纯单测：
mock requests.post / rerank 函数验证成功解析、失败降级、槽位保持与开关逻辑，
不加载 SemArt、不调 LLM、不联网，秒级完成。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.base import RetrievalResult
from src.retrieval import reranker as reranker_mod
from src.retrieval.hybrid import (
    HybridRetriever,
    _rerank_enabled,
    _rerank_fused,
    RERANK_POOL,
)


def _hit(content, source="semart", score=0.9, **meta) -> RetrievalResult:
    return RetrievalResult(content=content, source=source, score=score, metadata=meta)


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _PostRecorder:
    """替换 requests.post：记录 payload，按预设行为返回/抛错（可按 URL 分流）。"""

    def __init__(self, payload=None, error=None, by_url=None):
        self.payload = payload or {"results": []}
        self.error = error
        self.by_url = by_url or {}  # {url片段: payload 或 Exception}
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        for frag, behavior in self.by_url.items():
            if frag in url:
                if isinstance(behavior, Exception):
                    raise behavior
                return _FakeResp(behavior)
        if self.error:
            raise self.error
        return _FakeResp(self.payload)


def _patch(module, name, value):
    """with 风格的极简 patch：返回原值，调用方负责 finally 恢复。"""
    old = getattr(module, name)
    setattr(module, name, value)
    return old


# ── reranker.rerank ──────────────────────────────────────────────
def test_empty_documents_returns_empty_without_call():
    rec = _PostRecorder()
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        assert reranker_mod.rerank("q", []) == []
        assert rec.calls == []  # 空候选不发请求
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_unavailable_without_api_key():
    old_env = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = ""
    try:
        assert reranker_mod.rerank("q", ["doc"]) is None
    finally:
        if old_env is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_env


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
        ranked = reranker_mod.rerank("q", ["a", "b", "c"])
        assert ranked == [(0, 0.9), (1, 0.7), (2, 0.5)]
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_top_n_capped_by_document_count():
    rec = _PostRecorder({"results": [{"index": 0, "relevance_score": 1.0}]})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a", "b"], top_n=10)
        assert rec.calls[0]["json"]["top_n"] == 2  # top_n 不得超过候选数
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


def test_instruct_passed_through():
    rec = _PostRecorder({"results": []})
    old = _patch(reranker_mod.requests, "post", rec)
    try:
        reranker_mod.rerank("q", ["a"], instruct="按艺术史相关性排序")
        assert rec.calls[0]["json"]["instruct"] == "按艺术史相关性排序"
    finally:
        _patch(reranker_mod.requests, "post", old)


def test_http_error_retries_then_returns_none():
    rec = _PostRecorder(error=RuntimeError("boom"))
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_sleep = _patch(reranker_mod.time, "sleep", lambda s: None)
    try:
        assert reranker_mod.rerank("q", ["a"]) is None
        # 主模型（首次+MAX_RETRIES 重试）+ 后备模型（同次数），双失败才降级
        assert len(rec.calls) == (reranker_mod.MAX_RETRIES + 1) * 2
    finally:
        _patch(reranker_mod.requests, "post", old_post)
        _patch(reranker_mod.time, "sleep", old_sleep)


# ── 双端点与主备接力（2026-08-01：gte-rerank-v2 实测未下线）─────────
def test_fallback_model_takes_over_on_primary_failure():
    native_payload = {"output": {"results": [
        {"index": 1, "relevance_score": 0.8},
        {"index": 0, "relevance_score": 0.3},
    ]}}
    rec = _PostRecorder(by_url={
        "compatible-api": RuntimeError("403 FreeTierOnly"),
        "services/rerank": native_payload,
    })
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_sleep = _patch(reranker_mod.time, "sleep", lambda s: None)
    try:
        ranked = reranker_mod.rerank("q", ["a", "b"])
        assert ranked == [(1, 0.8), (0, 0.3)]  # 原生端点结果解析并按分降序
        urls = [c["url"] for c in rec.calls]
        assert any("services/rerank" in u for u in urls)  # 确实接力到了原生端点
    finally:
        _patch(reranker_mod.requests, "post", old_post)
        _patch(reranker_mod.time, "sleep", old_sleep)


def test_native_model_routes_to_native_endpoint():
    payload = {"output": {"results": [{"index": 0, "relevance_score": 1.0}]}}
    rec = _PostRecorder(payload)
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_model = _patch(reranker_mod, "RERANK_MODEL", "gte-rerank-v2")
    try:
        ranked = reranker_mod.rerank("q", ["a"])
        assert ranked == [(0, 1.0)]
        assert "services/rerank" in rec.calls[0]["url"]  # 按模型名自动选端点
        # 原生报文形状：input 包 query/documents，parameters 包 top_n
        assert rec.calls[0]["json"]["input"]["documents"] == ["a"]
        assert rec.calls[0]["json"]["parameters"]["top_n"] == 1
    finally:
        _patch(reranker_mod.requests, "post", old_post)
        _patch(reranker_mod, "RERANK_MODEL", old_model)


def test_fallback_skipped_when_same_as_primary():
    rec = _PostRecorder(error=RuntimeError("boom"))
    old_post = _patch(reranker_mod.requests, "post", rec)
    old_sleep = _patch(reranker_mod.time, "sleep", lambda s: None)
    old_fb = _patch(reranker_mod, "RERANK_FALLBACK_MODEL", reranker_mod.RERANK_MODEL)
    try:
        assert reranker_mod.rerank("q", ["a"]) is None
        # 主备同模型时不重复烧调用：只打一轮（首次+重试）
        assert len(rec.calls) == reranker_mod.MAX_RETRIES + 1
    finally:
        _patch(reranker_mod.requests, "post", old_post)
        _patch(reranker_mod.time, "sleep", old_sleep)
        _patch(reranker_mod, "RERANK_FALLBACK_MODEL", old_fb)


# ── hybrid._rerank_enabled ───────────────────────────────────────
def test_rerank_enabled_override_wins():
    os.environ["RERANK_ENABLED"] = "0"
    try:
        assert _rerank_enabled(True) is True  # 显式参数压过 env
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


# ── hybrid._rerank_fused ─────────────────────────────────────────
def test_pool_not_larger_than_top_k_skips_rerank():
    def _boom(*a, **kw):
        raise AssertionError("候选不足时不应调用 rerank")

    old = _patch(reranker_mod, "rerank", _boom)
    try:
        pool = [_hit("a"), _hit("b")]
        out = _rerank_fused("q", pool, top_k=5)
        assert [h.content for h in out] == ["a", "b"]
    finally:
        _patch(reranker_mod, "rerank", old)


def test_text_slots_reordered_image_keeps_slot():
    # 槽位 [semart, image, semart]：文本两位逆序，整页图原地不动
    pool = [
        _hit("t0", source="semart"),
        _hit("img", source="user_pdf_image", page_id="d-p1"),
        _hit("t1", source="semart"),
    ]

    def _fake_rerank(query, documents, top_n=None, instruct=None):
        assert documents == ["t0", "t1"]  # 整页图不送精排
        return [(1, 0.95), (0, 0.60)]  # t1 更相关

    old = _patch(reranker_mod, "rerank", _fake_rerank)
    try:
        out = _rerank_fused("q", pool, top_k=1)
        assert [h.content for h in out] == ["t1", "img", "t0"]
        assert out[0].metadata["rerank_score"] == 0.95  # 分数写进 metadata
        assert "rerank_score" not in out[1].metadata  # 整页图无精排分
    finally:
        _patch(reranker_mod, "rerank", old)


def test_rerank_failure_returns_coarse_order():
    old = _patch(reranker_mod, "rerank", lambda *a, **kw: None)
    try:
        pool = [_hit(f"t{i}") for i in range(6)]
        out = _rerank_fused("q", pool, top_k=2)
        assert [h.content for h in out] == [f"t{i}" for i in range(6)]
    finally:
        _patch(reranker_mod, "rerank", old)


def test_partial_rerank_response_returns_coarse_order():
    # API 少返回一个：按槽位重排会复制文档，必须整体回退原序
    old = _patch(reranker_mod, "rerank", lambda *a, **kw: [(2, 0.9), (0, 0.5)])
    try:
        pool = [_hit(f"t{i}") for i in range(4)]
        out = _rerank_fused("q", pool, top_k=1)
        assert [h.content for h in out] == [f"t{i}" for i in range(4)]
    finally:
        _patch(reranker_mod, "rerank", old)


# ── HybridRetriever.search 精排集成 ──────────────────────────────
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
    assert r.seen_top_k == [3]  # 不开精排不多取
    assert len(out) == 3


def test_search_rerank_on_fetches_pool():
    r = _RecordingRetriever("semart", n=RERANK_POOL)
    h = HybridRetriever()
    h.register("semart", r)

    def _fake_rerank(query, documents, top_n=None, instruct=None):
        # 把最后一个候选提到最前
        return [(len(documents) - 1, 0.99)] + [(i, 0.5) for i in range(len(documents) - 1)]

    old = _patch(reranker_mod, "rerank", _fake_rerank)
    try:
        out = h.search("q", top_k=2, rerank=True)
        assert r.seen_top_k == [RERANK_POOL]  # 开精排按池大小取候选
        assert out[0].content == f"semart{RERANK_POOL - 1}"  # 逆序末位被精排提到第一
        assert len(out) == 2
    finally:
        _patch(reranker_mod, "rerank", old)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 reranker 全部 {len(fns)} 个单测通过！")
