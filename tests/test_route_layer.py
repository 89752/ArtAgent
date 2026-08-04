"""§6.3 路由决策层 + P0-4 检索过滤纯单测（无 LLM / 无网络）。"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes.common import _prefilter_route
from src.agent.state import AgentState
from src.retrieval.structured_retriever import _metadata_hit_filters
from src.tools.retrieval import _result_hit_filters


def test_prefilter_direct_greeting_knowledge_arithmetic():
    assert _prefilter_route("你好") == ("direct", "prefilter:寒暄/常识定义/算术")
    assert _prefilter_route("谢谢")[0] == "direct"
    assert _prefilter_route("1+1等于几")[0] == "direct"
    assert _prefilter_route("什么是线性透视")[0] == "direct"
    # 介绍人物应走检索（exact_lookup/query_painter_knowledge），不做 direct 硬答
    assert _prefilter_route("介绍一下莫奈") is None
    assert _prefilter_route("介绍下梵高") is None


def test_prefilter_web_time_sensitive():
    assert _prefilter_route("今天北京天气怎么样") == ("web", "prefilter:时效/实时信息")
    assert _prefilter_route("最近有什么艺术新闻")[0] == "web"


def test_prefilter_comparison_strong_verb_only():
    assert _prefilter_route("对比莫奈和梵高的色彩") == ("comparison", "prefilter:强比较动词")
    # "有什么区别"不强短路（领域比较 vs 常识区别交给 LLM）
    assert _prefilter_route("巴洛克和洛可可的装饰风格有什么不同") is None
    assert _prefilter_route("油画颜料和丙烯颜料有什么区别") is None


def test_prefilter_timeline_and_recommendation():
    assert _prefilter_route("梳理伦勃朗的风格演变")[0] == "timeline"
    assert _prefilter_route("推荐类似卡拉瓦乔的画家")[0] == "recommendation"


def test_classify_intent_returns_route_from_llm():
    from src.agent.nodes.common import classify_intent

    with patch(
        "src.agent.intent_tree.classify_intents",
        return_value=([], "comparison", "comparison", "fake-reason"),
    ):
        res = classify_intent(AgentState(user_query="莫奈和梵高在色彩上有什么不同"))
    assert res["route"] == "comparison"
    assert res["intent"] == "comparison"
    assert res["route_reason"] == "fake-reason"


def test_classify_intent_prefilter_wins_without_llm():
    from src.agent.nodes.common import classify_intent

    res = classify_intent(AgentState(user_query="今天北京天气怎么样"))
    assert res["route"] == "web"


def test_result_hit_filters():
    docs = [
        {"title": "A", "author": "Claude Monet", "school": "Impressionism", "timeframe": "1851-1900"},
        {"title": "B", "author": "Rembrandt", "school": "Baroque", "timeframe": "1601-1700"},
    ]
    assert _result_hit_filters(docs[0], {"author": "monet"})
    assert not _result_hit_filters(docs[1], {"school": "impressionism"})
    assert _result_hit_filters(docs[0], {"timeframe": "1851"})


def test_metadata_hit_filters_aliases():
    meta = {"artist": "Claude Monet", "movement": "Impressionism", "year_bucket": "1851-1900"}
    assert _metadata_hit_filters(meta, {"author": "monet"})
    assert _metadata_hit_filters(meta, {"school": "impression"})
    assert not _metadata_hit_filters(meta, {"timeframe": "1901"})


def test_semantic_search_filters_source_and_fetches_more():
    from src.tools.retrieval import semantic_search

    captured = {}

    class _Hybrid:
        active_dataset = "core"

        def search(self, query, top_k=5, dataset_id=None, sources=None, rerank=None, filters=None):
            captured.update({"top_k": top_k, "sources": sources, "filters": filters})
            return []

    with patch("src.retrieval.hybrid.get_hybrid_retriever", return_value=_Hybrid()):
        semantic_search.invoke(
            {"query": "水景", "top_k": 3, "filters": {"author": "Monet", "source": "core"}}
        )
    assert captured["sources"] == ["core"]
    assert captured["filters"] is None  # 结构化过滤在工具层后置执行
    assert captured["top_k"] > 3  # 带过滤时多取候选


def test_build_graph_has_route_branch():
    from src.agent.graph import build_graph

    g = build_graph()
    assert g is not None
