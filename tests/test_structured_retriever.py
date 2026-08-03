# tests/test_structured_retriever.py
"""
结构化表检索器（src/retrieval/structured_retriever.py）纯单测：
构造小 DataFrame 与 fake 向量集合，不加载 SemArt、不调 LLM、不联网，秒级完成。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.structured_retriever import (
    CORE_SCHEMA,
    TableSchema,
    StructuredTableRetriever,
    get_structured_retriever,
    register_structured_dataset,
)

SCHEMA = TableSchema(
    entity_col="AUTHOR",
    group_axis_col="TIMEFRAME",
    description_col="DESCRIPTION",
    image_col="IMAGE_FILE",
)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"TITLE": "Irises", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "1851-1900",
             "DESCRIPTION": "Van Gogh's irises study", "IMAGE_FILE": "irises.jpg"},
            {"TITLE": "The Starry Night", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "1851-1900",
             "DESCRIPTION": "Swirling night sky", "IMAGE_FILE": "starry.jpg"},
            {"TITLE": "Sunflowers", "AUTHOR": "GOGH, Vincent van", "TIMEFRAME": "",
             "DESCRIPTION": "Yellow still life", "IMAGE_FILE": "sun.jpg"},
            {"TITLE": "Water Lilies", "AUTHOR": "MONET, Claude", "TIMEFRAME": "1901-1950",
             "DESCRIPTION": "Pond reflections", "IMAGE_FILE": "lilies.jpg"},
        ]
    )


def _retriever(**kwargs) -> StructuredTableRetriever:
    return StructuredTableRetriever("test_ds", SCHEMA, df=_df(), source="semart", **kwargs)


# ── TableSchema 能力推导 ─────────────────────────────────────────
def test_core_schema_capabilities_full():
    assert CORE_SCHEMA.supports_timeline is True
    assert CORE_SCHEMA.supports_recommendation is True


def test_schema_no_axis_no_timeline():
    schema = TableSchema(entity_col="NAME", description_col="BIO")
    assert schema.supports_timeline is False
    assert schema.supports_recommendation is True  # 实体+描述列齐全即可


# ── group_by_axis（timeline 用） ─────────────────────────────────
def test_group_by_axis_sorted_and_unknown_dropped():
    groups = _retriever().group_by_axis("Van Gogh")
    # "Sunflowers" 无 TIMEFRAME → 存在真实分组时 Unknown 组被丢弃
    assert list(groups.keys()) == ["1851-1900"]
    assert len(groups["1851-1900"]) == 2


def test_group_by_axis_only_unknown_kept():
    df = _df()
    df["TIMEFRAME"] = ""
    r = StructuredTableRetriever("t_only_unknown", SCHEMA, df=df)
    groups = r.group_by_axis("Van Gogh")
    assert list(groups.keys()) == ["Unknown"]
    assert len(groups["Unknown"]) == 3


def test_group_by_axis_entity_not_found():
    assert _retriever().group_by_axis("zzz-nobody") == {}


def test_group_by_axis_without_axis_col():
    schema = TableSchema(entity_col="AUTHOR", description_col="DESCRIPTION")
    r = StructuredTableRetriever("t_no_axis", schema, df=_df())
    assert r.group_by_axis("Van Gogh") == {}


def test_group_by_axis_sorted_multiple_groups():
    df = _df()
    df.loc[3, "TIMEFRAME"] = "1801-1850"  # Monet 挪到更早的分组
    df.loc[0, "AUTHOR"] = "MONET, Claude"
    r = StructuredTableRetriever("t_multi", SCHEMA, df=df)
    groups = r.group_by_axis("Monet")
    assert list(groups.keys()) == ["1801-1850", "1851-1900"]


# ── exclude_by_entity / exclude_from_results（recommendation 用） ──
def test_exclude_by_entity_dataframe():
    out = _retriever().exclude_by_entity(["Vincent van Gogh"])
    assert len(out) == 1 and out.iloc[0]["AUTHOR"] == "MONET, Claude"


def test_exclude_by_entity_empty_names_returns_all():
    assert len(_retriever().exclude_by_entity([])) == 4


def test_exclude_by_entity_short_tokens_ignored():
    # 长度 ≤ 2 的词不作排除 token（"AI" 不应排除任何行）
    assert len(_retriever().exclude_by_entity(["AI"])) == 4


def test_exclude_from_results_dicts():
    results = [
        {"title": "Irises", "author": "GOGH, Vincent van"},
        {"title": "Water Lilies", "author": "MONET, Claude"},
    ]
    out = _retriever().exclude_from_results(results, ["Van Gogh"])
    assert [r["title"] for r in out] == ["Water Lilies"]


def test_exclude_from_results_empty_names():
    results = [{"title": "Irises", "author": "GOGH, Vincent van"}]
    assert _retriever().exclude_from_results(results, []) == results


# ── search：fuzzy 兜底路径（无向量集合） ─────────────────────────
def test_fuzzy_search_by_entity():
    hits = _retriever().search("Monet", top_k=5)
    assert len(hits) == 1
    h = hits[0]
    assert h.source == "semart"
    assert h.score == 1.0
    assert h.metadata["dataset_id"] == "test_ds"
    assert h.metadata["title"] == "Water Lilies"
    assert h.content == "Pond reflections"
    assert h.image_refs == ["lilies.jpg"]


def test_fuzzy_search_description_fallback():
    # 实体列匹配不到时，退化描述列包含匹配
    hits = _retriever().search("night sky", top_k=5)
    assert len(hits) == 1 and hits[0].metadata["title"] == "The Starry Night"


def test_fuzzy_search_top_k():
    hits = _retriever().search("Van Gogh", top_k=2)
    assert len(hits) == 2


def test_fuzzy_search_filters_equality():
    hits = _retriever().search("Van Gogh", top_k=5, filters={"TIMEFRAME": "1851-1900"})
    assert {h.metadata["title"] for h in hits} == {"Irises", "The Starry Night"}


def test_fuzzy_search_no_match():
    assert _retriever().search("zzz-nothing") == []


# ── search：向量路径（fake collection + fake embed_fn） ──────────
class _FakeCollection:
    """模拟 Chroma collection 的最小接口。"""

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
        "t_vec", SCHEMA, df=_df(), source="semart",
        collection_loader=lambda: collection,
        embed_fn_loader=lambda: (lambda text: [0.1, 0.2, 0.3]),
    )
    return r, collection


def test_vector_search_score_and_metadata():
    r, _ = _vector_retriever()
    hits = r.search("anything", top_k=2)
    assert len(hits) == 2
    assert hits[0].metadata["title"] == "Irises"
    assert hits[0].score == 0.8  # 1 - 0.2
    assert hits[0].metadata["dataset_id"] == "t_vec"
    assert hits[0].image_refs == ["irises.jpg"]


def test_vector_search_n_results_capped_by_count():
    r, collection = _vector_retriever(count=1)
    hits = r.search("anything", top_k=5)
    assert collection.last_n_results == 1  # min(top_k, collection.count())
    assert len(hits) == 1


# ── 懒加载与注册表 ───────────────────────────────────────────────
def test_lazy_df_loader():
    calls = []
    r = StructuredTableRetriever(
        "t_lazy", SCHEMA, df_loader=lambda: calls.append(1) or _df()
    )
    assert calls == []          # 构造不触发加载
    _ = r.df
    _ = r.df
    assert calls == [1]         # 首次访问加载并缓存


def test_registry_register_and_get():
    register_structured_dataset("t_reg", SCHEMA, df=_df())
    assert get_structured_retriever("t_reg").dataset_id == "t_reg"


def test_registry_unknown_dataset_raises():
    try:
        get_structured_retriever("zzz-unregistered")
    except KeyError:
        return
    raise AssertionError("未注册的数据源应抛 KeyError")


# ── 路由层能力开关（graph._capability_supported） ─────────────────
def test_capability_gate():
    from src.agent.graph import _capability_supported

    no_axis = TableSchema(entity_col="NAME", description_col="BIO")
    register_structured_dataset(
        "t_gate", no_axis, df=pd.DataFrame({"NAME": ["x"], "BIO": ["y"]})
    )
    assert _capability_supported("timeline", "t_gate") is False
    assert _capability_supported("recommendation", "t_gate") is True
    assert _capability_supported("timeline", "zzz-unregistered") is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n🎉 structured_retriever 全部 {len(fns)} 个单测通过！")
