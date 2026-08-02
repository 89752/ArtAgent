# tests/test_table_ingest.py
"""
Stage 5 结构化表格上传纯单测：
文件类型路由 / 多 sheet 选择 / 编码兜底 / schema 推断（fake LLM）/
确认注册流程（隔离状态存储与 Hybrid 单例）/ 能力开关 / 空角色守卫。
不加载 SemArt、不调真实 LLM、不联网，秒级完成。
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data import documents_store
from src.ingestion import pipeline as pipeline_mod
from src.ingestion import table_pipeline as tp
from src.ingestion.schema_inference import infer_table_schema
from src.ingestion.table_loader import (
    _effective_columns,
    _sheet_score,
    classify_upload,
    load_table,
)
from src.retrieval import hybrid as hybrid_mod
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.structured_retriever import (
    StructuredTableRetriever,
    TableSchema,
    _REGISTRY,
)

FIXTURES = Path(__file__).parent / "fixtures"
BOOKS_CSV = FIXTURES / "plain_list_books.csv"
PLAN_XLSX = FIXTURES / "两周Python算法密集学习计划.xlsx"


class _FakeLLM:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def _tmpdir():
    return Path(tempfile.mkdtemp(prefix="t5_"))


def _cleanup(*paths: Path):
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass  # 沙箱 safe-delete 可能拦截，残留无害


def _isolate_state(tmp: Path):
    """把 SQLite 状态库与 uploads 指到临时目录，返回恢复函数。"""
    old_db = documents_store.DB_PATH
    old_uploads = tp.UPLOADS_DIR
    db_path = tmp / "documents.db"
    documents_store._reset_for_tests(db_path)
    documents_store.init_db()
    tp.UPLOADS_DIR = tmp / "uploads"
    return lambda: (
        setattr(documents_store, "DB_PATH", old_db),
        setattr(tp, "UPLOADS_DIR", old_uploads),
    )


def _fresh_hybrid():
    """把全局 Hybrid 单例换成新实例（测试持有引用），返回 (实例, 恢复函数)。"""
    fresh = HybridRetriever()
    old = hybrid_mod.get_hybrid_retriever
    hybrid_mod.get_hybrid_retriever = lambda: fresh
    return fresh, lambda: setattr(hybrid_mod, "get_hybrid_retriever", old)


# ── 文件类型路由 ─────────────────────────────────────────────────
def test_classify_upload_routing():
    assert classify_upload("a.pdf") == "pdf"
    assert classify_upload("b.CSV") == "table"
    assert classify_upload("c.xlsx") == "table"
    assert classify_upload("d.xls") == "table"
    assert classify_upload("e.txt") is None
    assert classify_upload("") is None


# ── 表格加载 ─────────────────────────────────────────────────────
def test_load_csv_fixture():
    loaded = load_table(str(BOOKS_CSV))
    assert loaded.sheet_name == ""
    assert len(loaded.df) == 12
    assert loaded.columns == ["书名", "作者", "出版社", "页数", "价格"]


def test_load_xlsx_picks_biggest_data_sheet():
    if not PLAN_XLSX.exists():
        return  # fixture 不在时跳过（不影响其余断言）
    loaded = load_table(str(PLAN_XLSX))
    assert loaded.sheet_name == "每日打卡表"  # 说明页/汇总页不应被选中
    assert len(loaded.df) == 14
    assert "学习主题" in loaded.columns


def test_sheet_score_ignores_unnamed_and_empty_cols():
    df = pd.DataFrame({"a": [1, 2], "Unnamed: 1": [None, None], "b": [3, 4]})
    assert _effective_columns(df) == ["a", "b"]
    assert _sheet_score(df) == 2 * 2


def test_csv_gbk_encoding_fallback():
    tmp = _tmpdir()
    try:
        p = tmp / "gbk.csv"
        p.write_bytes("书名,作者\n红楼梦,曹雪芹\n".encode("gbk"))
        loaded = load_table(str(p))
        assert loaded.df.iloc[0]["书名"] == "红楼梦"
    finally:
        _cleanup(tmp)


def test_load_table_rejects_thin_table():
    tmp = _tmpdir()
    try:
        p = tmp / "thin.csv"
        p.write_text("只有一列\n值1\n值2\n", encoding="utf-8")
        try:
            load_table(str(p))
            raise AssertionError("单列无表头表应报 ValueError")
        except ValueError:
            pass
    finally:
        _cleanup(tmp)


# ── schema 推断（fake LLM）───────────────────────────────────────
def _books_df():
    return pd.read_csv(BOOKS_CSV)


def test_infer_full_roles():
    llm = _FakeLLM(json.dumps({
        "entity_col": "书名", "group_axis_col": None,
        "description_col": "备注", "image_col": None,
        "display_name": "书单", "reasoning": "书名是实体",
    }, ensure_ascii=False))
    # "备注" 不在列里 → 应被置空；"书名" 存在 → 保留
    out = infer_table_schema(_books_df(), "books.csv", llm=llm)
    assert out.entity_col == "书名"
    assert out.description_col == ""  # 猜了不存在的列 → 置空
    assert out.group_axis_col is None
    assert out.display_name == "书单"


def test_infer_supports_flags():
    llm = _FakeLLM(json.dumps({
        "entity_col": "书名", "group_axis_col": "出版社",
        "description_col": "书名", "image_col": None,
    }, ensure_ascii=False))
    schema = infer_table_schema(_books_df(), "x", llm=llm).to_table_schema()
    assert schema.supports_timeline is True
    assert schema.supports_recommendation is True


def test_infer_all_null_means_unsupported():
    llm = _FakeLLM('{"entity_col": null, "group_axis_col": null, "description_col": null, "image_col": null}')
    schema = infer_table_schema(_books_df(), "x", llm=llm).to_table_schema()
    assert schema.supports_timeline is False
    assert schema.supports_recommendation is False


def test_infer_markdown_wrapped_json():
    llm = _FakeLLM("```json\n" + json.dumps({"entity_col": "作者"}) + "\n```")
    out = infer_table_schema(_books_df(), "x", llm=llm)
    assert out.entity_col == "作者"


def test_infer_llm_failure_returns_empty():
    out = infer_table_schema(_books_df(), "x", llm=_FakeLLM("", error=RuntimeError("boom")))
    assert out.entity_col == "" and "失败" in out.reasoning  # 不抛异常，允许手填


# ── 入库 → 确认 → 注册 ───────────────────────────────────────────
def _ingest_books(tmp, llm=None):
    csv_copy = tmp / "books.csv"
    shutil.copy(BOOKS_CSV, csv_copy)
    llm = llm or _FakeLLM(json.dumps({
        "entity_col": "书名", "group_axis_col": None,
        "description_col": None, "image_col": None,
        "display_name": "我的书单", "reasoning": "x",
    }, ensure_ascii=False))
    return tp.ingest_table(str(csv_copy), "doc1", doc_name="books.csv", llm=llm)


def test_ingest_then_pending_not_registered():
    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    fresh, restore_h = _fresh_hybrid()
    try:
        summary = _ingest_books(tmp)
        assert summary["status"] == "pending_confirm"
        assert summary["dataset_id"] == "table_doc1"
        assert summary["proposed_schema"]["entity_col"] == "书名"
        assert "table_doc1" not in _REGISTRY  # 确认前不注册
        assert "table_doc1" not in fresh._retrievers
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


def test_confirm_registers_and_activates():
    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    fresh, restore_h = _fresh_hybrid()
    try:
        _ingest_books(tmp)
        out = tp.confirm_table_schema("doc1", {
            "entity_col": "书名", "group_axis_col": "出版社",
            "description_col": None, "image_col": None, "display_name": "书单",
        })
        assert out["status"] == "active"
        r = _REGISTRY["table_doc1"]
        assert r.schema.entity_col == "书名"
        assert r.schema.group_axis_col == "出版社"
        assert r.schema.supports_timeline is True
        assert r.schema.supports_recommendation is False  # 无描述列
        assert "table_doc1" in fresh._retrievers  # 已挂进 Hybrid
        # df_loader 懒加载真能读出数据
        assert len(r.df) == 12
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


def test_confirm_requires_entity_col():
    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    _fresh, restore_h = _fresh_hybrid()
    try:
        _ingest_books(tmp)
        try:
            tp.confirm_table_schema("doc1", {"entity_col": ""})
            raise AssertionError("空 entity_col 应报 ValueError")
        except ValueError:
            pass
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


def test_confirm_rejects_unknown_column():
    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    _fresh, restore_h = _fresh_hybrid()
    try:
        _ingest_books(tmp)
        try:
            tp.confirm_table_schema("doc1", {"entity_col": "不存在的列"})
            raise AssertionError("幻觉列应报 ValueError")
        except ValueError:
            pass
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


def test_unregister_and_restore():
    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    fresh, restore_h = _fresh_hybrid()
    try:
        _ingest_books(tmp)
        tp.confirm_table_schema("doc1", {"entity_col": "书名"})
        tp.unregister_table("table_doc1")
        assert "table_doc1" not in _REGISTRY and "table_doc1" not in fresh._retrievers
        # 重启恢复：从状态存储重建
        n = tp.restore_active_tables()
        assert n == 1
        assert "table_doc1" in _REGISTRY and "table_doc1" in fresh._retrievers
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


# ── 能力开关（Stage 2 预留在 Stage 5 真正生效）─────────────────────
def test_capability_gate_downgrades_unsupported_table():
    from src.agent.graph import _capability_supported

    tmp = _tmpdir()
    restore = _isolate_state(tmp)
    _fresh, restore_h = _fresh_hybrid()
    try:
        _ingest_books(tmp)
        tp.confirm_table_schema("doc1", {"entity_col": "书名"})  # 无轴无描述
        assert _capability_supported("timeline", "table_doc1") is False
        assert _capability_supported("recommendation", "table_doc1") is False
        assert _capability_supported("timeline", "semart") is True  # SemArt 不受影响
        assert _capability_supported("timeline", "table_未注册") is False  # 未注册也降级
    finally:
        restore(); restore_h(); _REGISTRY.pop("table_doc1", None); _cleanup(tmp)


# ── 空角色守卫（负样本表不炸）─────────────────────────────────────
def test_empty_roles_do_not_crash_search():
    df = pd.DataFrame({"名称": ["甲", "乙"], "数值": [1, 2]})
    r = StructuredTableRetriever(
        "t_empty", TableSchema(entity_col="", description_col=""), df=df,
    )
    assert r.search("甲", top_k=3) == []  # 两路皆空 → 空结果而非 KeyError
    assert len(r.exclude_by_entity(["甲"])) == 2  # 空实体列 → 原样返回


# ── 第三级词重叠打分（recommendation 长特征 query 的救命路径）────────
def _overlap_retriever():
    df = pd.DataFrame({
        "NAME": ["A 画家", "B 画家", "C 画家"],
        "DESC": [
            "bold vivid colors and thick impasto brushwork",
            "quiet ink wash landscape with sparse composition",
            "emotional intensity with swirling dynamic movement",
        ],
    })
    return StructuredTableRetriever(
        "t_ov", TableSchema(entity_col="NAME", description_col="DESC"), df=df,
    )


def test_word_overlap_scores_long_feature_query():
    # 模拟 recommendation 的 extracted_features 长描述：整串包含必空，
    # 词重叠应把 bold colors/impasto 的 A 排第一
    hits = _overlap_retriever().search(
        "Intense saturated colors, heavy impasto brushstrokes, high emotional "
        "intensity, dynamic swirling movement, expressive style",
        top_k=3,
    )
    assert [h.metadata["name"] for h in hits] == ["C 画家", "A 画家"]
    assert hits[0].score > hits[1].score > 0  # 命中率递减且非扁平 1.0


def test_word_overlap_returns_empty_when_no_match():
    hits = _overlap_retriever().search("quantum chromodynamics lattice gauge", top_k=3)
    assert hits == []  # 零命中 → 空结果（下游如实说"匹配有限"而非乱塞）


# ── user_table 结果形状（消费方契约）──────────────────────────────
def test_user_table_result_shape():
    from src.tools.retrieval import _format_result

    df = pd.read_csv(BOOKS_CSV)
    schema = TableSchema(entity_col="书名", description_col="")
    r = StructuredTableRetriever("table_t", schema, df=df)
    _REGISTRY["table_t"] = r  # _format_result 要查 schema
    try:
        hits = r.search("算法导论", top_k=1)
        assert len(hits) == 1
        out = _format_result(hits[0])
        assert out["source"] == "user_table"  # 带 source 键 → 不进配图卡片
        assert out["title"] == "算法导论"
        assert "书名" in out  # 原始列小写键在——recommendation 排除靠它定位
        assert out["description_snippet"]  # 相关性过滤拼候选靠它
        # exclude_from_results 在格式化字典上真能用
        kept = r.exclude_from_results([out], ["算法导论"])
        assert kept == []
    finally:
        _REGISTRY.pop("table_t", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [OK] {fn.__name__}")
    print(f"\n[PASS] table_ingest 全部 {len(fns)} 个单测通过！")
