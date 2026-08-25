"""作品检索工具（semantic_search / exact_lookup）。

提供两种检索方式：
  - semantic_search: 语义向量检索（用于模糊查询、主题检索）
  - exact_lookup:    精确字段查询（用于按画家/标题/年代精确查找）

semantic_search 改走检索抽象层（HybridRetriever）；融合用户上传 PDF 的
文字/整页图两路结果，按 source 分形状返回：
  - semart → 画作字典（title/author/date/...，画作形状保持一致）
  - user_pdf_text / user_pdf_image → 文档片段字典（带 doc_name/page/内容）
web/service.py 的 ToolMessage 解析与各合成节点不受影响（画作形状未变）。
数据过滤/格式化统一走 src/data/access.py 数据访问层。
"""

from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.data.access import (
    EVIDENCE_SNIPPET_LEN,
    fuzzy_match,
    hit_filters_match,
    row_to_artwork_dict,
)
from src.retrieval.base import RetrievalResult

load_dotenv()

DEFAULT_TOP_K = 5

def _format_result(result: RetrievalResult) -> dict:
    """格式化单条检索结果，供 Agent 消费（按数据源分形状）。"""
    if result.source == "semart":
        artwork = row_to_artwork_dict(result.metadata)
        artwork["relevance_score"] = round(result.score, 4)
        return artwork
    if result.source == "core":
        # 核心库（M3）：与 semart 同契约（title/author/date/...），不带 source 键
        # → 可进 UI 配图卡片；image_file 为 URL，由 _thumb_data_uri 直通
        meta = result.metadata
        snippet = str(meta.get("description") or result.content or "")
        if len(snippet) > EVIDENCE_SNIPPET_LEN:
            snippet = snippet[:EVIDENCE_SNIPPET_LEN] + "..."
        return {
            "title": str(meta.get("title") or ""),
            "author": str(meta.get("artist") or ""),
            "date": str(meta.get("year_display") or (meta.get("year") or "")),
            "technique": str(meta.get("material") or ""),
            "school": str(meta.get("movement") or meta.get("school") or meta.get("genre") or ""),
            "timeframe": str(meta.get("year_bucket") or ""),
            "image_file": str(meta.get("image_url") or ""),
            "description_snippet": snippet,
            "relevance_score": round(result.score, 4),
        }
    meta = result.metadata
    if result.source == "user_table":
        # 用户表格：原始列全带上（小写键），通用
        # title/description_snippet 供证据模板与相关性过滤拼候选；
        # 带 source 键 → 不进 UI 配图卡片（陷阱 #13）
        dataset_id = str(meta.get("dataset_id") or "")
        entity, desc, axis = "", "", ""
        try:
            from src.retrieval.structured_retriever import get_structured_retriever

            schema = get_structured_retriever(dataset_id).schema
            if schema.entity_col:
                entity = str(meta.get(schema.entity_col.lower()) or "")
            if schema.description_col:
                desc = str(meta.get(schema.description_col.lower()) or "")
            if schema.group_axis_col:
                axis = str(meta.get(schema.group_axis_col.lower()) or "")
        except Exception:  # 表已被注销等异常：退化为通用形状，不拖垮检索
            pass
        title = str(meta.get("title") or entity or "(未命名记录)")
        if axis:
            title += f"（{axis}）"
        snippet = desc or result.content or title
        if len(snippet) > EVIDENCE_SNIPPET_LEN:
            snippet = snippet[:EVIDENCE_SNIPPET_LEN] + "..."
        return {
            **{k: v for k, v in meta.items() if k != "dataset_id"},
            "source": result.source,
            "title": title,
            "content": result.content,
            "description_snippet": snippet,
            "dataset_id": dataset_id,
            "relevance_score": round(result.score, 4),
        }
    # 用户文档（PDF）：title 形如"《画册》第3页"，供证据模板与溯源引用
    title = f"《{meta.get('doc_name') or '用户文档'}》第{meta.get('page', '?')}页"
    section = str(meta.get("section") or "").strip()
    if section:  # 上下文头展示侧：章节进标题（旧文档无此字段自动跳过）
        title += f" · {section[:40]}"
    snippet = result.content
    if len(snippet) > EVIDENCE_SNIPPET_LEN:
        snippet = snippet[:EVIDENCE_SNIPPET_LEN] + "..."
    out = {
        "source": result.source,
        "title": title,
        "doc_id": meta.get("doc_id", ""),
        "doc_name": meta.get("doc_name", ""),
        "page": meta.get("page", 0),
        "block_type": meta.get("block_type", ""),
        "content": result.content,
        "description_snippet": snippet,
        "image_path": result.image_refs[0] if result.image_refs else "",
        "relevance_score": round(result.score, 4),
    }
    if result.source == "user_pdf_image":
        out["title"] = title + "（整页图）"
        # 提示 Agent：这页图可以用 read_page_image 真正读取内容
        out["read_hint"] = "调用 read_page_image(image_path=...) 可读取此页图片的文字与图面内容"
    return out


def _artwork_from_schema_row(schema, row: dict) -> dict:
    """按 TableSchema 角色把 df 行/元数据 dict 转成工具契约（title/author/date/...）。

    字段规整与描述截断统一走 src/data/access.row_to_artwork_dict，
    避免两套"行 → 工具字典"实现漂移。
    """
    def g(col: Optional[str]) -> str:
        return str(row.get(col) or "") if col else ""

    mapped = {
        "title": g(schema.title_col) or g("title"),
        "author": g(schema.entity_col),
        "date": g(schema.date_col),
        "technique": g(schema.technique_col),
        "school": g(schema.school_col),
        "timeframe": g(schema.group_axis_col),
        "image_file": g(schema.image_col),
        "description": g(schema.description_col),
    }
    return row_to_artwork_dict(mapped, snippet_len=EVIDENCE_SNIPPET_LEN)


# ------------------------------------------------------------------ #
# LangChain Tools                                                      #
# ------------------------------------------------------------------ #


@tool
def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    通过自然语言语义检索相关画作与用户上传文档片段。

    默认检索全部已注册数据源：内置核心库、用户上传 PDF、已确认的
    用户表格都会自动参与（表格无需切换，确认 schema 即生效）。

    注意：用户上传文档（手稿/画册/回忆录等）的内容**只存在于本工具的结果中**，
    query_painter_knowledge / exact_lookup 等工具看不到文档内容；涉及文档细节
    的问题（如"莫奈在葛列尔画室的同学""布丹怎么发现莫奈"）必须调用本工具。
    常识/定义/算术类问题（如"什么是线性透视""1+1等于几"）不需要调用本工具。

    适用场景：
      - 按主题检索（如"描绘爱情的文艺复兴画作"）
      - 按风格检索（如"印象派风景画"）
      - 按内容描述检索（如"使用金箔的画作"）
      - 检索用户上传的 PDF 文档内容（结果 source 为 user_pdf_text/user_pdf_image，
        标题形如《文档名》第N页，引用时请注明来自用户文档）

    Args:
        query: 自然语言检索查询
        top_k: 返回结果数量（默认5）
        filters: 可选结构化过滤，如 {"author": "Monet", "school": "Impressionism",
                 "timeframe": "1851-1900", "source": "core"}；
                 source 只查指定通道（core / user_pdf_text / user_pdf_image），
                 author/school/timeframe 做大小写不敏感包含匹配

    Returns:
        匹配结果列表：画作（标题、画家、年代、技法、流派、图片路径、描述摘要）
        与用户文档片段（文档名、页码、内容）
    """
    from src.retrieval.hybrid import get_hybrid_retriever

    hybrid = get_hybrid_retriever()
    f = dict(filters or {})
    src_val = f.pop("source", None)
    sources = [str(src_val)] if src_val else None
    # 带结构化过滤时多取候选，后置过滤避免漏召回
    fetch_k = max(top_k * 3, 20) if f else top_k
    results = hybrid.search(
        query,
        top_k=fetch_k,
        sources=sources,
    )
    out = [_format_result(r) for r in results]
    if f:
        out = [d for d in out if _result_hit_filters(d, f)][:top_k]
    else:
        out = out[:top_k]
    return out


@tool
def agentic_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Evidence-first retrieval with one coverage-gated query rewrite.

    Use when a question has several concepts or the first normal retrieval was
    sparse. The tool performs at most one rewrite/retrieval pass and returns
    both merged evidence and a coverage audit, avoiding unbounded loops.
    """
    from src.retrieval.agentic import adaptive_retrieve

    def retrieve(q: str) -> list[dict]:
        return semantic_search.invoke({"query": q, "top_k": top_k})

    evidence, audit = adaptive_retrieve(query, retrieve)
    return {"evidence": evidence[:top_k], "coverage": audit}


def _result_hit_filters(d: dict, filters: dict) -> bool:
    """对格式化结果做结构化过滤（author/school/timeframe 包含匹配）。"""
    return hit_filters_match(d, filters)


@tool
def exact_lookup(
    author: Optional[str] = None,
    title: Optional[str] = None,
    timeframe: Optional[str] = None,
    school: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    按字段精确/模糊匹配查询画作。

    适用场景：
      - 查询特定画家的作品（如"找所有莫奈的画"）
      - 查询特定标题（如"找《星夜》"）
      - 按年代段筛选（如"1900-1950年的作品"）
      - 按流派筛选（如"意大利画派"）

    Args:
        author:    画家姓名（部分匹配）
        title:     画作标题（部分匹配）
        timeframe: 年代段，如 "1900-1950"
        school:    流派，如 "Italian", "French"
        top_k:     最多返回条数（默认5）

    Returns:
        匹配画作列表
    """
    # 2026-08-02：按当前生效数据源（semart / core / 用户表格）的角色列检索，
    # 不再硬编码 SemArt 列名——切换数据源后 exact_lookup 跟着走
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.structured_retriever import get_structured_retriever

    dataset_id = get_hybrid_retriever().active_dataset
    retriever = get_structured_retriever(dataset_id)
    schema = retriever.schema
    df = retriever.df

    # 标题/作者走统一的三级模糊匹配；枚举字段（年代段/流派）保持简单包含
    if author:
        df = fuzzy_match(df, schema.entity_col, author)
    if title and schema.title_col:
        df = fuzzy_match(df, schema.title_col, title)
    if timeframe and schema.group_axis_col:
        df = df[
            df[schema.group_axis_col].astype(str).str.lower().str.contains(
                timeframe.lower(), na=False, regex=False
            )
        ]
    if school and schema.school_col:
        df = df[
            df[schema.school_col].astype(str).str.lower().str.contains(
                school.lower(), na=False, regex=False
            )
        ]

    if df.empty:
        return [{"message": "No artworks found matching the given criteria."}]

    return [
        _artwork_from_schema_row(schema, row)
        for _, row in df.head(top_k).iterrows()
    ]
