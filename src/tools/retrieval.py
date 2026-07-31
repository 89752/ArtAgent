"""
Tool 1: Artwork Retrieval Tools

提供两种检索方式：
  - semantic_search: 语义向量检索（用于模糊查询、主题检索）
  - exact_lookup:    精确字段查询（用于按画家/标题/年代精确查找）

Stage 2 起 semantic_search 改走检索抽象层（HybridRetriever），Agent 工具层
无感知升级——返回形状保持现状（title/author/date/.../description_snippet/
relevance_score），web/service.py 的 ToolMessage 解析与各合成节点不受影响。
数据过滤/格式化统一走 src/data/access.py 数据访问层。
"""

from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.data.access import fuzzy_match, row_to_artwork_dict
from src.retrieval.base import RetrievalResult

load_dotenv()

DEFAULT_TOP_K = 5


def _format_result(result: RetrievalResult) -> dict:
    """格式化单条检索结果，供 Agent 消费（返回形状与 Stage 1 保持一致）。"""
    artwork = row_to_artwork_dict(result.metadata)
    artwork["relevance_score"] = round(result.score, 4)
    return artwork


# ------------------------------------------------------------------ #
# LangChain Tools                                                      #
# ------------------------------------------------------------------ #


@tool
def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    通过自然语言语义检索相关画作。

    适用场景：
      - 按主题检索（如"描绘爱情的文艺复兴画作"）
      - 按风格检索（如"印象派风景画"）
      - 按内容描述检索（如"使用金箔的画作"）

    Args:
        query: 自然语言检索查询
        top_k: 返回结果数量（默认5）

    Returns:
        匹配画作列表，每项包含标题、画家、年代、技法、流派、图片路径、描述摘要
    """
    from src.retrieval.hybrid import get_hybrid_retriever

    results = get_hybrid_retriever().search(query, top_k=top_k)
    return [_format_result(r) for r in results]


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
    from src.data.loader import get_dataset

    df = get_dataset().all

    # 标题/作者走统一的三级模糊匹配；枚举字段（年代段/流派）保持简单包含
    if author:
        df = fuzzy_match(df, "AUTHOR", author)
    if title:
        df = fuzzy_match(df, "TITLE", title)
    if timeframe:
        df = df[
            df["TIMEFRAME"]
            .str.lower()
            .str.contains(timeframe.lower(), na=False, regex=False)
        ]
    if school:
        df = df[
            df["SCHOOL"].str.lower().str.contains(school.lower(), na=False, regex=False)
        ]

    if df.empty:
        return [{"message": "No artworks found matching the given criteria."}]

    return [row_to_artwork_dict(row) for _, row in df.head(top_k).iterrows()]
