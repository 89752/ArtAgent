"""
Tool 1: Artwork Retrieval Tools

提供两种检索方式：
  - semantic_search: 语义向量检索（用于模糊查询、主题检索）
  - exact_lookup:    精确字段查询（用于按画家/标题/年代精确查找）

数据过滤/格式化统一走 src/data/access.py 数据访问层。
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.data.access import fuzzy_match, row_to_artwork_dict

load_dotenv()

CHROMA_DIR = Path(os.getenv("INDEX_DIR", "./data/index")) / "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 5


# ------------------------------------------------------------------ #
# 单例：Chroma collection + embedding model                           #
# ------------------------------------------------------------------ #


@lru_cache(maxsize=1)
def _get_collection():
    """加载持久化的 Chroma collection（全局单例）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("semart")


@lru_cache(maxsize=1)
def _get_embedding_model():
    """加载 BGE embedding 模型（全局单例）。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(text: str) -> list[float]:
    """将文本转为归一化向量。"""
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _format_result(meta: dict, distance: Optional[float] = None) -> dict:
    """格式化单条 Chroma 检索结果，供 Agent 消费。"""
    result = row_to_artwork_dict(meta)
    if distance is not None:
        result["relevance_score"] = round(1 - distance, 4)
    return result


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
    collection = _get_collection()
    query_embedding = _embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [_format_result(meta, dist) for meta, dist in zip(metadatas, distances)]


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
