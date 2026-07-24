"""
Tool 1: Artwork Retrieval Tools

提供两种检索方式：
  - semantic_search: 语义向量检索（用于模糊查询、主题检索）
  - exact_lookup:    精确字段查询（用于按画家/标题/年代精确查找）
"""

import os
import pickle
from pathlib import Path
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

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
    """格式化单条检索结果，供 Agent 消费。"""
    result = {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "date": meta.get("date", ""),
        "technique": meta.get("technique", ""),
        "school": meta.get("school", ""),
        "timeframe": meta.get("timeframe", ""),
        "image_file": meta.get("file", ""),
        "description_snippet": (
            meta.get("description", "")[:200] + "..."
            if len(meta.get("description", "")) > 200
            else meta.get("description", "")
        ),
    }
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

    dataset = get_dataset()
    df = dataset.all

    # 逐条件过滤
    if author:
        # 拆词匹配：取最长的词（通常是姓）优先匹配
        # "Vincent van Gogh" → ["Vincent", "van", "Gogh"] → 按长度排序 → 先试 "Gogh"
        # 这样能处理 "Van Gogh" / "Vincent van Gogh" / "gogh" 等各种格式
        tokens = [
            t for t in author.strip().split() if len(t) > 2
        ]  # 过滤掉 "van", "de", "di" 等介词
        tokens_sorted = sorted(tokens, key=len, reverse=True)  # 最长词（姓）优先
        mask = None
        for token in tokens_sorted:
            candidate = (
                df["AUTHOR"]
                .str.lower()
                .str.contains(token.lower(), na=False, regex=False)
            )
            if candidate.any():
                mask = candidate
                break  # 找到匹配就停止，优先用最长词
        if mask is None:
            # 所有词都没匹配到，fallback 用原始输入
            mask = (
                df["AUTHOR"]
                .str.lower()
                .str.contains(author.lower(), na=False, regex=False)
            )
        df = df[mask]
    if title:
        df = df[
            df["TITLE"].str.lower().str.contains(title.lower(), na=False, regex=False)
        ]
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

    results = []
    for _, row in df.head(top_k).iterrows():
        desc = str(row.get("DESCRIPTION", ""))
        results.append(
            {
                "title": row["TITLE"],  # ← 统一用小写key
                "author": row["AUTHOR"],
                "date": str(row.get("DATE", "")),
                "technique": str(row.get("TECHNIQUE", "")),
                "school": str(row.get("SCHOOL", "")),
                "timeframe": str(row.get("TIMEFRAME", "")),
                "image_file": str(row.get("IMAGE_FILE", "")),
                "description_snippet": desc[:200] + "..." if len(desc) > 200 else desc,
            }
        )

    return results
