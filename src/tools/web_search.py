"""联网搜索工具（web_search，知识库缺口兜底）。

当核心库检索不到 / 结果不相关时，转而联网搜索。

实现：
  - 优先使用 Tavily（需 TAVILY_API_KEY）
  - 无 key 时优雅降级：返回一条说明信息，不抛异常，
    这样 reflection 兜底逻辑仍能跑通，只是没有真实联网结果。
"""

import os
from functools import lru_cache

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

MAX_RESULTS = 5


@lru_cache(maxsize=1)
def _get_tavily_client():
    """返回 Tavily 客户端，无 key 时返回 None。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=api_key)
    except Exception:
        return None


def _search_impl(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """底层实现，供工具和节点直接调用（绕过 @tool 包装）。"""
    client = _get_tavily_client()

    if client is None:
        return [
            {
                "title": "网络搜索未配置",
                "snippet": (
                    "未检测到 TAVILY_API_KEY，无法联网搜索。"
                    "请在 .env 中配置 TAVILY_API_KEY 以启用兜底搜索。"
                    "本次将仅基于本地知识回答。"
                ),
                "url": "",
                "source": "system",
            }
        ]

    try:
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = []
        for item in resp.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:500],
                    "url": item.get("url", ""),
                    "source": "tavily",
                }
            )
        return results or [
            {
                "title": "无结果",
                "snippet": f"网络搜索 '{query}' 未返回结果。",
                "url": "",
                "source": "tavily",
            }
        ]
    except Exception as e:
        return [
            {
                "title": "搜索失败",
                "snippet": f"网络搜索出错：{e}",
                "url": "",
                "source": "error",
            }
        ]


@tool
def web_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    联网搜索，用于本地知识库检索不到或结果不相关时的兜底。

    适用场景：
      - SemArt 数据集中没有该画家/画作/流派的信息
      - 用户询问较新的、超出数据集时间范围（8-19世纪）的艺术信息
      - 需要最新的展览、拍卖、收藏地等时效性信息

    Args:
        query:       搜索查询（建议用英文以获得更多结果）
        max_results: 返回结果数量（默认5）

    Returns:
        搜索结果列表，每项含 title / snippet / url / source
    """
    return _search_impl(query, max_results)


def web_search_available() -> bool:
    """供 UI/诊断使用：判断联网搜索是否已配置。"""
    return _get_tavily_client() is not None
