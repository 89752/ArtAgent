"""museum_search：Metropolitan Museum 开放馆藏检索（免费、无需 key）。

Met API（CC0 公共领域）：
  - search: GET /public/collection/v1/search?q=...&hasImages=true
  - object: GET /public/collection/v1/objects/{id}
限速 80 req/s，内部无需令牌桶（单次工具调用最多 top_k+1 个请求）。
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from langchain_core.tools import tool

from src.utils.http import get_json

_MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
_MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
@tool
def museum_search(
    query: Optional[str] = None,
    artist: Optional[str] = None,
    title: Optional[str] = None,
    has_image: bool = True,
    top_k: int = 5,
) -> dict:
    """检索 Met 博物馆开放馆藏（CC0 公共领域，免费无需 key）。

    适用场景：用户询问数据集之外的画作/馆藏信息（20 世纪、非欧洲、当代），
    或需要"现藏于哪个博物馆"这类馆藏事实。本地库不足时作为外部知识补充。

    Args:
        query:    自由文本搜索词
        artist:   艺术家姓名
        title:    作品标题
        has_image: 只要带图片的结果（默认 True）
        top_k:    返回数量（默认5）

    Returns:
        {success, total, results: [{title, artist, date, medium, department,
        image_url, object_url, is_public_domain}], error?}
    """
    parts = [p for p in (query, artist, title) if p]
    if not parts:
        return {"success": False, "error": "需要提供 query / artist / title 之一"}
    search_q = " ".join(parts)
    params = {"q": search_q, "hasImages": "true" if has_image else "false"}
    url = _MET_SEARCH + "?" + urllib.parse.urlencode(params)
    try:
        data = get_json(url)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Met API 检索失败：{e}"}

    ids = (data.get("objectIDs") or [])[:top_k]
    results: list[dict] = []
    for oid in ids:
        try:
            obj = get_json(_MET_OBJECT.format(oid=oid))
        except Exception:  # noqa: BLE001 — 单条失败跳过
            continue
        results.append(
            {
                "title": str(obj.get("title") or ""),
                "artist": str(obj.get("artistDisplayName") or ""),
                "date": str(obj.get("objectDate") or ""),
                "medium": str(obj.get("medium") or ""),
                "department": str(obj.get("department") or ""),
                "image_url": str(obj.get("primaryImage") or ""),
                "object_url": str(obj.get("objectURL") or ""),
                "is_public_domain": bool(obj.get("isPublicDomain")),
            }
        )
    return {"success": True, "total": len(results), "results": results}
