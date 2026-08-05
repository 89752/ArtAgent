"""wiki_lookup：Wikipedia REST 摘要查询（免费、无需 key）。

与 web_search 分工：
- wiki_lookup：定义/生平/流派/术语的百科摘要（结构化、带来源 URL）；
- web_search：时效信息（展览/拍卖/新闻/价格）或百科摘要不足时的兜底。
"""

from __future__ import annotations

import re
import urllib.parse

from langchain_core.tools import tool

from src.utils.http import get_json

_SUMMARY_MAX_CHARS = 1600


def _lang_for(entity: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", entity) else "en"


@tool
def wiki_lookup(entity: str) -> dict:
    """查询维基百科摘要：画家/流派/艺术术语的定义与关键事实（免费）。

    适用场景：用户问"什么是巴洛克""莫奈是谁""印象派这个名称的来源"等
    百科型问题，本地库只有画作元数据时用它补定义/生平。中文实体自动查
    中文维基，英文实体查英文维基。

    Args:
        entity: 实体名（画家/流派/术语，建议给出全名）

    Returns:
        {success, entity, lang, title, description, extract, source_url}
    """
    lang = _lang_for(entity)
    title = urllib.parse.quote(entity.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        data = get_json(url)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "entity": entity, "error": f"Wikipedia 查询失败：{e}"}
    if data.get("type") == "disambiguation":
        return {
            "success": False,
            "entity": entity,
            "error": "命中消歧义页，请给出更具体的全名",
            "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    extract = str(data.get("extract") or "")
    if len(extract) > _SUMMARY_MAX_CHARS:
        extract = extract[:_SUMMARY_MAX_CHARS] + "..."
    return {
        "success": True,
        "entity": entity,
        "lang": lang,
        "title": str(data.get("title") or ""),
        "description": str(data.get("description") or ""),
        "extract": extract,
        "source_url": str(
            data.get("content_urls", {}).get("desktop", {}).get("page", "")
        ),
    }
