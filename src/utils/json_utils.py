"""LLM 输出 JSON 鲁棒解析：去 markdown fence、截取首个完整对象/数组。"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    """从 LLM 输出中鲁棒地解析 JSON（去 markdown 代码块、截取首个 {} 或 []）。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 兜底：截取第一个完整的对象或数组
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                continue
    return None
