"""
数据访问层：介于 loader.py 与各工具/节点之间的统一访问函数。

解决的问题：此前"模糊匹配标题/作者"、"DataFrame 行 → 展示用字典"、
"证据字典列表 → 拼给 LLM 看的文本块"三类逻辑在 retrieval.py /
image_lookup.py / style_comparison.py / image_analysis.py / 各管线节点里
各自独立实现且写法不一（截断长度 200/250/300 各自为政）。本模块是唯一实现，
所有工具与管线节点一律走这里，不再各自直接摸 DataFrame。
"""

from __future__ import annotations

import re
from typing import Union

import pandas as pd

# 统一的描述片段截断长度（替换掉 200/250/300 三个各自为政的数字）
EVIDENCE_SNIPPET_LEN = 200

# 标题类匹配需要去掉的英文冠词前缀
_ARTICLES = ("the ", "a ", "an ")


def _strip_article(text: str) -> str:
    """去掉开头的英文冠词（the/a/an），用于标题类匹配。"""
    stripped = text.strip()
    for article in _ARTICLES:
        if stripped.lower().startswith(article):
            return stripped[len(article):]
    return stripped


def _contains_mask(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    return df[column].str.lower().str.contains(value.lower(), na=False, regex=False)


def fuzzy_match(df: pd.DataFrame, column: str, query: str) -> pd.DataFrame:
    """
    统一的模糊匹配，三级递进，每一级命中即返回：

      1. 精确匹配（忽略大小写）
      2. 去冠词后精确匹配（处理 "The Kiss" vs "Kiss"）
      3. 分词包含匹配：长度 > 2 的词按长度降序逐个尝试包含匹配
         （最长词通常是姓，能处理 "Van Gogh" / "Vincent van Gogh" / "gogh"），
         全部落空则用整条 query 做包含匹配兜底。

    返回匹配到的 DataFrame（可能为空），调用方决定取第一行还是全部。
    """
    if df.empty or not query or not str(query).strip():
        return df.iloc[0:0]

    query = str(query).strip()

    # 1. 精确匹配（忽略大小写）
    exact = df[df[column].str.lower() == query.lower()]
    if not exact.empty:
        return exact

    # 2. 去冠词后精确匹配
    stripped = _strip_article(query)
    if stripped != query:
        exact2 = df[df[column].str.lower() == stripped.lower()]
        if not exact2.empty:
            return exact2

    # 3. 分词包含：最长词优先（跳过 van/de/di 等短介词）
    tokens = sorted((t for t in stripped.split() if len(t) > 2), key=len, reverse=True)
    for token in tokens:
        candidate = df[_contains_mask(df, column, token)]
        if not candidate.empty:
            return candidate

    # 兜底：整条 query 做包含匹配
    return df[_contains_mask(df, column, stripped)]


# Chroma metadata 用小写 key（title/file/description），
# DataFrame 用大写（TITLE/IMAGE_FILE/DESCRIPTION），此处做归一。
_KEY_ALIASES = {
    "title": ("TITLE", "title"),
    "author": ("AUTHOR", "author"),
    "date": ("DATE", "date"),
    "technique": ("TECHNIQUE", "technique"),
    "school": ("SCHOOL", "school"),
    "timeframe": ("TIMEFRAME", "timeframe"),
    "image_file": ("IMAGE_FILE", "file", "image_file"),
    "description": ("DESCRIPTION", "description"),
}


def _first_value(row: Union[pd.Series, dict], out_key: str):
    """按别名表取第一个存在且非空的字段值。"""
    for key in _KEY_ALIASES[out_key]:
        try:
            value = row[key]
        except (KeyError, IndexError, TypeError):
            continue
        if value is not None and str(value) != "":
            return value
    return ""


def row_to_artwork_dict(
    row: Union[pd.Series, dict],
    snippet_len: int | None = EVIDENCE_SNIPPET_LEN,
) -> dict:
    """
    统一的"画作记录 → 展示用字典"转换（小写 key），含描述截断。

    接受 DataFrame 行（pd.Series，大写列名）或 Chroma metadata（dict，小写 key）。
    snippet_len 传 None 时不截断描述（保留完整 description_snippet）。
    """
    desc = str(_first_value(row, "description"))
    if snippet_len is not None and len(desc) > snippet_len:
        snippet = desc[:snippet_len] + "..."
    else:
        snippet = desc
    return {
        "title": str(_first_value(row, "title")),
        "author": str(_first_value(row, "author")),
        "date": str(_first_value(row, "date")),
        "technique": str(_first_value(row, "technique")),
        "school": str(_first_value(row, "school")),
        "timeframe": str(_first_value(row, "timeframe")),
        "image_file": str(_first_value(row, "image_file")),
        "description_snippet": snippet,
    }


def format_evidence_block(
    docs: list[dict],
    template: str = "- {title} ({date}): {description_snippet}",
) -> str:
    """
    统一的"证据字典列表 → 拼给 LLM 看的文本块"。

    template 中用 {field} 引用 docs 里的 key，调用方指定要哪些字段、什么顺序、
    什么分隔符。字段缺失或为空时渲染为空串，并清理残留的空括号与孤立分隔符，
    避免给 LLM 看 "- 标题 (): " 这类脏文本。
    """
    lines = []
    for d in docs:
        line = template
        for field in re.findall(r"{(\w+)}", template):
            value = d.get(field, "")
            line = line.replace(
                "{" + field + "}", str(value) if value is not None else ""
            )
        # 清理空括号 / 连续或孤立的竖线 / 多余空白
        line = re.sub(r"\(\s*\)", "", line)
        line = re.sub(r"\[\s*\]", "", line)
        line = re.sub(r"(\s*\|\s*){2,}", " | ", line)
        line = re.sub(r"^(\s*[-•]?\s*)\|\s*", r"\1", line)
        line = re.sub(r"\s*\|\s*$", "", line)
        line = re.sub(r"\s+([:;])", r"\1", line)  # 空字段删除后残留的"空格+冒号"
        line = re.sub(r"(?<=.)[ \t]{2,}", " ", line)  # 压缩非行首的连续空白（保留缩进）
        # 去掉行尾因空字段留下的孤立分隔符（": " / " |" / " —" 等）
        line = line.rstrip(" :|—-")
        lines.append(line)
    return "\n".join(lines)
