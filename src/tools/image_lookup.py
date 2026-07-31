"""
Tool: Image Lookup（含视觉分析能力）

从 SemArt 本地图片资源中查找画作配图；analyze=True 时对定位到的
单幅画作调用视觉模型分析（构图/色彩/笔触）。

设计原则：默认只查找返回路径（快、免费），LLM 判断确实需要"看图
分析"时才传 analyze=True 触发视觉模型调用（慢、消耗 API 额度）。

由原 image_lookup 与 analyze_image 两个工具合并而来；模糊匹配与
行转字典统一走 src/data/access.py 数据访问层。
"""

import base64
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.data.access import fuzzy_match, row_to_artwork_dict

load_dotenv()

_DATA_DIR = Path(os.getenv("SEMART_DATA_DIR", "./SemArt"))


# ------------------------------------------------------------------ #
# 查找定位（不走视觉模型）                                              #
# ------------------------------------------------------------------ #


def _resolve_path(image_file: str) -> str:
    """把 IMAGE_FILE 字段解析成完整本地路径字符串。"""
    p = _DATA_DIR / "Images" / image_file
    return str(p) if p.exists() else ""


def lookup_images(
    title: Optional[str] = None,
    author: Optional[str] = None,
    timeframe: Optional[str] = None,
    top_k: int = 3,
) -> list[dict]:
    """底层实现，供节点直接调用（绕过 @tool 包装）。"""
    from src.data.loader import get_dataset

    df = get_dataset().all

    if title:
        df = fuzzy_match(df, "TITLE", title)
    if author:
        df = fuzzy_match(df, "AUTHOR", author)
    if timeframe:
        df = df[
            df["TIMEFRAME"].str.lower().str.contains(
                timeframe.lower(), na=False, regex=False
            )
        ]

    if df.empty:
        return []

    out = []
    for _, row in df.head(top_k).iterrows():
        d = row_to_artwork_dict(row)
        d["image_path"] = _resolve_path(d["image_file"])
        out.append(d)
    return out


# ------------------------------------------------------------------ #
# 视觉分析（analyze=True 时触发）                                       #
# ------------------------------------------------------------------ #

_ANALYSIS_FOCUS_PROMPTS = {
    "general": (
        "请综合分析这幅画：\n"
        "1) 画面主题与内容\n"
        "2) 艺术风格与流派\n"
        "3) 构图与空间布局\n"
        "4) 色彩与光影运用\n"
        "5) 整体情感与艺术价值"
    ),
    "style": (
        "请重点分析这幅画的艺术风格与技法：\n"
        "1) 风格归属（如巴洛克、印象派、文艺复兴等）\n"
        "2) 笔触与肌理特征\n"
        "3) 技法特点（厚涂法、明暗法等）\n"
        "4) 与同时期画家风格的关联"
    ),
    "composition": (
        "请重点分析这幅画的构图与视觉元素：\n"
        "1) 构图结构与平衡感\n"
        "2) 透视与空间深度\n"
        "3) 色彩搭配与冷暖对比\n"
        "4) 光影处理（明暗对比）\n"
        "5) 视觉引导与焦点"
    ),
    "content": (
        "请重点分析这幅画的内容与主题：\n"
        "1) 画面中的人物、场景或物体\n"
        "2) 叙事性或故事背景\n"
        "3) 象征元素与寓意\n"
        "4) 历史或神话背景\n"
        "5) 核心主题与艺术家意图"
    ),
}


def _get_vision_llm() -> ChatOpenAI:
    """返回支持视觉的模型实例。"""
    return ChatOpenAI(
        model="qwen3.5-omni-plus",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0.3,
    )


def _find_image_file(artwork_query: str) -> Optional[str]:
    """
    按标题或文件名查找 IMAGE_FILE。
    优先精确匹配文件名，其次走统一模糊匹配找标题。
    """
    from src.data.loader import get_dataset

    df = get_dataset().all

    # 1. 精确匹配文件名
    exact = df[df["IMAGE_FILE"].str.lower() == artwork_query.lower()]
    if not exact.empty:
        return exact.iloc[0]["IMAGE_FILE"]

    # 2. 标题模糊匹配（三级递进），取第一条
    matched = fuzzy_match(df, "TITLE", artwork_query)
    if not matched.empty:
        return matched.iloc[0]["IMAGE_FILE"]

    return None


def _analyze_image_file(image_file: str, analysis_focus: str) -> dict:
    """对已定位的 IMAGE_FILE 调用视觉模型分析，返回结构化结果。"""
    from src.data.loader import get_dataset

    df = get_dataset().all
    row = df[df["IMAGE_FILE"] == image_file].iloc[0]
    # 分析场景保留完整数据集描述（LLM 需要全文，不要 200 字截断）
    metadata = row_to_artwork_dict(row, snippet_len=None)

    image_path = _DATA_DIR / "Images" / image_file
    if not image_path.exists():
        return {
            "success": False,
            "error": f"图片文件不存在：{image_path}",
            "metadata": metadata,
        }

    prompt = _ANALYSIS_FOCUS_PROMPTS.get(
        analysis_focus, _ANALYSIS_FOCUS_PROMPTS["general"]
    )
    # 附上已知元数据引导模型
    metadata_hint = (
        f"已知信息：标题《{metadata['title']}》，"
        f"画家：{metadata['author']}，"
        f"年代：{metadata['date']}，"
        f"技法：{metadata['technique']}，"
        f"流派：{metadata['school']}。\n\n"
        f"{prompt}\n请用中文回答。"
    )

    image_ext = image_path.suffix.lstrip(".").lower()
    if image_ext == "jpg":
        image_ext = "jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    try:
        llm = _get_vision_llm()
        msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"},
                },
                {"type": "text", "text": metadata_hint},
            ]
        )
        response = llm.invoke([msg])
        analysis_text = response.content
    except Exception as e:
        analysis_text = f"视觉分析失败：{e}"

    return {
        "success": True,
        "image_file": image_file,
        "title": metadata["title"],
        "author": metadata["author"],
        "date": metadata["date"],
        "technique": metadata["technique"],
        "school": metadata["school"],
        "dataset_description": metadata["description_snippet"],  # snippet_len=None → 完整描述
        "analysis_focus": analysis_focus,
        "analysis": analysis_text,
    }


# ------------------------------------------------------------------ #
# LangChain Tool                                                       #
# ------------------------------------------------------------------ #


@tool
def image_lookup(
    title: Optional[str] = None,
    author: Optional[str] = None,
    timeframe: Optional[str] = None,
    top_k: int = 3,
    analyze: bool = False,
    analysis_focus: str = "general",
):
    """
    从 SemArt 本地图片库查找画作配图；analyze=True 时对定位到的画作做视觉分析。

    适用场景：
      - 需要为某画家/某时期的叙述配上代表作品图（默认模式，快、免费）
      - 按标题定位一幅画的图片文件
      - 用户要求"看图分析"某幅具体画作（analyze=True，分析构图/色彩/笔触）

    Args:
        title:          画作标题（部分匹配）
        author:         画家姓名（部分匹配）
        timeframe:      时期，如 "1851-1900"
        top_k:          返回数量（默认3，仅查找模式）
        analyze:        True 时对定位到的第一幅画调用视觉模型分析（慢、消耗 API 额度）
        analysis_focus: 分析侧重点："general" 综合（默认）/ "style" 风格技法 /
                        "composition" 构图色彩 / "content" 内容主题

    Returns:
        analyze=False: 画作列表，每项含 title/author/date/timeframe/image_file/image_path/description_snippet
        analyze=True:  单幅画的视觉分析结果（含元数据、完整数据集描述、分析文本）
    """
    if not analyze:
        return lookup_images(title, author, timeframe, top_k)

    # 视觉分析模式：先定位单一画作
    query = title or author or ""
    if not query:
        return {
            "success": False,
            "error": "视觉分析需要提供 title 或 author 参数来定位画作。",
        }
    image_file = _find_image_file(query)
    if image_file is None:
        return {
            "success": False,
            "error": f"未在数据集中找到画作：'{query}'，请尝试英文标题。",
        }
    return _analyze_image_file(image_file, analysis_focus)
