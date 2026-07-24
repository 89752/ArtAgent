"""
Tool 4: Style Comparison Tool

对比两幅画作的风格、技法、构图等特征，
输出结构化的对比分析报告。
"""

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


def _format_work_info(work, title_query: str) -> str:
    if work is None:
        return f"'{title_query}': Not found in database."
    return (
        f"Title: {work['TITLE']}\n"
        f"Artist: {work['AUTHOR']}\n"
        f"Date: {work.get('DATE', 'Unknown')}\n"
        f"Technique: {work.get('TECHNIQUE', 'Unknown')}\n"
        f"School: {work.get('SCHOOL', 'Unknown')}\n"
        f"Timeframe: {work.get('TIMEFRAME', 'Unknown')}\n"
        f"Description: {work.get('DESCRIPTION', '')[:300]}..."
    )


@tool
def compare_artwork_styles(
    artwork1_title: str,
    artwork2_title: str,
    comparison_aspects: str = "all",
) -> dict:
    """
    对比两幅画作的艺术风格与特征。

    适用场景：
      - 对比同一画家不同时期的作品
      - 对比不同画派的风格差异
      - 分析相似主题的不同表达方式

    Args:
        artwork1_title:      第一幅画作的标题（支持部分匹配）
        artwork2_title:      第二幅画作的标题（支持部分匹配）
        comparison_aspects:  对比维度，可选：
                             "all"         - 全面对比（默认）
                             "style"       - 风格与技法
                             "composition" - 构图与色彩
                             "historical"  - 历史背景与影响

    Returns:
        结构化对比分析报告
    """
    from src.data.loader import get_dataset
    from src.utils.llm import get_llm

    dataset = get_dataset()
    df = dataset.all

    def _fuzzy_find(title_query: str):
        """模糊查找画作：优先完整匹配，其次包含匹配，都没有返回 None。"""
        # 1. 完整匹配（忽略大小写）
        exact = df[df["TITLE"].str.lower() == title_query.lower()]
        if not exact.empty:
            return exact.iloc[0].to_dict()

        # 2. 去掉冠词后匹配（the / a / an）
        stripped = title_query.strip()
        for article in ("the ", "a ", "an "):
            if stripped.lower().startswith(article):
                stripped = stripped[len(article) :]
                break
        if stripped != title_query:
            exact2 = df[df["TITLE"].str.lower() == stripped.lower()]
            if not exact2.empty:
                return exact2.iloc[0].to_dict()

        # 3. 包含匹配（取第一条）
        contains = df[
            df["TITLE"]
            .str.lower()
            .str.contains(stripped.lower(), na=False, regex=False)
        ]
        if not contains.empty:
            return contains.iloc[0].to_dict()

        return None

    work1 = _fuzzy_find(artwork1_title)
    work2 = _fuzzy_find(artwork2_title)

    work1_info = _format_work_info(work1, artwork1_title)
    work2_info = _format_work_info(work2, artwork2_title)

    aspect_prompts = {
        "all": "Compare these two artworks across all major dimensions: artistic style, technique, composition, color use, historical context, and thematic content.",
        "style": "Focus on comparing the artistic style and painting technique of these two works.",
        "composition": "Focus on comparing the composition, spatial arrangement, and color palette of these two works.",
        "historical": "Focus on comparing the historical context, artistic movement, and cultural significance of these two works.",
    }

    aspect_prompt = aspect_prompts.get(comparison_aspects, aspect_prompts["all"])

    llm = get_llm(temperature=0.3)
    prompt = f"""You are an expert art historian and critic. {aspect_prompt}

=== Artwork 1 ===
{work1_info}

=== Artwork 2 ===
{work2_info}

Please provide a structured comparison with clear sections.
Highlight both similarities and differences.
End with a brief conclusion about what makes each work distinctive.
"""

    response = llm.invoke(prompt)

    return {
        "artwork1": artwork1_title,
        "artwork1_found": work1 is not None,
        "artwork1_matched": work1["TITLE"] if work1 else None,
        "artwork2": artwork2_title,
        "artwork2_found": work2 is not None,
        "artwork2_matched": work2["TITLE"] if work2 else None,
        "comparison_aspects": comparison_aspects,
        "comparison": response.content,
    }
