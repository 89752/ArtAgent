"""
Tool 3: Painter Knowledge Query Tool

结合 SemArt 数据集统计信息 + DeepSeek LLM 知识，
回答关于画家的专业问题。
"""

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def query_painter_knowledge(
    painter_name: str,
    question: str,
) -> dict:
    """
    查询画家的详细知识，结合数据集统计与 LLM 知识。

    适用场景：
      - 询问画家的生平、风格、历史地位
      - 询问画家的代表作品
      - 询问画家所属流派、时代背景

    Args:
        painter_name: 画家姓名
        question:     关于该画家的具体问题

    Returns:
        包含数据集统计信息和 LLM 分析的综合回答
    """
    from src.data.loader import get_dataset
    from src.utils.llm import get_llm

    dataset = get_dataset()

    # 1. 从数据集中获取统计信息
    painter_works = dataset.get_by_author(painter_name)
    dataset_context = ""

    if not painter_works.empty:
        works_count = len(painter_works)
        techniques = painter_works["TECHNIQUE"].value_counts().head(3).to_dict()
        timeframes = painter_works["TIMEFRAME"].value_counts().head(3).index.tolist()
        schools = painter_works["SCHOOL"].value_counts().head(1).index.tolist()
        sample_titles = painter_works["TITLE"].head(5).tolist()

        dataset_context = f"""
Dataset statistics for {painter_name}:
- Works in database: {works_count}
- Main school/origin: {', '.join(schools) if schools else 'Unknown'}
- Active timeframes: {', '.join(timeframes)}
- Common techniques: {', '.join([f"{k} ({v})" for k, v in techniques.items()])}
- Sample works: {', '.join(sample_titles)}
"""
    else:
        dataset_context = f"No works found for '{painter_name}' in the SemArt database."

    # 2. 调用 LLM 综合回答
    llm = get_llm(temperature=0.3)
    prompt = f"""You are an expert art historian. Answer the following question about the painter.

Painter: {painter_name}
Question: {question}

{dataset_context}

Please provide a comprehensive, accurate answer drawing on both the dataset information above 
and your art history knowledge. Be specific and informative.
"""

    response = llm.invoke(prompt)

    return {
        "painter": painter_name,
        "question": question,
        "dataset_stats": dataset_context.strip(),
        "answer": response.content,
    }
