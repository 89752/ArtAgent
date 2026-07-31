"""
场景2：时间线梳理 + 图像佐证子管线。

extract_subject → gather_periods → synthesize
  - extract_subject: 抽取要梳理的画家/流派（英文）
  - gather_periods:  按 timeframe 分组，收集每个时期的评论证据 + 代表作配图
  - synthesize:      按时间顺序组织连贯叙述，点名各时期代表作

数据访问：Stage 2 起 gather 通过当前数据源的 StructuredTableRetriever
定位并分组画家作品（内部走 access.fuzzy_match），row_to_artwork_dict
产出证据字典（含 description_snippet），synthesize 用 format_evidence_block
拼证据文本——同一批画作记录只查一次、只拼一次。
"""

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.prompts import TIMELINE_SUBJECT_PROMPT, TIMELINE_SYNTHESIZE_PROMPT
from src.data.access import format_evidence_block, row_to_artwork_dict
from src.utils.llm import get_llm, get_deterministic_llm
from src.utils.logging_config import get_logger, log_event

logger = get_logger("timeline")

# 每个时期取多少条评论证据 / 多少张配图
_EVIDENCE_PER_PERIOD = 2
_IMAGES_PER_PERIOD = 1
_MAX_PERIODS = 6


def timeline_extract_subject(state: AgentState) -> dict:
    """抽取单个画家/流派名称。"""
    prompt = TIMELINE_SUBJECT_PROMPT.format(user_query=state.user_query)
    subject = get_deterministic_llm().invoke(prompt).content.strip().strip('"').strip()
    if not subject:
        subject = state.user_query
    return {"subjects": [subject], "current_step": "timeline_extract_subject"}


def timeline_gather_periods(state: AgentState) -> dict:
    """
    按分组轴（SemArt 为 timeframe）分组画家作品，收集每个时期的证据与配图。

    数据访问走当前生效数据源的 StructuredTableRetriever（Stage 2）：
    本节点依赖的不再是 SemArt 字段名，而是 schema 声明的实体列与分组轴。
    """
    from src.retrieval.structured_retriever import get_structured_retriever
    from src.tools.image_lookup import lookup_images

    subject = state.subjects[0] if state.subjects else state.user_query
    retriever = get_structured_retriever(state.dataset_id)
    groups = retriever.group_by_axis(subject)
    log_event(
        logger, "gather_periods",
        subject=subject, dataset_id=state.dataset_id,
        works_found=sum(len(g) for g in groups.values()),
    )

    images: list[dict] = []
    docs_by_period: dict[str, list[dict]] = {}

    if not groups:
        return {
            "retrieved_docs": {},
            "images": [],
            "current_step": "timeline_gather_periods",
        }

    # groups 按分组轴值升序（形如 "1851-1900" 天然可按时间排序）
    for period, subset in list(groups.items())[:_MAX_PERIODS]:
        # 证据：取该时期若干条评论（row_to_artwork_dict 直接产出含
        # description_snippet 的字典，不再自己 iterrows() 拼第二遍）
        docs_by_period[period] = [
            row_to_artwork_dict(row)
            for _, row in subset.head(_EVIDENCE_PER_PERIOD).iterrows()
        ]

        # 配图：该时期取代表作
        imgs = lookup_images(author=subject, timeframe=period, top_k=_IMAGES_PER_PERIOD)
        images.extend(imgs)

    log_event(
        logger, "gather_periods",
        periods=list(groups.keys())[:_MAX_PERIODS], images=images,
    )
    return {
        "retrieved_docs": docs_by_period,
        "images": images,
        "artworks": [
            {
                "title": i["title"],
                "author": i["author"],
                "date": i.get("date", ""),
                "image_file": i.get("image_file", ""),
            }
            for i in images
        ],
        "current_step": "timeline_gather_periods",
    }


def timeline_synthesize(state: AgentState) -> dict:
    """按时间顺序组织叙述。"""
    subject = state.subjects[0] if state.subjects else state.user_query

    if not state.retrieved_docs:
        # 数据集没有该对象 → 交给反思/web 兜底
        msg = (
            f"SemArt 数据集中未收录 {subject} 的作品，"
            f"无法基于本地数据梳理其时间线。"
        )
        return {
            "final_answer": msg,
            "messages": [AIMessage(content=msg)],
            "current_step": "timeline_synthesize",
        }

    period_evidence = "\n\n".join(
        f"时期 {period}：\n"
        + format_evidence_block(docs, "  - {title}: {description_snippet}")
        for period, docs in state.retrieved_docs.items()
    )
    image_list = "\n".join(
        f"  - [{i.get('timeframe','')}] {i['title']} — {i['author']}"
        for i in state.images
    ) or "(无配图)"

    prompt = TIMELINE_SYNTHESIZE_PROMPT.format(
        subject=subject,
        user_query=state.user_query,
        period_evidence=period_evidence,
        image_list=image_list,
    )
    answer = get_llm(0.4).invoke(prompt).content
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "current_step": "timeline_synthesize",
    }
