"""
场景2：时间线梳理 + 图像佐证子管线。

extract_subject → gather_periods → synthesize
  - extract_subject: 抽取要梳理的画家/流派（英文）
  - gather_periods:  按 timeframe 分组，收集每个时期的评论证据 + 代表作配图
  - synthesize:      按时间顺序组织连贯叙述，点名各时期代表作
"""

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.prompts import TIMELINE_SUBJECT_PROMPT, TIMELINE_SYNTHESIZE_PROMPT
from src.utils.llm import get_llm, get_deterministic_llm

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
    按 timeframe 分组画家作品，收集每个时期的证据与配图。
    直接用数据集（比逐时期语义检索更可靠地拿到时期覆盖）。
    """
    from src.data.loader import get_dataset
    from src.tools.image_lookup import lookup_images

    subject = state.subjects[0] if state.subjects else state.user_query
    dataset = get_dataset()
    works = dataset.get_by_author(subject)

    period_evidence_blocks = []
    images: list[dict] = []
    docs_by_period: dict[str, list[dict]] = {}

    if works.empty:
        return {
            "retrieved_docs": {},
            "images": [],
            "current_step": "timeline_gather_periods",
        }

    # 按 timeframe 排序分组
    def _period_key(tf: str) -> str:
        return tf if tf else "Unknown"

    works = works.copy()
    works["_TF"] = works["TIMEFRAME"].fillna("").map(_period_key)
    # 按时期字符串排序（形如 "1851-1900" 天然可排序）
    periods = sorted([p for p in works["_TF"].unique() if p and p != "Unknown"])
    if not periods:
        periods = ["Unknown"]

    for period in periods[:_MAX_PERIODS]:
        subset = works[works["_TF"] == period]
        # 证据：取该时期若干条评论
        docs = []
        lines = [f"时期 {period}："]
        for _, row in subset.head(_EVIDENCE_PER_PERIOD).iterrows():
            desc = str(row.get("DESCRIPTION", ""))[:250]
            title = row.get("TITLE", "")
            lines.append(f"  - {title}: {desc}")
            docs.append(
                {
                    "title": title,
                    "author": row.get("AUTHOR", ""),
                    "date": str(row.get("DATE", "")),
                    "image_file": str(row.get("IMAGE_FILE", "")),
                    "description_snippet": desc,
                }
            )
        docs_by_period[period] = docs
        period_evidence_blocks.append("\n".join(lines))

        # 配图：该时期取代表作
        imgs = lookup_images(author=subject, timeframe=period, top_k=_IMAGES_PER_PERIOD)
        images.extend(imgs)

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
        "\n".join(
            [f"时期 {period}："]
            + [
                f"  - {d.get('title','')}: {d.get('description_snippet','')}"
                for d in docs
            ]
        )
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
