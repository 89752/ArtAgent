"""
查询改写与多问题拆分（P0-③，借鉴 ragent `MultiQuestionRewriteService`）。

职责：把多轮追问（含指代）改写成可独立理解的完整问题，并把复合问题
（如"对比莫奈和梵高，顺便推荐几幅类似的画"）拆分成多个子问题——
替代原来的 contextualize（指代消解），且为多意图并行检索提供子问题列表。

容错设计：
- LLM 调用失败 / 返回畸形 JSON → 回落（归一化原文 + 单子问题），永不报错；
- 可用环境变量 REWRITE_ENABLED=0 关闭 LLM 改写（走纯规则归一化）；
- 历史只带最近 2 轮（4 条用户/助手消息），控制 token 成本。

用法：
    from src.agent.rewrite import rewrite_and_split
    result = rewrite_and_split("这幅画现在在哪里？", history)
    result.rewritten_question  # 独立完整问题
    result.sub_questions       # 拆分出的子问题
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


REWRITE_SPLIT_PROMPT = """你是多轮对话的查询改写与拆分模块。

根据【对话历史】把用户【最新问题】处理成：
1. rewritten_question：改写后的完整问题——把所有指代（他/她/它/这幅/这位/刚才
   提到的/那幅画 等）替换成历史中明确指向的具体对象；如果最新问题本身已完整
   无需上下文，就原样保留；
2. sub_questions：如果一个问题包含多个可以独立检索的子问题（例如"对比X和Y，
   顺便推荐类似的画"），拆成多个；否则只有改写后的问题本身。
3. rewritten_question 还要做【压缩与抽取】：去掉口头禅、重复、语气词和客套
   （如"就是那个""嗯""你懂的""帮我看看"），只保留关键信息——实体（画家/画作/
   流派）、约束（时期/风格/数量）、意图动作；
4. key_entities：从问题中抽取的关键实体列表（画家名/画作名/流派名，英文优先）；
5. ambiguous：若问题语义不明、无法确定用户到底想要什么（如"给我看看那个"、
   "关于那幅画你了解吗"），置 true，否则 false。

只输出 JSON，不要解释、不要 markdown 代码块：
{{"rewritten_question": "...", "sub_questions": ["...", "..."],
  "key_entities": ["..."], "ambiguous": false}}

对话历史：
{history}

最新问题：
{question}"""


@dataclass
class RewriteResult:
    """改写结果：独立完整问题 + 子问题列表 + 关键实体 + 语义不明标记。"""

    rewritten_question: str
    sub_questions: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    ambiguous: bool = False


def normalize_query(question: str) -> str:
    """归一化：去首尾空白与常见引号。"""
    return (question or "").strip().strip('"').strip("“”").strip()


def rewrite_enabled() -> bool:
    """LLM 改写开关（REWRITE_ENABLED，默认开；0/false/no 关闭）。"""
    return os.getenv("REWRITE_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _parse_json(raw: str) -> Optional[dict]:
    """鲁棒解析改写 JSON（容错 markdown fence 与截断）。"""
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    try:
        data = json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def _recent_history(history) -> str:
    """只保留最近 2 轮（4 条）用户/助手消息，去掉工具消息与系统摘要。"""
    if not history:
        return "(无)"
    lines: list[str] = []
    for msg in history[-8:]:
        mtype = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "") or "")[:200]
        if mtype == "human":
            lines.append(f"用户：{content}")
        elif mtype == "ai":
            lines.append(f"助手：{content}")
    return "\n".join(lines[-4:]) or "(无)"


def rewrite_and_split(
    question: str,
    history=None,
    llm: Optional[Callable[[str], str]] = None,
) -> RewriteResult:
    """改写 + 拆分主入口；任何失败回落 (归一化原文, [归一化原文])。"""
    normalized = normalize_query(question)
    fallback = RewriteResult(normalized, [normalized])

    if not rewrite_enabled():
        return fallback
    if not normalized:
        return fallback

    prompt = REWRITE_SPLIT_PROMPT.format(
        history=_recent_history(history),
        question=question,
    )
    if llm is None:
        from src.utils.llm import get_deterministic_llm

        def _default_llm(p: str) -> str:
            return get_deterministic_llm().invoke(p).content

        llm = _default_llm

    try:
        raw = llm(prompt)
    except Exception:
        return fallback

    data = _parse_json(raw)
    if data is None:
        return fallback

    rq = normalize_query(data.get("rewritten_question") or "")
    if not rq:
        rq = normalized
    subs_raw = data.get("sub_questions")
    subs = (
        [normalize_query(s) for s in subs_raw if normalize_query(s)]
        if isinstance(subs_raw, list)
        else []
    )
    if not subs:
        subs = [rq]
    entities_raw = data.get("key_entities")
    entities = (
        [normalize_query(e) for e in entities_raw if normalize_query(e)]
        if isinstance(entities_raw, list)
        else []
    )
    ambiguous = bool(data.get("ambiguous"))
    return RewriteResult(rq, subs, entities, ambiguous)
