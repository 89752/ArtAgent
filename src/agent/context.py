"""上下文工程：ContextBuilder 的纯函数层（详见 docs/Agent化-规划.md）。

职责：把散落在 state 里的偏好、会话台账、检索证据、历史，组装成结构化的
上下文块（编号引用、按 artwork 去重、token/字符预算、历史窗口裁剪）。
本模块只做纯计算（不调 LLM、不读库），供 general_agent 组装 messages 使用，
也保证全部逻辑可单测。

块结构（按落地顺序）：
  system（角色/能力/技能/规则）→ profile（画像）→ summary（滚动摘要，
  未启用前为空）→ session（会话台账）→ evidence（编号引用 [N]）→ history。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── 预算常量（实测后可调） ─────────────────────────────────────
EVIDENCE_CHAR_BUDGET = 4500
SNIPPET_LEN = 200
PROFILE_CHAR_BUDGET = 800
MEMORY_CHAR_BUDGET = 800
SESSION_CHAR_BUDGET = 600
SUMMARY_CHAR_BUDGET = 1200
HISTORY_MAX_TURNS = 8


@dataclass
class ContextBudget:
    """统一上下文预算（v2）：总量上限 + 各块预算 + 历史窗口自适应。

    裁剪优先级（成熟平台惯例）：system/profile 不裁 → summary →
    evidence → subtasks → history（最后裁，保底 2 轮）。
    """

    total_chars: int = 12000
    history_max_turns: int = HISTORY_MAX_TURNS
    history_min_turns: int = 2
    evidence_chars: int = EVIDENCE_CHAR_BUDGET
    subtasks_chars: int = 3000
    summary_chars: int = SUMMARY_CHAR_BUDGET
    memory_chars: int = MEMORY_CHAR_BUDGET


def estimate_context_chars(blocks: ContextBlocks) -> int:
    """估算喂给 LLM 的上下文总量（字符近似，token 约 /2~3）。"""
    body = sum(
        len(s)
        for s in (blocks.system, blocks.profile, blocks.summary,
                  blocks.session, blocks.evidence, blocks.subtasks, blocks.memory)
    )
    history = sum(
        len(str(getattr(m, "content", "") or "")) for m in blocks.history
    )
    return body + history


def apply_budget(blocks: ContextBlocks, budget: ContextBudget | None = None) -> ContextBlocks:
    """把块压进总量预算：先自适应缩历史窗口（8→6→4→2，保底 2 轮），
    再按优先级把文本块截到各自预算（summary → evidence → subtasks）。"""
    budget = budget or ContextBudget()
    out = ContextBlocks(
        system=blocks.system,
        profile=blocks.profile,
        summary=blocks.summary,
        session=blocks.session,
        evidence=blocks.evidence,
        subtasks=blocks.subtasks,
        memory=blocks.memory,
        history=list(blocks.history),
    )
    # 1) 自适应历史窗口：从最大轮数递减到最小轮数，找到首个不超预算的窗口
    for turns in range(budget.history_max_turns, budget.history_min_turns - 1, -2):
        cand = trim_history(out.history, max_turns=turns)
        out.history = cand
        if estimate_context_chars(out) <= budget.total_chars:
            break
    # 2) 文本块按优先级截断（secondary 保险；主预算在 build 时已生效）
    for field, cap in (
        ("summary", budget.summary_chars),
        ("evidence", budget.evidence_chars),
        ("subtasks", budget.subtasks_chars),
        ("memory", budget.memory_chars),
    ):
        cur = getattr(out, field)
        if len(cur) > cap:
            if cap > 24:
                setattr(out, field, cur[: cap - 16] + "…（已截断）")
            else:
                setattr(out, field, cur[:cap])
    return out


def _artwork_key(item: dict) -> str:
    """去重键：artwork_id 优先，否则 title+author 小写。"""
    aid = str(item.get("artwork_id") or "").strip()
    if aid:
        return f"id:{aid}"
    title = str(item.get("title") or "").strip().lower()
    author = str(item.get("author") or str(item.get("artist") or "")).strip().lower()
    return f"{title}|{author}"


def dedup_artworks(items: list[dict]) -> list[dict]:
    """按去重键保留首个（relevance_score 最高者优先保留）。"""
    best: dict[str, dict] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = _artwork_key(item)
        if key == "|":
            continue  # 无任何标识的裸条目不参与去重展示
        prev = best.get(key)
        if prev is None:
            best[key] = item
            continue
        cur_score = item.get("relevance_score")
        prev_score = prev.get("relevance_score")
        try:
            if cur_score is not None and (
                prev_score is None or float(cur_score) > float(prev_score)
            ):
                best[key] = item
        except (TypeError, ValueError):
            pass
    return list(best.values())


def format_numbered_evidence_block(
    items: list[dict],
    budget: int = EVIDENCE_CHAR_BUDGET,
) -> str:
    """编号引用证据块：- [N] title（author, date）：snippet。

    先按 _artwork_key 去重，再编号；超出预算时按顺序截断（保留前面高分者）。
    通用模板版（无编号、可自定义字段）见 src/data/access.py 的
    format_evidence_block，两者分工：本函数用于上下文证据注入。
    """
    unique = dedup_artworks(items)
    lines: list[str] = []
    used = 0
    for i, item in enumerate(unique, 1):
        title = str(item.get("title") or "(未命名)")
        author = str(item.get("author") or item.get("artist") or "")
        date = str(item.get("date") or "")
        snippet = str(
            item.get("description_snippet")
            or item.get("description")
            or item.get("content")
            or ""
        ).strip()
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN] + "..."
        head = f"- [{i}] {title}"
        if author:
            head += f"（{author}"
            if date:
                head += f", {date}"
            head += "）"
        line = f"{head}：{snippet}" if snippet else head
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def format_multi_evidence(
    grouped: dict[str, list[dict]],
    budget: int = EVIDENCE_CHAR_BUDGET,
) -> str:
    """多子任务证据块：按子问题分组、全局编号引用。

    【子任务1】对比莫奈和梵高的色彩
    - [1] ...
    【子任务2】推荐几幅类似莫奈的风景画
    - [2] ...
    """
    blocks: list[str] = []
    used = 0
    seq = 0
    for idx, (sub, items) in enumerate((grouped or {}).items(), 1):
        lines: list[str] = []
        for item in dedup_artworks(items):
            seq += 1
            title = str(item.get("title") or "(未命名)")
            author = str(item.get("author") or item.get("artist") or "")
            date = str(item.get("date") or "")
            snippet = str(
                item.get("description_snippet")
                or item.get("description")
                or item.get("content")
                or ""
            ).strip()
            if len(snippet) > SNIPPET_LEN:
                snippet = snippet[:SNIPPET_LEN] + "..."
            head = f"- [{seq}] {title}"
            if author:
                head += f"（{author}"
                if date:
                    head += f", {date}"
                head += "）"
            line = f"{head}：{snippet}" if snippet else head
            if used + len(line) + 24 > budget:  # 留出子任务标题的余量
                break
            lines.append(line)
            used += len(line)
        if lines:
            blocks.append(f"【子任务{idx}】{sub}\n" + "\n".join(lines))
        if used >= budget:
            break
    return "\n\n".join(blocks)


def build_profile_block(
    preferences: dict,
    budget: int = PROFILE_CHAR_BUDGET,
) -> str:
    """用户画像块：喜欢的画家 / 风格（带权重）。"""
    parts: list[str] = []
    # 新语义优先：preferences 为完整陈述（v1 移除后 memory_items 存句子）
    prefs = preferences.get("preferences") or []
    if prefs:
        parts.append("偏好：" + "；".join(str(p) for p in prefs[:8]))
        block = "；".join(parts)
        return block[:budget]
    artists = preferences.get("artists") or []
    styles = preferences.get("styles") or []
    if artists:
        parts.append(f"喜欢画家：{', '.join(str(a) for a in artists[:8])}")
    if styles:
        parts.append(f"偏好风格：{', '.join(str(s) for s in styles[:8])}")
    if not parts:
        return ""
    block = "；".join(parts)
    return block[:budget]


def build_memory_block(
    items: list[dict],
    budget: int = MEMORY_CHAR_BUDGET,
) -> str:
    """用户记忆块：检索注入的相关记忆，带来源与时间。"""
    if not items:
        return ""
    lines: list[str] = []
    used = 0
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        source = "用户明确" if item.get("source") == "user_explicit" else (
            "自动抽取" if item.get("source") == "extracted" else "评估数据"
        )
        try:
            from datetime import datetime

            updated = datetime.fromisoformat(str(item.get("updated_at") or ""))
            days = max(0, int((datetime.now().astimezone() - updated).total_seconds() // 86400))
            time_hint = "今天" if days == 0 else f"{days} 天前"
        except (TypeError, ValueError):
            time_hint = ""
        suffix = f"（来源：{source}" + (f" · {time_hint}" if time_hint else "") + "）"
        line = f"- {content}{suffix}"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def build_session_block(
    ledger: dict,
    budget: int = SESSION_CHAR_BUDGET,
) -> str:
    """会话台账块：本轮已展示画作 / 已推荐画家 / 待澄清项。"""
    parts: list[str] = []
    shown = ledger.get("shown_artworks") or []
    recommended = ledger.get("recommended_artists") or []
    pending = ledger.get("pending_clarification") or ""
    if shown:
        parts.append(f"已展示画作：{', '.join(str(x) for x in shown[:6])}")
    if recommended:
        parts.append(f"已推荐画家：{', '.join(str(x) for x in recommended[:6])}")
    if pending:
        parts.append(f"待澄清：{str(pending)[:80]}")
    docs = ledger.get("uploaded_docs") or []
    if docs:
        doc_parts: list[str] = []
        for d in docs[:3]:
            name = str(d.get("doc_name") or d.get("doc_id") or "文档")
            pages = d.get("pages")
            img = int(d.get("image_pages") or 0)
            txt = int(d.get("text_chunks") or 0)
            hint = "（无文字索引，需 read_page_image 逐页读图）" if img > 0 and txt == 0 else ""
            doc_parts.append(f"{name}{f'({pages}页)' if pages else ''}{hint}")
        parts.append("已上传文档：" + "；".join(doc_parts))
    images = ledger.get("uploaded_images") or []
    if images:
        img_parts: list[str] = []
        for im in images[:3]:
            name = str(im.get("original_name") or im.get("image_id") or "图片")
            iid = str(im.get("image_id") or "")
            img_parts.append(f"{name}(id={iid})")
        parts.append("用户图片：" + "；".join(img_parts))
    reports = ledger.get("analysis_reports") or []
    if reports:
        rep_parts: list[str] = []
        for r in reports[:3]:
            iid = str(r.get("image_id") or "")
            fw = str(r.get("framework") or "")
            rep_parts.append(f"{iid}(framework={fw})")
        parts.append("已有分析报告：" + "；".join(rep_parts))
    sid = str(ledger.get("session_id") or "")
    if sid:
        parts.append(f"会话ID：{sid}")
    if not parts:
        return ""
    return "；".join(parts)[:budget]


def build_summary_block(
    summary: str,
    budget: int = SUMMARY_CHAR_BUDGET,
) -> str:
    """会话滚动摘要块（未生成时调用方传空字符串）。"""
    if not summary:
        return ""
    return str(summary)[:budget]


def format_skills_index(skills) -> str:
    """技能索引块：让 agent 知道系统具备哪些可点名的技能。"""
    if not skills:
        return ""
    lines = [
        "可用技能（按需调用 skill_<id> 执行完整流程；"
        "在消息开头输入 /<技能id> 可强制激活）："
    ]
    for skill in skills:
        desc = skill.description or skill.name
        lines.append(f"- skill_{skill.id}：{desc}")
    return "\n".join(lines)


def trim_history(messages, max_turns: int = HISTORY_MAX_TURNS):
    """历史窗口裁剪：保留开头的 system 消息 + 最近 max_turns 轮消息。

    简单保守策略（不拆散 tool_call 与 ToolMessage 的配对）：
    取尾部 max_turns*2 条消息，前缀的 system 消息保留。
    """
    if not messages:
        return []
    head: list = []
    rest: list = list(messages)
    while rest and getattr(rest[0], "type", "") == "system":
        head.append(rest.pop(0))
    tail = rest[-(max_turns * 2) :] if max_turns > 0 else []
    return head + tail


def _collect_artwork_dicts(obj, out: list[dict]) -> None:
    """递归收集"画作形状"的字典（有 title 且带 author/artist/描述/图任一字段）。"""
    if isinstance(obj, dict):
        if isinstance(obj.get("title"), str) and obj.get("title"):
            if any(
                k in obj
                for k in ("author", "artist", "description", "description_snippet", "image_file")
            ):
                out.append(obj)
        for value in obj.values():
            _collect_artwork_dicts(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_artwork_dicts(value, out)


def extract_evidence_from_messages(messages) -> list[dict]:
    """从历史 ToolMessage 中抽取画作证据（兼容列表、嵌套 evidence/candidates/periods）。"""
    import json

    out: list[dict] = []
    for msg in messages or []:
        if getattr(msg, "type", "") != "tool":
            continue
        try:
            data = json.loads(str(msg.content))
        except Exception:
            continue
        _collect_artwork_dicts(data, out)
    return out


def condense_tool_messages(messages, limit: int = 300):
    """把长 JSON 工具结果压成短占位（保持 tool_call_id 配对，完整证据走 evidence 块）。

    非 JSON 内容（如文本错误）原样保留；守卫消息短小不受影响。
    仅用于喂给 LLM 的副本，state.messages 保持不变（UI 解析仍读原始内容）。
    """
    import json

    from langchain_core.messages import ToolMessage

    out: list = []
    for msg in messages or []:
        if getattr(msg, "type", "") == "tool":
            content = str(msg.content or "")
            is_json = False
            try:
                json.loads(content)
                is_json = True
            except Exception:
                pass
            if is_json and len(content) > limit:
                out.append(
                    ToolMessage(
                        content=content[:limit] + "…（完整证据见【evidence】块）",
                        name=msg.name,
                        tool_call_id=msg.tool_call_id,
                        id=msg.id,
                    )
                )
                continue
        out.append(msg)
    return out


@dataclass
class ContextBlocks:
    """一次组装的全部上下文块（供 general_agent 拼接消息）。"""

    system: str = ""
    profile: str = ""
    summary: str = ""
    session: str = ""
    evidence: str = ""
    subtasks: str = ""
    memory: str = ""
    history: list = field(default_factory=list)

    def to_system_messages(self):
        """把非空块拼成 SystemMessage 列表（evidence 独立成块便于引用编号）。"""
        from langchain_core.messages import SystemMessage

        msgs: list = []
        for label, content in (
            ("system", self.system),
            ("profile", self.profile),
            ("summary", self.summary),
            ("session", self.session),
            ("evidence", self.evidence),
            ("subtasks", self.subtasks),
            ("memory", self.memory),
        ):
            if not content:
                continue
            body = content if label == "system" else f"【{label}】\n{content}"
            msgs.append(SystemMessage(content=body))
        return msgs
