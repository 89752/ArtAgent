"""Agent 行为化 eval（带指标，Phase 5）。

输出三类量化指标（写 eval/agentic_metrics.md）：
1. 行为触发率：每个能力用例跑 N 次，统计"期望行为是否发生"的比例
   （如澄清触发率、技能调用率、RAG-gate 关闭率）；
2. 效率指标：平均耗时、平均工具轮次（成本/延迟代理）；
3. 事实准确率：ground-truth 精确查询（长尾元数据）经完整 Agent 链路后
   答案是否命中期望实体，报告准确率。

用法：
    python eval/agentic_eval.py              # 行为用例 N=2 + 事实题
    python eval/agentic_eval.py --runs 1     # 快速模式
    python eval/agentic_eval.py --no-facts   # 只跑行为指标
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import get_graph


def _tool_names(messages) -> list[str]:
    out: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.extend(t["name"] for t in m.tool_calls)
    return out


BEHAVIOR_CASES: list[dict] = [
    {"name": "rag_gate", "question": "你好",
     "expect": {"rag_off": True, "no_tools": True}},
    {"name": "clarify", "question": "给我推荐几幅画",
     "expect": {"ask": True, "no_tools": True}},
    {"name": "multi_intent", "question": "对比莫奈和梵高，顺便推荐几幅类似的画",
     "expect": {"multi": True, "tools_any": ["compare_subjects", "recommend_with_exclusions"]}},
    {"name": "skill", "question": "帮我深度分析一下梵高的《星夜》",
     "expect": {"tools_any": ["skill_artwork_deep_analysis"]}},
    {"name": "memory_write", "question": "记住我特别喜欢莫奈的睡莲系列",
     "expect": {"tools_any": ["remember"]}},
    {"name": "grain_paintings", "question": "推荐几幅浓烈奔放的画",
     "expect": {"recommend": True, "answer_has_book_titles": True}},
    {"name": "collection", "question": "帮我把《睡莲》和《日出·印象》收藏为清单，名字叫印象派",
     "expect": {"tools_any": ["save_collection"]}},
]


# 事实准确率：长尾/精确元数据，期望实体必须出现在最终回答里
FACT_CASES: list[dict] = [
    {"q": "《The Assumption Altarpiece》的作者是谁？", "gt": ["moretto"]},
    {"q": "MET 收藏的《Cityscape》（2008.359.25）的作者是谁？", "gt": ["anonymous", "无名"]},
    {"q": "《Woman with a Pink》是哪一年创作的？", "gt": ["1660"]},
    {"q": "《The Davis Madonna》是用什么材料画的？", "gt": ["tempera", "蛋彩", "panel", "木板"]},
    {"q": "《Portrait of a Man with Gloves in Hand》是哪位画家画的？", "gt": ["rembrandt", "伦勃朗"]},
    {"q": "Gillis van Coninxloo 的森林风景画有什么特点？", "gt": ["forest", "森林", "wood", "树木"]},
]


def _invoke(graph, question: str, thread_id: str) -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "tool_results": [],
            "final_answer": "",
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def _check(case: dict, result: dict) -> tuple[bool, list[str]]:
    exp = case["expect"]
    fails: list[str] = []
    tools = _tool_names(result.get("messages") or [])
    answer = str(result.get("final_answer") or "")
    if exp.get("rag_off") and result.get("rag_needed") is not False:
        fails.append("期望 rag_needed=False")
    if exp.get("ask") and result.get("ask_user") != "ask":
        fails.append("期望触发澄清")
    if exp.get("no_tools") and tools:
        fails.append(f"期望零工具调用，实际 {tools}")
    if exp.get("multi"):
        if len(result.get("sub_questions") or []) < 2:
            fails.append("期望拆分子问题≥2")
        if not result.get("multi_evidence"):
            fails.append("期望 multi_evidence 非空")
    for t in exp.get("tools_any", []):
        if t not in tools:
            fails.append(f"期望工具 {t}，实际 {tools}")
    if exp.get("recommend") and "recommend_with_exclusions" not in tools:
        fails.append("期望 recommend_with_exclusions")
    if exp.get("answer_has_book_titles") and answer.count("《") < 2:
        fails.append("期望画作粒度（书名号≥2）")
    return not fails, fails


def _gt_hit(answer: str, gts: list[str]) -> bool:
    a = (answer or "").lower()
    return any(g.lower() in a for g in gts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 行为化 eval（带指标）")
    parser.add_argument("--runs", type=int, default=2, help="每个行为用例重复次数")
    parser.add_argument("--no-facts", action="store_true", help="跳过事实准确率")
    args = parser.parse_args()

    graph = get_graph()
    lines: list[str] = [
        "# Agent 行为化 eval 指标报告",
        "",
        f"- 行为用例重复次数：{args.runs}",
        f"- 事实题：{'跳过' if args.no_facts else f'{len(FACT_CASES)} 题'}",
        "",
    ]

    # ── 1. 行为触发率 + 效率指标 ──
    lines.append("## 1. 行为指标（触发率 / 平均耗时 / 平均工具轮次）")
    lines.append("")
    lines.append("| 用例 | 触发率 | 平均耗时(s) | 平均工具轮次 |")
    lines.append("|---|---|---|---|")
    all_ok = all_runs = 0
    for case in BEHAVIOR_CASES:
        name = case["name"]
        question = case["question"]
        ok_n = 0
        seconds: list[float] = []
        rounds: list[int] = []
        for i in range(args.runs):
            t0 = time.time()
            result = _invoke(graph, question, f"eval-{name}-{i}")
            seconds.append(time.time() - t0)
            rounds.append(result.get("tool_rounds") or 0)
            ok, fails = _check(case, result)
            ok_n += int(ok)
            all_runs += 1
            all_ok += int(ok)
            print(f"[{name}] run{i+1}/{args.runs} {'✅' if ok else '❌'} "
                  f"{'；'.join(fails) if fails else ''}")
        rate = ok_n / args.runs
        avg_s = sum(seconds) / len(seconds)
        avg_r = sum(rounds) / len(rounds)
        lines.append(f"| {name} | {rate:.0%} | {avg_s:.1f} | {avg_r:.1f} |")
    lines.append("")
    lines.append(f"行为整体通过率：{all_ok}/{all_runs}（{all_ok / max(all_runs, 1):.0%}）")
    lines.append("")

    # ── 2. 事实准确率 ──
    if not args.no_facts:
        lines.append("## 2. 事实准确率（长尾/精确元数据，完整 Agent 链路）")
        lines.append("")
        lines.append("| 问题 | 命中 |")
        lines.append("|---|---|")
        hits = 0
        for i, fact in enumerate(FACT_CASES):
            result = _invoke(graph, fact["q"], f"eval-fact-{i}")
            answer = str(result.get("final_answer") or "")
            hit = _gt_hit(answer, fact["gt"])
            hits += int(hit)
            lines.append(f"| {fact['q'][:36]} | {'✅' if hit else '❌'} |")
        lines.append("")
        lines.append(f"事实准确率：{hits}/{len(FACT_CASES)}（{hits / len(FACT_CASES):.0%}）")
        lines.append("")

    report = Path("eval/agentic_metrics.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n指标报告已写入：{report}")


if __name__ == "__main__":
    main()
