"""本地向量库 vs Web 检索 vs 纯 LLM 三方对比（2026-08-02）。

目的：回答"直接调 LLM / 网上检索，是否也能得到差不多质量的回答"。
两类问题：
  - exact：指名道姓/长尾元数据（部分带 ground_truth，自动核对）
  - general：名画通识（人工比对）
三路：
  - local：核心库语义检索 → 证据注入 → LLM
  - web：Tavily 联网检索 → 结果注入 → LLM
  - llm：直接调 LLM（不注入任何证据）
输出：eval/rag_threeway_report.md

用法：
    python eval/rag_threeway.py             # 全部 12 题
    python eval/rag_threeway.py --limit 2   # 只跑前 2 题
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.context import format_evidence_block
from src.retrieval.hybrid import get_hybrid_retriever
from src.tools.web_search import _search_impl
from src.utils.llm import get_llm


def _format_web_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"- [{i}] {r.get('title', '')}\n  {str(r.get('snippet', ''))[:400]}"
        )
    return "\n".join(lines)


EVIDENCE_PROMPT = """你是艺术史助手。请基于以下【检索证据】回答问题。
证据不足或证据与问题无关时，明确说明，不要编造。

【证据】
{evidence}

【问题】
{question}

回答："""

DIRECT_PROMPT = """你是艺术史助手。请直接回答下面的问题。

【问题】
{question}

回答："""


CASES: list[dict] = [
    # ── exact：指名道姓 / 长尾元数据 ──
    {"cat": "exact", "q": "《Portrait of a Man with Gloves in Hand》是哪位画家画的？",
     "gt": ["Rembrandt"]},
    {"cat": "exact", "q": "《The Assumption Altarpiece》的作者是谁？",
     "gt": ["Moretto"]},
    {"cat": "exact", "q": "《Woman with a Pink》是哪一年创作的？",
     "gt": ["1660"]},
    {"cat": "exact", "q": "Gillis van Coninxloo 的森林风景画有什么特点？",
     "gt": ["forest", "wood", "tree"]},
    {"cat": "exact", "q": "MET 收藏的《Cityscape》（2008.359.25）的作者是谁？",
     "gt": ["anonymous", "无名"]},
    {"cat": "exact", "q": "《The Davis Madonna》是用什么材料画的？",
     "gt": ["tempera", "panel", "木板"]},
    # ── general：名画通识 ──
    {"cat": "general", "q": "什么是巴洛克风格？", "gt": []},
    {"cat": "general", "q": "莫奈和梵高的色彩风格有什么不同？", "gt": []},
    {"cat": "general", "q": "印象派有哪些特点？", "gt": []},
    {"cat": "general", "q": "《星夜》的创作背景是什么？", "gt": []},
    {"cat": "general", "q": "伦勃朗的风格演变是怎样的？", "gt": []},
    {"cat": "general", "q": "提香的绘画有什么特点？", "gt": []},
]


def _gt_hit(answer: str, gts: list[str]) -> bool:
    a = (answer or "").lower()
    return any(g.lower() in a for g in gts)


def run_local(question: str) -> dict:
    t0 = time.time()
    hybrid = get_hybrid_retriever()
    hits = hybrid.search(question, top_k=5, rerank=False)
    evidence = format_evidence_block([h.metadata for h in hits])
    prompt = EVIDENCE_PROMPT.format(evidence=evidence or "(无检索结果)", question=question)
    answer = get_llm(0.3).invoke(prompt).content
    titles = [str(h.metadata.get("title") or "") for h in hits[:3]]
    return {"answer": answer, "seconds": round(time.time() - t0, 1), "titles": titles}


def run_web(question: str) -> dict:
    t0 = time.time()
    results = _search_impl(question)
    evidence = _format_web_results(results)
    prompt = EVIDENCE_PROMPT.format(evidence=evidence, question=question)
    answer = get_llm(0.3).invoke(prompt).content
    titles = [str(r.get("title") or "") for r in results[:3]]
    return {"answer": answer, "seconds": round(time.time() - t0, 1), "titles": titles}


def run_llm(question: str) -> dict:
    t0 = time.time()
    prompt = DIRECT_PROMPT.format(question=question)
    answer = get_llm(0.3).invoke(prompt).content
    return {"answer": answer, "seconds": round(time.time() - t0, 1), "titles": []}


RUNNERS = {"local": run_local, "web": run_web, "llm": run_llm}


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 vs Web vs 纯 LLM 三方对比")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = CASES[: args.limit] if args.limit > 0 else CASES
    lines: list[str] = ["# 本地向量库 vs Web 检索 vs 纯 LLM 对比报告", ""]
    summary: list[str] = ["| # | 类别 | 问题 | local | web | llm |"]
    summary.append("|---|---|---|---|---|---|")

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] ({case['cat']}) {case['q']}")
        results = {name: runner(case["q"]) for name, runner in RUNNERS.items()}

        lines.append(f"## {i}. {case['q']}")
        lines.append(f"- 类别：{case['cat']}｜ground_truth：{case['gt'] or '(人工比对)'}")
        for name in ("local", "web", "llm"):
            r = results[name]
            hit = "✅" if case["gt"] and _gt_hit(r["answer"], case["gt"]) else ""
            lines.append(
                f"- **{name}**（{r['seconds']}s{hit}）："
                f"{str(r['answer'])[:500]}"
            )
            if r["titles"]:
                lines.append(f"  - 来源标题：{'；'.join(r['titles'])}")
        lines.append("")

        marks = []
        for name in ("local", "web", "llm"):
            r = results[name]
            if case["gt"]:
                marks.append("✅" if _gt_hit(r["answer"], case["gt"]) else "❌")
            else:
                marks.append("—")
        summary.append(
            f"| {i} | {case['cat']} | {case['q'][:26]} | {marks[0]} | {marks[1]} | {marks[2]} |"
        )

    report = Path("eval/rag_threeway_report.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"\n报告已写入：{report}")


if __name__ == "__main__":
    main()
