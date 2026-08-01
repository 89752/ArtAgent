"""
ArtAgent 离线评估脚本。

两个可量化指标（简历用真实数字，非"能跑通"级别）：
  1. 意图分类   —— 40 条标注集，报总准确率 + 各类 P/R/F1 + 混淆矩阵。
                   只调用 classify_intent（每条一次确定性 LLM 调用），无需向量库。
  2. 已知项检索 —— 从 SemArt 随机抽 N 幅画，用其描述片段作 query，
                   看原画能否命中 semantic_search 的 top-k（Recall@k）。
                   全自动标注、客观，衡量向量检索质量。

用法：
    python eval/run_eval.py                 # 跑全部
    python eval/run_eval.py --no-retrieval  # 只跑意图分类（快，不加载向量库）
    python eval/run_eval.py --retrieval-n 30
结果同时打印到控制台并写入 eval/results.md。
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# 允许 `python eval/run_eval.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTENTS = ["comparison", "timeline", "recommendation", "general"]
EVAL_DIR = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════
# 指标 1：意图分类
# ══════════════════════════════════════════════════════════════════
def eval_intent() -> dict:
    from src.agent.state import AgentState
    from src.agent.nodes.common import classify_intent

    testset = json.loads((EVAL_DIR / "intent_testset.json").read_text(encoding="utf-8"))
    rows = []
    correct = 0
    # confusion[gold][pred] = count
    confusion = {g: defaultdict(int) for g in INTENTS}

    print(f"\n▶ 意图分类评估（{len(testset)} 条）...")
    for i, case in enumerate(testset, 1):
        state = AgentState(user_query=case["query"])
        pred = classify_intent(state)["intent"]
        gold = case["gold"]
        ok = pred == gold
        correct += ok
        confusion[gold][pred] += 1
        rows.append({"query": case["query"], "gold": gold, "pred": pred, "ok": ok})
        mark = "✓" if ok else "✗"
        print(f"  [{i:>2}/{len(testset)}] {mark} gold={gold:<14} pred={pred:<14} {case['query'][:34]}")

    accuracy = correct / len(testset)

    # 各类 precision / recall / f1
    per_class = {}
    for c in INTENTS:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in INTENTS if g != c)
        fn = sum(confusion[c][p] for p in INTENTS if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(confusion[c].values())
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(INTENTS)

    return {
        "total": len(testset),
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": {g: dict(confusion[g]) for g in INTENTS},
        "errors": [r for r in rows if not r["ok"]],
    }


# ══════════════════════════════════════════════════════════════════
# 指标 2：已知项检索（Recall@k）
# ══════════════════════════════════════════════════════════════════
def eval_retrieval(n: int = 20, top_k: int = 5, seed: int = 42) -> dict:
    from src.data.loader import get_dataset
    from src.retrieval.hybrid import get_hybrid_retriever

    df = get_dataset().all
    # 只取描述够长的样本，query 才有信息量
    usable = df[df["DESCRIPTION"].astype(str).str.len() > 120]
    random.seed(seed)
    idxs = random.sample(list(usable.index), min(n, len(usable)))

    # Stage 3 起 semantic_search 融合全部数据源；本指标衡量 SemArt 向量索引
    # 质量（基线 64.0%@n=25），必须锁定 semart 源，否则开发库里的用户 PDF
    # 会污染指标。
    hybrid = get_hybrid_retriever()

    print(f"\n▶ 已知项检索评估（{len(idxs)} 条，Recall@{top_k}，source=semart）...")
    hits = 0
    rows = []
    for i, idx in enumerate(idxs, 1):
        row = df.loc[idx]
        gold_title = str(row["TITLE"]).strip().lower()
        desc = str(row["DESCRIPTION"])
        # 用描述中段片段作 query，避免直接含标题
        query = desc[40:200]
        results = hybrid.search(query, top_k=top_k, sources=["semart"])
        returned = [
            str(r.metadata.get("title", "")).strip().lower() for r in results
        ]
        hit = gold_title in returned
        hits += hit
        rows.append({"gold": row["TITLE"], "hit": hit})
        mark = "✓" if hit else "✗"
        print(f"  [{i:>2}/{len(idxs)}] {mark} {str(row['TITLE'])[:44]}")

    return {"total": len(idxs), "hits": hits, "recall_at_k": hits / len(idxs) if idxs else 0.0, "top_k": top_k}


# ══════════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════════
def render_report(intent_res: dict, retrieval_res: dict | None) -> str:
    lines = []
    lines.append("# ArtAgent 评估结果\n")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # —— 意图分类 ——
    lines.append("## 1. 意图分类\n")
    lines.append(
        f"**准确率 {intent_res['accuracy']:.1%}**（{intent_res['correct']}/{intent_res['total']}）"
        f" · Macro-F1 {intent_res['macro_f1']:.3f}\n"
    )
    lines.append("| 意图 | Precision | Recall | F1 | 样本数 |")
    lines.append("|---|---|---|---|---|")
    for c, m in intent_res["per_class"].items():
        lines.append(
            f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |"
        )
    lines.append("")

    # 混淆矩阵
    lines.append("**混淆矩阵**（行=真实，列=预测）\n")
    header = "| gold \\ pred | " + " | ".join(INTENTS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(INTENTS) + 1))
    for g in INTENTS:
        cells = " | ".join(str(intent_res["confusion"][g].get(p, 0)) for p in INTENTS)
        lines.append(f"| **{g}** | {cells} |")
    lines.append("")

    if intent_res["errors"]:
        lines.append("**误分类样本**\n")
        for e in intent_res["errors"]:
            lines.append(f"- `{e['gold']}` → `{e['pred']}`：{e['query']}")
        lines.append("")

    # —— 检索 ——
    if retrieval_res:
        lines.append("## 2. 已知项检索\n")
        lines.append(
            f"**Recall@{retrieval_res['top_k']} = {retrieval_res['recall_at_k']:.1%}**"
            f"（{retrieval_res['hits']}/{retrieval_res['total']}）\n"
        )
        lines.append(
            "> 从 SemArt 随机抽画作，用其描述中段片段作 query，检验原画能否命中 "
            f"semantic_search 的 top-{retrieval_res['top_k']}。全自动标注，衡量向量检索质量。\n"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ArtAgent 离线评估")
    parser.add_argument("--no-retrieval", action="store_true", help="跳过检索评估（不加载向量库，更快）")
    parser.add_argument("--retrieval-n", type=int, default=20, help="检索评估抽样数量")
    args = parser.parse_args()

    t0 = time.time()
    intent_res = eval_intent()
    retrieval_res = None if args.no_retrieval else eval_retrieval(n=args.retrieval_n)

    report = render_report(intent_res, retrieval_res)
    out_path = EVAL_DIR / "results.md"
    out_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"意图分类准确率: {intent_res['accuracy']:.1%}  (Macro-F1 {intent_res['macro_f1']:.3f})")
    if retrieval_res:
        print(f"检索 Recall@{retrieval_res['top_k']}: {retrieval_res['recall_at_k']:.1%}")
    print(f"耗时 {time.time() - t0:.1f}s · 报告已写入 {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

