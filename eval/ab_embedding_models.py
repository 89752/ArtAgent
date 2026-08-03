"""用户文档文本通道 embedding 模型 A/B：bge-small-en（现状） vs bge-m3（候选）。

语料：莫奈手稿 OCR 40 chunks（doc 4fecfaa2b111）。
指标：raw cosine 排序下的 Recall@1/3/5、MRR、最佳 gold 平均/中位排名；
     附带生产重排（qwen3-rerank）对照。当候选池覆盖全部 40 chunks 时，
     rerank 输出与 embedding 模型无关，故只跑一次。

用法：python eval/ab_embedding_models.py
输出：eval/ab_embedding_models_report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid import get_or_create_chroma_collection
from src.retrieval.reranker import rerank, rerank_available
from src.retrieval.userdoc_text_retriever import COLLECTION_NAME

BASE = Path(__file__).resolve().parent
TESTSET = BASE / "embed_ab_testset.json"
REPORT = BASE / "ab_embedding_models_report.md"
MAX_SEQ = 1024  # 与生产 _get_bge_m3_model 一致
MODELS = ("bge-small-en", "bge-m3")


def load_corpus():
    col = get_or_create_chroma_collection(COLLECTION_NAME)
    hits = col.get(include=["documents", "metadatas"])
    ids = hits["ids"]
    docs = [(d or "") for d in hits["documents"]]
    metas = hits["metadatas"]
    return ids, docs, metas


def build_model(name: str):
    if name == "bge-small-en":
        return SentenceTransformer("BAAI/bge-small-en-v1.5")
    model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    model.max_seq_length = MAX_SEQ
    if torch.cuda.is_available():
        model.half()
    return model


def encode(model, texts):
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def best_rank(sim_row, golds) -> int | None:
    gold_set = set(golds)
    order = np.argsort(-sim_row)
    for pos, idx in enumerate(order):
        if int(idx) in gold_set:
            return pos + 1
    return None


def summarize(ranks: list) -> dict:
    n = len(ranks)
    valid = [r for r in ranks if r is not None]
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "recall@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1.0 / r for r in valid) / n,
        "mean_rank": (sum(valid) / len(valid)) if valid else float("nan"),
        "median_rank": (float(np.median(valid)) if valid else float("nan")),
    }


def fmt(s: dict) -> str:
    return (
        f"R@1 {s['recall@1']:.2f} | R@3 {s['recall@3']:.2f} | R@5 {s['recall@5']:.2f} "
        f"| MRR {s['mrr']:.3f} | meanRank {s['mean_rank']:.1f} | medRank {s['median_rank']:.0f}"
    )


def main() -> None:
    ids, docs, metas = load_corpus()
    tests = json.loads(TESTSET.read_text(encoding="utf-8"))["queries"]
    queries = [t["q"] for t in tests]
    golds = [t["gold"] for t in tests]
    groups = [t["group"] for t in tests]
    n = len(queries)
    print(f"corpus: {len(docs)} chunks | queries: {n} ({sum(1 for g in groups if g=='fragment')} fragment + {sum(1 for g in groups if g=='paraphrase')} paraphrase)")

    results: dict[str, dict] = {}
    for name in MODELS:
        t0 = time.time()
        model = build_model(name)
        doc_vec = encode(model, docs)
        q_vec = encode(model, queries)
        sim = q_vec @ doc_vec.T
        ranks = [best_rank(sim[i], golds[i]) for i in range(n)]
        results[name] = {"ranks": ranks, "load_sec": round(time.time() - t0, 1)}
        print(f"[{name}] load+encode {results[name]['load_sec']}s | overall {fmt(summarize(ranks))}")

    # 生产重排对照（候选池 = 全部 chunks；两个模型输入相同 → 输出相同，只跑一次）
    rr_ranks: list = []
    rr_sec = 0.0
    if rerank_available():
        t0 = time.time()
        for i in range(n):
            ranked = rerank(queries[i], docs, top_n=len(docs))
            if ranked is None:
                rr_ranks.append(None)
                continue
            gold_set = set(golds[i])
            rr_ranks.append(
                next((pos + 1 for pos, (idx, _s) in enumerate(ranked) if int(idx) in gold_set), None)
            )
        rr_sec = round(time.time() - t0, 1)
        print(f"[rerank(qwen3)] {rr_sec}s | overall {fmt(summarize(rr_ranks))}")

    # ---- 写报告 ----
    lines: list[str] = []
    A = lines.append
    A("# 用户文档文本通道 embedding 模型 A/B：bge-small-en vs bge-m3")
    A("")
    A(f"日期：2026-08-03　语料：莫奈手稿 OCR **{len(docs)} chunks**（doc `4fecfaa2b111`，16 页扫描件，中文）")
    A(f"测试集：**{n} 题**（{sum(1 for g in groups if g=='fragment')} 片段召回 + {sum(1 for g in groups if g=='paraphrase')} 改写提问），gold 由人工按内容标注。")
    A("")
    A("## 方法")
    A("- 同一语料分别用 bge-small-en（现状）与 bge-m3（候选，`max_seq_length=1024` + GPU fp16，与生产 `_get_bge_m3_model` 一致）编码；")
    A("- 指标按 **raw cosine 排序**（无重排）计算：Recall@1/3/5、MRR、最佳 gold 平均/中位排名；")
    A("- 另跑一次生产 qwen3-rerank 对照：候选池=全部 40 chunks 时，rerank 输出与 embedding 模型无关（输入相同），故只跑一次。")
    A("")
    A("## 总体结果（raw cosine，无重排）")
    A("")
    A("| 模型 | 分组 | R@1 | R@3 | R@5 | MRR | meanRank | medRank |")
    A("|---|---|---|---|---|---|---|---|")
    for name in MODELS:
        ranks = results[name]["ranks"]
        for gname in ("fragment", "paraphrase", "all"):
            idxs = [i for i in range(n) if (gname == "all" or groups[i] == gname)]
            s = summarize([ranks[i] for i in idxs])
            label = {"fragment": "片段召回", "paraphrase": "改写提问", "all": "全部"}[gname]
            A(f"| {name} | {label} | {s['recall@1']:.2f} | {s['recall@3']:.2f} | {s['recall@5']:.2f} | {s['mrr']:.3f} | {s['mean_rank']:.1f} | {s['median_rank']:.0f} |")
    if rr_ranks:
        s = summarize(rr_ranks)
        A(f"| rerank(qwen3) | 全部（一次） | {s['recall@1']:.2f} | {s['recall@3']:.2f} | {s['recall@5']:.2f} | {s['mrr']:.3f} | {s['mean_rank']:.1f} | {s['median_rank']:.0f} |")
    A("")
    A("## 逐题对照（最佳 gold 排名，越小越好）")
    A("")
    A("| # | 分组 | 问题 | gold | small-en | m3 | 胜者 |")
    A("|---|---|---|---|---|---|---|")
    for i in range(n):
        r_s = results["bge-small-en"]["ranks"][i]
        r_m = results["bge-m3"]["ranks"][i]
        winner = "平" if r_s == r_m else ("m3" if (r_m or 999) < (r_s or 999) else "small-en")
        q = queries[i] if len(queries[i]) <= 34 else queries[i][:33] + "…"
        A(f"| {i} | {'片段' if groups[i]=='fragment' else '改写'} | {q} | {golds[i]} | {r_s or '-'} | {r_m or '-'} | {winner} |")
    A("")
    A("## 耗时")
    A("")
    for name in MODELS:
        A(f"- {name}：模型加载 + 全部编码 {results[name]['load_sec']}s")
    if rr_ranks:
        A(f"- qwen3-rerank：{n} 题 × 40 候选 {rr_sec}s")
    A("")
    A("## 结论与建议")
    A("")
    A("（结论由脚本输出后人工补充：见最终答复。）")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
