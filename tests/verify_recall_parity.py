# tests/verify_recall_parity.py
"""
Stage 2 验收辅助：验证 semantic_search 改走 HybridRetriever 后与旧实现
（直接 Chroma 查询）结果逐位一致；并用 n=25 复现 64.0% 基线口径。

只调本地 BGE + Chroma，不走 LLM。
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def old_semantic_search(query: str, top_k: int = 5) -> list[str]:
    """Stage 1 时期的旧实现（内联复刻，用于对照）。"""
    from src.retrieval.hybrid import get_chroma_collection, _get_bge_model

    collection = get_chroma_collection("semart")
    emb = _get_bge_model().encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )
    return [m["title"] for m in results["metadatas"][0]]


def new_semantic_search(query: str, top_k: int = 5) -> list[str]:
    from src.tools.retrieval import semantic_search

    results = semantic_search.invoke({"query": query, "top_k": top_k})
    return [r["title"] for r in results]


def eval_queries(n: int, seed: int = 42) -> list[str]:
    """与 eval/run_eval.py 完全一致的抽样与 query 构造。"""
    from src.data.loader import get_dataset

    df = get_dataset().all
    usable = df[df["DESCRIPTION"].astype(str).str.len() > 120]
    random.seed(seed)
    idxs = random.sample(list(usable.index), min(n, len(usable)))
    return [str(df.loc[i]["DESCRIPTION"])[40:200] for i in idxs]


def main():
    # ── 1. 新旧路径逐位对比（eval 同款 20 条 query + 额外 5 条） ──
    queries = eval_queries(20) + [
        "Renaissance painting with golden background",
        "impressionist landscape with haystacks",
        "portrait of a young woman with pearl earring",
        "baroque dramatic light and shadow scene",
        "still life with flowers and insects",
    ]
    mismatches = 0
    for i, q in enumerate(queries, 1):
        old_titles = old_semantic_search(q)
        new_titles = new_semantic_search(q)
        if old_titles != new_titles:
            mismatches += 1
            print(f"  ✗ [{i}] 不一致：\n      old={old_titles}\n      new={new_titles}")
    print(f"新旧路径对比：{len(queries)} 条 query，{mismatches} 条不一致")
    assert mismatches == 0, "新旧检索路径结果不一致，存在真实行为差异！"
    print("  ✓ 新旧实现结果逐位一致（标题、顺序、条数完全相同）")

    # ── 2. n=25 复现基线口径（64.0% = 16/25） ──
    from eval.run_eval import eval_retrieval

    res = eval_retrieval(n=25)
    print(f"\nn=25 Recall@5 = {res['recall_at_k']:.1%}（{res['hits']}/{res['total']}）")


if __name__ == "__main__":
    main()
