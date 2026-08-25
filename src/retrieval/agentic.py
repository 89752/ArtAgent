"""Bounded agentic retrieval: coverage check, one rewrite, and evidence merge."""

from __future__ import annotations

import re
from typing import Callable

from src.utils.config import get_bool, get_int


def _query_terms(query: str) -> set[str]:
    """Extract comparable English words and Chinese two-character concepts."""
    latin = set(re.findall(r"[A-Za-z0-9_-]{3,}", query.lower()))
    chinese = {
        chunk[i:i + 2]
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", query)
        for i in range(len(chunk) - 1)
    }
    return latin | chinese


def coverage_check(query: str, items: list[dict], *, min_evidence: int = 3) -> dict:
    """Deterministically decide whether a first retrieval has enough evidence."""
    useful = [item for item in items if isinstance(item, dict) and (item.get("title") or item.get("content"))]
    query_terms = _query_terms(query)
    text = " ".join(str(i.get("title") or "") + " " + str(i.get("description_snippet") or i.get("content") or "") for i in useful).lower()
    covered = {term for term in query_terms if term in text}
    return {
        "sufficient": len(useful) >= min_evidence and (not query_terms or len(covered) >= max(1, len(query_terms) // 2)),
        "evidence_count": len(useful),
        "covered_terms": sorted(covered),
        "missing_terms": sorted(query_terms - covered),
    }


def rewrite_query(query: str, missing_terms: list[str], llm=None) -> str:
    """Generate a single conservative rewrite; failures use the original query."""
    if not missing_terms:
        return query
    prompt = (
        "重写艺术史检索词，保留原意，仅补足缺失概念。只输出检索词，不解释。\n"
        f"原查询：{query}\n缺失概念：{', '.join(missing_terms)}"
    )
    try:
        if llm is None:
            from src.utils.llm import get_cheap_llm
            llm = get_cheap_llm()
        candidate = str(llm.invoke(prompt).content).strip().replace("\n", " ")
        return candidate[:500] or query
    except Exception:
        return query


def adaptive_retrieve(
    query: str,
    retrieve: Callable[[str], list[dict]],
    *,
    llm=None,
) -> tuple[list[dict], dict]:
    """Run at most two retrievals, preserving evidence order and provenance."""
    first = retrieve(query)
    if not get_bool("retrieval.agentic_enabled", True):
        coverage = coverage_check(query, first, min_evidence=get_int("retrieval.agentic_min_evidence", 3, lo=1, hi=20))
        return first, {**coverage, "rewritten": False, "disabled": True}
    coverage = coverage_check(
        query,
        first,
        min_evidence=get_int("retrieval.agentic_min_evidence", 3, lo=1, hi=20),
    )
    if coverage["sufficient"]:
        return first, {**coverage, "rewritten": False}
    rewritten = rewrite_query(query, coverage["missing_terms"], llm=llm)
    if rewritten == query:
        return first, {**coverage, "rewritten": False}
    second = retrieve(rewritten)
    seen: set[tuple] = set()
    merged: list[dict] = []
    for item in first + second:
        key = (item.get("source"), item.get("doc_id"), item.get("page"), item.get("title"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged, {**coverage_check(query, merged), "rewritten": True, "rewrite_query": rewritten}
