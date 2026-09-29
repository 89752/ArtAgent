"""Explicit quality decisions; an empty or skipped evaluation cannot pass."""
import json
from pathlib import Path


def public_retrieval():
    from src.retrieval.lexical import _tokenize, _bm25_scores
    cases = json.loads((Path(__file__).parent/"sets"/"public_retrieval.json").read_text(encoding="utf-8"))
    docs = [_tokenize(c["text"]) for c in cases]
    hits = 0
    for i,case in enumerate(cases):
        scores = _bm25_scores(_tokenize(case["query"]),docs)
        hits += bool(scores) and scores[i]>0 and max(range(len(scores)),key=scores.__getitem__)==i
    return {"total":len(cases),"hits":hits,"recall_at_k":hits/len(cases) if cases else 0,"top_k":1,
            "source":"public_lexical_fixture", "scope":"词法检索契约；不代表完整混合检索质量"}


def gate_decision(parts, intent=None, *, min_success=.9, max_skip=.05):
    failures = []
    if not parts and not intent:
        failures.append("not_evaluated")
    if intent is not None and (not intent.get("total") or intent.get("match",0)/intent["total"] < min_success):
        failures.append("intent_quality")
    for name in ("retrieval","public_retrieval"):
        if name in parts and (not parts[name].get("total") or parts[name].get("recall_at_k",0)<min_success):
            failures.append(name)
    for name in ("cases","multi_turn"):
        if name not in parts:
            continue
        part = parts[name]
        total = part.get("total",0)
        if not total or part.get("skipped",0)/total > max_skip:
            failures.append(name+":coverage")
        rows = part.get("rows",[])
        successes = sum(bool(r.get("task_success",r.get("ok",False))) and r.get("grounded",True) and r.get("score",5)>=3 for r in rows)
        if not total or successes/total < min_success:
            failures.append(name+":quality")
    return {"passed":not failures,"failures":failures}
