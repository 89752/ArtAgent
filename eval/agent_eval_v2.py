"""Agent 评估 v2（2026-08-03 重构，按《评价体系设计》实现）。

验收口径 = 最终答案质量 + 状态校验；意图分类仅作诊断（--diag）。
评估器：规则判题器 / LLM 裁判 / 轨迹评估器 / 状态断言（记忆表、收藏表）。
单条 API 失败只跳过不中断，报告标注有效样本数。

数据集目录：eval/sets/（answer_golden / fact / behavior / tool / intent_diag /
adversarial / multi_turn）。

用法：
    python eval/agent_eval_v2.py                    # 全量
    python eval/agent_eval_v2.py --pr               # PR 门禁档（离线检索 20 条）
    python eval/agent_eval_v2.py --retrieval-n 100  # 只跑检索
    python eval/agent_eval_v2.py --answers 5 --behavior-runs 2 --limit 10
    python eval/agent_eval_v2.py --diag             # 附带意图诊断
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Windows GBK 控制台打印 ✅/❌ 会崩，统一重配置为 UTF-8 输出
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import get_graph

EVAL_DIR = Path(__file__).resolve().parent
SETS = EVAL_DIR / "sets"
OUT = EVAL_DIR / "agent_eval_report.md"
HISTORY = EVAL_DIR / "metrics_history.jsonl"
INTENTS = ["comparison", "timeline", "recommendation", "general"]
SEED = 42
# 记忆身份隔离：评估全程使用 eval-test 身份，生产 web_user 不被污染
EVAL_MEMORY_USER = os.getenv("MEMORY_USER_ID", "eval-test")
os.environ["MEMORY_USER_ID"] = EVAL_MEMORY_USER


def _load_json(name: str, fallback: list) -> list:
    path = SETS / name
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    print(f"[warn] 数据集缺失或损坏：{path}，使用空集")
    return fallback


def _load_case_cache(name: str) -> dict[int, dict]:
    """按用例下标持久化的结果缓存：分块跑（--offset/--limit）自动累计合并。"""
    path = EVAL_DIR / "case_cache" / f"{name}.json"
    if path.exists():
        try:
            return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_case_cache(name: str, data: dict[int, dict]) -> None:
    path = EVAL_DIR / "case_cache" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({str(k): v for k, v in data.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


ANSWER_GOLDEN = _load_json("answer_golden.json", [])
FACT_CASES = _load_json("fact_testset.json", [])
BEHAVIOR_CASES = _load_json("behavior_testset.json", [])
TOOL_CASES = _load_json("tool_testset.json", [])
INTENT_DIAG = _load_json("intent_diag.json", [])
ADVERSARIAL = _load_json("adversarial.json", [])
MULTI_TURN = _load_json("multi_turn_golden.json", [])
ROUTE_CASES = _load_json("route_diag.json", [])


# ── 基础工具 ─────────────────────────────────────────────────────
def _safe_invoke(graph, question: str, thread_id: str) -> dict:
    try:
        return graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "user_query": question,
                "tool_results": [],
                "final_answer": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:  # noqa: BLE001 —— 单条失败只跳过
        print(f"[skip] {thread_id}：{e}")
        return {"messages": [], "final_answer": "", "error": str(e)}


def _tool_names(messages) -> list[str]:
    out: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.extend(t["name"] for t in m.tool_calls)
    return out


def _gt_hit(answer: str, gts: list[str]) -> bool:
    a = (answer or "").lower()
    return any(g.lower() in a for g in gts)


def _check_behavior(case: dict, result: dict) -> tuple[bool, list[str]]:
    exp = case.get("expect") or {}
    fails: list[str] = []
    tools = _tool_names(result.get("messages") or [])
    answer = str(result.get("final_answer") or "")
    if exp.get("rag_off") and result.get("rag_needed") is not False:
        fails.append("期望 rag_needed=False")
    if exp.get("ask") and result.get("ask_user") != "ask":
        fails.append("期望触发澄清")
    if exp.get("no_tools") and tools:
        fails.append(f"期望零工具调用，实际 {tools}")
    if exp.get("web_fallback") and not (
        result.get("web_results") or "web_search" in tools
    ):
        fails.append("期望联网兜底/搜索，实际未发生")
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


# ── 状态断言（τ-bench 方法论：不只看文案，看落库）──────────────
def _check_state(sc: dict) -> list[str]:
    fails: list[str] = []
    if not sc:
        return fails
    from src.memory.collections import list_collections
    from src.memory.store import list_preferences

    if "memory" in sc:
        prefs = list_preferences(EVAL_MEMORY_USER)
        text = json.dumps(prefs, ensure_ascii=False).lower()
        for kw in sc["memory"]:
            if kw.lower() not in text:
                fails.append(f"记忆表缺失关键词：{kw}")
    if "collection" in sc:
        cols = list_collections(EVAL_MEMORY_USER)
        want = str(sc["collection"]["name"]).lower()
        hit = next(
            (c for c in cols if want in str(c.get("name", "")).lower()),
            None,
        )
        if hit is None:
            fails.append(f"收藏表缺失清单：{want}")
        else:
            items = [str(i).lower() for i in (hit.get("items") or [])]
            for t in sc["collection"].get("titles", []):
                if not any(t.lower() in i for i in items):
                    fails.append(f"收藏清单缺条目：{t}")
    return fails


def _snapshot_state() -> dict:
    try:
        from src.memory.collections import list_collections
        from src.memory.store import list_preferences

        return {
            "prefs": list_preferences(EVAL_MEMORY_USER),
            "cols": [c.get("name") for c in list_collections(EVAL_MEMORY_USER)],
        }
    except Exception:  # noqa: BLE001
        return {"prefs": [], "cols": []}


def _cleanup_state(snap: dict) -> None:
    try:
        from src.memory.collections import _get_conn
        from src.memory.memory_items import clear_user_memories
        from src.memory.store import delete_preference, list_preferences

        for pref in list_preferences(EVAL_MEMORY_USER):
            if pref not in snap.get("prefs", []):
                kind = pref.get("kind") if isinstance(pref, dict) else None
                value = pref.get("value") if isinstance(pref, dict) else None
                if kind and value:
                    delete_preference(EVAL_MEMORY_USER, kind, value)
        cur_cols = [
            c.get("name")
            for c in __import__("src.memory.collections", fromlist=["list_collections"]).list_collections(
                EVAL_MEMORY_USER
            )
        ]
        for name in cur_cols:
            if name not in snap.get("cols", []):
                _get_conn().execute(
                    "DELETE FROM collections WHERE user_id = ? AND name = ?",
                    (EVAL_MEMORY_USER, name),
                )
                _get_conn().commit()
        # 记忆主表全量清场（评估身份专用，不触碰生产）
        clear_user_memories(EVAL_MEMORY_USER)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 状态清理失败：{e}")


def _clear_eval_state() -> None:
    """用例前清场：清空 eval-test 的记忆主表与收藏（多轮/记忆用例隔离）。"""
    try:
        from src.memory.collections import _get_conn
        from src.memory.memory_items import clear_user_memories
        from src.memory.summary import delete_user_summaries

        clear_user_memories(EVAL_MEMORY_USER)
        delete_user_summaries(EVAL_MEMORY_USER)
        _get_conn().execute(
            "DELETE FROM collections WHERE user_id = ?", (EVAL_MEMORY_USER,)
        )
        _get_conn().commit()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 用例前清场失败：{e}")


# ── LLM 裁判 ─────────────────────────────────────────────────────
JUDGE_PROMPT = """你是严格的艺术助手评估员。请对 Agent 的最终回答打分。

用户问题：{question}

Agent 回答：
{answer}

评分维度（各 0-1 分，总分 1-5）：
1. 事实正确性：无事实错误、无编造；
2. 完整性：覆盖问题要点；
3. 证据支撑：内容可追溯到检索证据（画作/画家/年代具体）；
4. 表达：结构清晰、直接回答；
5. 无幻觉：未回答时如实说明，而不是编造。

只输出 JSON：{{"score": 1-5 整数, "grounded": true/false, "reasons": "一句话理由"}}"""

EXPECT_PROMPT = """你是严格的评估员。用户问题：{question}

Agent 回答：
{answer}

期望行为：{expect}

只输出 JSON：{{"ok": true/false, "reasons": "一句话理由"}}"""


def _judge(question: str, answer: str) -> dict:
    from src.utils.llm import get_deterministic_llm

    try:
        raw = get_deterministic_llm().invoke(
            JUDGE_PROMPT.format(question=question, answer=(answer or "")[:3000])
        ).content
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception as e:  # noqa: BLE001
        print(f"[judge-skip] {e}")
        return {}


def _judge_expect(question: str, answer: str, expect: str) -> dict:
    from src.utils.llm import get_deterministic_llm

    try:
        raw = get_deterministic_llm().invoke(
            EXPECT_PROMPT.format(question=question, answer=(answer or "")[:2000], expect=expect)
        ).content
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception as e:  # noqa: BLE001
        print(f"[judge-skip] {e}")
        return {}


# ── 1. 答案质量（核心）──────────────────────────────────────────
def run_answer_quality(graph, limit: int | None, offset: int = 0) -> dict:
    rows, skipped, judged = [], 0, 0
    total_s, total_rounds = 0.0, 0
    cases = ANSWER_GOLDEN[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        t0 = time.time()
        result = _safe_invoke(graph, case["query"], f"ans-{i}")
        dt = time.time() - t0
        if result.get("error"):
            skipped += 1
            continue
        total_s += dt
        total_rounds += result.get("tool_rounds") or 0
        answer = str(result.get("final_answer") or "")
        score = _judge(case["query"], answer)
        if score.get("score"):
            judged += 1
            state_fails = _check_state(case.get("state_check") or {})
            rows.append(
                {
                    "id": case.get("id", f"ans-{i}"),
                    "task_type": case.get("task_type", "?"),
                    "score": int(score["score"]),
                    "grounded": bool(score.get("grounded")),
                    "state_ok": not state_fails,
                    "state_fails": state_fails,
                    "seconds": dt,
                    "tools": _tool_names(result.get("messages") or []),
                }
            )
            print(
                f"[ans {judged}] {case.get('task_type')} score={score['score']} "
                f"state={'✅' if not state_fails else '❌'} {case['query'][:28]}"
            )
    return {
        "rows": rows,
        "total": len(ANSWER_GOLDEN),
        "skipped": skipped,
        "judged": judged,
        "avg_seconds": total_s / max(judged + skipped, 1),
        "avg_rounds": total_rounds / max(judged + skipped, 1),
    }


# ── 2. 事实准确率 ───────────────────────────────────────────────
def run_facts(graph, limit: int | None, offset: int = 0) -> dict:
    rows, hits, skipped = [], 0, 0
    cases = FACT_CASES[offset:]
    if limit:
        cases = cases[:limit]
    for i, fact in enumerate(cases, offset):
        result = _safe_invoke(graph, fact["q"], f"fact-{i}")
        if result.get("error"):
            skipped += 1
            continue
        answer = str(result.get("final_answer") or "")
        hit = _gt_hit(answer, fact["gt"])
        hits += int(hit)
        rows.append({"q": fact["q"], "hit": hit})
        print(f"[fact {i + 1}/{min(limit or len(FACT_CASES), len(FACT_CASES))}] {'✅' if hit else '❌'} {fact['q'][:34]}")
    return {"rows": rows, "total": len(rows) + skipped, "hits": hits, "skipped": skipped}


# ── 3. 行为化 ───────────────────────────────────────────────────
def run_behavior(graph, runs: int, limit: int | None, offset: int = 0) -> dict:
    out: list[dict] = []
    all_ok = all_runs = skipped = 0
    cases = BEHAVIOR_CASES[offset:]
    if limit:
        cases = cases[:limit]
    for ci, case in enumerate(cases, offset):
        ok_n, seconds, rounds, judge_scores, valid = 0, [], [], [], 0
        for i in range(runs):
            t0 = time.time()
            result = _safe_invoke(graph, case["question"], f"beh-{case['name']}-{i}")
            dt = time.time() - t0
            if result.get("error"):
                skipped += 1
                continue
            valid += 1
            seconds.append(dt)
            rounds.append(result.get("tool_rounds") or 0)
            ok, fails = _check_behavior(case, result)
            ok_n += int(ok)
            all_runs += 1
            all_ok += int(ok)
            s = _judge(case["question"], str(result.get("final_answer") or ""))
            if s.get("score"):
                judge_scores.append(int(s["score"]))
            print(f"[beh {case['name']} {i + 1}/{runs}] {'✅' if ok else '❌'} {'；'.join(fails)}")
        out.append(
            {
                "name": case["name"],
                "rate": ok_n / max(valid, 1),
                "ok": ok_n,
                "runs": runs,
                "avg_s": sum(seconds) / max(len(seconds), 1),
                "avg_rounds": sum(rounds) / max(len(rounds), 1),
                "judge_avg": (sum(judge_scores) / len(judge_scores)) if judge_scores else None,
            }
        )
    return {"rows": out, "all_ok": all_ok, "all_runs": all_runs, "skipped": skipped}


# ── 4. 工具选择 ─────────────────────────────────────────────────
def run_tools(graph, limit: int | None, offset: int = 0) -> dict:
    cache = _load_case_cache("tools")
    ok_n, skipped = 0, 0
    total_s, total_rounds = 0.0, 0
    cases = TOOL_CASES[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        t0 = time.time()
        result = _safe_invoke(graph, case["q"], f"tool-{i}")
        dt = time.time() - t0
        if result.get("error"):
            cache[i] = {"q": case["q"], "error": True}
            skipped += 1
            continue
        total_s += dt
        total_rounds += result.get("tool_rounds") or 0
        tools = _tool_names(result.get("messages") or [])
        expect = case.get("expect_any") or []
        ok = (not expect and not tools) or (bool(expect) and any(t in tools for t in expect))
        cache[i] = {"q": case["q"], "expect": expect, "tools": tools, "ok": ok}
        print(f"[tool {i + 1}/{min(limit or len(TOOL_CASES), len(TOOL_CASES))}] {'✅' if ok else '❌'} {case['q'][:26]}")
    _save_case_cache("tools", cache)
    rows = [cache[k] for k in sorted(cache)]
    ok_n = sum(1 for r in rows if r.get("ok"))
    skipped = sum(1 for r in rows if r.get("error"))
    return {
        "rows": rows,
        "ok": ok_n,
        "total": len(rows),
        "skipped": skipped,
        "avg_seconds": total_s / max(len(rows), 1),
        "avg_rounds": total_rounds / max(len(rows), 1),
    }


# ── 5. 多轮 ─────────────────────────────────────────────────────
def run_multi_turn(graph, limit: int | None, offset: int = 0) -> dict:
    cache = _load_case_cache("multi_turn")
    ok_n, skipped = 0, 0
    cases = MULTI_TURN[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        _clear_eval_state()  # 多轮用例前清场，防串扰
        final_answer, all_tools, error = "", [], False
        for ti, turn in enumerate(case["turns"]):
            # 多轮必须共享同一 thread_id，否则 LangGraph 记忆/历史无法跨轮传递
            result = _safe_invoke(graph, turn["user"], f"mt-{case['id']}")
            if result.get("error"):
                error = True
                break
            final_answer = str(result.get("final_answer") or "")
            all_tools.extend(_tool_names(result.get("messages") or []))
        if error:
            cache[i] = {"id": case["id"], "error": True}
            skipped += 1
            continue
        ans_ok = any(k in final_answer for k in case.get("expect_answer_any", [])) or not case.get(
            "expect_answer_any"
        )
        state_fails = _check_state(case.get("final_state") or {})
        ok = ans_ok and not state_fails
        cache[i] = {
            "id": case["id"],
            "ans_ok": ans_ok,
            "state_fails": state_fails,
            "tools": all_tools,
            "final_answer": final_answer[:120],
            "ok": ok,
        }
        print(f"[mt {case['id']}] {'✅' if ok else '❌'} ans={ans_ok} state={state_fails or '✅'}")
    _save_case_cache("multi_turn", cache)
    rows = [cache[k] for k in sorted(cache)]
    ok_n = sum(1 for r in rows if r.get("ok"))
    skipped = sum(1 for r in rows if r.get("error"))
    return {"rows": rows, "ok": ok_n, "total": len(rows), "skipped": skipped}


# ── 6. 对抗 ─────────────────────────────────────────────────────
def run_adversarial(graph, limit: int | None, offset: int = 0) -> dict:
    cache = _load_case_cache("adversarial")
    ok_n, skipped = 0, 0
    cases = ADVERSARIAL[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        result = _safe_invoke(graph, case["query"], f"adv-{case['id']}")
        if result.get("error"):
            cache[i] = {"id": case["id"], "error": True}
            skipped += 1
            continue
        answer = str(result.get("final_answer") or "")
        gt_ok = _gt_hit(answer, case.get("gold_facts", [])) if case.get("gold_facts") else True
        j = _judge_expect(case["query"], answer, case["expect"])
        ok = bool(j.get("ok")) and gt_ok
        cache[i] = {"id": case["id"], "type": case["type"], "ok": ok, "reasons": j.get("reasons", "")}
        print(f"[adv {case['id']}] {'✅' if ok else '❌'} {case['type']}")
    _save_case_cache("adversarial", cache)
    rows = [cache[k] for k in sorted(cache)]
    ok_n = sum(1 for r in rows if r.get("ok"))
    skipped = sum(1 for r in rows if r.get("error"))
    return {"rows": rows, "ok": ok_n, "total": len(rows), "skipped": skipped}


# ── 7. 检索（离线）──────────────────────────────────────────────
def run_retrieval(n: int, top_k: int = 5) -> dict:
    import pandas as pd

    from src.retrieval.hybrid import get_hybrid_retriever

    core_path = Path(os.getenv("CORE_DATA_PATH", "./data/core/artworks_core.csv"))
    df = pd.read_csv(core_path, encoding="utf-8-sig", keep_default_na=False)
    usable = df[df["description"].astype(str).str.len() > 120]
    rng = random.Random(SEED)
    idxs = rng.sample(list(usable.index), min(n, len(usable)))
    hybrid = get_hybrid_retriever()
    print(f"\n▶ 已知项检索评估（{len(idxs)} 条，Recall@{top_k}，source=core）...")
    hits = 0
    for i, idx in enumerate(idxs, 1):
        row = df.loc[idx]
        gold_title = str(row["title"]).strip().lower()
        query = str(row["description"])[40:200]
        results = hybrid.search(query, top_k=top_k, sources=["core"])
        returned = [str(r.metadata.get("title", "")).strip().lower() for r in results]
        hit = gold_title in returned or any(
            (len(gold_title) >= 12 and gold_title in t)
            or (len(t) >= 12 and t in gold_title)
            for t in returned
        )
        hits += int(hit)
        print(f"  [{i:>2}/{len(idxs)}] {'✓' if hit else '✗'} {str(row['title'])[:44]}")
    rerank_on = os.getenv("RERANK_ENABLED", "1").strip().lower() not in ("0", "false", "no")
    return {
        "total": len(idxs),
        "hits": hits,
        "recall_at_k": hits / len(idxs) if idxs else 0.0,
        "top_k": top_k,
        "seed": SEED,
        "rerank": rerank_on,
        "reranker": "jina-reranker-v3.5",
        "source": "core",
    }


# ── 8. 意图诊断（软信号，非验收）──────────────────────────────
def run_intent_diag(limit: int | None, offset: int = 0) -> dict:
    from src.agent.nodes.common import classify_intent
    from src.agent.state import AgentState

    cache = _load_case_cache("intent_diag")
    cases = INTENT_DIAG[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        try:
            state = AgentState(user_query=case["query"])
            res = classify_intent(state)
            cache[i] = {
                "query": case["query"],
                "gold": case["gold"],
                "primary": res.get("intent"),
                "route": res.get("route"),
                "route_reason": res.get("route_reason"),
                "scores": {s["id"]: round(float(s.get("score", 0)), 2) for s in res.get("intent_scores", [])[:3]},
            }
        except Exception as e:  # noqa: BLE001
            cache[i] = {"query": case["query"], "gold": case["gold"], "error": str(e)}
    _save_case_cache("intent_diag", cache)
    rows = [cache[k] for k in sorted(cache, key=int)]
    ok = sum(1 for r in rows if r.get("primary") == r.get("gold"))
    return {"rows": rows, "total": len(rows), "match": ok}


# ── 9. 路由决策诊断（软信号：direct/rag/web/comparison 等误判率）──
def run_route_diag(limit: int | None, offset: int = 0) -> dict:
    from src.agent.nodes.common import classify_intent
    from src.agent.state import AgentState

    cache = _load_case_cache("route_diag")
    cases = ROUTE_CASES[offset:]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases, offset):
        try:
            state = AgentState(user_query=case["query"])
            res = classify_intent(state)
            cache[i] = {
                "query": case["query"],
                "gold": case.get("gold_route", ""),
                "route": res.get("route", ""),
                "reason": res.get("route_reason", ""),
            }
        except Exception as e:  # noqa: BLE001
            cache[i] = {
                "query": case["query"],
                "gold": case.get("gold_route", ""),
                "route": "",
                "reason": str(e),
            }
    _save_case_cache("route_diag", cache)
    rows = [cache[k] for k in sorted(cache, key=int)]
    ok = sum(1 for r in rows if r.get("route") == r.get("gold"))
    return {"rows": rows, "total": len(rows), "match": ok}


# ── 报告 ────────────────────────────────────────────────────────
def _pct(a: int, b: int) -> str:
    return f"{a / max(b, 1):.0%}"


def render_sections(
    parts: dict,
    intent_diag: dict | None,
    route_diag: dict | None,
    runs: int,
) -> list[tuple[str, str]]:
    """按 '## 标题' 生成报告分节（供整写与 --append 增量合并共用）。"""
    sections: list[tuple[str, str]] = []

    if "answers" in parts:
        a = parts["answers"]
        L = ["## 1. 答案质量（核心）", ""]
        if a["rows"]:
            avg = sum(r["score"] for r in a["rows"]) / len(a["rows"])
            pass4 = sum(1 for r in a["rows"] if r["score"] >= 4) / len(a["rows"])
            grounded = sum(1 for r in a["rows"] if r["grounded"]) / len(a["rows"])
            state_ok = sum(1 for r in a["rows"] if r["state_ok"]) / len(a["rows"])
            by_type: dict[str, list[int]] = {}
            for r in a["rows"]:
                by_type.setdefault(r["task_type"], []).append(r["score"])
            L.append(
                f"**平均分 {avg:.2f}/5 · 通过率(≥4) {pass4:.0%} · 证据支撑 {grounded:.0%} · "
                f"状态校验 {state_ok:.0%}**（有效 {a['judged']}/{a['total']}，跳过 {a['skipped']}）"
            )
            L.append("")
            L.append("| 任务类型 | 样本 | 平均分 | 通过率(≥4) |")
            L.append("|---|---|---|---|")
            for t, scores in sorted(by_type.items()):
                L.append(f"| {t} | {len(scores)} | {sum(scores) / len(scores):.2f} | {sum(1 for s in scores if s >= 4) / len(scores):.0%} |")
        else:
            L.append(f"（无有效样本，跳过 {a['skipped']}——检查 API 可用性）")
        L.append("")
        sections.append(("答案质量（核心）", "\n".join(L)))

    if "facts" in parts:
        f = parts["facts"]
        L = ["## 2. 事实准确率", "", f"**{f['hits']}/{f['total']}（{_pct(f['hits'], f['total'])}）**（跳过 {f['skipped']}）", ""]
        sections.append(("事实准确率", "\n".join(L)))

    if "behavior" in parts:
        b = parts["behavior"]
        L = [f"## 3. 行为化（{len(BEHAVIOR_CASES)} 类 × {runs} 次）", "",
             f"行为整体通过率：**{b['all_ok']}/{b['all_runs']}**（跳过 {b['skipped']}）", "",
             "| 用例 | 触发率 | 平均耗时(s) | 平均工具轮次 | 裁判均分 |",
             "|---|---|---|---|---|"]
        for r in b["rows"]:
            j = f"{r['judge_avg']:.1f}" if r["judge_avg"] else "-"
            L.append(f"| {r['name']} | {r['rate']:.0%} | {r['avg_s']:.1f} | {r['avg_rounds']:.1f} | {j} |")
        L.append("")
        sections.append(("行为化", "\n".join(L)))

    if "tools" in parts:
        t = parts["tools"]
        L = ["## 4. 工具选择", "",
             f"**{t['ok']}/{t['total']}（{_pct(t['ok'], t['total'])}）**"
             f"（跳过 {t['skipped']}；平均 {t['avg_seconds']:.1f}s / {t['avg_rounds']:.1f} 轮）", ""]
        sections.append(("工具选择", "\n".join(L)))

    if "multi_turn" in parts:
        m = parts["multi_turn"]
        L = ["## 5. 多轮对话", "", f"**{m['ok']}/{m['total']}（{_pct(m['ok'], m['total'])}）**（跳过 {m['skipped']}）", ""]
        rows = m.get("rows") or []
        passed = [r["id"] for r in rows if r.get("ok")]
        failed = [r["id"] for r in rows if not r.get("ok") and not r.get("error")]
        if passed:
            L.append("通过：" + " · ".join(passed))
        if failed:
            L.append("")
            L.append("未通过：" + " · ".join(failed))
        L.append("")
        sections.append(("多轮对话", "\n".join(L)))

    if "adversarial" in parts:
        a = parts["adversarial"]
        L = ["## 6. 对抗与安全", "", f"**{a['ok']}/{a['total']}（{_pct(a['ok'], a['total'])}）**（跳过 {a['skipped']}）", ""]
        sections.append(("对抗与安全", "\n".join(L)))

    if "retrieval" in parts:
        r = parts["retrieval"]
        L = ["## 7. 已知项检索 Recall@5", "",
             f"**{r['recall_at_k']:.1%}**（{r['hits']}/{r['total']}）· source=core · seed={r['seed']} "
             f"· rerank={'on' if r['rerank'] else 'off'} · reranker={r['reranker']}", ""]
        sections.append(("已知项检索 Recall@5", "\n".join(L)))

    if intent_diag and intent_diag["rows"]:
        L = ["## 8. 意图诊断（软提示参考，非验收指标）", "",
             f"主意图与 gold 一致率：**{intent_diag['match']}/{intent_diag['total']}**", "",
             "| 问题 | gold | 主意图 | 前三叶子分数 |", "|---|---|---|---|"]
        for r in intent_diag["rows"]:
            scores = "；".join(f"{k}={v}" for k, v in (r.get("scores") or {}).items())
            L.append(f"| {r['query'][:24]} | {r.get('gold', '-')} | {r.get('primary', '-')} | {scores} |")
        L.append("")
        sections.append(("意图诊断", "\n".join(L)))

    if route_diag and route_diag["rows"]:
        r = route_diag
        L = ["## 9. 路由决策（软信号，非验收指标）", "",
             f"路由与 gold 一致率：**{r['match']}/{r['total']}**", "",
             "| 问题 | gold | 实际路由 | 理由 |", "|---|---|---|---|"]
        for x in r["rows"]:
            L.append(f"| {x['query'][:26]} | {x.get('gold', '-')} | {x.get('route', '-')} | {str(x.get('reason', ''))[:40]} |")
        L.append("")
        sections.append(("路由决策", "\n".join(L)))

    skip_total = sum(
        p.get("skipped", 0) for p in parts.values() if isinstance(p, dict)
    )
    if skip_total:
        L = ["## 附：数据有效性说明", "",
             f"本场共 {skip_total} 条因 API 失败跳过（API 不可用/内容审核）。",
             "跳过占比高时请先恢复 API 可用性再重跑，勿将本报告视为正式基线。", ""]
        sections.append(("数据有效性说明", "\n".join(L)))
    return sections


def render(parts: dict, intent_diag: dict | None, route_diag: dict | None, runs: int) -> str:
    header = [
        "# ArtAgent Agent 评估报告（v2）", "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 对话模型：{os.getenv('LLM_MODEL', '-')} · 视觉：{os.getenv('VISION_MODEL', '-')} · 精排：{os.getenv('RERANK_MODEL', 'jina-reranker-v3.5')}",
        "", "> 验收口径：最终答案质量 + 状态校验；意图分类仅作诊断。", "",
    ]
    return "\n".join(
        header + [body for _, body in render_sections(parts, intent_diag, route_diag, runs)]
    )


def _merge_append(out_path: Path, new_sections: list[tuple[str, str]]) -> str:
    """把新分节合并进已有报告：同标题替换，新标题追加，头部保留。"""
    import re

    def norm(title: str) -> str:
        return re.sub(r"^\d+\.\s*", "", title.strip())

    def body_without_heading(body: str) -> str:
        lines = body.splitlines()
        if lines and lines[0].startswith("## "):
            return "\n".join(lines[1:]).strip("\n")
        return body.strip("\n")

    existing = out_path.read_text(encoding="utf-8").splitlines() if out_path.exists() else []
    header: list[str] = []
    old: list[tuple[str, str]] = []
    cur_title: str | None = None
    cur_body: list[str] = []

    def flush() -> None:
        nonlocal cur_title, cur_body
        if cur_title is not None:
            old.append((cur_title, "\n".join(cur_body)))
        cur_title, cur_body = None, []

    for line in existing:
        if line.startswith("## "):
            flush()
            cur_title = norm(line[3:])
        elif cur_title is None:
            header.append(line)
        else:
            cur_body.append(line)
    flush()

    merged = [(t, b) for t, b in old if t not in dict(new_sections)]
    merged.extend(new_sections)
    merged.sort(key=lambda x: _SECTION_ORDER.get(x[0], 99))
    if not header:
        header = [
            "# ArtAgent Agent 评估报告（v2）", "",
            f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            "> 验收口径：最终答案质量 + 状态校验；意图分类仅作诊断。", "",
        ]
    return "\n".join(
        header + [f"## {t}\n\n{body_without_heading(b)}" for t, b in merged]
    )


_SECTION_ORDER = {
    "答案质量（核心）": 1,
    "事实准确率": 2,
    "行为化": 3,
    "工具选择": 4,
    "多轮对话": 5,
    "对抗与安全": 6,
    "已知项检索 Recall@5": 7,
    "意图诊断": 8,
    "路由决策": 9,
    "数据有效性说明": 10,
}


def _append_history(parts: dict) -> None:
    rec: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    for key, p in parts.items():
        if isinstance(p, dict):
            rec[key] = {
                k: p[k]
                for k in ("total", "hits", "ok", "skipped", "judged", "recall_at_k")
                if k in p
            }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ArtAgent Agent 评估 v2")
    parser.add_argument("--answers", type=int, default=0, help="答案质量题数（默认 0；0=关闭）")
    parser.add_argument("--facts", action="store_true", help="跑事实准确率")
    parser.add_argument("--behavior-runs", type=int, default=0, help="每行为用例重复次数（0=关闭）")
    parser.add_argument("--tools", action="store_true", help="跑工具选择")
    parser.add_argument("--multi-turn", action="store_true", help="跑多轮评估")
    parser.add_argument("--adversarial", action="store_true", help="跑对抗与安全")
    parser.add_argument("--retrieval-n", type=int, default=0, help="检索抽样数（0=关闭）")
    parser.add_argument("--diag", action="store_true", help="附带意图诊断")
    parser.add_argument("--route", action="store_true", help="附带路由决策诊断（少量分类调用）")
    parser.add_argument("--limit", type=int, default=None, help="每部分最大用例数（调试用）")
    parser.add_argument("--pr", action="store_true", help="PR 门禁档：离线检索 20 条 + 意图诊断")
    parser.add_argument("--out", default=str(OUT), help="报告输出路径")
    parser.add_argument("--append", action="store_true", help="增量合并到已有报告（同标题替换）")
    parser.add_argument("--offset", type=int, default=0, help="本部分起始下标（分批续跑时用）")
    args = parser.parse_args()

    if args.pr:
        args.retrieval_n = args.retrieval_n or 20
        args.diag = True

    parts: dict = {}
    graph = None
    need_agent = bool(
        args.answers
        or args.facts
        or args.behavior_runs
        or args.tools
        or args.multi_turn
        or args.adversarial
    )
    snap = _snapshot_state() if need_agent else None
    if need_agent:
        graph = get_graph()

    if args.answers and graph:
        print(f"▶ 答案质量（{args.answers} 条黄金集）")
        parts["answers"] = run_answer_quality(graph, args.limit or args.answers, args.offset)
    if args.facts and graph:
        print(f"▶ 事实准确率（{len(FACT_CASES)} 条）")
        parts["facts"] = run_facts(graph, args.limit, args.offset)
    if args.behavior_runs and graph:
        print(f"▶ 行为化（{len(BEHAVIOR_CASES)} 类 × {args.behavior_runs}）")
        parts["behavior"] = run_behavior(graph, args.behavior_runs, args.limit, args.offset)
    if args.tools and graph:
        print(f"▶ 工具选择（{len(TOOL_CASES)} 条）")
        parts["tools"] = run_tools(graph, args.limit, args.offset)
    if args.multi_turn and graph:
        print(f"▶ 多轮（{len(MULTI_TURN)} 条）")
        parts["multi_turn"] = run_multi_turn(graph, args.limit, args.offset)
    if args.adversarial and graph:
        print(f"▶ 对抗与安全（{len(ADVERSARIAL)} 条）")
        parts["adversarial"] = run_adversarial(graph, args.limit, args.offset)
    if args.retrieval_n:
        print(f"▶ 检索 Recall@5（{args.retrieval_n} 条，离线）")
        parts["retrieval"] = run_retrieval(args.retrieval_n)

    # --diag 与 --route 解耦：各自只跑自己的用例集，避免 --route 连带跑
    # 40 条意图诊断（会意外覆盖报告分节）
    intent_diag = run_intent_diag(args.limit, args.offset) if args.diag else None
    route_diag = run_route_diag(args.limit, args.offset) if args.route else None
    if snap is not None:
        _cleanup_state(snap)

    if args.append:
        report = _merge_append(
            Path(args.out),
            render_sections(parts, intent_diag, route_diag, args.behavior_runs),
        )
    else:
        report = render(parts, intent_diag, route_diag, args.behavior_runs)
    Path(args.out).write_text(report, encoding="utf-8")
    _append_history(parts)
    print(f"\n报告已写入：{args.out}")


if __name__ == "__main__":
    main()
