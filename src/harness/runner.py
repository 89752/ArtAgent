"""Research jobs: leased execution, immutable results, and explicit acceptance."""
import hashlib
import json
import os
import threading
import time
import uuid

from langchain_core.messages import HumanMessage, ToolMessage

from src.harness.context import RunContext, RunStopped, run_scope
from src.harness.storage import list_artifacts, put_artifact
from src.harness.verification import TaskSpec, collect_evidence, verify_result
from src.tasks import store


def run_job(graph, task_id: str, user_id: str, *, attempt: str = ""):
    job = store.get_task(task_id)
    if not job or job["payload"].get("user_id") != user_id:
        return
    try:
        spec = TaskSpec(**(job["payload"].get("spec") or {"min_sources": 0}))
    except (ValueError, TypeError):
        store.update_task(task_id, status="failed", error="任务规格无效，请重新创建任务")
        return
    if job["cancel_requested"]:
        if not job.get("lease_owner"):
            store.update_task(task_id, status="interrupted", error="用户取消")
        return
    if attempt:
        if job.get("attempt_id") != attempt or job.get("lease_until", 0) <= time.time():
            return
    else:
        attempt = store.claim_job(task_id, user_id, f"{os.getpid()}:{uuid.uuid4().hex}")
    if not attempt:
        return
    from src.memory.memory_items import set_active_user_id, clear_active_user_id
    from src.agent.nodes.general import _skill_definitions
    from src.tools.registry import TOOL_BY_NAME
    from src.utils.governance import tool_spec
    from src.utils.config import get
    from src.data.documents_store import get_document
    from src.observability.runs import record_run

    started = time.monotonic()
    previous_elapsed = float((job.get("usage") or {}).get("elapsed_seconds", 0))
    stop_heartbeat = threading.Event()
    lost_lease = threading.Event()

    def pulse():
        while not stop_heartbeat.wait(15):
            with context._lock:
                context.usage["elapsed_seconds"] = previous_elapsed + time.monotonic()-started
                alive = store.heartbeat(task_id, attempt, dict(context.usage))
            if not alive:
                lost_lease.set()
                return

    threading.Thread(target=pulse, daemon=True, name="job-lease").start()

    def cancelled():
        fresh = store.get_task(task_id)
        return lost_lease.is_set() or not fresh or fresh["attempt_id"] != attempt or fresh["cancel_requested"] or fresh["pause_requested"]

    def persist(usage):
        usage["elapsed_seconds"] = previous_elapsed + time.monotonic()-started
        if not store.heartbeat(task_id, attempt, usage):
            raise RunStopped("interrupted")

    context = RunContext(run_id=attempt, user_id=user_id, task_id=task_id, attempt_id=attempt,
        max_model_calls=spec.max_model_calls, max_tool_calls=spec.max_tool_calls, max_tokens=spec.max_tokens,
        max_output_tokens=2048 if len(job.get("plan") or []) == 1 else 4096,
        deadline=started+max(0, spec.max_seconds-previous_elapsed),
        document_ids=set(spec.document_ids) if spec.document_ids else None,
        cancelled=cancelled, usage=job.get("usage") or {}, persist=persist,
        allowed_tools={name for name, tool in TOOL_BY_NAME.items() if tool_spec(tool).read_only} | {"delegate_task"} | {"skill_"+s.id for s in _skill_definitions},
        versions={"model": get("models.llm_model"), "skills": {s.id:s.content_hash for s in _skill_definitions}})
    if context.document_ids is not None:
        # Selected-material tasks cannot silently use unrelated core/web sources.
        context.allowed_tools &= {"semantic_search", "agentic_retrieve", "read_page_image", "read_artifact"}
    set_active_user_id(user_id)
    status, error = "failed", ""
    try:
        with run_scope(context):
            docs = [get_document(did, user_id) for did in spec.document_ids]
            material_tasks = [store.get_task(did) for did in spec.document_ids]
            if any(not doc or doc.get("status") not in {"done", "active", "ready", "completed"} for doc in docs) or any(task and task["status"] != "done" for task in material_tasks):
                status, error = "waiting_input", "所选资料尚未就绪，请先完成上传与解析后继续。"
                return
            while True:
                context.check()
                job = store.get_task(task_id)
                index, plan = int(job["step_index"]), job["plan"]
                if index >= len(plan):
                    status, error = "verification_failed", "计划为空，未执行任务。"
                    return
                fingerprint = hashlib.sha256(json.dumps([job["payload"], plan], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
                context.step_id = f"{index}:{fingerprint}"
                previous = [a for a in list_artifacts(task_id, user_id) if a["kind"] in {"report", "step_answer"}]
                committed = next((a for a in reversed(previous) if a["step_id"] == str(index) and a["metadata"].get("fingerprint") == fingerprint and a["metadata"].get("verification", {}).get("passed")), None)
                if committed:
                    artifact = committed
                else:
                    evidence = []
                    for item in previous:
                        for source in item["metadata"].get("evidence", []):
                            if source["locator"] not in {e["locator"] for e in evidence}:
                                evidence.append({**source, "id": f"E{len(evidence)+1}"})
                    from src.learning.store import active_guidance
                    guidance = active_guidance(user_id, job["payload"]["objective"])
                    compact_docs = [{
                        "doc_id": doc.get("doc_id"), "doc_name": doc.get("doc_name"),
                        "kind": doc.get("kind"), "pages": doc.get("pages"),
                        "text_chunks": doc.get("text_chunks"), "image_pages": doc.get("image_pages"),
                    } for doc in docs]
                    compact_previous = [
                        {"id": item["id"], "content": item["content"][:2500]}
                        for item in previous[-2:]
                    ]
                    compact_evidence = [
                        {**item, "excerpt": str(item.get("excerpt") or "")[:800]}
                        for item in evidence[:12]
                    ]
                    prefetched = False
                    if len(plan) == 1 and (
                        not spec.document_ids
                        or all(not (doc.get("route_distribution") or {}).get("multimodal") for doc in docs)
                    ):
                        # One-step research has a deterministic retrieval path. Prefetch
                        # either the selected text corpus or the core collection once so
                        # the model cannot skip the evidence contract or loop over tools.
                        from src.tools.retrieval import semantic_search
                        from src.utils.governance import governed_invoke

                        prefetched_messages = []
                        for doc_id in (spec.document_ids or [None]):
                            args = {"query": job["payload"]["objective"], "top_k": 8}
                            if doc_id:
                                args["filters"] = {"doc_id": doc_id}
                            raw = governed_invoke(
                                semantic_search,
                                args,
                                context="main", user_id=user_id,
                            )
                            prefetched_messages.append(ToolMessage(
                                content=raw, name="semantic_search",
                                tool_call_id=f"prefetch:{doc_id}",
                            ))
                        for source in collect_evidence(prefetched_messages):
                            if source["locator"] not in {e["locator"] for e in evidence}:
                                evidence.append({**source, "id": f"E{len(evidence)+1}"})
                        compact_evidence = [
                            {**item, "excerpt": str(item.get("excerpt") or "")[:800]}
                            for item in evidence[:12]
                        ]
                        context.allowed_tools = set()
                        prefetched = bool(evidence)
                    prompt = (
                        f"研究目标：{job['payload']['objective']}\n当前第 {index+1}/{len(plan)} 步：{plan[index]}\n"
                        f"用户补充：{job['payload'].get('additional_input', '')}\n"
                        f"资料：{json.dumps(compact_docs, ensure_ascii=False, default=str)}\n"
                        f"验收要求：{spec.acceptance}\n"
                        f"前序产物摘要：{json.dumps(compact_previous,ensure_ascii=False)}\n"
                        f"证据索引：{json.dumps(compact_evidence, ensure_ascii=False)}\n"
                        f"经过评测的执行经验（只作流程参考，不作为事实）：{guidance}\n"
                        "请完成当前步骤，明确区分事实、观察与推断。引用使用证据索引中的 [E编号]。"
                        "不要捏造来源或页码；资料不足时明确说明缺口。当前步骤必须直接产出结果，"
                        "证据充分后不要重复检索。"
                        + ("证据已经由系统从所选文字资料中预取；请直接回答，不要请求工具。" if prefetched else "")
                    )
                    result = graph.invoke({"messages": [HumanMessage(content=prompt, name="user-input")],
                        "user_query": prompt, "original_user_query": prompt, "user_id": user_id,
                        "conversation_id": f"job:{task_id}", "uploaded_docs": docs, "final_answer": "",
                        "ask_user": "continue", "reflection_notes": "", "verification": {}, "execution_status": "running",
                        "tool_rounds": 0, "executed_tool_signatures": [], "retry_count": 0,
                        "execution_mode": "job_quality" if len(plan) > 1 and index == len(plan)-1 else "job_fast"},
                        config={"configurable": {"thread_id": f"{user_id}:job:{task_id}:{context.step_id}:{attempt}"}, "recursion_limit": 40})
                    context.check()
                    messages = list(result.get("messages") or [])
                    for item in list_artifacts(task_id, user_id):
                        if item["kind"] == "tool_result":
                            messages.append(ToolMessage(content=item["content"], tool_call_id=item["id"]))
                    for source in collect_evidence(messages):
                        if source["locator"] not in {e["locator"] for e in evidence}:
                            evidence.append({**source, "id": f"E{len(evidence)+1}"})
                    content = str(result.get("final_answer") or "").strip()
                    if spec.min_sources and evidence and "[E" not in content:
                        refs = "\n".join(
                            f"- [{item['id']}] {item.get('title') or item['locator']} — {item['locator']}"
                            for item in evidence[: max(spec.min_sources, 3)]
                        )
                        result["final_answer"] = f"{content}\n\n参考证据：\n{refs}".strip()
                    verification = verify_result(result, min_sources=spec.min_sources if index == len(plan)-1 else 0, evidence=evidence)
                    artifact = put_artifact(user_id, task_id, str(index), str(result.get("final_answer") or ""),
                        kind="report" if index == len(plan)-1 else "step_answer",
                        metadata={"evidence": evidence, "verification": verification, "run_id": attempt,
                                  "versions": context.versions, "step": plan[index], "step_index": index, "fingerprint": fingerprint,
                                  "guidance": [g["id"] for g in guidance]})
                    if not verification["passed"]:
                        status, error = verification["status"], "；".join(verification["issues"])
                        return
                if not store.advance_agent_job(task_id, artifact={"id": artifact["id"], "kind": artifact["kind"],
                    "content": artifact["content"], "step_index": index, "revision": artifact["revision"]}, attempt=attempt):
                    raise RunStopped("interrupted")
                if index == len(plan)-1:
                    status = "done"
                    return
    except RunStopped as exc:
        fresh = store.get_task(task_id) or {}
        status = "paused" if fresh.get("pause_requested") else ("interrupted" if exc.status == "cancelled" else exc.status)
        error = {"budget_exhausted": "本次任务预算已耗尽", "unknown_execution_state": "上次写操作结果未知，需要核对后再继续"}.get(status, "执行已停止")
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        stop_heartbeat.set()
        with context._lock:
            context.usage["elapsed_seconds"] = previous_elapsed + time.monotonic()-started
            store.heartbeat(task_id, attempt, dict(context.usage))
        store.finish_attempt(task_id, attempt, status, error)
        clear_active_user_id()
        record_run(request_id=attempt, user_id=user_id, session_id=f"job:{task_id}", intent="research",
                   latency_ms=(time.monotonic()-started)*1000, error=error,
                   model_calls=[{"model": context.versions.get("model"), "role": "job_total", "input_tokens": context.usage.get("tokens", 0), "token_source": "estimated"}])
