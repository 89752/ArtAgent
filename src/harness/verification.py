"""Task contracts and deterministic evidence checks; semantics remain separately judged."""
import json
import re
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    workflow: str = "research"
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    acceptance: list[str] = Field(default_factory=list, max_length=20)
    min_sources: int = Field(default=1, ge=0, le=20)
    max_model_calls: int = Field(default=40, ge=1, le=100)
    max_tool_calls: int = Field(default=60, ge=1, le=200)
    max_tokens: int = Field(default=80000, ge=1000, le=500000)
    max_seconds: int = Field(default=600, ge=10, le=3600)


def collect_evidence(messages):
    evidence, seen = [], set()
    def visit(value):
        if isinstance(value, dict):
            locator = value.get("source_url") or value.get("url") or value.get("artwork_id")
            if value.get("doc_id") and (value.get("page") is not None or value.get("page_idx") is not None):
                locator = f"{value['doc_id']}#page={value.get('page', value.get('page_idx'))}"
            elif value.get("doc_id") and value.get("dataset_id"):
                locator = f"{value['doc_id']}#dataset={value['dataset_id']}"
            if locator and str(locator) not in seen:
                seen.add(str(locator))
                evidence.append({"id": f"E{len(evidence)+1}", "locator": str(locator),
                                 "title": str(value.get("title") or value.get("doc_name") or locator),
                                 "excerpt": str(value.get("description_snippet") or value.get("content") or value.get("snippet") or "")[:1200]})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for message in messages or []:
        if getattr(message, "type", "") == "tool":
            try:
                visit(json.loads(str(message.content)))
            except (ValueError, TypeError):
                pass
    return evidence


def verify_result(result, *, min_sources=0, evidence=None):
    content = str(result.get("final_answer") or "").strip()
    if result.get("ask_user") == "ask":
        return {"passed": False, "status": "waiting_input", "issues": [content or "需要补充材料"]}
    issues = []
    if result.get("execution_status") in {"budget_exhausted", "unknown_execution_state", "cancelled"}:
        return {"passed": False, "status": result["execution_status"], "issues": ["执行未完成，不能验收"]}
    if not content:
        issues.append("产物为空")
    if result.get("reflection_notes") in ("FAIL", "RETRY") or (result.get("verification") or {}).get("passed") is False:
        issues.append("语义验收未通过")
    sources = evidence or []
    if len(sources) < min_sources:
        issues.append(f"可定位来源不足：需要 {min_sources}，实际 {len(sources)}")
    if min_sources and sources:
        cited = set(re.findall(r"\[(E\d+)\]", content))
        valid = {item["id"] for item in sources}
        if not cited or cited - valid:
            issues.append("报告应使用有效的 [E编号] 引用")
    return {"passed": not issues, "status": "done" if not issues else "verification_failed", "issues": issues}
