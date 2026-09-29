"""Configuration and local capability checks; never imply provider connectivity."""
import importlib.util
import os


def report(user_id):
    from src.data.documents_store import list_documents
    from src.utils.config import get
    docs = list_documents(user_id)
    checks = {
        "model_configured": bool(get("models.llm_api_key") and get("models.llm_base_url")),
        "durable_checkpoints": importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None,
        "process_isolation": os.getenv("ARTAGENT_JOB_EXECUTION", "process") == "process",
        "stream_process_isolation": os.getenv("ARTAGENT_STREAM_EXECUTION", "process") == "process",
        "available_documents": sum(d.get("status") in {"done", "active"} or bool(d.get("active_index_id")) for d in docs),
        "failed_documents": sum(d.get("status") == "failed" for d in docs),
    }
    return {"ok": True, "checks": checks, "provider_tested": False,
            "scope": "本地配置与持久层检查；不代表供应商连通性或研究质量已验收"}
