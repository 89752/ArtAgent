from langchain_core.tools import tool


@tool
def read_artifact(artifact_id: str, offset: int = 0) -> dict:
    """Read a saved tool result or report, 1500 characters at a time, using its artifact_id."""
    from src.harness.storage import get_artifact
    from src.memory.memory_items import get_memory_user_id
    item = get_artifact(artifact_id, get_memory_user_id())
    if not item:
        return {"status": "NOT_FOUND"}
    offset = max(0, offset)
    return {"content": item["content"][offset:offset+1500], "next_offset": offset+1500 if len(item["content"]) > offset+1500 else None}
