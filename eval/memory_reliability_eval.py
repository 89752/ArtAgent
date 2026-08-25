"""Deterministic long-term-memory reliability benchmark.

Runs the plan's minimum matrix: 50 writes, 50 recalls, and 20 each for
conflict resolution, stale suppression, and forget.  It exercises the durable
memory store under a dedicated user and never touches production memories.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import memory_items as mem

OUT = Path(__file__).resolve().parent / "memory_reliability_report.json"
USER = "eval-memory-reliability"


def _ok(result: bool) -> int:
    return 1 if result else 0


def run() -> dict:
    # The benchmark checks storage lifecycle deterministically; embeddings are
    # intentionally disabled so local model availability cannot change results.
    original_embed = mem._embed
    mem._embed = lambda _text: None
    try:
        mem.clear_user_memories(USER)
        writes = recalls = conflicts = stales = forgets = 0

        for i in range(50):
            content = f"偏好测试-[{i:03d}]"
            item = mem.add_memory(USER, content, entity=f"write-{i}", importance=0.8)
            writes += _ok(bool(item.get("id")))
        for i in range(50):
            content = f"偏好测试-[{i:03d}]"
            hits = mem.search_memories(USER, content, top_k=3)
            recalls += _ok(any(hit.get("content") == content for hit in hits))

        for i in range(20):
            entity = f"conflict-{i}"
            mem.add_memory(USER, "旧偏好", kind="fact", entity=entity)
            mem.add_memory(USER, "新偏好", kind="fact", entity=entity)
            active = mem.list_memories(USER)
            conflicts += _ok(any(x.get("entity") == entity and x.get("content") == "新偏好" for x in active))

        conn = mem._get_conn()
        for i in range(20):
            entity = f"stale-{i}"
            item = mem.add_memory(USER, f"过时信息-{i}", entity=entity, expected_valid_days=1)
            conn.execute("UPDATE memory_items SET updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (item["id"],))
            conn.commit()
            hits = mem.search_memories(USER, f"过时信息-{i}", top_k=3)
            stales += _ok(not any(hit.get("id") == item["id"] for hit in hits))

        for i in range(20):
            item = mem.add_memory(USER, f"遗忘信息-{i}", entity=f"forget-{i}")
            deleted = mem.delete_memory(USER, item["id"])
            hits = mem.search_memories(USER, f"遗忘信息-{i}", top_k=3)
            forgets += _ok(deleted and not any(hit.get("id") == item["id"] for hit in hits))

        result = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matrix": {"write": 50, "recall": 50, "conflict": 20, "stale": 20, "forget": 20},
            "passed": {"write": writes, "recall": recalls, "conflict": conflicts, "stale": stales, "forget": forgets},
            "metrics": {
                "write_precision": writes / 50,
                "recall_precision": recalls / 50,
                "conflict_accuracy": conflicts / 20,
                "stale_accuracy": stales / 20,
                "forget_success": forgets / 20,
            },
        }
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        mem._embed = original_embed
        mem.clear_user_memories(USER)


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["metrics"], ensure_ascii=False))
    if any(value < 1.0 for value in report["metrics"].values()):
        raise SystemExit("memory reliability gate failed")
