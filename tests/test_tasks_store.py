"""任务表与持久执行测试。"""

import tempfile
from pathlib import Path

import pytest

from src.tasks import store as tasks


@pytest.fixture(autouse=True)
def _isolated_db():
    tmp = Path(tempfile.mkdtemp(prefix="artagent_tasks_test_"))
    tasks._reset_for_tests(tmp / "tasks.db")
    yield
    tasks._reset_for_tests()


def test_lifecycle_and_payload():
    tid = tasks.create_task("ingest_pdf", {"doc_id": "d1", "kind": "pdf"})
    assert tasks.get_task(tid)["status"] == "pending"
    tasks.update_task(tid, status="processing", progress=30)
    t = tasks.get_task(tid)
    assert t["status"] == "processing"
    assert t["payload"]["doc_id"] == "d1"
    tasks.update_task(tid, status="done", progress=100)
    assert tasks.get_task(tid)["status"] == "done"
    assert tasks.get_task(tid)["finished_at"]


def test_interrupted_recovery_and_retry():
    tid = tasks.create_task("ingest_table", {"kind": "table"})
    tasks.update_task(tid, status="processing")
    n = tasks.mark_interrupted_on_startup()
    assert n == 1
    assert tasks.get_task(tid)["status"] == "interrupted"

    # 只有 failed/interrupted 可重置
    assert tasks.reset_task(tid) is True
    assert tasks.get_task(tid)["status"] == "pending"
    assert tasks.reset_task(tid) is False   # pending 不能重复重置


def test_invalid_status_rejected():
    tid = tasks.create_task("x")
    with pytest.raises(ValueError):
        tasks.update_task(tid, status="bogus")
