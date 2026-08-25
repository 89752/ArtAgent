"""Regression checks for post-P2 reliability and evaluation alignment."""

from pathlib import Path

from eval.agent_eval_v2 import failure_taxonomy, reproducibility_metadata
from src.memory import memory_items as mem
from src.observability import runs
from src.retrieval.agentic import adaptive_retrieve
from src.tasks import store as tasks


def test_agentic_rag_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AGENTIC_RAG_ENABLED", "0")
    seen = []

    def retrieve(query):
        seen.append(query)
        return [{"title": "one", "content": "one"}]

    evidence, audit = adaptive_retrieve("two concepts", retrieve)
    assert len(seen) == 1
    assert evidence
    assert audit["disabled"] is True


def test_agent_job_pause_resume_and_restart_recovery(tmp_path: Path):
    tasks._reset_for_tests(tmp_path / "tasks.db")
    task_id = tasks.create_agent_job("目标", "u1", ["步骤一", "步骤二"])
    assert tasks.pause_agent_job(task_id)
    assert tasks.get_task(task_id)["status"] == "paused"
    assert tasks.resume_agent_job(task_id)
    tasks.update_task(task_id, status="processing")
    assert tasks.mark_interrupted_on_startup() == 1
    resumed = tasks.recover_interrupted_agent_jobs()
    assert resumed == [{"task_id": task_id, "user_id": "u1"}]
    assert tasks.get_task(task_id)["status"] == "pending"


def test_stale_memory_is_not_recalled():
    user = "test-stale-memory"
    old_embed = mem._embed
    mem._embed = lambda _text: None
    try:
        mem.clear_user_memories(user)
        item = mem.add_memory(user, "临时旅行计划", entity="trip", expected_valid_days=1)
        mem._get_conn().execute(
            "UPDATE memory_items SET updated_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (item["id"],)
        )
        mem._get_conn().commit()
        assert not mem.search_memories(user, "临时旅行计划")
    finally:
        mem._embed = old_embed
        mem.clear_user_memories(user)


def test_trace_records_model_role(tmp_path: Path):
    runs._reset_for_tests(tmp_path / "observability.db")
    run_id = runs.record_run(user_id="u1", model_calls=[{"role": "reasoning", "model": "r", "input_tokens": 2}])
    assert runs.get_run_detail(run_id, "u1")["model_calls"][0]["role"] == "reasoning"
    assert runs.metrics(user_id="u1")["model_roles"] == {"reasoning": 1}


def test_eval_failure_taxonomy_and_reproducibility_metadata():
    labels = failure_taxonomy({"answer": "", "fact_hit": False, "trajectory": {"loop_rate": 1}})
    assert {"F1_EMPTY_ANSWER", "F2_FACTUALITY", "F7_TOOL_LOOP"} <= set(labels)
    assert reproducibility_metadata()["datasets"]["cases.json"] != ""
