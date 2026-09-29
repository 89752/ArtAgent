import json
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.harness.context import RunContext, RunStopped, invoke_model, run_scope
from src.harness.storage import put_artifact, get_artifact
from src.harness.verification import verify_result
from src.tasks import store


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTAGENT_HARNESS_DB_PATH",str(tmp_path/"harness.db"))
    store._reset_for_tests(tmp_path/"tasks.db")


def test_real_reflection_loop_repairs_and_saves_latest(monkeypatch,tmp_path):
    from src.agent import nodes
    from src.agent.nodes import common
    from src.agent.graph import build_graph
    monkeypatch.setenv("ARTAGENT_CHECKPOINT_DB_PATH",str(tmp_path/"graph.db"))
    monkeypatch.setattr(nodes,"load_memory",lambda state:{})
    monkeypatch.setattr(nodes,"ask_user",lambda state:{"ask_user":"continue"})
    monkeypatch.setattr(nodes,"save_memory",lambda state:{})
    seen=[]
    def generate(state):
        seen.append(state)
        return {"messages":[AIMessage(content="corrected" if len(seen)>1 else "wrong")]}
    monkeypatch.setattr(nodes,"general_agent",generate)
    verdicts=iter(['{"passed":false,"issues":["incorrect"],"repair":"correct it"}','{"passed":true}'])
    monkeypatch.setattr(common,"get_deterministic_llm",lambda:SimpleNamespace(invoke=lambda _p:AIMessage(content=next(verdicts))))
    result=build_graph().invoke({"messages":[HumanMessage(content="question")],"user_query":"question"},
        config={"configurable":{"thread_id":"audit"}})
    assert len(seen)==2
    assert "correct it" in seen[1].messages[-1].content
    assert result["final_answer"]=="corrected"
    assert result["verification"]["passed"]


def test_empty_and_clarification_never_complete():
    assert not verify_result({})["passed"]
    assert verify_result({"ask_user":"ask","final_answer":"请上传材料"})["status"]=="waiting_input"
    tid=store.create_agent_job("test","u",["one"])
    assert not store.advance_agent_job(tid,artifact={"content":""})
    assert store.get_task(tid)["step_index"]==0


def test_lease_fences_other_workers_and_startup():
    tid=store.create_agent_job("test","u",["one"])
    first=store.claim_job(tid,"u","worker-a")
    assert first
    assert store.claim_job(tid,"u","worker-b") is None
    assert store.mark_interrupted_on_startup()==0
    assert not store.advance_agent_job(tid,artifact={"content":"result"},attempt="stale")
    assert store.advance_agent_job(tid,artifact={"content":"result"},attempt=first)


def test_expired_worker_cannot_commit():
    tid=store.create_agent_job("test","u",["one"])
    first=store.claim_job(tid,"u","a",ttl=-1)
    second=store.claim_job(tid,"u","b")
    assert first!=second
    assert not store.advance_agent_job(tid,artifact={"content":"stale"},attempt=first)
    assert store.advance_agent_job(tid,artifact={"content":"fresh"},attempt=second)


def test_artifact_replay_versions_and_isolation():
    one=put_artifact("u","t","step","first")
    assert put_artifact("u","t","step","first")["id"]==one["id"]
    assert put_artifact("u","t","step","second")["revision"]==2
    assert get_artifact(one["id"],"other") is None


def test_artifact_reverification_is_a_new_revision():
    failed = put_artifact("u", "t", "step", "same", metadata={"verification": {"passed": False}})
    passed = put_artifact("u", "t", "step", "same", metadata={"verification": {"passed": True}})
    assert passed["revision"] == failed["revision"] + 1
    assert passed["metadata"]["verification"]["passed"]


def test_dispatcher_picks_up_lease_expired_after_startup():
    tid = store.create_agent_job("test", "u", ["one"])
    attempt = store.claim_job(tid, "u", "worker")
    assert store.runnable_agent_jobs() == []
    with store._lock:
        store._get_conn().execute("UPDATE tasks SET lease_until=0 WHERE task_id=?", (tid,))
        store._get_conn().commit()
    assert store.runnable_agent_jobs() == [{"task_id": tid, "user_id": "u"}]
    assert store.claim_job(tid, "u", "replacement") != attempt


def test_unfinished_execution_cannot_pass_acceptance():
    result = verify_result({"final_answer": "A plausible report", "execution_status": "budget_exhausted"})
    assert not result["passed"]
    assert result["status"] == "budget_exhausted"


def test_cancelled_pending_job_is_settled_without_execution():
    tid = store.create_agent_job("test", "u", ["one"])
    assert store.cancel_agent_job(tid)
    assert store.runnable_agent_jobs() == []
    assert store.get_task(tid)["status"] == "interrupted"


def test_history_trimming_keeps_complete_tool_exchange():
    from langchain_core.messages import ToolMessage
    from src.agent.context import trim_history
    exchange = [HumanMessage(content="new"), AIMessage(content="", tool_calls=[{"id":"c", "name":"search", "args":{}}]),
                ToolMessage(content="evidence", tool_call_id="c"), AIMessage(content="answer")]
    messages = [HumanMessage(content="old"), AIMessage(content="old answer"), *exchange]
    assert trim_history(messages, max_turns=1) == exchange


def test_shared_budget_and_cancellation_prevent_model_start():
    calls=[]
    model=SimpleNamespace(invoke=lambda _m: calls.append(1) or AIMessage(content="ok"))
    with run_scope(RunContext("r","u",max_model_calls=1)):
        invoke_model(model,"one")
        with pytest.raises(RunStopped):
            invoke_model(model,"two")
    assert calls==[1]
    with run_scope(RunContext("r","u",cancelled=lambda:True)):
        with pytest.raises(RunStopped):
            invoke_model(model,"never")
    assert calls==[1]


def test_unknown_write_is_not_replayed():
    from src.harness.storage import start_tool
    from src.utils.governance import governed_invoke
    calls=[]
    tool=SimpleNamespace(name="mutate",invoke=lambda _a:calls.append(1) or "saved")
    run=RunContext("r","u",task_id="t",step_id="s")
    with run_scope(run):
        assert governed_invoke(tool,{})=="saved"
        assert governed_invoke(tool,{})=="saved"
    assert calls==[1]
    import hashlib
    key=hashlib.sha256(json.dumps(["u","t","s","mutate",{"x":1}],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    start_tool(key,"u","t")
    with run_scope(run), pytest.raises(RunStopped,match="unknown_execution_state"):
        governed_invoke(tool,{"x":1})
    assert calls==[1]


def test_gate_fails_empty_or_regressed_results():
    from eval.gate import gate_decision,public_retrieval
    assert not gate_decision({})["passed"]
    assert not gate_decision({"retrieval":{"total":10,"recall_at_k":.1}})["passed"]
    assert gate_decision({"public_retrieval":public_retrieval()})["passed"]


def test_candidate_cannot_promote_without_evaluation():
    from src.learning import store as learning
    tid=store.create_agent_job("test","u",["one"])
    put_artifact("u",tid,"s","report",metadata={"verification":{"passed":True}})
    store.advance_agent_job(tid,artifact={"content":"report"})
    item=learning.create_candidate("u",tid,"method","Verify evidence before drafting",["test"])
    with pytest.raises(ValueError):
        learning.change_status(item["id"],"u","promote")
    assert learning.list_experiences("other")==[]
    learning.save_evaluation(item["id"],"u",{"passed":True,"content_hash":item["content_hash"],"suite_hash":"obsolete"})
    with pytest.raises(ValueError):
        learning.change_status(item["id"],"u","promote")
    learning.save_evaluation(item["id"],"u",{"passed":True,"content_hash":item["content_hash"],"suite_hash":learning.suite_hash()})
    learning.change_status(item["id"],"u","promote")
    assert learning.active_guidance("u","test")
    learning.change_status(item["id"],"u","rollback")
    assert learning.active_guidance("u","test")==[]


def test_runner_requires_evidence_and_commits_verified_report(monkeypatch):
    from langchain_core.messages import ToolMessage
    from src.harness.runner import run_job
    from src.harness.storage import list_artifacts
    from src.observability import runs
    monkeypatch.setattr(runs, "record_run", lambda **kwargs: None)
    tid = store.create_agent_job("research", "u", ["report"], spec={"min_sources": 1})
    output = {"final_answer": "Supported finding [E1]", "verification": {"passed": True},
              "messages": [ToolMessage(content=json.dumps({"source_url":"https://example.org/source", "title":"Source", "content":"Finding"}), tool_call_id="c")]}
    run_job(SimpleNamespace(invoke=lambda *args, **kwargs: output), tid, "u")
    assert store.get_task(tid)["status"] == "done"
    artifact = next(a for a in list_artifacts(tid, "u") if a["kind"] == "report")
    assert artifact["metadata"]["verification"]["passed"]
    assert artifact["metadata"]["evidence"][0]["id"] == "E1"
    missing = store.create_agent_job("research", "u", ["gather", "report"], spec={"min_sources": 1})
    run_job(SimpleNamespace(invoke=lambda *args, **kwargs: {"final_answer":"Unsupported finding"}), missing, "u")
    assert store.get_task(missing)["status"] == "verification_failed"
    assert store.get_task(missing)["step_index"] == 1


def test_default_plan_keeps_lookup_fast_and_deep_research_explicit():
    from src.harness.planning import default_plan

    assert len(default_plan("请列出《静港》的作者、年份、媒介和尺寸")) == 1
    assert len(default_plan("比较 A01 和 A02 的年份与媒介")) == 1
    assert len(default_plan("请写一份完整研究报告，全面分析三件作品")) == 3


def test_fast_job_uses_smaller_per_call_output_budget():
    from src.harness.context import RunContext

    assert RunContext("r", "u").max_output_tokens == 4096


def test_skill_propagates_stop_instead_of_returning_error_text(monkeypatch):
    from src.skills import loader
    skill = next(s for s in loader.load_skills() if s.id == "artwork_deep_analysis")
    model = SimpleNamespace(bind_tools=lambda _tools: model)
    monkeypatch.setattr(loader, "get_deterministic_llm", lambda: model)
    with run_scope(RunContext("r", "u", cancelled=lambda: True)):
        with pytest.raises(RunStopped, match="cancelled"):
            loader._skill_runner(skill)("test")
