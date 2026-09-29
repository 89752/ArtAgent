import { useEffect, useState } from "react";
import { getJson, sendJson } from "../api/client";
import type { AgentTask, OperationsMetrics, RunDetail, RunSummary } from "../api/types";
import { toast } from "../lib/dialogs";
import { useUiStore } from "../store/uiStore";
import { ModalShell } from "./ModalShell";
import { ResearchResults, ExperienceLibrary } from "./ResearchResults";

const ACTIVE = new Set(["pending", "processing"]);

export function OperationsCenter() {
  const modal = useUiStore((s) => s.modal);
  const closeModal = useUiStore((s) => s.closeModal);
  const [metrics, setMetrics] = useState<OperationsMetrics | null>(null);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [objective, setObjective] = useState("");
  const [plan, setPlan] = useState("");
  const [busy, setBusy] = useState(false);
  const [openedTask, setOpenedTask] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [docIds, setDocIds] = useState<string[]>([]);
  const [availableDocs, setAvailableDocs] = useState<Array<{doc_id: string; doc_name?: string}>>([]);

  const refresh = async () => {
    try {
      const [m, t, r] = await Promise.all([
        getJson<OperationsMetrics>("/api/metrics?limit=100"),
        getJson<{ items: AgentTask[] }>("/api/tasks"),
        getJson<{ items: RunSummary[] }>("/api/runs?limit=15"),
      ]);
      setMetrics(m);
      setTasks((t.items || []).filter((item) => item.type === "agent_job"));
      setRuns(r.items || []);
      setRefreshKey(k => k + 1);
    } catch (error) {
      toast("加载运行中心失败：" + (error instanceof Error ? error.message : error), "err");
    }
  };

  useEffect(() => {
    if (modal !== "operations") return;
    void getJson<{items: Array<{doc_id: string; doc_name?: string}>}>("/api/documents").then(d => setAvailableDocs(d.items || [])).catch(() => setAvailableDocs([]));
    void refresh();
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [modal]);

  const createJob = async () => {
    if (!objective.trim() || busy) return;
    setBusy(true);
    try {
      await sendJson("/api/jobs", "POST", {
        objective: objective.trim(),
        plan: plan.split("\n").map((x) => x.trim()).filter(Boolean),
        spec: {document_ids: docIds, min_sources: 1},
      });
      setObjective("");
      setPlan("");
      toast("长期任务已开始；每一步都会保留可恢复产物。");
      await refresh();
    } catch (error) {
      toast("创建任务失败：" + (error instanceof Error ? error.message : error), "err");
    } finally {
      setBusy(false);
    }
  };

  const action = async (task: AgentTask, name: "pause" | "resume" | "retry" | "cancel") => {
    try {
      await sendJson(`/api/jobs/${encodeURIComponent(task.task_id)}/${name}`, "POST");
      await refresh();
    } catch (error) {
      toast("任务操作失败：" + (error instanceof Error ? error.message : error), "err");
    }
  };

  const showRun = async (run: RunSummary) => {
    try {
      const data = await getJson<{ ok: boolean; run: RunDetail }>(`/api/runs/${run.id}`);
      setSelected(data.run);
    } catch (error) {
      toast("加载运行轨迹失败：" + (error instanceof Error ? error.message : error), "err");
    }
  };

  return (
    <ModalShell open={modal === "operations"} outerClass="ops-modal" labelledBy="ops-title" onBackdrop={() => closeModal("operations")}>
      <section className="ops-card">
        <header className="ops-header">
          <div><h2 id="ops-title">运行中心</h2><p>查看本账号的模型路由、调用轨迹与可恢复任务。</p></div>
          <button type="button" className="df-close" aria-label="关闭运行中心" onClick={() => closeModal("operations")}>×</button>
        </header>
        <div className="ops-body">
          <div className="ops-metrics">
            <Metric label="运行次数" value={String(metrics?.count || 0)} />
            <Metric label="P95 延迟" value={metrics?.latency_ms ? `${(metrics.latency_ms.p95 / 1000).toFixed(1)}s` : "—"} />
            <Metric label="错误率" value={metrics ? `${((metrics.error_rate || 0) * 100).toFixed(1)}%` : "—"} />
            <Metric label="估算成本" value={metrics?.est_cost_total?.toFixed(4) || "0"} />
          </div>
          <div className="ops-grid">
            <section className="ops-section">
              <h3>长期任务</h3>
              <label htmlFor="research-objective">研究目标</label>
              <textarea id="research-objective" maxLength={4000} value={objective} onChange={(e) => setObjective(e.target.value)} placeholder="例如：比较莫奈与透纳的光色语言，形成带来源的报告" rows={2} />
              <fieldset className="research-docs"><legend>选择研究资料（可选）</legend>{availableDocs.length ? availableDocs.map(doc => <label key={doc.doc_id}><input type="checkbox" checked={docIds.includes(doc.doc_id)} onChange={e => setDocIds(ids => e.target.checked ? [...ids, doc.doc_id] : ids.filter(id => id !== doc.doc_id))} />{doc.doc_name || doc.doc_id}</label>) : <p>可先在资料库上传画册，也可以使用现有艺术资料研究。</p>}</fieldset>
              <label htmlFor="research-plan">自定义步骤（可选）</label>
              <textarea id="research-plan" value={plan} onChange={(e) => setPlan(e.target.value)} placeholder="每行一个步骤；留空将依次收集证据、比较论点、生成报告" rows={3} />
              <button className="df-btn" type="button" disabled={!objective.trim() || busy} onClick={() => void createJob()}>{busy ? "创建中…" : "创建可恢复任务"}</button>
              <div className="ops-list">{tasks.length === 0 ? <p className="ops-empty">暂无长期任务。</p> : tasks.map((task) => <div className="ops-task" key={task.task_id}>
                <div><b>{({pending:"排队中",processing:"执行中",paused:"已暂停",done:"已完成",failed:"失败",interrupted:"已中断",waiting_input:"待补充",verification_failed:"待修订",budget_exhausted:"预算耗尽",unknown_execution_state:"需核对执行结果"} as Record<string,string>)[task.status]}</b><span>{task.payload?.objective || task.task_id}</span>{task.error && <small>{task.error}</small>}</div>
                <div className="ops-actions">
                  <button type="button" aria-expanded={openedTask === task.task_id} onClick={() => setOpenedTask(openedTask === task.task_id ? "" : task.task_id)}>查看成果</button>
                  {ACTIVE.has(task.status) && <button type="button" onClick={() => void action(task, "pause")}>暂停</button>}
                  {task.status === "paused" && <button type="button" onClick={() => void action(task, "resume")}>继续</button>}
                  {(task.status === "failed" || task.status === "interrupted") && <button type="button" onClick={() => void action(task, "retry")}>重试</button>}
                  {ACTIVE.has(task.status) && <button type="button" onClick={() => void action(task, "cancel")}>取消</button>}
                </div>
              </div>)}</div>
              {tasks.filter(t => t.task_id === openedTask).map(t => <ResearchResults key={t.task_id} task={t} onChange={refresh} />)}
            </section>
            <section className="ops-section">
              <h3>模型路由与近期运行</h3>
              <p className="ops-route">{Object.entries(metrics?.model_roles || {}).map(([role, count]) => `${role} ${count}`).join(" · ") || "尚无模型调用记录"}</p>
              <div className="ops-list">{runs.length === 0 ? <p className="ops-empty">暂无运行轨迹。</p> : runs.map((run) => <button type="button" className="ops-run" key={run.id} onClick={() => void showRun(run)}>
                <span>#{run.id} · {run.intent || "general"}</span><span>{(run.latency_ms / 1000).toFixed(1)}s</span><small>{(run.tools || []).join("、") || "无工具"}{run.error ? " · 异常" : ""}</small>
              </button>)}</div>
              {selected && <div className="ops-trace"><b>运行 #{selected.id} 轨迹</b><p>节点：{selected.node_events.map((x) => `${x.node_name} ${Math.round(x.latency_ms)}ms`).join(" · ") || "—"}</p><p>模型：{selected.model_calls.map((x) => `${x.role || "main"}/${x.model || "provider"}`).join(" · ") || "—"}</p><p>工具：{selected.tool_calls.map((x) => `${x.tool_name} (${x.status})`).join(" · ") || "—"}</p></div>}
            </section>
          </div>
          <ExperienceLibrary refreshKey={refreshKey} />
        </div>
      </section>
    </ModalShell>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
