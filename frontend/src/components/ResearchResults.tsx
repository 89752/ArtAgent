import { useEffect, useState } from "react";
import { getJson, sendJson } from "../api/client";
import type { AgentTask } from "../api/types";
import { renderMarkdown } from "../lib/markdown";
import { toast } from "../lib/dialogs";
import { SafeHtml } from "./SafeHtml";

interface Artifact {
  id: string; kind: string; content: string; revision: number;
  metadata: { step?: string; verification?: { passed: boolean; issues: string[] }; evidence?: Array<{id: string; locator: string; title: string; excerpt: string}> };
}
interface Experience {
  id: string; title: string; instructions: string; status: string;
  evaluation: { passed?: boolean; scope?: string; rows?: Array<{case_id: string; baseline: boolean; candidate: boolean}> };
}

export function ResearchResults({ task, onChange }: { task: AgentTask; onChange: () => Promise<void> }) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  useEffect(() => {
    let live = true;
    setLoading(true);
    getJson<{items: Artifact[]}>(`/api/jobs/${task.task_id}/artifacts`).then(data => {
      if (live) { setArtifacts(data.items.filter(a => a.kind !== "tool_result")); setError(""); }
    }).catch(e => { if (live) setError(String(e.message)); }).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [task.task_id, task.status, task.step_index]);

  const act = async (action: string) => {
    if (busy) return;
    setBusy(true); setError("");
    try {
      await sendJson(`/api/jobs/${task.task_id}/${action}`, "POST", action === "learn" ? undefined : {text: input});
      setInput("");
      toast(action === "learn" ? "经验候选已保存，请在经验区评测后启用。" : "任务已加入执行队列。");
      await onChange();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const download = (artifact: Artifact) => {
    const evidence = (artifact.metadata.evidence || []).map(e => `- [${e.id}] ${e.title}：${e.locator}`).join("\n");
    const url = URL.createObjectURL(new Blob([artifact.content + "\n\n## 证据来源\n" + evidence], {type: "text/markdown;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url; link.download = `research-${task.task_id}-v${artifact.revision}.md`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <div className="research-results" aria-busy={loading || busy}>
    <h4>{task.payload?.objective || "研究成果"}</h4>
    <ol className="research-steps">{task.steps?.map((step, i) => <li key={i}>{step.title} · {step.status === "done" ? "已完成" : step.status === "failed" ? "需处理" : "待完成"}</li>)}</ol>
    {loading && <p role="status">正在读取成果…</p>}
    {error && <p className="research-error" role="alert">{error}</p>}
    {!loading && !error && artifacts.length === 0 && <p>成果将在步骤执行后显示；未通过验收的草稿也会保留。</p>}
    {artifacts.map((a, i) => <details key={a.id} open={i === artifacts.length - 1}>
      <summary>{a.metadata.step || "研究报告"} · 第 {a.revision} 版 · {a.metadata.verification?.passed ? "验收通过" : "待修订"}</summary>
      {!!a.metadata.verification?.issues?.length && <p className="research-error">{a.metadata.verification.issues.join("；")}</p>}
      <div className="research-report"><SafeHtml html={renderMarkdown(a.content)} /></div>
      {!!a.metadata.evidence?.length && <section aria-label="证据来源"><h5>证据来源</h5><ul>{a.metadata.evidence.map(e => <li key={e.id}>
        <strong>[{e.id}] {e.title}</strong><p>{e.excerpt}</p>{/^https?:\/\//i.test(e.locator) ? <a href={e.locator} target="_blank" rel="noreferrer">查看原始来源</a> : <span>{e.locator}</span>}
      </li>)}</ul></section>}
      <button className="df-btn" type="button" onClick={() => download(a)}>下载此版本</button>
    </details>)}
    {["waiting_input", "failed", "verification_failed", "done", "budget_exhausted"].includes(task.status) && <div className="research-edit">
      <label htmlFor={`input-${task.task_id}`}>{task.status === "done" ? "修订要求" : "补充信息或修订要求"}</label>
      <textarea id={`input-${task.task_id}`} value={input} maxLength={8000} onChange={e => setInput(e.target.value)} rows={3} />
      <button className="df-btn" type="button" disabled={busy || !input.trim()} onClick={() => void act(["done", "budget_exhausted", "verification_failed"].includes(task.status) ? "revise" : "input")}>{busy ? "处理中…" : "提交并继续"}</button>
    </div>}
    {task.status === "done" && <button className="df-btn" type="button" disabled={busy} onClick={() => void act("learn")}>{busy ? "正在处理…" : "提炼为经验候选"}</button>}
  </div>;
}

export function ExperienceLibrary({ refreshKey }: { refreshKey: number }) {
  const [items, setItems] = useState<Experience[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let live = true;
    getJson<{items: Experience[]}>("/api/experiences").then(data => {if(live) setItems(data.items);}).catch(e => {if(live) setError(e.message);});
    return () => {live = false;};
  }, [version, refreshKey]);
  const act = async (id: string, action: string) => {
    setBusy(id); setError("");
    try { await sendJson(`/api/experiences/${id}/${action}`, "POST"); setVersion(v => v + 1); }
    catch(e) { setError(e instanceof Error ? e.message : String(e)); }
    finally {setBusy("");}
  };
  return <section className="research-learning"><h3>经验与技能改进</h3>
    <p>经验仅用于本账号。候选经过基线对比评测后才能启用；启用后的方法可随时停用。评测会调用模型。</p>
    {error && <p role="alert" className="research-error">{error}</p>}
    {!items.length && <p>完成研究任务后，可从成果区提炼第一条经验。</p>}
    {items.map(item => <details key={item.id}><summary>{item.title} · {({candidate:"候选",evaluated:"已评测",active:"已启用",rolled_back:"已停用"} as Record<string,string>)[item.status] || item.status}</summary>
      <p className="research-instructions">{item.instructions}</p>
      {!!item.evaluation.rows?.length && <><p>{item.evaluation.passed ? "对比评测通过" : "尚未证明改善，不能启用"}。{item.evaluation.scope}</p><ul>{item.evaluation.rows.map(r => <li key={r.case_id}>{r.case_id}：基线 {r.baseline ? "通过" : "未通过"} / 候选 {r.candidate ? "通过" : "未通过"}</li>)}</ul></>}
      <div className="ops-actions">
        {["candidate", "evaluated"].includes(item.status) && <button type="button" disabled={!!busy} onClick={() => void act(item.id,"evaluate")}>{busy === item.id ? "评测中…" : "运行对比评测"}</button>}
        {item.status === "evaluated" && <button type="button" disabled={!!busy || !item.evaluation.passed} onClick={() => void act(item.id,"promote")}>启用此经验</button>}
        {item.status === "active" && <button type="button" disabled={!!busy} onClick={() => void act(item.id,"rollback")}>停用并回滚</button>}
      </div></details>)}
  </section>;
}
