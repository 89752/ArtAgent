import type { AnalysisEvent } from "./types";

export interface StreamAnalysisOptions {
  imageId: string;
  focus?: string;
  signal: AbortSignal;
  onEvent: (evt: AnalysisEvent) => void;
}

/** POST /api/painting-analysis/{id} 并按 SSE 帧解析分析事件。 */
export async function streamPaintingAnalysis(
  opts: StreamAnalysisOptions,
): Promise<void> {
  const params = new URLSearchParams();
  if (opts.focus && opts.focus !== "all") params.set("focus", opts.focus);
  const qs = params.toString();
  const url = `/api/painting-analysis/${encodeURIComponent(opts.imageId)}${
    qs ? `?${qs}` : ""
  }`;

  let res: Response;
  try {
    res = await fetch(url, { method: "POST", signal: opts.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new Error("网络中断或服务未响应，请稍后重试。");
  }
  if (!res.ok) {
    let msg = `分析请求失败（HTTP ${res.status}）`;
    try {
      const j = (await res.json()) as { error?: string };
      if (j.error) msg = j.error;
    } catch {
      /* 忽略 */
    }
    throw new Error(msg);
  }
  if (!res.body) throw new Error("连接中断，未收到完整结果，请重试。");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let terminal = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt: AnalysisEvent;
      try {
        evt = JSON.parse(line.slice(5).trim()) as AnalysisEvent;
      } catch {
        continue;
      }
      opts.onEvent(evt);
      if (evt.type === "done" || evt.type === "rejected" || evt.type === "error") {
        terminal = true;
      }
    }
    if (terminal) break;
  }
  if (!terminal) throw new Error("连接中断，未收到完整结果，请重试。");
}

/** 把分析（含拒绝）写入会话历史：用户回合 + assistant 回合。 */
export async function persistAnalysisMessage(
  imageId: string,
  sid: string,
  opts: { userText?: string; html?: string; title?: string } = {},
): Promise<void> {
  const res = await fetch(
    `/api/painting-analysis/${encodeURIComponent(imageId)}/message`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sid,
        user_text: opts.userText || "",
        html: opts.html || "",
        title: opts.title || "",
      }),
    },
  );
  if (!res.ok) {
    throw new Error(`分析结果保存失败（HTTP ${res.status}）`);
  }
}
