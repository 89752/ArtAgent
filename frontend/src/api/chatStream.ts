import type { ChatDone, ChatEvent } from "./types";

export interface StreamChatOptions {
  message: string;
  sessionId: string;
  regenerate: boolean;
  signal: AbortSignal;
  onDelta: (html: string) => void;
  onDone: (evt: ChatDone) => void;
}

/**
 * POST /api/chat 并按 SSE 帧（\n\n）解析 data: 行。
 * 与原生版行为一致：未收到 done 即断开视为网络中断；AbortError 原样抛出。
 */
export async function streamChat(opts: StreamChatOptions): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: opts.signal,
      body: JSON.stringify({
        message: opts.message,
        session_id: opts.sessionId,
        regenerate: opts.regenerate,
      }),
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new Error("网络中断或服务未响应，请稍后重试。");
  }
  if (!res.ok) {
    let msg = `服务请求失败（HTTP ${res.status}）`;
    try {
      const j = (await res.json()) as { error?: string; detail?: unknown };
      if (j.error) msg = String(j.error);
      else if (typeof j.detail === "string") msg = j.detail;
    } catch {
      /* 忽略 */
    }
    throw new Error(msg);
  }
  if (!res.body) throw new Error("连接中断，未收到完整回复，请重试。");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let gotDone = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt: ChatEvent;
      try {
        evt = JSON.parse(line.slice(5).trim()) as ChatEvent;
      } catch {
        continue;
      }
      if (evt.type === "delta") {
        opts.onDelta(evt.html);
      } else if (evt.type === "done") {
        gotDone = true;
        opts.onDone(evt);
      }
    }
  }
  if (!gotDone) throw new Error("连接中断，未收到完整回复，请重试。");
}
