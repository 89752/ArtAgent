import { create } from "zustand";
import { getJson, sendJson } from "../api/client";
import { streamChat } from "../api/chatStream";
import {
  persistAnalysisMessage,
  streamPaintingAnalysis,
} from "../api/paintingAnalysis";
import type {
  ArtworkAnalysisReport,
  ChatDone,
  ChipEntry,
  HistoryMessage,
  SceneCard,
  SessionDetailData,
  SessionItem,
  SessionListData,
  Source,
} from "../api/types";
import { toast } from "../lib/dialogs";
import { genId, updateUrl } from "../lib/utils";
import { useDocStore } from "./docStore";
import { useUiStore } from "./uiStore";

const SESSIONS_PAGE = 50;
const SESSIONS_SEARCH_CAP = 500;

export interface UserTurn {
  id: string;
  role: "user";
  text: string;
  files: ChipEntry[];
}

export interface AssistantTurn {
  id: string;
  role: "assistant";
  html: string;
  sources: Source[];
  streaming: boolean;
  cancelled: boolean;
  error: string;
  note?: string;
  report?: ArtworkAnalysisReport;
  title?: string;
  analysis?: boolean;
}

export interface AttachmentTurn {
  id: string;
  role: "attachment";
  docId: string;
  docName: string;
  kind: string;
}

export type Turn = UserTurn | AssistantTurn | AttachmentTurn;

interface StreamEntry {
  sid: string;
  turnId: string;
  abort: () => void;
  userText: string;
  files: ChipEntry[];
}

export interface ChatState {
  sid: string;
  title: string;
  view: "welcome" | "chat";
  turnsBySid: Record<string, Turn[]>;
  pendingRegenerate: boolean;
  editingTurnId: string | null;
  feedbackTurnId: string | null;
  streams: Record<string, StreamEntry>;
  sessionMap: Record<string, SessionItem>;
  sessions: SessionItem[];
  sessionsTotal: number;
  sessionsOffset: number;
  sessionsHasMore: boolean;
  searchLimited: boolean;
  cards: SceneCard[];
  memoryCount: number;
  uploadMaxBytes: number;
  ratedBySid: Record<string, boolean>;
  bootstrapLoaded: boolean;
  loadBootstrap: () => Promise<void>;
  loadSessions: () => Promise<void>;
  loadMoreSessions: () => Promise<void>;
  searchSessions: (q: string) => Promise<void>;
  openSession: (sid: string) => Promise<void>;
  newSession: () => void;
  deleteSession: (sid: string, title: string) => Promise<void>;
  renameSession: (sid: string, title: string) => Promise<boolean>;
  send: (
    text: string,
    opts?: { regenerate?: boolean; editInPlace?: boolean },
  ) => Promise<void>;
  stopGeneration: () => void;
  analyzeImage: (
    imageId: string,
    name: string,
    sid: string,
    focus?: string,
  ) => Promise<void>;
  regenerateLast: (turnId: string) => Promise<void>;
  editLast: (turnId: string) => void;
  cancelEdit: () => void;
  commitEdit: (text: string) => Promise<void>;
  setFeedbackTurn: (id: string | null) => void;
  markRated: (sid: string) => void;
  removeAttachmentTurns: (docId: string) => void;
  restoreFeedback: (sid: string) => Promise<void>;
}

function toTurns(messages: HistoryMessage[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      turns.push({
        id: genId(),
        role: "user",
        text: m.content || "",
        files: [],
      });
    } else if (m.role === "attachment") {
      turns.push({
        id: genId(),
        role: "attachment",
        docId: m.doc_id || "",
        docName: m.doc_name || "文档",
        kind: m.kind || "pdf",
      });
    } else {
      turns.push({
        id: genId(),
        role: "assistant",
        html: m.content || "",
        sources: m.sources || [],
        report: m.report,
        title: m.title,
        analysis: m.analysis === true,
        streaming: false,
        cancelled: false,
        error: "",
      });
    }
  }
  return turns;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  sid: genId(),
  title: "新对话",
  view: "welcome",
  turnsBySid: {},
  pendingRegenerate: false,
  editingTurnId: null,
  feedbackTurnId: null,
  streams: {},
  sessionMap: {},
  sessions: [],
  sessionsTotal: 0,
  sessionsOffset: 0,
  sessionsHasMore: false,
  searchLimited: false,
  cards: [],
  memoryCount: 0,
  uploadMaxBytes: 50 * 1024 * 1024,
  ratedBySid: {},
  bootstrapLoaded: false,

  loadBootstrap: async () => {
    try {
      const data = await getJson<{
        cards: SceneCard[];
        memory: number;
        upload_max_bytes: number;
      }>("/api/bootstrap");
      set({
        cards: data.cards || [],
        memoryCount: data.memory || 0,
        uploadMaxBytes:
          data.upload_max_bytes > 0 ? data.upload_max_bytes : get().uploadMaxBytes,
        bootstrapLoaded: true,
      });
    } catch (e) {
      console.error(e);
    }
  },

  loadSessions: async () => {
    try {
      const data = await getJson<SessionListData>(
        `/api/sessions?offset=0&limit=${SESSIONS_PAGE}`,
      );
      const map: Record<string, SessionItem> = {};
      for (const s of data.items || []) map[s.session_id] = s;
      set({
        sessionMap: map,
        sessions: data.items || [],
        sessionsTotal: data.total || (data.items || []).length,
        sessionsOffset: (data.items || []).length,
        sessionsHasMore: data.has_more === true,
      });
    } catch (e) {
      console.error(e);
    }
  },

  loadMoreSessions: async () => {
    try {
      const data = await getJson<SessionListData>(
        `/api/sessions?offset=${get().sessionsOffset}&limit=${SESSIONS_PAGE}`,
      );
      const map = { ...get().sessionMap };
      for (const s of data.items || []) map[s.session_id] = s;
      const sessions = [...get().sessions];
      for (const s of data.items || []) {
        if (!sessions.some((x) => x.session_id === s.session_id)) sessions.push(s);
      }
      set({
        sessionMap: map,
        sessions,
        sessionsTotal: data.total || Object.keys(map).length,
        sessionsOffset: get().sessionsOffset + (data.items || []).length,
        sessionsHasMore: data.has_more === true,
      });
    } catch (e) {
      console.error(e);
    }
  },

  searchSessions: async (q) => {
    const query = q.trim().toLowerCase();
    if (!query) {
      set({ searchLimited: false });
      await get().loadSessions();
      return;
    }
    try {
      let offset = get().sessionsOffset;
      const map = { ...get().sessionMap };
      while (offset < get().sessionsTotal && offset < SESSIONS_SEARCH_CAP) {
        const data = await getJson<SessionListData>(
          `/api/sessions?offset=${offset}&limit=100`,
        );
        for (const s of data.items || []) map[s.session_id] = s;
        const total = data.total || Object.keys(map).length;
        offset += (data.items || []).length;
        if (!data.has_more) {
          set({ sessionsTotal: total });
          break;
        }
        set({ sessionsTotal: total });
      }
      const list = Object.values(map).filter((s) =>
        (s.title || "").toLowerCase().includes(query),
      );
      set({
        sessionMap: map,
        sessions: list,
        sessionsOffset: offset,
        searchLimited: get().sessionsTotal > SESSIONS_SEARCH_CAP,
      });
    } catch (e) {
      console.error(e);
    }
  },

  openSession: async (sid) => {
    useDocStore.getState().stashChips(get().sid);
    set({ editingTurnId: null, feedbackTurnId: null });
    const live = get().streams[sid];
    if (live) {
      set((s) => ({
        sid,
        title: s.sessionMap[sid]?.title || s.title,
        view: "chat",
        pendingRegenerate: false,
      }));
      let turns: Turn[] = [];
      try {
        const data = await getJson<SessionDetailData>(
          `/api/sessions/${encodeURIComponent(sid)}`,
        );
        turns = toTurns(data.messages || []);
      } catch {
        /* 历史读取失败不影响实时气泡 */
      }
      turns.push({
        id: genId(),
        role: "user",
        text: live.userText || "请查阅我上传的文档并回答。",
        files: live.files,
      });
      const liveTurn = (get().turnsBySid[sid] || []).find(
        (t) => t.id === live.turnId,
      );
      if (liveTurn && liveTurn.role === "assistant") turns.push(liveTurn);
      set((s) => ({
        sid,
        title: s.sessionMap[sid]?.title || s.title,
        view: "chat",
        pendingRegenerate: false,
        turnsBySid: { ...s.turnsBySid, [sid]: turns },
      }));
      useDocStore.getState().loadChipsFor(sid);
      updateUrl(sid);
      return;
    }
    try {
      const data = await getJson<SessionDetailData>(
        `/api/sessions/${encodeURIComponent(sid)}`,
      );
      if (!(data.messages || []).length) {
        get().newSession();
        return;
      }
      const title = get().sessionMap[sid]?.title || "未命名对话";
      set((s) => ({
        sid,
        title,
        view: "chat",
        pendingRegenerate: false,
        turnsBySid: { ...s.turnsBySid, [sid]: toTurns(data.messages || []) },
      }));
      useDocStore.getState().loadChipsFor(sid);
      updateUrl(sid);
      void get().restoreFeedback(sid);
    } catch (e) {
      console.error(e);
      toast("加载会话失败", "err");
    }
  },

  newSession: () => {
    useDocStore.getState().stashChips(get().sid);
    const sid = genId();
    set({
      sid,
      title: "新对话",
      view: "welcome",
      pendingRegenerate: false,
      editingTurnId: null,
      feedbackTurnId: null,
      turnsBySid: { ...get().turnsBySid, [sid]: [] },
    });
    useDocStore.getState().loadChipsFor(sid);
    updateUrl("");
  },

  deleteSession: async (sid, title) => {
    const ok = await useUiStore
      .getState()
      .confirmAsk({
        title: "删除对话",
        text: `确定删除「${title || "未命名对话"}」？删除后不可恢复。`,
        okText: "删除",
        danger: true,
      });
    if (!ok) return;
    useDocStore.getState().clearPending(sid);
    try {
      await sendJson<{ ok: boolean }>(
        `/api/sessions/${encodeURIComponent(sid)}`,
        "DELETE",
      );
    } catch {
      /* 忽略：删除失败不阻塞本地切换 */
    }
    if (get().sid === sid) {
      useDocStore.getState().clearChips(sid);
      get().newSession();
    }
    void get().loadSessions();
  },

  renameSession: async (sid, title) => {
    try {
      await sendJson<{ ok: boolean }>(
        `/api/sessions/${encodeURIComponent(sid)}`,
        "PATCH",
        { title },
      );
      await get().loadSessions();
      if (get().sid === sid) set({ title });
      return true;
    } catch (e) {
      toast("重命名失败：" + (e instanceof Error ? e.message : e), "err");
      return false;
    }
  },

  send: async (textRaw, opts = {}) => {
    const text = (textRaw || "").trim();
    const docStore = useDocStore.getState();
    const sidAtSend = get().sid;
    const chips = docStore.chipsBySid[sidAtSend] || [];
    const hasFiles = chips.length > 0;
    if ((!text && !hasFiles) || get().streams[sidAtSend]) return;
    if (opts.regenerate || opts.editInPlace) set({ pendingRegenerate: true });

    if (!get().sessionMap[sidAtSend]) {
      const t = (text || "上传文档").slice(0, 24) || "新对话";
      const item: SessionItem = {
        session_id: sidAtSend,
        title: t,
        updated_at: new Date().toISOString(),
        relative: "刚刚",
      };
      set((s) => ({
        title: t,
        sessionMap: { ...s.sessionMap, [sidAtSend]: item },
        sessions: [item, ...s.sessions],
        sessionsHasMore: s.sessionsHasMore || Object.keys(s.sessionMap).length + 1 >= SESSIONS_PAGE,
      }));
    }
    set({ view: "chat" });

    if (!opts.editInPlace && !opts.regenerate) {
      const userTurn: UserTurn = {
        id: genId(),
        role: "user",
        text,
        files: [...chips],
      };
      set((s) => ({
        turnsBySid: {
          ...s.turnsBySid,
          [sidAtSend]: [...(s.turnsBySid[sidAtSend] || []), userTurn],
        },
      }));
    }
    docStore.clearChips(sidAtSend);

    const turnId = genId();
    const pendingHtml =
      '<div class="think-box" open><summary>正在思考…</summary></div>';
    set((s) => ({
      turnsBySid: {
        ...s.turnsBySid,
        [sidAtSend]: [
          ...(s.turnsBySid[sidAtSend] || []),
          {
            id: turnId,
            role: "assistant" as const,
            html: pendingHtml,
            sources: [],
            streaming: true,
            cancelled: false,
            error: "",
          },
        ],
      },
    }));

    const abortCtrl = new AbortController();
    set((s) => ({
      streams: {
        ...s.streams,
        [sidAtSend]: {
          sid: sidAtSend,
          turnId,
          abort: () => abortCtrl.abort(),
          userText: text,
          files: [...chips],
        },
      },
    }));

    try {
      await streamChat({
        message: text || "请查阅我上传的文档并回答。",
        sessionId: sidAtSend,
        regenerate: get().pendingRegenerate,
        signal: abortCtrl.signal,
        onDelta: (html) => updateTurn(sidAtSend, turnId, { html }),
        onDone: (evt: ChatDone) => {
          updateTurn(sidAtSend, turnId, {
            html: evt.html,
            sources: evt.sources || [],
            streaming: false,
            cancelled: evt.cancelled,
            error: evt.error || "",
          });
          window.dispatchEvent(new Event("artagent:memory-updated"));
          if (evt.session_id && get().sid === sidAtSend) {
            set({ sid: evt.session_id });
            updateUrl(evt.session_id);
          }
        },
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        updateTurn(sidAtSend, turnId, {
          note: "已停止生成",
          streaming: false,
        });
      } else {
        const msg = err instanceof Error ? err.message : "网络中断或服务未响应，请稍后重试。";
        updateTurn(sidAtSend, turnId, {
          html: "😔 " + msg,
          error: msg,
          streaming: false,
        });
      }
      console.error(err);
    } finally {
      set((s) => {
        const streams = { ...s.streams };
        delete streams[sidAtSend];
        return { streams };
      });
      if (get().sid === sidAtSend) set({ pendingRegenerate: false });
      void get().loadSessions();
    }
  },

  stopGeneration: () => {
    const entry = get().streams[get().sid];
    if (entry) entry.abort();
  },

  analyzeImage: async (imageId, name, sid, focus = "all") => {
    if (get().streams[sid]) {
      toast("请先完成当前生成");
      return;
    }
    set({ view: "chat" });
    const turnId = genId();
    const displayName = name || "画作";
    const userText = `分析画作：${displayName}`;
    const pendingHtml = `<div class="think-box" open><summary>正在分析《${escapeHtml(displayName)}》…</summary></div>`;
    set((s) => ({
      turnsBySid: {
        ...s.turnsBySid,
        [sid]: [
          ...(s.turnsBySid[sid] || []),
          { id: genId(), role: "user" as const, text: userText, files: [] },
        ],
      },
    }));
    set((s) => ({
      turnsBySid: {
        ...s.turnsBySid,
        [sid]: [
          ...(s.turnsBySid[sid] || []),
          {
            id: turnId,
            role: "assistant" as const,
            html: pendingHtml,
            sources: [],
            streaming: true,
            cancelled: false,
            error: "",
            report: undefined,
            title: displayName,
            analysis: true,
          },
        ],
      },
    }));

    const abortCtrl = new AbortController();
    set((s) => ({
      streams: {
        ...s.streams,
        [sid]: {
          sid,
          turnId,
          abort: () => abortCtrl.abort(),
          userText,
          files: [],
        },
      },
    }));

    try {
      await streamPaintingAnalysis({
        imageId,
        focus,
        signal: abortCtrl.signal,
        onEvent: (evt) => {
          if (evt.type === "stage") {
            updateTurn(sid, turnId, {
              html: `<div class="think-box" open><summary>正在分析…</summary><div class="think-body"><div class="chain-step"><span class="chain-dot pending"></span><div class="chain-name">${escapeHtml(evt.label)}</div></div></div></div>`,
            });
          } else if (evt.type === "done") {
            const doneHtml = `<div class="md-answer">分析完成：${escapeHtml(
              evt.report?.overall_assessment || "已生成三层分析报告",
            )}</div>`;
            updateTurn(sid, turnId, {
              html: doneHtml,
              report: evt.report,
              streaming: false,
              cancelled: false,
              error: "",
            });
            void persistAnalysisMessage(imageId, sid, {
              userText,
              html: doneHtml,
              title: displayName,
            }).catch(() => undefined);
          } else if (evt.type === "rejected") {
            const guide = evt.guide
              ? `<div class="stop-note">${escapeHtml(evt.guide)}</div>`
              : "";
            const rejectedHtml = `<div class="md-answer">😔 ${escapeHtml(evt.reason)}</div>${guide}`;
            updateTurn(sid, turnId, {
              html: rejectedHtml,
              error: evt.reason,
              streaming: false,
            });
            void persistAnalysisMessage(imageId, sid, {
              userText,
              html: rejectedHtml,
              title: displayName,
            }).catch(() => undefined);
          } else if (evt.type === "error") {
            updateTurn(sid, turnId, {
              html: `😔 ${escapeHtml(evt.message)}`,
              error: evt.message,
              streaming: false,
            });
          }
        },
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        updateTurn(sid, turnId, { note: "已停止分析", streaming: false });
      } else {
        const msg = err instanceof Error ? err.message : "分析失败，请稍后重试。";
        updateTurn(sid, turnId, {
          html: "😔 " + escapeHtml(msg),
          error: msg,
          streaming: false,
        });
      }
      console.error(err);
    } finally {
      set((s) => {
        const streams = { ...s.streams };
        delete streams[sid];
        return { streams };
      });
      void get().loadSessions();
    }
  },

  regenerateLast: async (turnId) => {
    const cur = get();
    if (cur.streams[cur.sid]) return;
    const turns = cur.turnsBySid[cur.sid] || [];
    const idx = turns.findIndex((t) => t.id === turnId);
    if (idx < 0) return;
    const turn = turns[idx];
    if (turn.role === "assistant" && turn.analysis) {
      toast("分析结果不支持重新生成，可点击「分析画作」重新分析");
      return;
    }
    let lastAssistantIdx = -1;
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
    if (idx !== lastAssistantIdx) {
      toast("只能重新生成最后一条回答");
      return;
    }
    let userText = "";
    for (let i = idx - 1; i >= 0; i--) {
      if (turns[i].role === "user") {
        userText = (turns[i] as UserTurn).text;
        break;
      }
    }
    if (!userText) return;
    set((s) => ({
      turnsBySid: {
        ...s.turnsBySid,
        [cur.sid]: s.turnsBySid[cur.sid].filter((t) => t.id !== turnId),
      },
    }));
    await get().send(userText, { regenerate: true });
  },

  editLast: (turnId) => {
    const cur = get();
    if (cur.streams[cur.sid]) return;
    if (cur.editingTurnId) {
      toast("请先完成当前编辑");
      return;
    }
    const turns = cur.turnsBySid[cur.sid] || [];
    const idx = turns.findIndex((t) => t.id === turnId);
    if (idx < 0) return;
    let lastUserIdx = -1;
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === "user") {
        lastUserIdx = i;
        break;
      }
    }
    if (idx !== lastUserIdx) {
      toast("只能编辑最后一条消息");
      return;
    }
    set({ editingTurnId: turnId, feedbackTurnId: null });
  },

  cancelEdit: () => set({ editingTurnId: null }),

  commitEdit: async (text) => {
    const cur = get();
    const id = cur.editingTurnId;
    if (!id) return;
    const trimmed = text.trim();
    if (!trimmed) {
      toast("消息不能为空", "err");
      return;
    }
    set((s) => {
      const turns = s.turnsBySid[s.sid] || [];
      const idx = turns.findIndex((t) => t.id === id);
      if (idx < 0) return s;
      return {
        editingTurnId: null,
        turnsBySid: {
          ...s.turnsBySid,
          [s.sid]: turns
            .map((t, i) =>
              i === idx && t.role === "user" ? { ...t, text: trimmed } : t,
            )
            .slice(0, idx + 1),
        },
      };
    });
    await get().send(trimmed, { editInPlace: true });
  },

  setFeedbackTurn: (id) => set({ feedbackTurnId: id }),

  markRated: (sid) =>
    set((s) => ({ ratedBySid: { ...s.ratedBySid, [sid]: true } })),

  restoreFeedback: async (sid) => {
    try {
      const data = await getJson<{ items: Array<{ session_id: string }> }>(
        "/api/feedback?limit=200",
      );
      if ((data.items || []).some((f) => f.session_id === sid)) {
        get().markRated(sid);
      }
    } catch {
      /* 忽略：反馈服务不可用时不影响主流程 */
    }
  },

  removeAttachmentTurns: (docId) => {
    set((s) => {
      const turnsBySid: Record<string, Turn[]> = {};
      for (const [sid, turns] of Object.entries(s.turnsBySid)) {
        turnsBySid[sid] = turns.filter(
          (t) => !(t.role === "attachment" && t.docId === docId),
        );
      }
      return { turnsBySid };
    });
  },
}));

function updateTurn(
  sid: string,
  turnId: string,
  patch: Partial<AssistantTurn>,
): void {
  useChatStore.setState((s) => {
    const turns = s.turnsBySid[sid];
    if (!turns) return s;
    return {
      turnsBySid: {
        ...s.turnsBySid,
        [sid]: turns.map((t) =>
          t.id === turnId && t.role === "assistant" ? { ...t, ...patch } : t,
        ),
      },
    };
  });
}

function escapeHtml(text: string): string {
  return String(text).replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ] as string,
  );
}
