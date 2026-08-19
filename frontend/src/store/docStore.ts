import { create } from "zustand";
import { getJson, sendJson } from "../api/client";
import type { ChipEntry, Doc } from "../api/types";
import { uploadWithProgress } from "../api/upload";
import { attachUserImage, uploadUserImage } from "../api/userImages";
import { askOversize, toast } from "../lib/dialogs";
import { genId, updateUrl } from "../lib/utils";
import { useUiStore } from "./uiStore";

const PENDING_PREFIX = "artagent.pending.v1.";

function toChipStatus(status: string): ChipEntry["status"] {
  if (
    status === "uploading" ||
    status === "processing" ||
    status === "pending" ||
    status === "pending_confirm" ||
    status === "done" ||
    status === "active" ||
    status === "failed"
  ) {
    return status;
  }
  return "processing";
}

interface DocState {
  docs: Doc[];
  docsById: Record<string, Doc>;
  chipsBySid: Record<string, ChipEntry[]>;
  schemaDoc: Doc | null;
  uploadBusy: number;
  docPollDelay: number;
  pollTimer: number | null;
  loadDocuments: () => Promise<void>;
  syncChips: (list: Doc[]) => void;
  loadChipsFor: (sid: string) => void;
  stashChips: (sid: string) => void;
  persistChips: (sid: string) => void;
  restoreChips: (sid: string) => ChipEntry[];
  clearChips: (sid: string) => void;
  clearPending: (sid: string) => void;
  uploadFiles: (files: FileList | File[], sid: string, maxBytes: number) => Promise<void>;
  uploadImageFile: (file: File, entry: ChipEntry, sid: string) => void;
  uploadDoc: (
    file: File,
    entry: ChipEntry,
    sid: string,
    maxBytes: number,
    oversize: string,
  ) => void;
  finishUpload: () => void;
  chipStillPending: (entry: ChipEntry, sid: string) => boolean;
  updateChip: (sid: string, id: string, patch: Partial<ChipEntry>) => void;
  removeChipEntry: (sid: string, id: string) => void;
  recordAttachment: (entry: ChipEntry, sid: string) => Promise<void>;
  removeChip: (
    id: string,
    sid: string,
    onDocDeleted?: (docId: string) => void,
  ) => Promise<void>;
  retryDocument: (docId: string) => Promise<void>;
  deleteDocument: (
    d: Doc,
    onDocDeleted?: (docId: string) => void,
  ) => Promise<void>;
  openSchema: (d: Doc) => void;
  closeSchema: () => void;
}

export const useDocStore = create<DocState>()((set, get) => ({
  docs: [],
  docsById: {},
  chipsBySid: {},
  schemaDoc: null,
  uploadBusy: 0,
  docPollDelay: 3000,
  pollTimer: null,

  loadDocuments: async () => {
    try {
      const list = await getJson<Doc[]>("/api/documents");
      const byId: Record<string, Doc> = {};
      for (const d of list) byId[d.doc_id] = d;
      set({ docs: list, docsById: byId });
      get().syncChips(list);
      const pending = list.some(
        (d) => d.status === "processing" || d.status === "pending",
      );
      const timer = get().pollTimer;
      if (pending) {
        if (timer == null) {
          const id = window.setTimeout(() => {
            set({ pollTimer: null });
            const delay = Math.min(Math.round(get().docPollDelay * 1.6), 30000);
            set({ docPollDelay: delay });
            void get().loadDocuments();
          }, get().docPollDelay);
          set({ pollTimer: id });
        }
      } else if (timer != null) {
        window.clearTimeout(timer);
        set({ pollTimer: null, docPollDelay: 3000 });
      }
    } catch (e) {
      console.error(e);
    }
  },

  syncChips: (list: Doc[]) => {
    const server = new Map(list.map((d) => [d.doc_id, d]));
    const chipsBySid = { ...get().chipsBySid };
    let changed = false;
    for (const sid of Object.keys(chipsBySid)) {
      const chips = chipsBySid[sid].map((c) => {
        if (!c.doc_id) return c;
        const d = server.get(c.doc_id);
        if (!d) return c;
        const ns = d.status === "done" ? "done" : toChipStatus(d.status);
        const next: ChipEntry = { ...c, error: d.error || "" };
        if (ns !== c.status) {
          next.status = ns;
          changed = true;
        }
        if (d.kind) next.kind = d.kind;
        return next;
      });
      chipsBySid[sid] = chips;
    }
    if (changed) set({ chipsBySid });
  },

  loadChipsFor: (sid) => {
    let chips = get().chipsBySid[sid];
    if (!chips) {
      chips = get().restoreChips(sid);
      set({ chipsBySid: { ...get().chipsBySid, [sid]: chips } });
    }
    get().syncChips(Object.values(get().docsById));
    get().persistChips(sid);
  },

  stashChips: (sid) => get().persistChips(sid),

  persistChips: (sid) => {
    const chips = get().chipsBySid[sid] || [];
    try {
      const arr = chips
        .filter((c) => c.doc_id)
        .map((c) => ({
          id: c.id,
          doc_id: c.doc_id,
          name: c.name,
          size: c.size,
          kind: c.kind,
          status: c.status,
          error: c.error,
          thumb_url: c.thumb_url,
        }));
      localStorage.setItem(PENDING_PREFIX + sid, JSON.stringify(arr));
    } catch {
      /* 忽略 */
    }
  },

  restoreChips: (sid: string): ChipEntry[] => {
    try {
      const raw = localStorage.getItem(PENDING_PREFIX + sid);
      if (!raw) return [];
      const arr = (JSON.parse(raw) || []) as Array<Partial<ChipEntry>>;
      return arr
        .filter((c): c is ChipEntry => Boolean(c.id && c.name))
        .map((c) => ({
          id: String(c.id),
          doc_id: c.doc_id ?? null,
          name: String(c.name),
          size: typeof c.size === "number" ? c.size : null,
          kind:
            c.kind === "table"
              ? "table"
              : c.kind === "image"
                ? "image"
                : "pdf",
          status: c.status ? toChipStatus(c.status) : "processing",
          error: c.error || "",
          thumb_url:
            typeof c.thumb_url === "string"
              ? c.thumb_url
              : c.doc_id
                ? `/api/user-images/${c.doc_id}/file`
                : undefined,
        }));
    } catch {
      return [];
    }
  },

  clearChips: (sid) => {
    set({ chipsBySid: { ...get().chipsBySid, [sid]: [] } });
    get().persistChips(sid);
  },

  clearPending: (sid) => {
    try {
      localStorage.removeItem(PENDING_PREFIX + sid);
    } catch {
      /* 忽略 */
    }
    const chipsBySid = { ...get().chipsBySid };
    delete chipsBySid[sid];
    set({ chipsBySid });
  },

  uploadFiles: async (files, sid, maxBytes) => {
    const list = Array.from(files);
    const all: Array<{ f: File; entry: ChipEntry }> = [];
    const oversized: Array<{ f: File; entry: ChipEntry }> = [];
    for (const f of list) {
      const ext = (f.name.split(".").pop() || "").toLowerCase();
      const isImage = [".jpg", ".jpeg", ".png", ".webp"].includes("." + ext);
      const kind: ChipEntry["kind"] = isImage
        ? "image"
        : [".csv", ".xlsx", ".xls"].includes("." + ext)
          ? "table"
          : "pdf";
      const entry: ChipEntry = {
        id: "tmp-" + genId(),
        doc_id: null,
        name: f.name,
        size: f.size,
        kind,
        status: "uploading",
        error: "",
      };
      all.push({ f, entry });
      if (entry.kind === "pdf" && f.size > maxBytes) oversized.push({ f, entry });
      if (entry.kind === "image" && f.size > maxBytes) {
        entry.status = "failed";
        entry.error = "图片超过大小限制";
      }
    }
    set({
      chipsBySid: {
        ...get().chipsBySid,
        [sid]: [...(get().chipsBySid[sid] || []), ...all.map((a) => a.entry)],
      },
    });
    let mode = "";
    if (oversized.length) {
      mode = await askOversize(
        oversized[0].f.name,
        oversized.length,
        Math.round(maxBytes / (1024 * 1024)),
      );
    }
    for (const { f, entry } of all) {
      if (entry.kind === "image" && f.size > maxBytes) {
        toast(entry.error || "图片超过大小限制", "err");
        continue;
      }
      if (entry.kind === "pdf" && f.size > maxBytes) {
        if (!mode) {
          get().removeChipEntry(sid, entry.id);
          continue;
        }
        void get().uploadDoc(f, entry, sid, maxBytes, mode);
      } else {
        void get().uploadDoc(f, entry, sid, maxBytes, "");
      }
    }
  },

  uploadDoc: (file, entry, sid, maxBytes, oversize) => {
    if (entry.kind === "image") {
      get().uploadImageFile(file, entry, sid);
      return;
    }
    set({ uploadBusy: get().uploadBusy + 1 });
    uploadWithProgress(file, oversize, (pct) => {
      get().updateChip(sid, entry.id, { progress: pct });
    }).then((j) => {
      if (!j.ok) {
        if (j.code === "oversized") {
          get().finishUpload();
          const maxMb = Math.max(
            1,
            Math.round((j.max_bytes || maxBytes) / (1024 * 1024)),
          );
          void askOversize(entry.name, 1, maxMb).then((m) => {
            if (m) {
              void get().uploadDoc(file, entry, sid, maxBytes, m);
            } else {
              get().removeChipEntry(sid, entry.id);
              get().persistChips(sid);
              toast("已取消上传", "err");
            }
          });
          return;
        }
        get().updateChip(sid, entry.id, {
          status: "failed",
          error: j.error || `上传失败（HTTP）`,
          progress: undefined,
        });
        toast(j.error || "上传失败", "err");
        get().finishUpload();
        return;
      }
      if (j.split) {
        get().removeChipEntry(sid, entry.id);
        const parts: ChipEntry[] = (j.documents || []).map((d) => ({
          id: "part-" + genId(),
          doc_id: d.doc_id,
          name: d.doc_name,
          size: null,
          kind: "pdf",
          status: "processing",
          error: "",
        }));
        set({
          chipsBySid: {
            ...get().chipsBySid,
            [sid]: [...(get().chipsBySid[sid] || []), ...parts],
          },
        });
        for (const p of parts) {
          if (get().chipStillPending(p, sid)) void get().recordAttachment(p, sid);
        }
        get().persistChips(sid);
        set({ docPollDelay: 3000 });
        toast(`已拆分为 ${j.count || 0} 份，正在后台解析…`);
        void get().loadDocuments();
        get().finishUpload();
        return;
      }
      const completedEntry: ChipEntry = {
        ...entry,
        doc_id: j.doc_id || null,
        kind: j.kind || entry.kind,
        status: "processing",
        progress: undefined,
      };
      get().updateChip(sid, entry.id, completedEntry);
      set({ docPollDelay: 3000 });
      get().persistChips(sid);
      // entry 是上传开始时的旧对象，doc_id 仍为 null。使用服务器响应构造的
      // 新对象记录附件，确保刷新后会话仍能恢复这次上传。
      if (get().chipStillPending(completedEntry, sid)) {
        void get().recordAttachment(completedEntry, sid);
      }
      toast(`已上传《${j.doc_name || entry.name}》，正在解析…`);
      void get().loadDocuments();
      get().finishUpload();
    });
  },

  uploadImageFile: (file, entry, sid) => {
    set({ uploadBusy: get().uploadBusy + 1 });
    uploadUserImage(file, sid, (pct) => {
      get().updateChip(sid, entry.id, { progress: pct });
    }).then((j) => {
      if (!j.ok) {
        get().updateChip(sid, entry.id, {
          status: "failed",
          error: j.error || "上传失败",
          progress: undefined,
        });
        toast(j.error || "上传失败", "err");
        get().finishUpload();
        return;
      }
      get().updateChip(sid, entry.id, {
        doc_id: j.image_id || null,
        thumb_url: j.thumb_url || "",
        status: "done",
        progress: undefined,
      });
      get().persistChips(sid);
      if (j.image_id && get().chipStillPending(entry, sid)) {
        void attachUserImage(sid, j.image_id).catch(() => undefined);
      }
      toast("图片已上传，可点击「分析画作」或直接在聊天中提问");
      get().finishUpload();
    });
  },

  finishUpload: () => {
    set({ uploadBusy: Math.max(0, get().uploadBusy - 1) });
  },

  chipStillPending: (entry, sid) =>
    Boolean((get().chipsBySid[sid] || []).some((c) => c.id === entry.id)),

  updateChip: (sid, id, patch) => {
    set({
      chipsBySid: {
        ...get().chipsBySid,
        [sid]: (get().chipsBySid[sid] || []).map((c) =>
          c.id === id ? { ...c, ...patch } : c,
        ),
      },
    });
  },

  removeChipEntry: (sid, id) => {
    set({
      chipsBySid: {
        ...get().chipsBySid,
        [sid]: (get().chipsBySid[sid] || []).filter((c) => c.id !== id),
      },
    });
  },

  recordAttachment: async (entry, sid) => {
    if (!entry.doc_id) return;
    try {
      await sendJson<{ ok: boolean }>(
        `/api/sessions/${encodeURIComponent(sid)}/attachment`,
        "POST",
        { doc_id: entry.doc_id },
      );
      // 仅上传附件、尚未发消息的会话也必须可通过刷新恢复。
      updateUrl(sid);
    } catch (e) {
      console.error("记录附件失败", e);
    }
  },

  removeChip: async (id, sid, onDocDeleted) => {
    const c = (get().chipsBySid[sid] || []).find((x) => x.id === id);
    if (!c) return;
    if (c.kind === "image" && c.doc_id) {
      const ok = await confirmDelete(
        "删除图片",
        `确定删除「${c.name}」？将同时删除上传图片与分析结果，不可恢复。`,
      );
      if (!ok) return;
      try {
        const j = await sendJson<{ ok: boolean; error?: string }>(
          `/api/user-images/${encodeURIComponent(c.doc_id)}`,
          "DELETE",
        );
        if (!j.ok) {
          toast(j.error || "删除失败", "err");
          return;
        }
        onDocDeleted?.(c.doc_id);
      } catch (e) {
        toast("删除失败：" + (e instanceof Error ? e.message : e), "err");
        return;
      }
    } else if (c.doc_id && c.status !== "processing" && c.status !== "uploading") {
      const ok = await confirmDelete(
        "删除文档",
        `确定删除「${c.name}」？将同时删除上传文件和索引向量，不可恢复。`,
      );
      if (!ok) return;
      try {
        const j = await sendJson<{ ok: boolean; error?: string }>(
          `/api/documents/${encodeURIComponent(c.doc_id)}`,
          "DELETE",
        );
        if (!j.ok) {
          toast(j.error || "删除失败", "err");
          return;
        }
        onDocDeleted?.(c.doc_id);
        void get().loadDocuments();
      } catch (e) {
        toast("删除失败：" + (e instanceof Error ? e.message : e), "err");
        return;
      }
    } else if (c.status === "processing") {
      toast("文档仍在后台解析，可在资料库中管理");
    } else {
      toast("已移除附件");
    }
    get().removeChipEntry(sid, id);
    get().persistChips(sid);
  },

  retryDocument: async (docId) => {
    try {
      const j = await sendJson<{ ok: boolean; error?: string }>(
        `/api/tasks/${encodeURIComponent(docId)}/retry`,
        "POST",
      );
      if (!j.ok) {
        toast(j.error || "重试失败", "err");
        return;
      }
      toast("已重新加入解析队列");
      const chipsBySid = { ...get().chipsBySid };
      for (const sid of Object.keys(chipsBySid)) {
        chipsBySid[sid] = chipsBySid[sid].map((c) =>
          c.doc_id === docId ? { ...c, status: "processing" as const, error: "" } : c,
        );
      }
      set({ chipsBySid });
      void get().loadDocuments();
    } catch (e) {
      toast("重试失败：" + (e instanceof Error ? e.message : e), "err");
    }
  },

  deleteDocument: async (d, onDocDeleted) => {
    const siblings = splitSiblings(d, get().docs);
    let targets = [d];
    if (siblings.length) {
      const source = d.split_source_name || inferredSplitSource(d.doc_name || "");
      const together = await confirmDelete(
        "删除同源文件",
        `检测到另外 ${siblings.length} 个文件可能与「${d.doc_name || "当前文件"}」来自同一次拆分${
          source ? `（原文件可能为「${source}」）` : ""
        }。是否一起删除？\n\n选择“删除”将删除这组文件；选择“取消”后可继续仅删除当前文件。`,
      );
      if (together) targets = [d, ...siblings];
    }
    const ok = await confirmDelete(
      targets.length > 1 ? "删除整组文档" : "删除文档",
      targets.length > 1
        ? `确定删除这 ${targets.length} 个可能同源的拆分文件？\n将同时删除上传文件和索引向量，不可恢复。`
        : `确定删除「${d.doc_name || "未命名文档"}」？\n将同时删除上传文件和索引向量，不可恢复。`,
    );
    if (!ok) return;
    try {
      const deletedIds: string[] = [];
      for (const target of targets) {
        const j = await sendJson<{ ok: boolean; error?: string }>(
          `/api/documents/${encodeURIComponent(target.doc_id)}`,
          "DELETE",
        );
        if (!j.ok) throw new Error(j.error || `删除「${target.doc_name || "文档"}」失败`);
        deletedIds.push(target.doc_id);
        onDocDeleted?.(target.doc_id);
      }
      toast(deletedIds.length > 1 ? `已删除 ${deletedIds.length} 个同源文件` : "已删除文档");
      const chipsBySid = { ...get().chipsBySid };
      for (const sid of Object.keys(chipsBySid)) {
        chipsBySid[sid] = chipsBySid[sid].filter(
          (c) => !c.doc_id || !deletedIds.includes(c.doc_id),
        );
      }
      set({ chipsBySid });
      void get().loadDocuments();
    } catch (e) {
      toast("删除失败：" + (e instanceof Error ? e.message : e), "err");
    }
  },

  openSchema: (d) => {
    useUiStore.getState().rememberFocus();
    set({ schemaDoc: d });
  },

  closeSchema: () => {
    set({ schemaDoc: null });
    useUiStore.getState().restoreFocus();
  },
}));

async function confirmDelete(title: string, text: string): Promise<boolean> {
  const { confirmAsk } = useUiStore.getState();
  return confirmAsk({ title, text, okText: "删除", danger: true });
}

function inferredSplitSource(name: string): string {
  return name.replace(/_part\d+\.pdf$/i, ".pdf");
}

function splitSiblings(doc: Doc, docs: Doc[]): Doc[] {
  if (doc.kind !== "pdf") return [];
  if (doc.split_group_id) {
    return docs.filter(
      (candidate) =>
        candidate.doc_id !== doc.doc_id &&
        candidate.split_group_id === doc.split_group_id,
    );
  }
  // 兼容旧版本：仅把严格符合 xxx_partN.pdf 且基础名一致的文件视为“可能同源”。
  const match = (doc.doc_name || "").match(/^(.*)_part\d+\.pdf$/i);
  if (!match) return [];
  const base = match[1].toLocaleLowerCase();
  return docs.filter((candidate) => {
    if (candidate.doc_id === doc.doc_id || candidate.kind !== "pdf") return false;
    const other = (candidate.doc_name || "").match(/^(.*)_part\d+\.pdf$/i);
    return Boolean(other && other[1].toLocaleLowerCase() === base);
  });
}
