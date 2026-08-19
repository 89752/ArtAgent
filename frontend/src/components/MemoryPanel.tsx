import { useEffect, useRef, useState } from "react";
import { getJson, sendJson } from "../api/client";
import type { MemoryItem, MemoryListData } from "../api/types";
import { importMemoryFile } from "../api/upload";
import { confirmAsk, toast } from "../lib/dialogs";
import { relTime } from "../lib/utils";
import { useUiStore } from "../store/uiStore";
import { ModalShell } from "./ModalShell";

const KIND_LABEL: Record<string, string> = {
  preference: "偏好",
  fact: "事实",
  profile: "画像",
  event: "事件",
  correction: "纠正",
  artist: "画家",
  style: "风格",
};

const SOURCE_LABEL: Record<string, string> = {
  user_explicit: "用户明确",
  extracted: "自动抽取",
  eval: "评估数据",
};

export function MemoryPanel({ embedded = false }: { embedded?: boolean }) {
  const modal = useUiStore((s) => s.modal);
  const closeModal = useUiStore((s) => s.closeModal);
  const [items, setItems] = useState<MemoryItem[] | null>(null);
  const [importText, setImportText] = useState("");
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = async () => {
    try {
      const data = await getJson<MemoryListData>("/api/memory");
      setItems(data.items || []);
    } catch (e) {
      console.error(e);
      toast("加载记忆失败", "err");
      setItems([]);
    }
  };

  useEffect(() => {
    // 设置页按需挂载 embedded 面板；挂载时也必须立即读取记忆。
    if (embedded || modal === "memory") {
      setItems(null);
      void load();
    }
  }, [modal, embedded]);

  useEffect(() => {
    let delayedReload: number | null = null;
    const onMemoryUpdated = () => {
      if (useUiStore.getState().modal !== "memory" && !embedded) return;
      void load();
      // 自动记忆在后台有约 2 秒防抖；即时刷新后再补一次，展示刚落库的条目。
      if (delayedReload !== null) window.clearTimeout(delayedReload);
      delayedReload = window.setTimeout(() => void load(), 3000);
    };
    window.addEventListener("artagent:memory-updated", onMemoryUpdated);
    return () => {
      window.removeEventListener("artagent:memory-updated", onMemoryUpdated);
      if (delayedReload !== null) window.clearTimeout(delayedReload);
    };
  }, [embedded]);

  const del = async (it: MemoryItem) => {
    const ok = await confirmAsk({
      title: "删除记忆",
      text: `确定删除「${it.content || it.value}」这条记忆？删除后不可恢复。`,
      okText: "删除",
      danger: true,
    });
    if (!ok) return;
    try {
      const j = await sendJson<{ ok: boolean; error?: string }>(
        `/api/memory/${encodeURIComponent(it.id)}`,
        "DELETE",
      );
      if (!j.ok) {
        toast(j.error || "删除失败", "err");
        return;
      }
      toast("已删除这条记忆");
      void load();
    } catch (e) {
      toast("删除失败：" + (e instanceof Error ? e.message : e), "err");
    }
  };

  const clearAll = async () => {
    const ok = await confirmAsk({
      title: "清空全部记忆",
      text: "确定清空 Agent 记住的全部偏好？不可恢复。",
      okText: "清空",
      danger: true,
    });
    if (!ok) return;
    try {
      const j = await sendJson<{ ok: boolean; error?: string }>(
        "/api/memory",
        "DELETE",
      );
      if (!j.ok) {
        toast(j.error || "清空失败", "err");
        return;
      }
      toast("记忆已清空");
      void load();
    } catch (e) {
      toast("清空失败：" + (e instanceof Error ? e.message : e), "err");
    }
  };

  const doImport = async () => {
    if (!importText.trim() || importing) return;
    setImporting(true);
    try {
      const j = await sendJson<{ ok: boolean; error?: string }>(
        "/api/memory/import",
        "POST",
        { text: importText },
      );
      if (!j.ok) {
        toast(j.error || "导入失败", "err");
        return;
      }
      toast("记忆已导入");
      setImportText("");
      void load();
    } catch (e) {
      toast("导入失败：" + (e instanceof Error ? e.message : e), "err");
    } finally {
      setImporting(false);
    }
  };

  const doImportFile = async (file: File) => {
    if (importing) return;
    setImporting(true);
    try {
      const j = await importMemoryFile(file);
      if (!j.ok) {
        toast(j.error || "导入失败", "err");
        return;
      }
      const s = j.stats;
      const detail =
        s && (s.added || s.dup || s.invalid)
          ? `新增 ${s.added}，重复 ${s.dup}，无效 ${s.invalid}`
          : "";
      toast(detail ? `记忆已导入：${detail}` : "记忆已导入");
      void load();
    } catch (e) {
      toast("导入失败：" + (e instanceof Error ? e.message : e), "err");
    } finally {
      setImporting(false);
    }
  };

  const header = (
    <div className="df-dialog-header">
      <h3 id={embedded ? "mem-settings-title" : "mem-title"}>记忆</h3>
      <p className="df-dialog-desc">
        这些记忆来自您的对话（含自动抽取），跨会话累积，可单项删除或一键清空。
      </p>
    </div>
  );

  const list = (
    <>
      <div className="mem-import">
        <textarea
          className="mem-import-input"
          value={importText}
          rows={3}
          placeholder={"每行一条记忆，例如：\n用户喜欢莫奈的睡莲\n用户偏好简洁回复"}
          onChange={(e) => setImportText(e.target.value)}
        />
        <div className="mem-import-row">
          <button
            type="button"
            className="df-btn"
            disabled={importing}
            onClick={() => void doImport()}
          >
            {importing ? "导入中…" : "导入记忆"}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.md,.json,.csv"
            className="mem-file-input"
            onChange={(e) => {
              const f = e.target.files && e.target.files[0];
              if (f) void doImportFile(f);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            className="df-btn"
            disabled={importing}
            onClick={() => fileRef.current?.click()}
          >
            从文件导入
          </button>
          <span className="mem-import-hint">支持 .txt / .md / .json / .csv</span>
        </div>
      </div>
      <div id="mem-list" className="mem-list">
        {items === null && <div className="mem-empty">加载中…</div>}
        {items !== null && items.length === 0 && (
          <div className="mem-empty">
            还没有记忆。多聊聊喜欢的画家、风格和偏好，Agent 会记住（含自动抽取）。
          </div>
        )}
        {(items || []).map((it) => (
          <div key={it.id} className="mem-item">
            <div className="mem-main">
              <span className="mem-kind">
                {KIND_LABEL[it.kind || ""] || it.kind || "记忆"}
              </span>
              <span className="mem-value" data-tip={it.content || it.value || ""}>
                {it.content || it.value || ""}
              </span>
            </div>
            <div className="mem-meta">
              {(it.entity ? "实体：" + it.entity + " · " : "") +
                (SOURCE_LABEL[it.source || ""] || it.source || "用户明确") +
                (relTime(it.updated_at) ? " · " + relTime(it.updated_at) : "")}
            </div>
            <button
              type="button"
              className="mem-del"
              aria-label={`删除记忆 ${it.content || it.value}`}
              data-tip="删除这条记忆"
              onClick={() => void del(it)}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </>
  );

  const footer = (
    <div className="df-dialog-footer">
      <button
        id="mem-clear"
        className="mem-clear danger"
        type="button"
        onClick={() => void clearAll()}
      >
        清空全部记忆
      </button>
    </div>
  );

  if (embedded) {
    return (
      <div className="settings-page">
        <div className="settings-page-head">{header}</div>
        <div className="settings-page-body">{list}</div>
        <div className="settings-page-foot">{footer}</div>
      </div>
    );
  }

  return (
    <ModalShell
      open={modal === "memory"}
      outerClass="mem-panel"
      labelledBy="mem-title"
      onBackdrop={() => closeModal("memory")}
    >
      <div className="mem-card df-dialog">
        <button
          id="mem-close"
          className="drawer-close df-close"
          type="button"
          aria-label="关闭记忆面板"
          onClick={() => closeModal("memory")}
        >
          ×
        </button>
        {header}
        <div className="df-dialog-body">{list}</div>
        {footer}
      </div>
    </ModalShell>
  );
}
