import {
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { useChatStore } from "../store/chatStore";
import { useDocStore } from "../store/docStore";
import { IconAttach, IconSend, IconStop } from "./icons";

const FILE_STATUS_LABEL: Record<string, string> = {
  uploading: "上传中…",
  processing: "解析中…",
  pending: "解析中…",
  pending_confirm: "待确认 schema",
  done: "已就绪",
  active: "已就绪",
  failed: "失败",
};

export function Composer() {
  const sid = useChatStore((s) => s.sid);
  const streaming = useChatStore((s) => !!s.streams[s.sid]);
  const send = useChatStore((s) => s.send);
  const stopGeneration = useChatStore((s) => s.stopGeneration);
  const chips = useDocStore((s) => s.chipsBySid[sid]);
  const maxBytes = useChatStore((s) => s.uploadMaxBytes);
  const docsById = useDocStore((s) => s.docsById);
  const uploadFiles = useDocStore((s) => s.uploadFiles);
  const removeChip = useDocStore((s) => s.removeChip);
  const retryDocument = useDocStore((s) => s.retryDocument);
  const openSchema = useDocStore((s) => s.openSchema);
  const removeAttachmentTurns = useChatStore((s) => s.removeAttachmentTurns);
  const analyzeImage = useChatStore((s) => s.analyzeImage);

  const [value, setValue] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const chipList = chips ?? [];

  useLayoutEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
  }, [value]);

  const doSend = () => {
    if (!value.trim() && !chipList.length) return;
    const text = value;
    setValue("");
    void send(text);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      doSend();
    }
  };

  const onFiles = (files: FileList | null) => {
    if (files?.length) void uploadFiles(files, sid, maxBytes);
  };

  return (
    <div className="composer">
      <div id="chips" className="chips" hidden={!chipList.length}>
        {chipList.map((c) => {
          const doc = c.doc_id ? docsById[c.doc_id] : undefined;
          const label = FILE_STATUS_LABEL[c.status] || c.status || "";
          const canConfirmSchema = c.status === "pending_confirm" && !!doc;
          const statusClass =
            "chip-status" +
            (c.status === "done" || c.status === "active"
              ? " ok"
              : c.status === "failed"
                ? " err"
                : c.status === "pending_confirm"
                  ? " warn action"
                  : " warn");
          return (
            <div key={c.id} className="chip" data-tip={c.error || undefined}>
              {c.kind === "image" ? (
                <img
                  className="chip-thumb"
                  src={
                    c.thumb_url ||
                    (c.doc_id ? `/api/user-images/${c.doc_id}/file` : "")
                  }
                  alt=""
                />
              ) : (
                <span className="chip-ico">{c.kind === "table" ? "📊" : "📄"}</span>
              )}
              <span className="chip-name" data-tip={c.name}>
                {c.name}
              </span>
              {canConfirmSchema ? (
                <button
                  type="button"
                  className={statusClass}
                  aria-label={`确认 ${c.name} 的列角色`}
                  data-tip="点击确认列角色"
                  onClick={() => openSchema(doc!)}
                >
                  {label}
                </button>
              ) : (
                <span className={statusClass}>
                  {c.status === "uploading" && typeof c.progress === "number"
                    ? `${c.progress}%`
                    : label}
                </span>
              )}
              {c.kind === "image" && c.status === "done" && c.doc_id && (
                <button
                  type="button"
                  className="chip-analyze"
                  data-tip="开始三层技法分析"
                  aria-label={`分析画作 ${c.name}`}
                  onClick={() => void analyzeImage(c.doc_id!, c.name, sid)}
                >
                  分析画作
                </button>
              )}
              {c.status === "failed" && c.doc_id && (
                <button
                  type="button"
                  className="chip-retry"
                  data-tip="重新解析"
                  aria-label={`重新解析 ${c.name}`}
                  onClick={() => void retryDocument(c.doc_id!)}
                >
                  ↻
                </button>
              )}
              <button
                type="button"
                className="chip-x"
                data-tip="移除附件"
                aria-label={`移除附件 ${c.name}`}
                onClick={() =>
                  void removeChip(c.id, sid, (docId) =>
                    removeAttachmentTurns(docId),
                  )
                }
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
      <div
        className={"composer-inner" + (dragOver ? " drag-over" : "")}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onFiles(e.dataTransfer?.files ?? null);
        }}
      >
        <label
          id="btn-attach"
          className="btn-attach"
          htmlFor="file-input"
          role="button"
          tabIndex={0}
          aria-label="上传文档"
          data-tip="上传 PDF / CSV / Excel 文档或绘画图片"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              document.getElementById("file-input")?.click();
            }
          }}
        >
          <IconAttach />
        </label>
        <textarea
          id="msg"
          ref={taRef}
          rows={1}
          placeholder="输入您的问题或想法…"
          aria-label="消息输入框"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button
          id="btn-send"
          className={"btn-send" + (streaming ? " stopping" : "")}
          type="button"
          aria-label={streaming ? "停止生成" : "发送"}
          onClick={() => {
            if (streaming) {
              stopGeneration();
            } else {
              doSend();
            }
          }}
        >
          <IconSend />
          <IconStop />
        </button>
      </div>
      <div className="disclaimer">
        内容由 AI 生成，仅供学习参考，请结合权威资料深入研究。
      </div>
    </div>
  );
}
