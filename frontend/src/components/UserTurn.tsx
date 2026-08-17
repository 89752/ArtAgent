import { memo, useLayoutEffect, useRef, useState } from "react";
import type { ChipEntry } from "../api/types";
import { useChatStore, type UserTurn as UserTurnType } from "../store/chatStore";
import { IconEdit } from "./icons";

const FILE_STATUS_LABEL: Record<string, string> = {
  uploading: "上传中…",
  processing: "解析中…",
  pending: "解析中…",
  pending_confirm: "待确认 schema",
  done: "已就绪",
  active: "已就绪",
  failed: "失败",
};

function FileCard({ entry }: { entry: ChipEntry }) {
  return (
    <div className="file-card">
      {entry.kind === "image" ? (
        <img
          className="fc-thumb"
          src={
            entry.thumb_url ||
            (entry.doc_id ? `/api/user-images/${entry.doc_id}/file` : "")
          }
          alt=""
        />
      ) : (
        <span className="fc-ico">{entry.kind === "table" ? "📊" : "📄"}</span>
      )}
      <span className="fc-name" data-tip={entry.name}>
        {entry.name}
      </span>
      <span className="fc-badge">
        {FILE_STATUS_LABEL[entry.status] || entry.status || ""}
      </span>
    </div>
  );
}

export const UserTurn = memo(function UserTurn({
  turn,
  editing,
}: {
  turn: UserTurnType;
  editing: boolean;
}) {
  const editLast = useChatStore((s) => s.editLast);
  const cancelEdit = useChatStore((s) => s.cancelEdit);
  const commitEdit = useChatStore((s) => s.commitEdit);
  const [draft, setDraft] = useState(turn.text);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    if (editing) {
      setDraft(turn.text);
      const ta = taRef.current;
      if (ta) {
        ta.style.height = "auto";
        ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
        ta.focus();
        const end = ta.value.length;
        ta.setSelectionRange(end, end);
      }
    }
  }, [editing, turn.text]);

  const onEditInput = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
  };

  return (
    <div className="turn user">
      <div className="avatar">我</div>
      <div className="bubble">
        {editing ? (
          <>
            <textarea
              ref={taRef}
              className="edit-input"
              value={draft}
              maxLength={8000}
              aria-label="编辑消息内容"
              onChange={(e) => setDraft(e.target.value)}
              onInput={onEditInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void commitEdit(draft);
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  cancelEdit();
                }
              }}
            />
            <div className="edit-row">
              <button type="button" className="edit-cancel" onClick={cancelEdit}>
                取消
              </button>
              <button
                type="button"
                className="edit-save"
                onClick={() => void commitEdit(draft)}
              >
                保存
              </button>
            </div>
          </>
        ) : (
          <>
            {turn.text && <p>{turn.text}</p>}
            {turn.files.map((f) => (
              <FileCard key={f.id} entry={f} />
            ))}
          </>
        )}
      </div>
      {!editing && (
        <div className="msg-actions">
          <button
            type="button"
            className="msg-act"
            data-tip="编辑这条消息"
            aria-label="编辑这条消息"
            onClick={() => editLast(turn.id)}
          >
            <IconEdit />
          </button>
        </div>
      )}
    </div>
  );
});
