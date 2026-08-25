import type { ReactNode } from "react";
import type { Doc } from "../api/types";
import { useChatStore } from "../store/chatStore";
import { useDocStore } from "../store/docStore";
import { useUiStore } from "../store/uiStore";
import { ModalShell } from "./ModalShell";

function badgeForDoc(d: Doc): ReactNode {
  const tip = (t: string) => ({ "data-tip": t });
  if (d.kind === "table") {
    if (d.status === "pending_confirm") {
      return <span className="doc-badge pending_confirm">待确认 schema</span>;
    }
    if (d.status === "active") {
      const caps = [
        d.supports_timeline ? "时间线" : null,
      ]
        .filter(Boolean)
        .join("/");
      return (
        <span
          className="doc-badge active"
          {...tip(`已启用 · 支持：${caps || "仅检索"}`)}
        >
          {d.rows || 0} 行
        </span>
      );
    }
    if (d.status === "failed") {
      return (
        <span className="doc-badge failed" {...tip(d.error || "")}>
          失败
        </span>
      );
    }
    return <span className="doc-badge">解析中…</span>;
  }
  if (d.status === "done") {
    const chunks = d.text_chunks || 0;
    const imgs = d.image_pages || 0;
    const label =
      chunks && imgs
        ? `${chunks} 片段 · ${imgs} 图`
        : imgs
          ? `${imgs} 整页图`
          : `${chunks} 片段`;
    return (
      <span
        className="doc-badge done"
        {...tip(`${d.pages || 0} 页 · 路由 ${JSON.stringify(d.route_distribution || {})}`)}
      >
        {label}
      </span>
    );
  }
  if (d.status === "failed") {
    return (
      <span className="doc-badge failed" {...tip(d.error || "")}>
        失败
      </span>
    );
  }
  return <span className="doc-badge">解析中…</span>;
}

export function LibraryDrawer({ embedded = false }: { embedded?: boolean }) {
  const modal = useUiStore((s) => s.modal);
  const closeModal = useUiStore((s) => s.closeModal);
  const docs = useDocStore((s) => s.docs);
  const retryDocument = useDocStore((s) => s.retryDocument);
  const deleteDocument = useDocStore((s) => s.deleteDocument);
  const openSchema = useDocStore((s) => s.openSchema);
  const removeAttachmentTurns = useChatStore((s) => s.removeAttachmentTurns);

  const onDocDeleted = (docId: string) => removeAttachmentTurns(docId);

  const header = (
    <div className="df-dialog-header">
      <h2 id={embedded ? "library-settings-title" : "library-title"}>资料库</h2>
      <p className="df-dialog-desc">
        这里只记录你上传过的文件；上传请使用输入框左侧的附件按钮。
      </p>
    </div>
  );

  const body = (
    <>
      <section className="lib-block">
        <h3>
          我的文档 <span id="doc-count" className="lib-count">{docs.length ? `(${docs.length})` : ""}</span>
        </h3>
        <nav id="doc-list" className="doc-list" aria-label="文档列表">
          {!docs.length && <div className="doc-empty">暂无上传文档</div>}
          {docs.map((d) => (
            <div key={d.doc_id} className="doc-item">
              <span className="doc-ico">{d.kind === "table" ? "📊" : "📄"}</span>
              <div className="doc-name" data-tip={d.doc_name || ""}>
                {d.doc_name || "未命名"}
              </div>
              {badgeForDoc(d)}
              <div className="doc-actions">
                {d.kind === "table" && d.status === "pending_confirm" && (
                  <button
                    type="button"
                    className="doc-confirm"
                    onClick={() => openSchema(d)}
                  >
                    确认
                  </button>
                )}
                {d.status === "failed" && (
                  <button
                    type="button"
                    className="doc-retry"
                    data-tip={d.error || "重新解析"}
                    onClick={() => void retryDocument(d.doc_id)}
                  >
                    重试
                  </button>
                )}
                {d.status !== "processing" && d.status !== "pending" && (
                  <button
                    type="button"
                    className="doc-delete"
                    aria-label={`删除文档 ${d.doc_name || ""}`}
                    data-tip="删除文档及关联向量"
                    onClick={() => void deleteDocument(d, onDocDeleted)}
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          ))}
        </nav>
      </section>
    </>
  );

  if (embedded) {
    return (
      <div className="settings-page">
        <div className="settings-page-head">{header}</div>
        <div className="settings-page-body">{body}</div>
      </div>
    );
  }

  return (
    <ModalShell
      open={modal === "library"}
      outerClass="drawer"
      labelledBy="library-title"
      onBackdrop={() => closeModal("library")}
    >
      <div className="drawer-backdrop" onClick={() => closeModal("library")} />
      <aside className="drawer-panel df-dialog">
        <button
          className="drawer-close df-close"
          type="button"
          aria-label="关闭资料库"
          onClick={() => closeModal("library")}
        >
          ×
        </button>
        {header}
        <div className="df-dialog-body">{body}</div>
        <div className="df-dialog-footer">
          <button
            type="button"
            className="df-btn"
            onClick={() => closeModal("library")}
          >
            关闭
          </button>
        </div>
      </aside>
    </ModalShell>
  );
}
