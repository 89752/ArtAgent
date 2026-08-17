import { useEffect, useState } from "react";
import { toast } from "../lib/dialogs";
import { useChatStore } from "../store/chatStore";
import { IconSearch } from "./icons";

export function HistoryList() {
  const sessions = useChatStore((s) => s.sessions);
  const sid = useChatStore((s) => s.sid);
  const hasMore = useChatStore((s) => s.sessionsHasMore);
  const searchLimited = useChatStore((s) => s.searchLimited);
  const searchSessions = useChatStore((s) => s.searchSessions);
  const loadMore = useChatStore((s) => s.loadMoreSessions);
  const openSession = useChatStore((s) => s.openSession);
  const deleteSession = useChatStore((s) => s.deleteSession);
  const renameSession = useChatStore((s) => s.renameSession);

  const [q, setQ] = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const searching = q.trim().length > 0;

  useEffect(() => {
    const t = window.setTimeout(() => void searchSessions(q), 300);
    return () => window.clearTimeout(t);
  }, [q, searchSessions]);

  const commitRename = async (save: boolean, item: { session_id: string; title: string }) => {
    if (renaming !== item.session_id) return;
    setRenaming(null);
    const val = draft.trim();
    if (save && val && val !== item.title) {
      const ok = await renameSession(item.session_id, val);
      if (ok) toast("已重命名");
    }
  };

  return (
    <>
      <div className="side-search">
        <IconSearch />
        <input
          id="hist-filter"
          type="search"
          placeholder="搜索历史对话"
          aria-label="搜索历史对话"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>
      <nav id="history" className="history" aria-label="历史对话">
        {!sessions.length && (
          <div className="hist-empty">
            {searching
              ? searchLimited
                ? "无匹配的对话（已搜索最近 500 条，更早的会话可能未覆盖）"
                : "无匹配的对话"
              : "暂无历史对话"}
          </div>
        )}
        {sessions.length > 0 &&
          sessions.map((s) => (
            <div
              key={s.session_id}
              className={"hist-btn" + (s.session_id === sid ? " active" : "")}
              data-sid={s.session_id}
              role="button"
              tabIndex={0}
              onClick={() => void openSession(s.session_id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  void openSession(s.session_id);
                }
              }}
            >
              <span className="h-body">
                {renaming === s.session_id ? (
                  <input
                    className="hist-rename-input"
                    type="text"
                    maxLength={60}
                    aria-label="重命名对话"
                    value={draft}
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void commitRename(true, s);
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        void commitRename(false, s);
                      }
                    }}
                    onBlur={() => void commitRename(true, s)}
                  />
                ) : (
                  <>
                    <span className="h-title">{s.title || "未命名对话"}</span>
                    <span className="h-time">{s.relative || ""}</span>
                  </>
                )}
              </span>
              <button
                type="button"
                className="hist-ren"
                aria-label="重命名对话"
                data-tip="重命名"
                onClick={(ev) => {
                  ev.stopPropagation();
                  setRenaming(s.session_id);
                  setDraft(s.title || "");
                }}
              >
                ✎
              </button>
              <button
                type="button"
                className="hist-del"
                aria-label="删除对话"
                data-tip="删除该对话"
                onClick={(ev) => {
                  ev.stopPropagation();
                  void deleteSession(s.session_id, s.title);
                }}
              >
                ×
              </button>
            </div>
          ))}
        {sessions.length > 0 && hasMore && (
          <button type="button" className="hist-more" onClick={() => void loadMore()}>
            加载更多
          </button>
        )}
        {searchLimited && sessions.length > 0 && (
          <div className="hist-hint">
            已搜索最近 500 条；更早的会话不在搜索范围内。
          </div>
        )}
      </nav>
    </>
  );
}
