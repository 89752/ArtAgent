import { useChatStore } from "../store/chatStore";
import { useUiStore } from "../store/uiStore";
import { HistoryList } from "./HistoryList";
import {
  IconChevronLeft,
  IconChevronRight,
  IconMessageSquarePlus,
  IconInfo,
  IconSettings,
} from "./icons";

export function Sidebar() {
  const collapsed = useUiStore((s) => s.collapsed);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const toggleCollapsed = useUiStore((s) => s.toggleCollapsed);
  const closeSidebar = useUiStore((s) => s.closeSidebar);
  const openModal = useUiStore((s) => s.openModal);
  const newSession = useChatStore((s) => s.newSession);

  return (
    <>
      <div
        id="side-overlay"
        className="side-overlay"
        hidden={!sidebarOpen}
        onClick={closeSidebar}
      />
      <aside id="sidebar" aria-label="主导航">
        <div className="side-top">
          {collapsed ? (
            <button
              type="button"
              className="side-logo-collapsed"
              aria-label="展开侧边栏"
              data-tip="展开侧边栏"
              onClick={toggleCollapsed}
            >
              <IconChevronRight width={18} height={18} />
            </button>
          ) : (
            <>
              <div className="brand">
                <img className="brand-mark" src="/static/emblem.svg" alt="" />
                <div className="brand-name">ArtAgent</div>
              </div>
              <button
                id="btn-side-collapse"
                className="btn-side-collapse"
                type="button"
                aria-label="收起侧边栏"
                data-tip="收起侧边栏"
                onClick={toggleCollapsed}
              >
                <IconChevronLeft width={18} height={18} />
              </button>
            </>
          )}
        </div>

        <button id="btn-new" className="side-new" type="button" onClick={newSession}>
          <IconMessageSquarePlus width={16} height={16} />
          {!collapsed && <span>新建对话</span>}
        </button>

        {!collapsed && (
          <>
            <div className="side-group-label">历史对话</div>
            <HistoryList />
          </>
        )}

        <div className="side-footer">
          <button
            type="button"
            className="side-foot-btn"
            onClick={() => openModal("operations")}
          >
            <IconInfo width={16} height={16} />
            {!collapsed && <span>运行中心</span>}
          </button>
          <button
            type="button"
            className="side-foot-btn"
            onClick={() => openModal("settings")}
          >
            <IconSettings width={16} height={16} />
            {!collapsed && <span>设置与更多</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
