import { useChatStore } from "../store/chatStore";
import { useUiStore } from "../store/uiStore";
import { IconThemeMoon } from "./icons";

export function Topbar() {
  const title = useChatStore((s) => s.title);
  const theme = useUiStore((s) => s.theme);
  const toggleTheme = useUiStore((s) => s.toggleTheme);

  return (
    <header id="topbar" className="topbar">
      <div className="tb-title" id="tb-title" data-tip={title}>
        {title}
      </div>
      <div className="tb-actions">
        <button
          id="btn-theme"
          className="icon-btn"
          type="button"
          aria-label="切换深色模式"
          data-tip="切换深色模式"
          onClick={toggleTheme}
        >
          <span className="theme-ico-moon" aria-hidden="true">
            <IconThemeMoon />
          </span>
          <span className="theme-ico-sun" aria-hidden="true">
            {theme === "dark" ? "☀" : "☾"}
          </span>
        </button>
      </div>
    </header>
  );
}
