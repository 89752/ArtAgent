import { useEffect, useState, type ChangeEvent } from "react";
import { fetchMe } from "./api/auth";
import { Chat } from "./components/Chat";
import { Composer } from "./components/Composer";
import { ConfirmModal } from "./components/ConfirmModal";
import { LibraryDrawer } from "./components/LibraryDrawer";
import { LoginScreen } from "./components/LoginScreen";
import { MemoryPanel } from "./components/MemoryPanel";
import { OversizeModal } from "./components/OversizeModal";
import { SchemaModal } from "./components/SchemaModal";
import { SettingsDialog } from "./components/SettingsDialog";
import { Sidebar } from "./components/Sidebar";
import { ToastHost } from "./components/ToastHost";
import { TooltipHost } from "./components/TooltipHost";
import { Topbar } from "./components/Topbar";
import { Welcome } from "./components/Welcome";
import { currentSidFromUrl } from "./lib/utils";
import { clearToken, getToken } from "./lib/user";
import { useChatStore } from "./store/chatStore";
import { useDocStore } from "./store/docStore";
import { useUiStore } from "./store/uiStore";

export default function App() {
  const [authed, setAuthed] = useState(() => Boolean(getToken()));
  const view = useChatStore((s) => s.view);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const collapsed = useUiStore((s) => s.collapsed);

  useEffect(() => {
    if (!authed) return;
    useUiStore.getState().initTheme();
    void useChatStore.getState().loadBootstrap();
    void useChatStore.getState().loadSessions();
    void useDocStore.getState().loadDocuments();
    const urlSid = currentSidFromUrl();
    if (urlSid) void useChatStore.getState().openSession(urlSid);
  }, [authed]);

  useEffect(() => {
    if (!getToken()) return;
    void fetchMe().then((u) => {
      if (!u) {
        clearToken();
        setAuthed(false);
      }
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const chat = useChatStore.getState();
      if (chat.feedbackTurnId) {
        chat.setFeedbackTurn(null);
        return;
      }
      const ui = useUiStore.getState();
      if (ui.confirm) {
        ui.resolveConfirm(false);
        return;
      }
      const doc = useDocStore.getState();
      if (doc.schemaDoc) {
        doc.closeSchema();
        return;
      }
      if (ui.oversize) {
        ui.resolveOversize("");
        return;
      }
      if (ui.modal === "library") {
        ui.closeModal("library");
        return;
      }
      if (ui.modal === "memory") {
        ui.closeModal("memory");
        return;
      }
      if (ui.modal === "settings") {
        ui.closeModal("settings");
        return;
      }
      if (ui.sidebarOpen) ui.closeSidebar();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  if (!authed) {
    return <LoginScreen onLogin={() => setAuthed(true)} />;
  }

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    const { sid, uploadMaxBytes } = useChatStore.getState();
    if (files?.length) {
      void useDocStore.getState().uploadFiles(files, sid, uploadMaxBytes);
    }
    e.target.value = "";
  };

  return (
    <div
      id="app"
      className={
        (collapsed ? "side-collapsed " : "") + (sidebarOpen ? "side-open" : "")
      }
    >
      <div className="grain" aria-hidden="true" />
      <Sidebar />
      <main id="main">
        <Topbar />
        {view === "welcome" ? <Welcome /> : <Chat />}
        <Composer />
      </main>
      <input
        id="file-input"
        type="file"
        accept=".pdf,.csv,.xlsx,.xls,.jpg,.jpeg,.png,.webp"
        hidden
        multiple
        onChange={onFileChange}
      />
      <MemoryPanel />
      <LibraryDrawer />
      <SchemaModal />
      <SettingsDialog />
      <ConfirmModal />
      <OversizeModal />
      <ToastHost />
      <TooltipHost />
    </div>
  );
}
