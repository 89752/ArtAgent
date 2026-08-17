import { useEffect, useState, type FormEvent } from "react";
import { changePassword, logout } from "../api/auth";
import { getUser } from "../lib/user";
import { useUiStore } from "../store/uiStore";
import { IconInfo, IconLibrary, IconMemory, IconUser } from "./icons";
import { LibraryDrawer } from "./LibraryDrawer";
import { MemoryPanel } from "./MemoryPanel";
import { ModalShell } from "./ModalShell";

type Section = "account" | "memory" | "library" | "about";

function ChangePasswordForm() {
  const [oldPwd, setOldPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [confirmPwd, setConfirmPwd] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (newPwd !== confirmPwd) {
      setMsg({ ok: false, text: "两次输入的新密码不一致" });
      return;
    }
    setBusy(true);
    const r = await changePassword(oldPwd, newPwd);
    setBusy(false);
    if (r.ok) {
      setMsg({ ok: true, text: "密码已修改，其他登录会话已失效" });
      setOldPwd("");
      setNewPwd("");
      setConfirmPwd("");
    } else {
      setMsg({ ok: false, text: r.error || "修改密码失败" });
    }
  };

  return (
    <form className="change-pwd-form" onSubmit={submit}>
      <div className="change-pwd-title">修改密码</div>
      <input
        className="account-input"
        type="password"
        value={oldPwd}
        placeholder="当前密码"
        autoComplete="current-password"
        onChange={(e) => {
          setOldPwd(e.target.value);
          setMsg(null);
        }}
      />
      <input
        className="account-input"
        type="password"
        value={newPwd}
        placeholder="新密码（至少 8 位）"
        autoComplete="new-password"
        onChange={(e) => {
          setNewPwd(e.target.value);
          setMsg(null);
        }}
      />
      <input
        className="account-input"
        type="password"
        value={confirmPwd}
        placeholder="确认新密码"
        autoComplete="new-password"
        onChange={(e) => {
          setConfirmPwd(e.target.value);
          setMsg(null);
        }}
      />
      <button className="login-btn" type="submit" disabled={busy}>
        {busy ? "提交中…" : "修改密码"}
      </button>
      {msg && (
        <p className={"login-error" + (msg.ok ? " account-ok" : "")}>{msg.text}</p>
      )}
    </form>
  );
}

function AccountContent() {
  const user = getUser();
  return (
    <div className="settings-page">
      <div className="settings-page-head">
        <div className="df-dialog-header">
          <h3>账户</h3>
          <p className="df-dialog-desc">
            正式多用户模式：记忆、资料库、会话与文档按用户隔离。
          </p>
        </div>
      </div>
      <div className="settings-page-body">
        <div className="account-info">
          <div className="account-line">
            用户名：{user?.username || user?.user_id || "user"}
          </div>
          <div className="account-line">用户 ID：{user?.user_id || "user"}</div>
        </div>
        <ChangePasswordForm />
      </div>
      <div className="settings-page-foot">
        <div className="df-dialog-footer">
          <button
            type="button"
            className="mem-clear danger"
            onClick={async () => {
              await logout();
              window.location.reload();
            }}
          >
            退出登录
          </button>
        </div>
      </div>
    </div>
  );
}

function AboutContent() {
  return (
    <div className="settings-page">
      <div className="settings-page-head">
        <div className="df-dialog-header">
          <h3>关于</h3>
          <p className="df-dialog-desc">ArtAgent · 西方艺术智能助手</p>
        </div>
      </div>
      <div className="settings-page-body">
        <p className="about-text">
          基于 LangGraph 的西方艺术史问答助手：本地 5.5 万+ 条作品库、混合检索、
          技能化对比/时间线/推荐、子智能体并行调研与跨会话记忆。
          版本 0.1.0。
        </p>
      </div>
    </div>
  );
}

export function SettingsDialog() {
  const modal = useUiStore((s) => s.modal);
  const closeModal = useUiStore((s) => s.closeModal);
  const [section, setSection] = useState<Section>("account");

  useEffect(() => {
    if (modal === "settings") setSection("account");
  }, [modal]);

  const navBtn = (key: Section, label: string, icon: React.ReactNode) => (
    <button
      type="button"
      className={"settings-nav-btn" + (section === key ? " active" : "")}
      onClick={() => setSection(key)}
    >
      {icon}
      <span>{label}</span>
    </button>
  );

  return (
    <ModalShell
      open={modal === "settings"}
      outerClass="modal settings-modal"
      labelledBy="settings-title"
      onBackdrop={() => closeModal("settings")}
    >
      <div className="settings-card df-dialog">
        <button
          className="drawer-close df-close"
          type="button"
          aria-label="关闭设置"
          onClick={() => closeModal("settings")}
        >
          ×
        </button>
        <div className="df-dialog-header">
          <h3 id="settings-title">设置</h3>
          <p className="df-dialog-desc">管理记忆、资料库与应用信息。</p>
        </div>
        <div className="settings-layout">
          <nav className="settings-nav" aria-label="设置分类">
            {navBtn("account", "账户", <IconUser width={16} height={16} />)}
            {navBtn("memory", "记忆", <IconMemory width={16} height={16} />)}
            {navBtn("library", "资料库", <IconLibrary width={16} height={16} />)}
            {navBtn("about", "关于", <IconInfo width={16} height={16} />)}
          </nav>
          <div className="settings-content">
            {section === "account" && <AccountContent />}
            {section === "memory" && <MemoryPanel embedded />}
            {section === "library" && <LibraryDrawer embedded />}
            {section === "about" && <AboutContent />}
          </div>
        </div>
      </div>
    </ModalShell>
  );
}
