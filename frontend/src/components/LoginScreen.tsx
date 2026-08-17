import { useState, type FormEvent } from "react";
import { login, register } from "../api/auth";

type Mode = "login" | "register";

export function LoginScreen({ onLogin }: { onLogin: () => void }) {
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (mode === "register" && password !== confirm) {
      setError("两次输入的密码不一致");
      return;
    }
    setBusy(true);
    const r =
      mode === "register"
        ? await register(username.trim(), password, name.trim())
        : await login(username.trim() || "user", password);
    setBusy(false);
    if (r.ok) {
      onLogin();
    } else {
      setError(r.error || "登录失败");
    }
  };

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={submit}>
        <img src="/static/emblem.svg" alt="" className="login-logo" />
        <h1 className="login-title">ArtAgent</h1>
        <p className="login-desc">西方艺术智能助手</p>
        <div className="login-tabs" role="tablist" aria-label="登录或注册">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "login"}
            className={"login-tab" + (mode === "login" ? " active" : "")}
            onClick={() => {
              setMode("login");
              setError("");
            }}
          >
            登录
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "register"}
            className={"login-tab" + (mode === "register" ? " active" : "")}
            onClick={() => {
              setMode("register");
              setError("");
            }}
          >
            注册
          </button>
        </div>
        {mode === "register" && (
          <input
            className="account-input"
            value={name}
            placeholder="昵称（可选）"
            autoComplete="name"
            onChange={(e) => {
              setName(e.target.value);
              setError("");
            }}
          />
        )}
        <input
          className="account-input"
          value={username}
          autoFocus
          placeholder="用户名"
          autoComplete="username"
          onChange={(e) => {
            setUsername(e.target.value);
            setError("");
          }}
        />
        <input
          className="account-input"
          type="password"
          value={password}
          placeholder="密码"
          autoComplete={mode === "register" ? "new-password" : "current-password"}
          onChange={(e) => {
            setPassword(e.target.value);
            setError("");
          }}
        />
        {mode === "register" && (
          <input
            className="account-input"
            type="password"
            value={confirm}
            placeholder="确认密码"
            autoComplete="new-password"
            onChange={(e) => {
              setConfirm(e.target.value);
              setError("");
            }}
          />
        )}
        <button className="login-btn" type="submit" disabled={busy}>
          {busy ? "处理中…" : mode === "register" ? "注册并登录" : "登录"}
        </button>
        {error && <p className="login-error">{error}</p>}
        <p className="login-hint">
          {mode === "register"
            ? "用户名 3-40 位（字母/数字/._-），密码至少 8 位"
            : "默认账号：user · 密码：11111111"}
        </p>
      </form>
    </div>
  );
}
