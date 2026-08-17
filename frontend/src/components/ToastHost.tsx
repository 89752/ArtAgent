import { useEffect, useRef } from "react";
import { useUiStore } from "../store/uiStore";

export function ToastHost() {
  const toastState = useUiStore((s) => s.toast);
  const clearToast = useUiStore((s) => s.clearToast);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!toastState) return;
    if (timer.current != null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      clearToast();
    }, 3400);
    return () => {
      if (timer.current != null) window.clearTimeout(timer.current);
    };
  }, [toastState, clearToast]);

  if (!toastState) return null;
  return (
    <div
      id="toast"
      className={"toast show" + (toastState.type ? " " + toastState.type : "")}
      role="status"
      aria-live="polite"
    >
      {toastState.msg}
    </div>
  );
}
