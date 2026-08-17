import { useEffect, useRef } from "react";
import { useUiStore } from "../store/uiStore";
import { ModalShell } from "./ModalShell";

export function ConfirmModal() {
  const confirm = useUiStore((s) => s.confirm);
  const resolveConfirm = useUiStore((s) => s.resolveConfirm);
  const okRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (confirm) {
      const t = window.setTimeout(() => okRef.current?.focus(), 60);
      return () => window.clearTimeout(t);
    }
  }, [confirm]);

  return (
    <ModalShell
      open={!!confirm}
      outerClass="modal"
      labelledBy="confirm-title"
      onBackdrop={() => resolveConfirm(false)}
    >
      <div className="modal-backdrop" onClick={() => resolveConfirm(false)} />
      <div className="schema-card confirm-card">
        <div className="schema-title" id="confirm-title">
          {confirm?.title}
        </div>
        <div id="confirm-text" className="confirm-text">
          {confirm?.text}
        </div>
        <div className="schema-actions">
          <button
            id="confirm-cancel"
            className="btn-schema-cancel"
            type="button"
            onClick={() => resolveConfirm(false)}
          >
            取消
          </button>
          <button
            ref={okRef}
            id="confirm-ok"
            className={"btn-schema-ok" + (confirm?.danger ? " danger" : "")}
            type="button"
            onClick={() => resolveConfirm(true)}
          >
            {confirm?.okText || "确认"}
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
