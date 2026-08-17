import { useEffect, useRef } from "react";
import { useUiStore } from "../store/uiStore";
import { ModalShell } from "./ModalShell";

export function OversizeModal() {
  const oversize = useUiStore((s) => s.oversize);
  const resolveOversize = useUiStore((s) => s.resolveOversize);
  const splitRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (oversize) {
      const t = window.setTimeout(() => splitRef.current?.focus(), 60);
      return () => window.clearTimeout(t);
    }
  }, [oversize]);

  const title = oversize
    ? oversize.count > 1
      ? `${oversize.count} 个文件超过 ${oversize.maxMb}MB`
      : `《${oversize.name}》超过 ${oversize.maxMb}MB`
    : "";

  return (
    <ModalShell
      open={!!oversize}
      outerClass="modal"
      labelledBy="oversize-title"
      onBackdrop={() => resolveOversize("")}
    >
      <div className="modal-backdrop" onClick={() => resolveOversize("")} />
      <div className="schema-card confirm-card">
        <div className="schema-title" id="oversize-title">
          {title}
        </div>
        <div id="oversize-text" className="confirm-text">
          拆分上传会保留高质量解析（MinerU），并按页拆成多份文档入库；
          直接上传则改用 pdfplumber 本地解析（不调 MinerU，版面/表格质量较低）。
        </div>
        <div className="schema-actions">
          <button
            ref={splitRef}
            id="oversize-split"
            className="btn-schema-ok"
            type="button"
            onClick={() => resolveOversize("split")}
          >
            拆分上传（推荐，保质量）
          </button>
          <button
            id="oversize-pdfplumber"
            className="btn-schema-cancel"
            type="button"
            onClick={() => resolveOversize("pdfplumber")}
          >
            用 pdfplumber 直接传
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
