import { useEffect, useState } from "react";
import { sendJson } from "../api/client";
import { toast } from "../lib/dialogs";
import { useDocStore } from "../store/docStore";
import { ModalShell } from "./ModalShell";

export function SchemaModal() {
  const schemaDoc = useDocStore((s) => s.schemaDoc);
  const closeSchema = useDocStore((s) => s.closeSchema);
  const loadDocuments = useDocStore((s) => s.loadDocuments);
  const [entity, setEntity] = useState("");
  const [axis, setAxis] = useState("");
  const [desc, setDesc] = useState("");
  const [image, setImage] = useState("");
  const [display, setDisplay] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!schemaDoc) return;
    const cols = schemaDoc.columns || [];
    const p = schemaDoc.proposed_schema || {};
    setEntity(p.entity_col || (cols[0] ?? ""));
    setAxis(p.group_axis_col || "");
    setDesc(p.description_col || "");
    setImage(p.image_col || "");
    setDisplay(p.display_name || (schemaDoc.doc_name || "").replace(/\.[^.]+$/, ""));
    setReasoning(p.reasoning || "");
    setError("");
  }, [schemaDoc]);

  const submit = async () => {
    if (!schemaDoc || submitting) return;
    setSubmitting(true);
    try {
      const j = await sendJson<{ ok: boolean; error?: string }>(
        `/api/documents/${encodeURIComponent(schemaDoc.doc_id)}/schema`,
        "POST",
        {
          entity_col: entity,
          group_axis_col: axis || null,
          description_col: desc || null,
          image_col: image || null,
          display_name: display,
        },
      );
      if (!j.ok) {
        setError(j.error || "确认失败");
        return;
      }
      closeSchema();
      toast("数据源已启用");
      void loadDocuments();
    } catch (e) {
      setError("确认失败：" + (e instanceof Error ? e.message : e));
    } finally {
      setSubmitting(false);
    }
  };

  const cols = schemaDoc?.columns || [];
  const select = (
    label: string,
    value: string,
    onChange: (v: string) => void,
    allowEmpty: boolean,
  ) => (
    <label>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {allowEmpty && <option value="">（无）</option>}
        {cols.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <ModalShell
      open={!!schemaDoc}
      outerClass="modal"
      labelledBy="schema-title"
      onBackdrop={closeSchema}
    >
      <div className="modal-backdrop" onClick={closeSchema} />
      <div className="schema-card">
        <div className="schema-title" id="schema-title">
          确认表格列角色
        </div>
        <div className="schema-meta">
          {schemaDoc?.doc_name}
          {schemaDoc?.sheet_name ? ` · 子表「${schemaDoc.sheet_name}」 · ` : " · "}
          {schemaDoc ? `${schemaDoc.rows || 0} 行 × ${schemaDoc.cols || cols.length} 列` : ""}
        </div>
        <div className="schema-grid">
          {select("实体名列（必填）", entity, setEntity, false)}
          {select("时间/分类轴列", axis, setAxis, true)}
          {select("描述文本列", desc, setDesc, true)}
          {select("图片列", image, setImage, true)}
        </div>
        <label className="schema-name-label">
          显示名
          <input
            id="schema-display"
            type="text"
            maxLength={20}
            value={display}
            onChange={(e) => setDisplay(e.target.value)}
          />
        </label>
        {reasoning && <div className="schema-reason">推断依据：{reasoning}</div>}
        <div className="schema-error" hidden={!error}>
          {error}
        </div>
        <div className="schema-actions">
          <button
            id="schema-cancel"
            className="btn-schema-cancel"
            type="button"
            onClick={closeSchema}
          >
            取消
          </button>
          <button
            id="schema-ok"
            className="btn-schema-ok"
            type="button"
            disabled={submitting}
            onClick={() => void submit()}
          >
            确认并启用
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
