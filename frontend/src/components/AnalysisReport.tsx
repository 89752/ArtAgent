import type {
  ArtworkAnalysisReport,
  DimensionAnalysis,
} from "../api/types";

const FRAMEWORK_LABELS: Record<string, string> = {
  realistic: "写实/具象",
  abstract: "抽象/表现",
  childlike: "儿童画/涂鸦",
  decorative: "装饰/图案",
};

function Evidence({ items }: { items?: string[] }) {
  if (!items?.length) return null;
  return (
    <ul className="ar-evidence">
      {items.map((e, i) => (
        <li key={i}>{e}</li>
      ))}
    </ul>
  );
}

function DimensionCard({
  title,
  item,
}: {
  title: string;
  item?: DimensionAnalysis;
}) {
  if (!item) {
    return (
      <div className="ar-dim">
        <b>{title}</b>
        <p className="ar-empty">未分析</p>
      </div>
    );
  }
  const notes: string[] = [];
  if (item.applies === false) notes.push("不适用");
  const kind = typeof item.kind === "string" ? item.kind : "";
  if (kind && kind !== "not_applicable") notes.push(kind);
  return (
    <div className="ar-dim">
      <div className="ar-dim-head">
        <b>{title}</b>
        {notes.length > 0 && <span className="ar-dim-notes">{notes.join(" · ")}</span>}
        {typeof item.confidence === "number" && (
          <span className="ar-conf">置信度 {Math.round(item.confidence * 100)}%</span>
        )}
      </div>
      {item.assessment ? <p className="ar-dim-text">{item.assessment}</p> : null}
      <Evidence items={item.evidence} />
    </div>
  );
}

export function AnalysisReport({ report }: { report: ArtworkAnalysisReport }) {
  const l1 = report.layer1_technique;
  const l2 = report.layer2_style_mood;
  const suggestions = report.layer3_suggestions?.priority_items;
  const framework = report.framework || "";
  return (
    <div className="analysis-report">
      <div className="ar-head">
        <span className="ar-framework">
          {FRAMEWORK_LABELS[framework] || framework || "框架未判定"}
        </span>
        {report.overall_assessment && (
          <p className="ar-overall">{report.overall_assessment}</p>
        )}
      </div>

      <details className="ar-layer" open>
        <summary>第一层 · 客观技法</summary>
        <div className="ar-grid">
          <DimensionCard title="透视" item={l1?.perspective} />
          <DimensionCard title="构图" item={l1?.composition} />
          <DimensionCard title="色彩" item={l1?.color} />
          <DimensionCard title="线条与笔触" item={l1?.line_brushwork} />
        </div>
      </details>

      <details className="ar-layer">
        <summary>第二层 · 风格与情绪基调</summary>
        {l2 ? (
          <div className="ar-mood">
            {l2.mood && <p className="ar-mood-text">{l2.mood}</p>}
            {l2.style_affinity?.length ? (
              <p className="ar-affinity">风格倾向：{l2.style_affinity.join("、")}</p>
            ) : null}
            {l2.caveat ? <p className="ar-caveat">{l2.caveat}</p> : null}
          </div>
        ) : (
          <p className="ar-empty">暂无内容</p>
        )}
      </details>

      <details className="ar-layer">
        <summary>第三层 · 专业指导建议</summary>
        {suggestions?.length ? (
          <ol className="ar-suggestions">
            {suggestions.map((s, i) => (
              <li key={i} className="ar-sug">
                <div className="ar-sug-issue">{s.issue}</div>
                {s.location_hint && (
                  <div className="ar-sug-loc">位置：{s.location_hint}</div>
                )}
                <div className="ar-sug-principle">依据：{s.principle}</div>
                <div className="ar-sug-action">{s.action}</div>
                {s.difficulty && (
                  <span className="ar-sug-diff">{s.difficulty}</span>
                )}
              </li>
            ))}
          </ol>
        ) : (
          <p className="ar-empty">暂无建议</p>
        )}
      </details>

      {report.disclaimer && (
        <div className="ar-disclaimer">{report.disclaimer}</div>
      )}
    </div>
  );
}
