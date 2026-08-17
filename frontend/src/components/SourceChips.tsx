import type { Source } from "../api/types";
import { toast } from "../lib/dialogs";

export function SourceChips({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <div className="sources">
      <div className="sources-head">参考来源</div>
      {sources.map((s, i) => (
        <button
          key={i}
          type="button"
          className="source-chip"
          data-i={String(i + 1)}
          data-tip={s.label}
          onClick={() => {
            void navigator.clipboard
              .writeText(s.label || "")
              .then(() => toast("已复制来源"))
              .catch(() => {
                /* 剪贴板不可用时忽略 */
              });
          }}
        >
          {i + 1} {s.label}
        </button>
      ))}
    </div>
  );
}
