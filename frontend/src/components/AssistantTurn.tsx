import { memo, useLayoutEffect, useRef, useState } from "react";
import { sendJson } from "../api/client";
import { attachCitations } from "../lib/citations";
import { toast } from "../lib/dialogs";
import { useChatStore, type AssistantTurn as AssistantTurnType } from "../store/chatStore";
import { IconCopy, IconRegen, IconThumb } from "./icons";
import { SafeHtml } from "./SafeHtml";
import { SourceChips } from "./SourceChips";
import { AnalysisReport } from "./AnalysisReport";

const FB_REASONS = ["回答不准确", "引用不充分", "过于冗长", "其他"];

function FeedbackPanel({ sid }: { sid: string }) {
  const setFeedbackTurn = useChatStore((s) => s.setFeedbackTurn);
  const markRated = useChatStore((s) => s.markRated);
  const [reason, setReason] = useState("");
  const [comment, setComment] = useState("");
  const commentRef = useRef<HTMLInputElement>(null);

  useLayoutEffect(() => {
    const t = window.setTimeout(() => commentRef.current?.focus(), 50);
    return () => window.clearTimeout(t);
  }, []);

  const submit = async () => {
    if (!reason) {
      toast("请先选择一个原因", "err");
      return;
    }
    try {
      await sendJson<{ ok: boolean }>("/api/feedback", "POST", {
        session_id: sid,
        rating: -1,
        reason,
        comment: comment.trim(),
      });
      markRated(sid);
      setFeedbackTurn(null);
      toast("已记录，我们会改进");
    } catch (e) {
      toast("反馈提交失败：" + (e instanceof Error ? e.message : e), "err");
    }
  };

  return (
    <div className="fb-panel">
      <div className="fb-head">哪里有问题？</div>
      <div className="fb-options">
        {FB_REASONS.map((r) => (
          <button
            key={r}
            type="button"
            className={"fb-opt" + (reason === r ? " on" : "")}
            onClick={() => setReason(r)}
          >
            {r}
          </button>
        ))}
      </div>
      <input
        ref={commentRef}
        className="fb-comment"
        type="text"
        maxLength={500}
        placeholder="补充说明（可选）"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
      <div className="fb-submit-row">
        <button
          type="button"
          className="fb-cancel"
          onClick={() => setFeedbackTurn(null)}
        >
          取消
        </button>
        <button type="button" className="fb-submit" onClick={() => void submit()}>
          提交反馈
        </button>
      </div>
    </div>
  );
}

export const AssistantTurn = memo(function AssistantTurn({
  turn,
  sid,
}: {
  turn: AssistantTurnType;
  sid: string;
}) {
  const bubbleRef = useRef<HTMLDivElement>(null);
  const regenerateLast = useChatStore((s) => s.regenerateLast);
  const setFeedbackTurn = useChatStore((s) => s.setFeedbackTurn);
  const markRated = useChatStore((s) => s.markRated);
  const rated = useChatStore((s) => !!s.ratedBySid[sid]);
  const feedbackOpen = useChatStore((s) => s.feedbackTurnId === turn.id);

  useLayoutEffect(() => {
    if (bubbleRef.current) attachCitations(bubbleRef.current, turn.sources);
  }, [turn.html, turn.sources]);

  const copy = async () => {
    const el = bubbleRef.current;
    if (!el) return;
    const md = el.querySelector(".md-answer");
    const raw = md ? md.textContent : el.textContent;
    try {
      await navigator.clipboard.writeText(raw || "");
      toast("已复制到剪贴板");
    } catch (e) {
      toast("复制失败：" + (e instanceof Error ? e.message : e), "err");
    }
  };

  const submitFeedback = async (
    rating: number,
    reason: string,
    comment: string,
  ) => {
    try {
      const res = await sendJson<{ ok: boolean; error?: string }>(
        "/api/feedback",
        "POST",
        { session_id: sid, rating, reason, comment },
      );
      if (!res.ok) {
        toast(res.error || "反馈提交失败", "err");
        return;
      }
      toast(rating === 1 ? "感谢反馈 👍" : "已记录，我们会改进");
      markRated(sid);
    } catch (e) {
      toast("反馈提交失败：" + (e instanceof Error ? e.message : e), "err");
    }
  };

  return (
    <div className="turn assistant">
      <div className="avatar">
        <img src="/static/emblem.svg" alt="" />
      </div>
      <div
        className={"bubble" + (turn.error ? " bubble-error" : "")}
        ref={bubbleRef}
        data-sid={sid}
      >
        {turn.title && <div className="analysis-title">《{turn.title}》</div>}
        <SafeHtml html={turn.html} />
        {turn.report && <AnalysisReport report={turn.report} />}
        {turn.note && <div className="stop-note">{turn.note}</div>}
        <SourceChips sources={turn.sources} />
      </div>
      {!turn.streaming && (
        <div className="msg-actions">
          <button
            type="button"
            className="msg-act"
            data-tip="复制回答"
            aria-label="复制回答"
            onClick={() => void copy()}
          >
            <IconCopy />
          </button>
          {!turn.analysis && (
            <button
              type="button"
              className="msg-act"
              data-action="regenerate"
              data-tip="重新生成回答"
              aria-label="重新生成回答"
              onClick={() => void regenerateLast(turn.id)}
            >
              <IconRegen />
            </button>
          )}
          <button
            type="button"
            className="msg-act fb"
            data-tip={rated ? "已评价" : "回答有帮助"}
            aria-label="回答有帮助"
            disabled={rated}
            onClick={() => void submitFeedback(1, "", "")}
          >
            <IconThumb />
          </button>
          <button
            type="button"
            className="msg-act fb"
            data-tip={rated ? "已评价" : "回答有问题"}
            aria-label="回答有问题"
            disabled={rated}
            onClick={() => {
              if (rated) return;
              setFeedbackTurn(turn.id);
            }}
          >
            <IconThumb flip />
          </button>
        </div>
      )}
      {feedbackOpen && <FeedbackPanel sid={sid} />}
    </div>
  );
});
