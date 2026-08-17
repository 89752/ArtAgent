import type { Turn } from "../store/chatStore";
import { AssistantTurn } from "./AssistantTurn";
import { UserTurn } from "./UserTurn";

function AttachmentTurn({ turn }: { turn: Extract<Turn, { role: "attachment" }> }) {
  return (
    <div className="turn system">
      <div className="bubble system-note">
        <div className="file-card attach-note" data-doc-id={turn.docId}>
          {turn.kind === "image" ? (
            <img
              className="fc-thumb"
              src={`/api/user-images/${encodeURIComponent(turn.docId)}/file`}
              alt=""
            />
          ) : (
            <span className="fc-ico">{turn.kind === "table" ? "📊" : "📄"}</span>
          )}
          <span className="fc-name" data-tip={turn.docName}>
            已上传《{turn.docName}》
          </span>
        </div>
      </div>
    </div>
  );
}

export function TurnRenderer({
  turn,
  sid,
  editingTurnId,
  hidden,
}: {
  turn: Turn;
  sid: string;
  editingTurnId: string | null;
  hidden: boolean;
}) {
  if (hidden) return null;
  return (
    <div className="chat-wrap">
      {turn.role === "user" && (
        <UserTurn turn={turn} editing={editingTurnId === turn.id} />
      )}
      {turn.role === "assistant" && <AssistantTurn turn={turn} sid={sid} />}
      {turn.role === "attachment" && <AttachmentTurn turn={turn} />}
    </div>
  );
}
