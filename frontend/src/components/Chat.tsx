import { useEffect, useRef, useState } from "react";
import { useAutoScroll } from "../hooks/useAutoScroll";
import { useChatStore } from "../store/chatStore";
import { IconJump } from "./icons";
import { TurnRenderer } from "./TurnRenderer";

export function Chat() {
  const sid = useChatStore((s) => s.sid);
  const turns = useChatStore((s) => s.turnsBySid[s.sid]);
  const editingTurnId = useChatStore((s) => s.editingTurnId);
  const ref = useRef<HTMLElement>(null);
  const { onScroll, scrollToBottom, isStuck } = useAutoScroll();
  const [jumpHidden, setJumpHidden] = useState(true);

  const turnList = turns ?? [];
  const editingIdx = editingTurnId
    ? turnList.findIndex((t) => t.id === editingTurnId)
    : -1;

  useEffect(() => {
    setJumpHidden(!turnList.length || isStuck());
    scrollToBottom(ref.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turnList]);

  const handleScroll = () => {
    onScroll(ref.current);
    setJumpHidden(!turnList.length || isStuck());
  };

  return (
    <>
      <section
        id="chat"
        className="chat"
        ref={ref}
        aria-label="对话内容"
        onScroll={handleScroll}
      >
        {turnList.map((t, i) => (
          <TurnRenderer
            key={t.id}
            turn={t}
            sid={sid}
            editingTurnId={editingTurnId}
            hidden={editingIdx >= 0 && i > editingIdx}
          />
        ))}
      </section>
      <button
        id="btn-jump"
        className="jump-bottom"
        type="button"
        aria-label="回到最新消息"
        data-tip="回到最新消息"
        hidden={jumpHidden}
        onClick={() => {
          scrollToBottom(ref.current, true);
          setJumpHidden(true);
        }}
      >
        <IconJump />
      </button>
    </>
  );
}
