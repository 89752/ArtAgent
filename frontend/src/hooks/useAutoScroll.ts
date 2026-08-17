import { useCallback, useRef } from "react";

export function useAutoScroll() {
  const stickBottom = useRef(true);

  const onScroll = useCallback((el: HTMLElement | null) => {
    if (!el) return;
    stickBottom.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }, []);

  const scrollToBottom = useCallback((el: HTMLElement | null, smooth = false) => {
    if (!el) return;
    stickBottom.current = true;
    if (smooth) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    } else {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  const isStuck = useCallback(() => stickBottom.current, []);

  return { onScroll, scrollToBottom, isStuck };
}
