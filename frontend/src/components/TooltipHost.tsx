import { useCallback, useEffect, useRef, useState } from "react";

interface TipState {
  text: string;
  x: number;
  y: number;
  show: boolean;
}

export function TooltipHost() {
  const [tip, setTip] = useState<TipState | null>(null);
  const timer = useRef<number | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  const hide = useCallback(() => {
    if (timer.current != null) window.clearTimeout(timer.current);
    setTip(null);
  }, []);

  const show = useCallback(
    (el: Element) => {
      const text = el.getAttribute("data-tip") || "";
      const disabled = Boolean(
        (el as HTMLElement & { disabled?: boolean }).disabled,
      );
      if (!text || disabled) {
        hide();
        return;
      }
      if (timer.current != null) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => {
        setTip({ text, x: 0, y: 0, show: false });
        window.requestAnimationFrame(() => {
          const r = el.getBoundingClientRect();
          const tipEl = tipRef.current;
          if (!tipEl) return;
          const w = tipEl.offsetWidth;
          const h = tipEl.offsetHeight;
          let x = r.left + r.width / 2 - w / 2;
          let y = r.top - h - 8;
          if (y < 6) y = r.bottom + 8;
          x = Math.max(8, Math.min(x, window.innerWidth - w - 8));
          setTip({ text, x, y, show: true });
        });
      }, 120);
    },
    [hide],
  );

  useEffect(() => {
    const closest = (target: EventTarget | null): Element | null =>
      target instanceof Element ? (target.closest("[data-tip]") ?? null) : null;
    const onPointerOver = (e: Event) => {
      const t = closest(e.target);
      if (t) show(t);
    };
    const onPointerOut = (e: Event) => {
      const from = closest(e.target);
      const related = (e as MouseEvent).relatedTarget;
      const to = related ? closest(related) : null;
      if (from && from !== to) hide();
    };
    const onFocusIn = (e: Event) => {
      const t = closest(e.target);
      if (t) show(t);
    };
    document.addEventListener("pointerover", onPointerOver);
    document.addEventListener("pointerout", onPointerOut);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", hide);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      document.removeEventListener("pointerover", onPointerOver);
      document.removeEventListener("pointerout", onPointerOut);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", hide);
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [show, hide]);

  return (
    <div
      ref={tipRef}
      className={"tooltip" + (tip?.show ? " show" : "")}
      hidden={!tip}
      style={tip ? { left: tip.x, top: tip.y } : undefined}
      role="tooltip"
    >
      {tip?.text}
    </div>
  );
}
