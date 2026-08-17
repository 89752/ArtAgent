import { useEffect, type RefObject } from "react";

export function useFocusTrap(
  open: boolean,
  containerRef: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!open) return;
    const container = containerRef.current;
    if (!container) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const focusables = container.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), ' +
          'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, containerRef]);
}
