import { useRef, type ReactNode } from "react";
import { useFocusTrap } from "../hooks/useFocusTrap";

interface ModalShellProps {
  open: boolean;
  outerClass: string;
  labelledBy?: string;
  onBackdrop?: () => void;
  children: ReactNode;
}

export function ModalShell({
  open,
  outerClass,
  labelledBy,
  onBackdrop,
  children,
}: ModalShellProps) {
  const ref = useRef<HTMLDivElement>(null);
  useFocusTrap(open, ref);
  if (!open) return null;
  return (
    <div
      ref={ref}
      className={outerClass}
      role="dialog"
      aria-modal="true"
      aria-labelledby={labelledBy}
      onClick={(e) => {
        if (e.target === ref.current && onBackdrop) onBackdrop();
      }}
    >
      {children}
    </div>
  );
}
