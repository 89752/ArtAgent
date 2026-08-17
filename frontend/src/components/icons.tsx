import type { ReactNode } from "react";

interface IconProps {
  className?: string;
  width?: number | string;
  height?: number | string;
  strokeWidth?: number;
}

function Svg({
  className,
  width,
  height,
  strokeWidth = 1.7,
  children,
  flip,
}: IconProps & { children: ReactNode; flip?: boolean }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={flip ? { transform: "scale(-1, -1)" } : undefined}
      width={width}
      height={height}
    >
      {children}
    </svg>
  );
}

export function IconCopy(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M9 9h11a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V10a1 1 0 0 1 1-1Z" />
      <path d="M5 15V5a1 1 0 0 1 1-1h10" />
    </Svg>
  );
}

export function IconRegen(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M23 4v6h-6" />
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
    </Svg>
  );
}

export function IconThumb(props: IconProps & { flip?: boolean }) {
  return (
    <Svg {...props} flip={props.flip}>
      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3z" />
      <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
    </Svg>
  );
}

export function IconEdit(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </Svg>
  );
}

export function IconMenu(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.8}>
      <path d="M4 7h16M4 12h16M4 17h16" />
    </Svg>
  );
}

export function IconPanelLeftClose(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
      <path d="m14 9-3 3 3 3" />
    </Svg>
  );
}

export function IconPanelLeftOpen(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
      <path d="m11 9 3 3-3 3" />
    </Svg>
  );
}

export function IconChevronLeft(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={2}>
      <path d="m15 18-6-6 6-6" />
    </Svg>
  );
}

export function IconChevronRight(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={2}>
      <path d="m9 18 6-6-6-6" />
    </Svg>
  );
}

export function IconMessageSquarePlus(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      <path d="M9 10h6M12 7v6" />
    </Svg>
  );
}

export function IconChat(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </Svg>
  );
}

export function IconMemory(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="M12 3a3 3 0 0 0-3 3v1H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-2V6a3 3 0 0 0-3-3Z" />
      <path d="M9 12h6M9 16h4" />
    </Svg>
  );
}

export function IconSettings(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
    </Svg>
  );
}

export function IconChevronsUpDown(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="m7 15 5 5 5-5M7 9l5-5 5 5" />
    </Svg>
  );
}

export function IconInfo(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8h.01M12 12v4" />
    </Svg>
  );
}

export function IconUser(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c0-4 3.6-6 8-6s8 2 8 6" />
    </Svg>
  );
}

export function IconSearch(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.8}>
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.2-3.2" />
    </Svg>
  );
}

export function IconThemeMoon(props: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={props.width || 16}
      height={props.height || 16}
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z" />
    </svg>
  );
}

export function IconLibrary(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z" />
      <path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20" />
    </Svg>
  );
}

export function IconAttach(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.7}>
      <path d="M21.4 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </Svg>
  );
}

export function IconSend(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.6} className={(props.className || "") + " ico-send"}>
      <path d="M4 12h15M13 6l6 6-6 6" />
    </Svg>
  );
}

export function IconStop(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={2.2} className={(props.className || "") + " ico-stop"}>
      <path d="M8 8h8v8H8z" />
    </Svg>
  );
}

export function IconJump(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={2}>
      <path d="M12 4v16M6 14l6 6 6-6" />
    </Svg>
  );
}

export function IconUpload(props: IconProps) {
  return (
    <Svg {...props} strokeWidth={1.8}>
      <path d="M12 16V4M6 10l6-6 6 6" />
      <path d="M4 20h16" />
    </Svg>
  );
}

export function IconUserAvatar() {
  return (
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="4">
        <circle cx="50" cy="36" r="18" />
        <path d="M18 88 C18 64 34 56 50 56 C66 56 82 64 82 88" />
      </g>
    </svg>
  );
}
