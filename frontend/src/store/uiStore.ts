import { create } from "zustand";

export type ModalName =
  | "library"
  | "memory"
  | "settings"
  | "confirm"
  | "oversize";

export interface ConfirmState {
  title: string;
  text: string;
  okText: string;
  danger: boolean;
  returnModal: ModalName | null;
  resolve: (v: boolean) => void;
}

export interface OversizeState {
  name: string;
  count: number;
  maxMb: number;
  resolve: (mode: string) => void;
}

export interface ToastState {
  msg: string;
  type: string;
  id: number;
}

export interface UiState {
  theme: "light" | "dark";
  sidebarOpen: boolean;
  collapsed: boolean;
  sidebarWidth: number;
  modal: ModalName | null;
  toast: ToastState | null;
  confirm: ConfirmState | null;
  oversize: OversizeState | null;
  lastFocus: Element | null;
  initTheme: () => void;
  toggleTheme: () => void;
  openSidebar: () => void;
  closeSidebar: () => void;
  toggleCollapsed: () => void;
  setSidebarWidth: (px: number) => void;
  openModal: (name: ModalName) => void;
  closeModal: (name: ModalName) => void;
  rememberFocus: () => void;
  restoreFocus: () => void;
  showToast: (msg: string, type?: string) => void;
  clearToast: () => void;
  confirmAsk: (opts: {
    title: string;
    text: string;
    okText?: string;
    danger?: boolean;
  }) => Promise<boolean>;
  resolveConfirm: (v: boolean) => void;
  askOversize: (name: string, count?: number, maxMb?: number) => Promise<string>;
  resolveOversize: (mode: string) => void;
}

const SIDEBAR_MIN = 220;
const SIDEBAR_MAX = 480;
const SIDEBAR_W_KEY = "artagent.sidebar.width";
const SIDEBAR_COLLAPSED_KEY = "artagent.sidebar.collapsed";
const THEME_KEY = "artagent.theme";

function loadStoredTheme(): "light" | "dark" {
  let t: string | null = null;
  try {
    t = localStorage.getItem(THEME_KEY);
  } catch {
    /* 忽略 */
  }
  if (!t) {
    t =
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
  }
  return t === "dark" ? "dark" : "light";
}

function loadSidebarWidth(): number {
  try {
    const saved = Number(localStorage.getItem(SIDEBAR_W_KEY));
    if (saved >= SIDEBAR_MIN && saved <= SIDEBAR_MAX) return saved;
  } catch {
    /* 忽略 */
  }
  return 280;
}

function loadCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export const useUiStore = create<UiState>()((set, get) => ({
  theme: "light",
  sidebarOpen: false,
  collapsed: false,
  sidebarWidth: 280,
  modal: null,
  toast: null,
  confirm: null,
  oversize: null,
  lastFocus: null,

  initTheme: () => {
    const theme = loadStoredTheme();
    document.documentElement.dataset.theme = theme;
    set({ theme, sidebarWidth: loadSidebarWidth(), collapsed: loadCollapsed() });
  },

  toggleTheme: () => {
    const next = get().theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* 忽略 */
    }
    set({ theme: next });
  },

  openSidebar: () => set({ sidebarOpen: true }),
  closeSidebar: () => set({ sidebarOpen: false }),

  toggleCollapsed: () => {
    const next = !get().collapsed;
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
    } catch {
      /* 忽略 */
    }
    set({ collapsed: next });
  },

  setSidebarWidth: (px) => {
    const w = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(px)));
    try {
      localStorage.setItem(SIDEBAR_W_KEY, String(w));
    } catch {
      /* 忽略 */
    }
    set({ sidebarWidth: w });
  },

  openModal: (name) => {
    get().rememberFocus();
    set({ modal: name });
  },

  closeModal: (name) => {
    if (get().modal === name) set({ modal: null });
    get().restoreFocus();
  },

  rememberFocus: () => {
    set({ lastFocus: document.activeElement });
  },

  restoreFocus: () => {
    const el = get().lastFocus;
    if (el && "focus" in el && typeof el.focus === "function") {
      try {
        (el as HTMLElement).focus();
      } catch {
        /* 忽略 */
      }
    }
    set({ lastFocus: null });
  },

  showToast: (msg, type = "") => {
    set({ toast: { msg, type, id: Date.now() + Math.random() } });
  },

  clearToast: () => set({ toast: null }),

  confirmAsk: ({ title, text, okText = "确认", danger = false }) =>
    new Promise<boolean>((resolve) => {
      const returnModal = get().modal;
      get().rememberFocus();
      set({
        confirm: { title, text, okText, danger, returnModal, resolve },
        modal: "confirm",
      });
    }),

  resolveConfirm: (v) => {
    const c = get().confirm;
    set({ confirm: null, modal: c?.returnModal || null });
    c?.resolve(v);
    get().restoreFocus();
  },

  askOversize: (name, count = 1, maxMb = 50) =>
    new Promise<string>((resolve) => {
      get().rememberFocus();
      set({ oversize: { name, count, maxMb, resolve }, modal: "oversize" });
    }),

  resolveOversize: (mode) => {
    const o = get().oversize;
    set({ oversize: null, modal: null });
    o?.resolve(mode);
    get().restoreFocus();
  },
}));
