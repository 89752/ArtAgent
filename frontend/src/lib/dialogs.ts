import { useUiStore } from "../store/uiStore";

export interface ConfirmOptions {
  title: string;
  text: string;
  okText?: string;
  danger?: boolean;
}

export function confirmAsk(opts: ConfirmOptions): Promise<boolean> {
  return useUiStore.getState().confirmAsk(opts);
}

export function askOversize(
  name: string,
  count = 1,
  maxMb = 50,
): Promise<string> {
  return useUiStore.getState().askOversize(name, count, maxMb);
}

export function toast(msg: string, type = ""): void {
  useUiStore.getState().showToast(msg, type);
}
