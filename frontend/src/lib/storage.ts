import type { Message } from "./types";

const KEY = "cvbot.messages";

export function loadMessages(): Message[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m): m is Message =>
        m && typeof m.id === "string" && (m.role === "user" || m.role === "assistant"),
    );
  } catch {
    return [];
  }
}

export function saveMessages(messages: Message[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(messages));
  } catch {
    /* storage unavailable (private mode / quota) — non-fatal */
  }
}

export function clearMessages(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* non-fatal */
  }
}
