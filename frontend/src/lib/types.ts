export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** Set when this assistant message failed to generate. */
  error?: boolean;
}

/** Wire format for prior turns sent to the backend. */
export interface ChatTurn {
  role: Role;
  content: string;
}
