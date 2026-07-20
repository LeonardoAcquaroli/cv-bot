import { useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import type { Message } from "../lib/types";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";
import "./ChatWindow.css";

interface Props {
  messages: Message[];
  /** True while awaiting the first token of the latest assistant reply. */
  awaitingFirstToken: boolean;
}

export default function ChatWindow({ messages, awaitingFirstToken }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, awaitingFirstToken]);

  if (messages.length === 0) {
    return (
      <div className="chat-window chat-window--empty">
        <p className="chat-window__hint">
          👋 Say hi, or ask about Leonardo&rsquo;s experience, skills, and projects.
        </p>
      </div>
    );
  }

  return (
    <div className="chat-window" role="log" aria-live="polite">
      <AnimatePresence initial={false}>
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
      </AnimatePresence>
      {awaitingFirstToken && <TypingIndicator />}
      <div ref={endRef} />
    </div>
  );
}
