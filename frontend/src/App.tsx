import { useCallback, useEffect, useRef, useState } from "react";
import Hero from "./components/Hero";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import HowItsDone from "./components/HowItsDone";
import { streamChat } from "./lib/api";
import { clearMessages, loadMessages, saveMessages } from "./lib/storage";
import type { Message } from "./lib/types";
import "./App.css";

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>(() => loadMessages());
  const [streaming, setStreaming] = useState(false);
  const [awaitingFirstToken, setAwaitingFirstToken] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    saveMessages(messages);
  }, [messages]);

  const send = useCallback(
    (text: string) => {
      if (streaming) return;

      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const userMsg: Message = { id: newId(), role: "user", content: text };
      const assistantId = newId();

      setMessages((prev) => [...prev, userMsg]);
      setStreaming(true);
      setAwaitingFirstToken(true);

      const controller = new AbortController();
      abortRef.current = controller;

      // The assistant bubble is created lazily on the first token (or error)
      // so only the typing indicator shows while we wait — no empty bubble.
      let created = false;
      const patch = (fn: (m: Message) => Message) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

      void streamChat(
        text,
        history,
        {
          onToken: (chunk) => {
            setAwaitingFirstToken(false);
            if (!created) {
              created = true;
              setMessages((prev) => [
                ...prev,
                { id: assistantId, role: "assistant", content: chunk },
              ]);
            } else {
              patch((m) => ({ ...m, content: m.content + chunk }));
            }
          },
          onError: (message) => {
            setAwaitingFirstToken(false);
            if (!created) {
              created = true;
              setMessages((prev) => [
                ...prev,
                { id: assistantId, role: "assistant", content: message, error: true },
              ]);
            } else {
              patch((m) => ({ ...m, error: true }));
            }
          },
          onDone: () => {
            setStreaming(false);
            setAwaitingFirstToken(false);
            abortRef.current = null;
          },
        },
        controller.signal,
      );
    },
    [messages, streaming],
  );

  const handleClear = () => {
    abortRef.current?.abort();
    clearMessages();
    setMessages([]);
    setStreaming(false);
    setAwaitingFirstToken(false);
  };

  return (
    <div className="app">
      <div className="app__inner">
        <Hero />
        <ChatWindow messages={messages} awaitingFirstToken={awaitingFirstToken} />
        <div className="app__footer">
          <ChatInput disabled={streaming} onSend={send} />
          {messages.length > 0 && (
            <button className="app__clear" onClick={handleClear} disabled={streaming}>
              Clear chat history
            </button>
          )}
        </div>
      </div>
      <HowItsDone />
    </div>
  );
}
