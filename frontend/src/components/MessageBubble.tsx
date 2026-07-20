import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import type { Message } from "../lib/types";
import "./MessageBubble.css";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isAssistant = message.role === "assistant";

  return (
    <motion.div
      className={`bubble-row bubble-row--${message.role}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
    >
      {isAssistant && (
        <img className="bubble__avatar" src="/avatar.webp" alt="Leo" width={34} height={34} />
      )}
      <div
        className={`bubble bubble--${message.role}${message.error ? " bubble--error" : ""}`}
        data-testid={`bubble-${message.role}`}
      >
        {message.content ? (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        ) : (
          <span className="bubble__placeholder" aria-hidden="true" />
        )}
      </div>
    </motion.div>
  );
}
