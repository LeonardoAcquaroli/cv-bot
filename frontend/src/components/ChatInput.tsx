import { useRef, useState } from "react";
import "./ChatInput.css";

interface Props {
  disabled: boolean;
  onSend: (text: string) => void;
}

const PLACEHOLDER = 'For example: "What are Leonardo\'s soft skills?"';

export default function ChatInput({ disabled, onSend }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="chat-input">
      <textarea
        ref={textareaRef}
        className="chat-input__field"
        rows={1}
        value={value}
        placeholder={PLACEHOLDER}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        aria-label="Message"
      />
      <button
        className="chat-input__send"
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send message"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M3.4 20.4 21 12 3.4 3.6 3 10l12 2-12 2z"
            fill="currentColor"
          />
        </svg>
      </button>
    </div>
  );
}
