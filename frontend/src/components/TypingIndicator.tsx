import "./TypingIndicator.css";

export default function TypingIndicator() {
  return (
    <div className="typing-row" aria-label="Leo is typing">
      <img className="typing__avatar" src="/avatar.webp" alt="" width={34} height={34} />
      <div className="typing">
        <span className="typing__dot" />
        <span className="typing__dot" />
        <span className="typing__dot" />
      </div>
    </div>
  );
}
