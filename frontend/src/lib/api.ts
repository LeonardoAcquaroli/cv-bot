import type { ChatTurn } from "./types";

export interface StreamHandlers {
  onToken: (text: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
}

interface ParsedEvent {
  event: string;
  data: string;
}

// Frames may be separated by "\n\n" or "\r\n\r\n" depending on the server
// (sse-starlette uses CRLF); lines likewise use "\n" or "\r\n".
const FRAME_SEP = /\r?\n\r?\n/;

/** Parse a single SSE frame (a block of lines) into event + data. */
function parseFrame(frame: string): ParsedEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * POST a chat message and stream the assistant reply over SSE.
 *
 * Resolves once the stream ends (after invoking onDone exactly once). Network,
 * HTTP, and parse failures are reported via onError, then onDone.
 */
export async function streamChat(
  message: string,
  history: ChatTurn[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let finished = false;
  const finish = () => {
    if (!finished) {
      finished = true;
      handlers.onDone();
    }
  };

  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
      signal,
    });
  } catch {
    handlers.onError("Couldn't reach the server. Check your connection and try again.");
    finish();
    return;
  }

  if (!response.ok || !response.body) {
    handlers.onError("The server returned an unexpected response. Please try again.");
    finish();
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  /** Returns "done" when a terminating event was seen. */
  const dispatch = (frame: string): "done" | void => {
    const parsed = parseFrame(frame);
    if (!parsed) return;
    if (parsed.event === "token") {
      try {
        const { text } = JSON.parse(parsed.data) as { text: string };
        if (text) handlers.onToken(text);
      } catch {
        /* ignore malformed token frame */
      }
    } else if (parsed.event === "error") {
      let msg = "Something went wrong while generating the answer.";
      try {
        msg = (JSON.parse(parsed.data) as { message: string }).message || msg;
      } catch {
        /* keep default */
      }
      handlers.onError(msg);
    } else if (parsed.event === "done") {
      return "done";
    }
  };

  try {
    outer: for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let match: RegExpExecArray | null;
      while ((match = FRAME_SEP.exec(buffer)) !== null) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        if (dispatch(frame) === "done") {
          await reader.cancel().catch(() => {});
          break outer;
        }
      }
    }
    // Flush any trailing frame that wasn't terminated by a separator.
    if (buffer.trim() && !finished) dispatch(buffer);
  } catch {
    if (!finished) {
      handlers.onError("The connection was interrupted. Please try again.");
    }
  } finally {
    finish();
  }
}
