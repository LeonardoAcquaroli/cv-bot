import { afterEach, describe, expect, it, vi } from "vitest";
import { streamChat } from "../api";

/** Build a Response whose body streams the given string as UTF-8 chunks. */
function sseResponse(body: string, chunkSize = 7): Response {
  const bytes = new TextEncoder().encode(body);
  let offset = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      controller.enqueue(bytes.slice(offset, offset + chunkSize));
      offset += chunkSize;
    },
  });
  return new Response(stream, { status: 200 });
}

afterEach(() => vi.restoreAllMocks());

describe("streamChat", () => {
  it("parses CRLF-separated SSE frames and reports tokens then done", async () => {
    // Exactly the wire format sse-starlette emits (\r\n separators).
    const body =
      'event: token\r\ndata: {"text": "Hello"}\r\n\r\n' +
      'event: token\r\ndata: {"text": " world"}\r\n\r\n' +
      "event: done\r\ndata: {}\r\n\r\n";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(sseResponse(body));

    const tokens: string[] = [];
    let done = 0;
    let errors = 0;

    await streamChat("hi", [], {
      onToken: (t) => tokens.push(t),
      onError: () => (errors += 1),
      onDone: () => (done += 1),
    });

    expect(tokens.join("")).toBe("Hello world");
    expect(done).toBe(1);
    expect(errors).toBe(0);
  });

  it("surfaces an error event message", async () => {
    const body = 'event: error\r\ndata: {"message": "boom"}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n';
    vi.spyOn(globalThis, "fetch").mockResolvedValue(sseResponse(body));

    let errorMsg = "";
    let done = 0;
    await streamChat("hi", [], {
      onToken: () => {},
      onError: (m) => (errorMsg = m),
      onDone: () => (done += 1),
    });

    expect(errorMsg).toBe("boom");
    expect(done).toBe(1);
  });

  it("reports a network failure via onError then onDone", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    let errorMsg = "";
    let done = 0;
    await streamChat("hi", [], {
      onToken: () => {},
      onError: (m) => (errorMsg = m),
      onDone: () => (done += 1),
    });

    expect(errorMsg).toMatch(/couldn't reach the server/i);
    expect(done).toBe(1);
  });
});
