import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MessageBubble from "../MessageBubble";

describe("MessageBubble", () => {
  it("renders user message content", () => {
    render(<MessageBubble message={{ id: "1", role: "user", content: "Hello Leo" }} />);
    expect(screen.getByTestId("bubble-user")).toHaveTextContent("Hello Leo");
  });

  it("renders assistant markdown with avatar", () => {
    render(
      <MessageBubble message={{ id: "2", role: "assistant", content: "**bold** answer" }} />,
    );
    const bubble = screen.getByTestId("bubble-assistant");
    expect(bubble.querySelector("strong")).toHaveTextContent("bold");
    expect(screen.getByAltText("Leo")).toBeInTheDocument();
  });
});
