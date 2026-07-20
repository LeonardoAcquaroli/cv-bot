import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import HowItsDone from "../HowItsDone";

describe("HowItsDone", () => {
  it("opens the stack diagram modal on click and closes it", async () => {
    const user = userEvent.setup();
    render(<HowItsDone />);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /how is it done/i }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveTextContent(/FastAPI/i);
    expect(dialog).toHaveTextContent(/Qdrant/i);

    await user.click(screen.getByRole("button", { name: /close/i }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
