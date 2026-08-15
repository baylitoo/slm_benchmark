import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Dialog, Sheet } from "@/components/ui";

describe("Dialog", () => {
  it("renders nothing when closed", () => {
    render(
      <Dialog open={false} onClose={() => {}} title="Seed dataset">
        body
      </Dialog>,
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders title, subtitle and children when open", () => {
    render(
      <Dialog
        open
        onClose={() => {}}
        title="Seed dataset"
        subtitle="Choose a source"
      >
        <p>form body</p>
      </Dialog>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("Seed dataset")).toBeInTheDocument();
    expect(screen.getByText("Choose a source")).toBeInTheDocument();
    expect(screen.getByText("form body")).toBeInTheDocument();
  });

  it("moves focus to the close button on open", () => {
    render(
      <Dialog open onClose={() => {}} title="Seed dataset">
        body
      </Dialog>,
    );
    expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  });

  it("calls onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Dialog open onClose={onClose} title="Seed dataset">
        body
      </Dialog>,
    );
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when the backdrop is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container } = render(
      <Dialog open onClose={onClose} title="Seed dataset">
        body
      </Dialog>,
    );
    const backdrop = container.querySelector(".absolute.inset-0");
    expect(backdrop).not.toBeNull();
    await user.click(backdrop as Element);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when Escape is pressed", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <Dialog open onClose={onClose} title="Seed dataset">
        body
      </Dialog>,
    );
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("Sheet", () => {
  it("stays mounted but inert while closed", () => {
    const { container } = render(
      <Sheet open={false} onClose={() => {}}>
        <p>panel content</p>
      </Sheet>,
    );
    expect(screen.getByText("panel content")).toBeInTheDocument();
    const aside = container.querySelector("aside");
    expect(aside).toHaveAttribute("inert");
    expect(aside).toHaveAttribute("aria-hidden", "true");
    expect(aside).toHaveClass("translate-x-full");
  });

  it("clears inert and slides in when opened", () => {
    const { container, rerender } = render(
      <Sheet open={false} onClose={() => {}}>
        <p>panel content</p>
      </Sheet>,
    );
    rerender(
      <Sheet open onClose={() => {}}>
        <p>panel content</p>
      </Sheet>,
    );
    const aside = container.querySelector("aside");
    expect(aside).not.toHaveAttribute("inert");
    expect(aside).toHaveAttribute("aria-hidden", "false");
    expect(aside).toHaveClass("translate-x-0");
  });

  it("moves focus to the close button when it transitions to open", () => {
    const { rerender } = render(
      <Sheet open={false} onClose={() => {}}>
        <p>panel content</p>
      </Sheet>,
    );
    rerender(
      <Sheet open onClose={() => {}}>
        <p>panel content</p>
      </Sheet>,
    );
    expect(screen.getByRole("button", { name: "Close panel" })).toHaveFocus();
  });

  it("calls onClose when the backdrop is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container } = render(
      <Sheet open onClose={onClose}>
        <p>panel content</p>
      </Sheet>,
    );
    const backdrop = container.querySelector("[aria-hidden].fixed.inset-0");
    expect(backdrop).not.toBeNull();
    await user.click(backdrop as Element);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
