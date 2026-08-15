import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Segmented } from "@/components/ui";

const OPTIONS = [
  { value: "grid" as const, label: "Grid" },
  { value: "list" as const, label: "List" },
];

describe("Segmented", () => {
  it("marks the active option as pressed and the rest as not pressed", () => {
    render(<Segmented value="grid" onChange={() => {}} options={OPTIONS} />);
    expect(screen.getByRole("button", { name: "Grid" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "List" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("calls onChange with the clicked option's value", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Segmented value="grid" onChange={onChange} options={OPTIONS} />);

    await user.click(screen.getByRole("button", { name: "List" }));

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith("list");
  });

  it("still fires onChange when the already-active option is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Segmented value="grid" onChange={onChange} options={OPTIONS} />);

    await user.click(screen.getByRole("button", { name: "Grid" }));

    expect(onChange).toHaveBeenCalledWith("grid");
  });
});
