import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RangeSlider } from "@/components/ui";

const STEPS = ["0", "500M", "1B", "3B", "7B", "15B", "35B", "70B", "70B+"];

describe("RangeSlider", () => {
  it("renders a labeled tick for every step, bolding the two selected ones", () => {
    render(
      <RangeSlider steps={STEPS} loIndex={2} hiIndex={5} onChange={() => {}} ariaLabel="Params" />,
    );
    for (const label of STEPS) expect(screen.getByText(label)).toBeInTheDocument();
    expect(screen.getByText("1B")).toHaveClass("font-semibold");
    expect(screen.getByText("15B")).toHaveClass("font-semibold");
    expect(screen.getByText("3B")).not.toHaveClass("font-semibold");
  });

  it("moving the minimum handle past the maximum clamps to it, not past it", () => {
    const onChange = vi.fn();
    render(
      <RangeSlider steps={STEPS} loIndex={2} hiIndex={4} onChange={onChange} ariaLabel="Params" />,
    );
    fireEvent.change(screen.getByRole("slider", { name: "Params — minimum" }), {
      target: { value: "7" },
    });
    expect(onChange).toHaveBeenCalledWith(4, 4);
  });

  it("moving the maximum handle below the minimum clamps to it, not below it", () => {
    const onChange = vi.fn();
    render(
      <RangeSlider steps={STEPS} loIndex={3} hiIndex={6} onChange={onChange} ariaLabel="Params" />,
    );
    fireEvent.change(screen.getByRole("slider", { name: "Params — maximum" }), {
      target: { value: "0" },
    });
    expect(onChange).toHaveBeenCalledWith(3, 3);
  });

  it("moving a handle within bounds reports the new index and keeps the other unchanged", () => {
    const onChange = vi.fn();
    render(
      <RangeSlider steps={STEPS} loIndex={1} hiIndex={5} onChange={onChange} ariaLabel="Params" />,
    );
    fireEvent.change(screen.getByRole("slider", { name: "Params — minimum" }), {
      target: { value: "3" },
    });
    expect(onChange).toHaveBeenCalledWith(3, 5);
  });
});
