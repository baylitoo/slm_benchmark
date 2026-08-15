import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Badge, StatusDot } from "@/components/ui";

describe("Badge", () => {
  it("renders its children", () => {
    render(<Badge>Running</Badge>);
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("defaults to the neutral tone", () => {
    render(<Badge>Idle</Badge>);
    expect(screen.getByText("Idle")).toHaveClass("bg-muted", "text-muted-foreground");
  });

  it.each([
    ["ok", "bg-emerald-500/10"],
    ["warn", "bg-amber-500/10"],
    ["err", "bg-rose-500/10"],
    ["info", "bg-accent/10"],
  ] as const)("applies the %s tone's classes", (tone, expectedClass) => {
    render(<Badge tone={tone}>Status</Badge>);
    expect(screen.getByText("Status")).toHaveClass(expectedClass);
  });

  it("merges a caller-supplied className without dropping tone classes", () => {
    render(
      <Badge tone="err" className="ml-2">
        Failed
      </Badge>,
    );
    const badge = screen.getByText("Failed");
    expect(badge).toHaveClass("ml-2", "bg-rose-500/10");
  });
});

describe("StatusDot", () => {
  it("renders with the neutral tone by default", () => {
    const { container } = render(<StatusDot />);
    expect(container.firstChild).toHaveClass("bg-muted-foreground");
  });

  it("applies the pulse animation class when pulse is set", () => {
    const { container } = render(<StatusDot tone="ok" pulse />);
    expect(container.firstChild).toHaveClass("bg-emerald-500", "animate-pulse-dot");
  });
});
