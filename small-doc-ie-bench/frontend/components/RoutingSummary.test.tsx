import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RoutingSummary } from "@/components/RoutingSummary";

describe("RoutingSummary", () => {
  it("renders nothing for a single-model result (no routing audit)", () => {
    const { container } = render(<RoutingSummary result={{ result: {}, routing: null }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for a non-object result", () => {
    const { container } = render(<RoutingSummary result="oops" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("makes the escalation chain legible: policy, winner, attempts, per-stage decisions", () => {
    render(
      <RoutingSummary
        result={{
          result: {},
          routing: {
            policy: "cheap-then-strong",
            selected_stage: "strong",
            terminal_decision: "accept",
            attempts: 2,
            total_tokens: 20,
            stages: [
              { stage: "cheap", decision: "escalate", avg_confidence: 0.4, reason: "below 0.8" },
              { stage: "strong", decision: "accept", avg_confidence: 0.95, reason: "valid" },
            ],
          },
        }}
      />,
    );
    expect(screen.getByText("cheap-then-strong")).toBeInTheDocument();
    expect(screen.getByText("strong", { selector: "span.font-mono.text-foreground" })).toBeInTheDocument();
    expect(screen.getByText("2 attempts")).toBeInTheDocument();
    expect(screen.getByText("escalate")).toBeInTheDocument();
    expect(screen.getByText("accept")).toBeInTheDocument();
    expect(screen.getByText("0.40")).toBeInTheDocument();
    expect(screen.getByText("0.95")).toBeInTheDocument();
    expect(screen.getByText("20 tok")).toBeInTheDocument();
  });

  it("flags an exhausted budget", () => {
    render(
      <RoutingSummary
        result={{ routing: { policy: "p", attempts: 3, budget_exhausted: true, stages: [] } }}
      />,
    );
    expect(screen.getByText("budget exhausted")).toBeInTheDocument();
  });
});
