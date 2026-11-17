import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PortsAdmin } from "@/components/deploy/PortsAdmin";
import type { PortsView } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

function polling(data: PortsView): PollingState<PortsView> {
  return {
    data,
    error: null,
    loading: false,
    refreshing: false,
    lastUpdated: Date.now(),
    live: true,
    refresh: () => {},
  };
}

describe("PortsAdmin", () => {
  it("lists another node's ports next to the local records", () => {
    render(
      <PortsAdmin
        ports={polling({
          range: { start: 8088, end: 8188 },
          deployments: [{ name: "lfm2.5-350m", port: 8088, state: "running" }],
          used: [8088],
          free_sample: [8089],
          recommended_next: 8089,
          by_node: {
            "rocky-llm": [
              { name: "lfm2.5-350m", port: 8088, state: "running" },
              { name: "nuextract3", port: 8088, state: "running" },
            ],
          },
        })}
      />,
    );
    expect(screen.getByText("nuextract3")).toBeInTheDocument();
    expect(screen.getByText("rocky-llm")).toBeInTheDocument();
    expect(screen.getAllByText("lfm2.5-350m")).toHaveLength(1);
  });
});
