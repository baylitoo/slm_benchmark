import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsageCard, formatCount } from "@/components/Observability";
import * as api from "@/lib/api";
import type { UsageDeployment } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getUsageSummary: vi.fn(),
  };
});

function makeEntry(overrides: Partial<UsageDeployment> = {}): UsageDeployment {
  return {
    deployment: "lfm2.5-350m",
    requests: 12,
    errors: 0,
    prompt_tokens: 3400,
    completion_tokens: 800,
    avg_latency_ms: 240.5,
    p95_latency_ms: 910,
    last_used_at: new Date().toISOString(),
    ...overrides,
  };
}

describe("formatCount", () => {
  it("keeps small counts verbatim and compacts thousands/millions", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(950)).toBe("950");
    expect(formatCount(1000)).toBe("1k");
    expect(formatCount(12400)).toBe("12.4k");
    expect(formatCount(3_200_000)).toBe("3.2M");
  });
});

describe("UsageCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders per-deployment aggregates from the ledger", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({
      window: "24h",
      deployments: [
        makeEntry(),
        makeEntry({ deployment: "nuextract3", requests: 3, errors: 2, prompt_tokens: 0, completion_tokens: 0 }),
      ],
    });
    render(<UsageCard active />);
    expect(await screen.findAllByText("lfm2.5-350m")).not.toHaveLength(0);
    expect(screen.getAllByText("nuextract3").length).toBeGreaterThan(0);
    // Token columns are compact-formatted.
    expect(screen.getAllByText("3.4k").length).toBeGreaterThan(0);
    // Errors render as a badge when non-zero.
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(api.getUsageSummary).toHaveBeenCalledWith("24h");
  });

  it("refetches with the selected window", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({ window: "24h", deployments: [] });
    render(<UsageCard active />);
    await screen.findByText(/No usage recorded/);
    await userEvent.click(screen.getByRole("button", { name: "7d" }));
    await waitFor(() => expect(api.getUsageSummary).toHaveBeenCalledWith("7d"));
  });

  it("shows the empty state when the window has no rows", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({ window: "24h", deployments: [] });
    render(<UsageCard active />);
    expect(await screen.findByText(/No usage recorded/)).toBeInTheDocument();
  });
});
