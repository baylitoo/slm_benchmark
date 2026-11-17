import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DeploymentsView } from "@/components/deploy/DeploymentsView";
import { ToastProvider } from "@/components/Toast";
import type { DeploymentRecord, StoreEntry } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

function polling<T>(data: T): PollingState<T> {
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

function record(name: string, node: string, servedMinutesAgo: number | null): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llamacpp", model: name, port: 8088 } },
    state: "ready",
    endpoint: `http://${node}:8088/v1`,
    node_id: node,
    observed: {
      phase: "hot",
      last_served_at:
        servedMinutesAgo == null
          ? null
          : new Date(Date.now() - servedMinutesAgo * 60_000).toISOString(),
    },
  } as DeploymentRecord;
}

function renderView(rows: DeploymentRecord[]) {
  render(
    <ToastProvider>
      <DeploymentsView
        deployments={polling(rows)}
        embeddingNames={new Set()}
        rerankerNames={new Set()}
        store={polling<StoreEntry[]>([])}
      />
    </ToastProvider>,
  );
}

function rowNames(): string[] {
  return screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).queryAllByText(/^(lfm2\.5-350m|nuextract3|lfm3-1\.7b)$/)[0]?.textContent ?? "")
    .filter(Boolean);
}

describe("DeploymentsView on a fleet", () => {
  it("lists the most recently used deployment first, whatever node runs it", () => {
    renderView([
      record("nuextract3", "central", 120),
      record("lfm3-1.7b", "rocky-llm", 2),
      record("lfm2.5-350m", "central", null),
    ]);
    expect(rowNames()).toEqual(["lfm3-1.7b", "nuextract3", "lfm2.5-350m"]);
  });

  it("filters the table to one node", async () => {
    renderView([record("nuextract3", "central", 5), record("lfm3-1.7b", "rocky-llm", 2)]);
    await userEvent.selectOptions(screen.getByLabelText("Node"), "rocky-llm");
    expect(rowNames()).toEqual(["lfm3-1.7b"]);
  });

  it("gives a remote row the same actions as a local one, with its queued request", () => {
    const remote = {
      ...record("lfm3-1.7b", "rocky-llm", 2),
      record_source: "catalog",
      node_request: { id: "r1", node_id: "rocky-llm", verb: "pin", status: "pending" },
    } as DeploymentRecord;
    renderView([record("nuextract3", "central", 5), remote]);

    expect(screen.getAllByTitle("Edit deployment runtime settings")).toHaveLength(2);
    expect(screen.getAllByTitle(/^Delete: stop the process/)).toHaveLength(2);
    expect(screen.getByText("pin pending")).toBeInTheDocument();
    expect(screen.queryByText("remote")).not.toBeInTheDocument();
  });
});
