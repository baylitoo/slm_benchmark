import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BatchView } from "@/components/benchmark/BatchView";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { BatchRunSummary } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listBatches: vi.fn(),
    getBatch: vi.fn(),
    getDeployments: vi.fn(),
    triggerBatchExtract: vi.fn(),
    downloadBatchResults: vi.fn(),
    // fileToBase64 reads a File via FileReader; stub it to a deterministic
    // marker so the assembled payload is checkable without jsdom's async
    // FileReader dance.
    fileToBase64: vi.fn(async (f: File) => `B64<${f.name}>`),
  };
});

function makeBatch(overrides: Partial<BatchRunSummary> = {}): BatchRunSummary {
  return {
    event_id: "ev-1",
    channel: "batch:1",
    tenant_id: "t",
    name: "Q3 invoices",
    schema_name: "invoice",
    model_selector: "lfm2.5-350m",
    status: "running",
    total_items: 10,
    done_items: 4,
    failed_items: 1,
    error: null,
    artifacts: [],
    created_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    updated_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    ...overrides,
  };
}

function renderView() {
  return render(
    <ToastProvider>
      <BatchView />
    </ToastProvider>,
  );
}

describe("BatchView", () => {
  beforeEach(() => {
    vi.mocked(api.listBatches).mockReset().mockResolvedValue([]);
    vi.mocked(api.getDeployments).mockReset().mockResolvedValue([]);
    vi.mocked(api.triggerBatchExtract).mockReset().mockResolvedValue({
      event_ids: ["ev-1"],
      channel: "batch:1",
      topics: [],
    });
    vi.mocked(api.downloadBatchResults).mockReset().mockResolvedValue(undefined);
  });

  it("sends a single .zip as zip_b64", async () => {
    renderView();
    const input = screen.getByLabelText("Batch documents") as HTMLInputElement;
    const zip = new File([new Uint8Array([80, 75, 3, 4])], "invoices.zip", { type: "application/zip" });
    await userEvent.upload(input, zip);
    await userEvent.type(screen.getByPlaceholderText("Q3 invoices"), "Q3");
    await userEvent.click(screen.getByRole("button", { name: /Start batch/ }));

    await waitFor(() => expect(api.triggerBatchExtract).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(api.triggerBatchExtract).mock.calls[0][0];
    expect(payload.zip_b64).toBe("B64<invoices.zip>");
    expect(payload.documents).toBeUndefined();
    expect(payload.name).toBe("Q3");
    expect(payload.schema_name).toBe("invoice");
  });

  it("sends several files as inline documents (never as a zip)", async () => {
    renderView();
    const input = screen.getByLabelText("Batch documents") as HTMLInputElement;
    await userEvent.upload(input, [
      new File(["a"], "a.pdf", { type: "application/pdf" }),
      new File(["b"], "b.pdf", { type: "application/pdf" }),
    ]);
    await userEvent.click(screen.getByRole("button", { name: /Start batch/ }));

    await waitFor(() => expect(api.triggerBatchExtract).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(api.triggerBatchExtract).mock.calls[0][0];
    expect(payload.zip_b64).toBeUndefined();
    expect(payload.documents).toEqual([
      { filename: "a.pdf", content_b64: "B64<a.pdf>" },
      { filename: "b.pdf", content_b64: "B64<b.pdf>" },
    ]);
  });

  it("disables Start until files are picked", () => {
    renderView();
    expect(screen.getByRole("button", { name: /Start batch/ })).toBeDisabled();
  });

  it("shows live progress for a running batch and download buttons only once artifacts exist", async () => {
    vi.mocked(api.listBatches).mockResolvedValue([
      makeBatch(),
      makeBatch({
        event_id: "ev-2",
        name: "done one",
        status: "completed",
        done_items: 10,
        failed_items: 0,
        artifacts: [
          { name: "results.jsonl", relkey: "k1", media_type: "application/x-ndjson", size_bytes: 1, sha256: "x" },
          { name: "results.csv", relkey: "k2", media_type: "text/csv", size_bytes: 1, sha256: "y" },
        ],
      }),
    ]);
    renderView();
    // Running row: 5/10 with 1 failed, no download buttons.
    expect(await screen.findByText("5/10")).toBeInTheDocument();
    expect(screen.getByText("1 failed")).toBeInTheDocument();
    // Settled row: JSONL + CSV buttons.
    const jsonl = screen.getByRole("button", { name: /JSONL/ });
    expect(screen.getByRole("button", { name: /CSV/ })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /JSONL/ })).toHaveLength(1); // only the settled one

    await userEvent.click(jsonl);
    expect(api.downloadBatchResults).toHaveBeenCalledWith(
      expect.objectContaining({ event_id: "ev-2", name: "done one" }),
      "jsonl",
    );
  });
});
