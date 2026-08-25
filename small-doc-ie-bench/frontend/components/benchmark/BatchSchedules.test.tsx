import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BatchSchedules, intervalLabel } from "@/components/benchmark/BatchSchedules";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { BatchScheduleView } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listBatchSchedules: vi.fn(),
    createBatchSchedule: vi.fn(),
    updateBatchSchedule: vi.fn(),
    deleteBatchSchedule: vi.fn(),
    runBatchScheduleNow: vi.fn(),
  };
});

function makeSchedule(overrides: Partial<BatchScheduleView> = {}): BatchScheduleView {
  return {
    id: "s-1",
    tenant_id: "t",
    name: "nightly re-run",
    source_event_id: "ev-1",
    schema_name: "invoice",
    selectors: { model_profile: "lfm2.5-350m" },
    interval: "daily",
    every_n_minutes: null,
    enabled: true,
    next_run_at: new Date("2026-01-02T03:00:00Z").toISOString(),
    last_run_at: new Date("2026-01-01T03:00:00Z").toISOString(),
    last_event_id: "ev-9",
    last_error: null,
    created_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    updated_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    ...overrides,
  };
}

function renderSchedules(props: Partial<Parameters<typeof BatchSchedules>[0]> = {}) {
  return render(
    <ToastProvider>
      <BatchSchedules source={null} onSourceHandled={() => undefined} {...props} />
    </ToastProvider>,
  );
}

describe("BatchSchedules", () => {
  beforeEach(() => {
    vi.mocked(api.listBatchSchedules).mockReset().mockResolvedValue([]);
    vi.mocked(api.createBatchSchedule).mockReset();
    vi.mocked(api.updateBatchSchedule).mockReset().mockResolvedValue(makeSchedule());
    vi.mocked(api.deleteBatchSchedule).mockReset().mockResolvedValue({ deleted: "s-1" });
    vi.mocked(api.runBatchScheduleNow).mockReset().mockResolvedValue({
      event_ids: ["ev-now"],
      channel: "batch:now",
      topics: [],
    });
  });

  it("labels intervals for humans", () => {
    expect(intervalLabel({ interval: "daily", every_n_minutes: null })).toBe("daily");
    expect(intervalLabel({ interval: "every_n_minutes", every_n_minutes: 30 })).toBe(
      "every 30 min",
    );
  });

  it("lists schedules with interval, next and last run", async () => {
    vi.mocked(api.listBatchSchedules).mockResolvedValue([
      makeSchedule(),
      makeSchedule({
        id: "s-2",
        name: "paused one",
        enabled: false,
        interval: "every_n_minutes",
        every_n_minutes: 30,
        last_error: "input documents no longer in the store: a.pdf",
      }),
    ]);
    renderSchedules();
    expect(await screen.findByText("nightly re-run")).toBeInTheDocument();
    expect(screen.getByText("daily")).toBeInTheDocument();
    expect(screen.getByText("every 30 min")).toBeInTheDocument();
    expect(screen.getByText("enabled")).toBeInTheDocument();
    expect(screen.getByText("paused")).toBeInTheDocument();
    // The broken schedule surfaces its last_error as a warning tooltip.
    expect(screen.getByTitle(/no longer in the store/)).toBeInTheDocument();
  });

  it("pauses and resumes via PATCH enabled", async () => {
    vi.mocked(api.listBatchSchedules).mockResolvedValue([makeSchedule()]);
    renderSchedules();
    await userEvent.click(await screen.findByRole("button", { name: /Pause/ }));
    await waitFor(() =>
      expect(api.updateBatchSchedule).toHaveBeenCalledWith("s-1", { enabled: false }),
    );
  });

  it("runs a schedule immediately", async () => {
    vi.mocked(api.listBatchSchedules).mockResolvedValue([makeSchedule()]);
    renderSchedules();
    await userEvent.click(await screen.findByRole("button", { name: /Run now/ }));
    await waitFor(() => expect(api.runBatchScheduleNow).toHaveBeenCalledWith("s-1"));
  });

  it("deletes a schedule", async () => {
    vi.mocked(api.listBatchSchedules).mockResolvedValue([makeSchedule()]);
    renderSchedules();
    await userEvent.click(await screen.findByRole("button", { name: /Delete/ }));
    await waitFor(() => expect(api.deleteBatchSchedule).toHaveBeenCalledWith("s-1"));
  });

  it("creates from a source batch with an every-N-minutes cadence", async () => {
    vi.mocked(api.createBatchSchedule).mockResolvedValue(makeSchedule());
    const onSourceHandled = vi.fn();
    renderSchedules({
      source: {
        event_id: "ev-src",
        channel: "batch:src",
        tenant_id: "t",
        name: "Q3 invoices",
        schema_name: "invoice",
        model_selector: "lfm2.5-350m",
        status: "completed",
        total_items: 3,
        done_items: 3,
        failed_items: 0,
        error: null,
        artifacts: [],
        created_at: new Date("2026-01-01T00:00:00Z").toISOString(),
        updated_at: new Date("2026-01-01T00:00:00Z").toISOString(),
      },
      onSourceHandled,
    });
    await userEvent.selectOptions(screen.getByLabelText("Interval"), "every_n_minutes");
    const minutes = screen.getByLabelText("Minutes between runs");
    await userEvent.clear(minutes);
    await userEvent.type(minutes, "45");
    await userEvent.click(screen.getByRole("button", { name: /Create schedule/ }));

    await waitFor(() => expect(api.createBatchSchedule).toHaveBeenCalledTimes(1));
    expect(api.createBatchSchedule).toHaveBeenCalledWith({
      source_event_id: "ev-src",
      name: "re-run: Q3 invoices",
      interval: "every_n_minutes",
      every_n_minutes: 45,
    });
    expect(onSourceHandled).toHaveBeenCalled();
  });

  it("surfaces a create failure inline and keeps the form open", async () => {
    vi.mocked(api.createBatchSchedule).mockRejectedValue(
      new Error("batch 'ev-src' not found"),
    );
    const onSourceHandled = vi.fn();
    renderSchedules({
      source: {
        event_id: "ev-src",
        channel: "batch:src",
        tenant_id: "t",
        name: "Q3",
        schema_name: "invoice",
        model_selector: null,
        status: "completed",
        total_items: 1,
        done_items: 1,
        failed_items: 0,
        error: null,
        artifacts: [],
        created_at: new Date("2026-01-01T00:00:00Z").toISOString(),
        updated_at: new Date("2026-01-01T00:00:00Z").toISOString(),
      },
      onSourceHandled,
    });
    await userEvent.click(screen.getByRole("button", { name: /Create schedule/ }));
    expect(await screen.findByText(/not found/)).toBeInTheDocument();
    expect(onSourceHandled).not.toHaveBeenCalled();
  });
});
