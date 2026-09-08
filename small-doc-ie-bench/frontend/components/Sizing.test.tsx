import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Sizing } from "@/components/Sizing";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { SizingView, WhatIfView } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getSizing: vi.fn(),
    whatifSizing: vi.fn(),
  };
});

function makeSizing(): SizingView {
  return {
    observed_available: true,
    total_bytes: 16 * 1024 ** 3,
    free_bytes: 6 * 1024 ** 3,
    per_model: [
      {
        name: "small",
        footprint_bytes: 2 * 1024 ** 3,
        fits_now: 2,
        running_instances: 0,
      },
    ],
  };
}

function makeWhatIfResult(): WhatIfView {
  return {
    observed_available: true,
    total_predicted_bytes: 2 * 1024 ** 3,
    ok: true,
    per_item: [
      {
        model: "small",
        instances: 1,
        context_length: 8192,
        footprint_bytes: 2 * 1024 ** 3,
        subtotal_bytes: 2 * 1024 ** 3,
        calibrated: false,
      },
    ],
  };
}

describe("Sizing — What if parallel slots", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getSizing).mockResolvedValue(makeSizing());
    vi.mocked(api.whatifSizing).mockResolvedValue(makeWhatIfResult());
  });

  it("checks fit with the default parallel slots of 1 (not sent)", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <Sizing active />
      </ToastProvider>,
    );

    await user.click(await screen.findByText("Add model"));
    await user.click(screen.getByText("Check fit"));

    await waitFor(() => expect(api.whatifSizing).toHaveBeenCalled());
    const [plan, nParallel] = vi.mocked(api.whatifSizing).mock.calls[0];
    expect(plan).toEqual([{ model: "small", instances: 1 }]);
    expect(nParallel).toBe(1);
  });

  it("passes an operator-raised parallel slots value to whatifSizing", async () => {
    const user = userEvent.setup();
    render(
      <ToastProvider>
        <Sizing active />
      </ToastProvider>,
    );

    await user.click(await screen.findByText("Add model"));
    const input = screen.getByLabelText("Parallel slots");
    await user.clear(input);
    await user.type(input, "4");
    await user.click(screen.getByText("Check fit"));

    await waitFor(() => expect(api.whatifSizing).toHaveBeenCalled());
    const [, nParallel] = vi.mocked(api.whatifSizing).mock.calls[0];
    expect(nParallel).toBe(4);
  });
});
