import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HfSearchSeed } from "./HfSearchSeed";
import { ToastProvider } from "../../Toast";
import * as api from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    searchHf: vi.fn(),
    inspectHf: vi.fn(),
    seedHf: vi.fn(),
  };
});

function renderSearch(existingStoreNames: string[] = []) {
  return render(
    <ToastProvider>
      <HfSearchSeed
        families={[{ name: "openai_chat" }]}
        existingStoreNames={existingStoreNames}
        onSeeded={vi.fn()}
      />
    </ToastProvider>,
  );
}

describe("HfSearchSeed preflight", () => {
  beforeEach(() => {
    vi.mocked(api.searchHf).mockReset();
    vi.mocked(api.inspectHf).mockReset();
    vi.mocked(api.seedHf).mockReset();
  });

  it("shows exact estimates and lets a smaller fitting quant unblock seeding", async () => {
    vi.mocked(api.searchHf).mockResolvedValue([{ id: "owner/model-GGUF" }]);
    vi.mocked(api.inspectHf).mockResolvedValue({
      repo: "owner/model-GGUF",
      architecture: "qwen2",
      verdict: "supported",
      readiness: "blocked",
      family: "openai_chat",
      runtime: "llama.cpp",
      reason: "architecture 'qwen2' → openai_chat",
      suggested_name: "model",
      recommended_quant: "Q4_K_M",
      context_length: 8192,
      fits_node: false,
      blockers: [{ code: "insufficient_memory", message: "The recommended quant is too large." }],
      recommendations: ["Choose a smaller artifact that fits this node: Q4_K_S."],
      artifact_options: [
        {
          kind: "gguf",
          label: "Q4_K_M",
          quant: "Q4_K_M",
          filename: "model-Q4_K_M.gguf",
          required_files: [
            { filename: "model-Q4_K_M.gguf", role: "model", size_bytes: 2_000_000_000 },
          ],
          download_size_bytes: 2_000_000_000,
          estimated_ram_bytes: 3_073_741_824,
          node_available_bytes: 1_800_000_000,
          fits_node: false,
          recommended: true,
        },
        {
          kind: "gguf",
          label: "Q4_K_S",
          quant: "Q4_K_S",
          filename: "model-Q4_K_S.gguf",
          required_files: [
            { filename: "model-Q4_K_S.gguf", role: "model", size_bytes: 500_000_000 },
          ],
          download_size_bytes: 500_000_000,
          estimated_ram_bytes: 1_573_741_824,
          node_available_bytes: 1_800_000_000,
          fits_node: true,
          recommended: false,
        },
      ],
    });

    renderSearch();
    await userEvent.type(
      screen.getByPlaceholderText(/Search models/),
      "model",
    );
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await userEvent.click(await screen.findByText("owner/model-GGUF"));

    expect(await screen.findByText("Does not fit")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Download & seed/ })).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: /Q4_K_S · fits/ }));

    expect(screen.getByText("Fits")).toBeInTheDocument();
    expect(screen.getAllByText("476.8 MB")).toHaveLength(2);
    expect(screen.getByRole("button", { name: /Download & seed/ })).toBeEnabled();
  });

  it("requires an explicit trust-remote-code family selection", async () => {
    vi.mocked(api.searchHf).mockResolvedValue([{ id: "owner/custom-model" }]);
    vi.mocked(api.inspectHf).mockResolvedValue({
      repo: "owner/custom-model",
      verdict: "supported",
      readiness: "blocked",
      family: "transformers",
      runtime: "transformers",
      reason: "safetensors fallback",
      needs_trust_remote_code: true,
      blockers: [
        {
          code: "remote_code_approval_required",
          message: "This checkpoint executes repository code.",
        },
      ],
      artifact_options: [
        {
          kind: "snapshot",
          label: "safetensors snapshot",
          quant: null,
          required_files: [
            { filename: "model.safetensors", role: "weights", size_bytes: 100_000_000 },
          ],
          download_size_bytes: 100_000_000,
          estimated_ram_bytes: 1_173_741_824,
          node_available_bytes: 2_000_000_000,
          fits_node: true,
          recommended: true,
        },
      ],
    });

    render(
      <ToastProvider>
        <HfSearchSeed
          families={[
            { name: "transformers", transformers_runtime: true },
            {
              name: "transformers_trust_remote_code",
              transformers_runtime: true,
              trust_remote_code: true,
            },
          ]}
          existingStoreNames={[]}
          onSeeded={vi.fn()}
        />
      </ToastProvider>,
    );
    await userEvent.type(screen.getByPlaceholderText(/Search models/), "custom");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await userEvent.click(await screen.findByText("owner/custom-model"));

    expect(screen.getByRole("button", { name: /Download & seed/ })).toBeDisabled();
    await userEvent.selectOptions(
      screen.getByRole("combobox"),
      "transformers_trust_remote_code",
    );
    expect(screen.getByRole("button", { name: /Download & seed/ })).toBeEnabled();
  });

  it("asks before reusing an existing store name", async () => {
    vi.mocked(api.searchHf).mockResolvedValue([{ id: "owner/model-GGUF" }]);
    vi.mocked(api.inspectHf).mockResolvedValue({
      repo: "owner/model-GGUF",
      verdict: "supported",
      readiness: "ready",
      family: "openai_chat",
      reason: "supported",
      suggested_name: "model",
    });
    vi.mocked(api.seedHf).mockResolvedValue({
      event_ids: ["event-1"],
      channel: "seed:model",
      topics: ["status", "result", "error"],
    });

    renderSearch(["model"]);
    await userEvent.type(screen.getByPlaceholderText(/Search models/), "model");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await userEvent.click(await screen.findByText("owner/model-GGUF"));
    await userEvent.click(screen.getByRole("button", { name: /Download & seed/ }));

    expect(screen.getByRole("dialog", { name: "This store name already exists" })).toBeVisible();
    expect(screen.getByText("No new files will be downloaded or replaced.")).toBeVisible();
    expect(api.seedHf).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: "Use existing model" }));
    expect(api.seedHf).toHaveBeenCalledOnce();
  });
});
