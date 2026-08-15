import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Review } from "@/components/Review";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { ReviewEvidenceView, ReviewTaskView } from "@/lib/api";

function renderReview() {
  return render(
    <ToastProvider>
      <Review />
    </ToastProvider>,
  );
}

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listReviews: vi.fn(),
    claimReview: vi.fn(),
    correctReview: vi.fn(),
    releaseReview: vi.fn(),
    approveReview: vi.fn(),
    rejectReview: vi.fn(),
    getReviewEvidence: vi.fn(),
  };
});

function makeTask(overrides: Partial<ReviewTaskView> = {}): ReviewTaskView {
  return {
    id: 1,
    source_request_id: "req-1",
    schema_name: "invoice",
    model_profile: "model-a",
    document_hash: "sha256:abc",
    status: "pending",
    priority: 12.5,
    priority_reasons: [{ code: "low_confidence", score: 0.5, detail: "mean confidence low" }],
    original_prediction: {
      invoice_number: { value: "WRONG", confidence: 0.2, evidence_ids: ["b1"] },
      total_ttc: { amount: "100.00", currency: "EUR", confidence: 0.9, evidence_ids: ["b2"] },
    },
    latest_prediction: {
      invoice_number: { value: "WRONG", confidence: 0.2, evidence_ids: ["b1"] },
      total_ttc: { amount: "100.00", currency: "EUR", confidence: 0.9, evidence_ids: ["b2"] },
    },
    validation_errors: [],
    validation_warnings: [],
    dynamic_schema: null,
    metadata: {},
    claimed_by: null,
    claim_expires_at: null,
    version: 1,
    created_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    updated_at: new Date("2026-01-01T00:00:00Z").toISOString(),
    decided_at: null,
    decided_by: null,
    decision_comment: null,
    corrections: [],
    events: [],
    suggested_corrections: [],
    evidence_available: true,
    ...overrides,
  };
}

describe("Review", () => {
  beforeEach(() => {
    vi.mocked(api.listReviews).mockReset();
    vi.mocked(api.correctReview).mockReset();
    vi.mocked(api.getReviewEvidence).mockReset();
    vi.mocked(api.getReviewEvidence).mockResolvedValue({
      task_id: 1,
      retention: "ocr_text",
      blocks: [],
    });
  });

  it("renders schema-generated field rows pre-filled with their current values", async () => {
    vi.mocked(api.listReviews).mockResolvedValue([makeTask()]);
    renderReview();

    await userEvent.click(await screen.findByText("req-1"));

    expect(await screen.findByDisplayValue("WRONG")).toBeInTheDocument();
    expect(screen.getByDisplayValue("100.00")).toBeInTheDocument();
    expect(screen.getByDisplayValue("EUR")).toBeInTheDocument();
  });

  it("fetches evidence and highlights the block matching the selected field", async () => {
    vi.mocked(api.listReviews).mockResolvedValue([makeTask()]);
    vi.mocked(api.getReviewEvidence).mockResolvedValue({
      task_id: 1,
      retention: "ocr_text",
      blocks: [
        { id: "b1", text: "WRONG-ON-DOC", page: 1, bbox: null, source: "pdf_text", confidence: 0.9 },
        { id: "b2", text: "100.00 EUR", page: 1, bbox: null, source: "pdf_text", confidence: 0.9 },
      ],
    } satisfies ReviewEvidenceView);
    renderReview();

    await userEvent.click(await screen.findByText("req-1"));
    // Row selection is driven by clicking the row, not the input inside it
    // (the input stops propagation so cursor placement doesn't reselect).
    await userEvent.click(screen.getByText("invoice_number"));

    expect(await screen.findByText("WRONG-ON-DOC")).toBeInTheDocument();
    expect(api.getReviewEvidence).toHaveBeenCalledWith(1);
  });

  it("submits only the field that actually changed, diffed against its current value", async () => {
    const task = makeTask({ status: "claimed", claimed_by: "studio-operator" });
    vi.mocked(api.listReviews).mockResolvedValue([task]);
    vi.mocked(api.correctReview).mockResolvedValue({ ...task, version: 2 });
    renderReview();

    await userEvent.click(await screen.findByText("req-1"));
    const input = await screen.findByDisplayValue("WRONG");
    await userEvent.clear(input);
    await userEvent.type(input, "INV-1");

    await userEvent.click(screen.getByRole("button", { name: /Submit 1 correction/ }));

    expect(api.correctReview).toHaveBeenCalledWith(1, 1, [
      { field_path: "invoice_number.value", value: "INV-1" },
    ]);
  });

  it("disables the Submit button until an edit actually differs from the original", async () => {
    const task = makeTask({ status: "claimed", claimed_by: "studio-operator" });
    vi.mocked(api.listReviews).mockResolvedValue([task]);
    renderReview();

    await userEvent.click(await screen.findByText("req-1"));
    expect(await screen.findByRole("button", { name: /Submit 0 corrections/ })).toBeDisabled();

    const input = screen.getByDisplayValue("WRONG");
    await userEvent.clear(input);
    await userEvent.type(input, "WRONG");

    expect(screen.getByRole("button", { name: /Submit 0 corrections/ })).toBeDisabled();
  });
});
