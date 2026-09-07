import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

const { reload, remove, update, create } = vi.hoisted(() => ({
  reload: vi.fn(), remove: vi.fn(), update: vi.fn(), create: vi.fn(),
}));
vi.mock("@/lib/api", async (original) => ({
  ...(await original<typeof import("@/lib/api")>()),
  deleteDynamicSchema: remove, updateDynamicSchema: update, createDynamicSchema: create,
}));
vi.mock("@/lib/useAsync", () => ({ useAsync: (key: string) => ({
  data: key === "dynamic-schemas" ? [{ name: "resume", spec: {
    document_type: "resume", fields: [{ name: "experience", type: "list", fields: [{ name: "company", type: "string" }] }],
  }, created_at: "2026-09-07T00:00:00Z", updated_at: "2026-09-07T00:00:00Z" }] : ["invoice"],
  error: null, loading: false, reload,
}) }));
import { Schemas } from "./Schemas";

beforeEach(() => { vi.clearAllMocks(); });

it("edits a saved schema and refreshes schema choices", async () => {
  update.mockResolvedValue({ name: "resume" });
  render(<Schemas />);
  await userEvent.click(screen.getByRole("button", { name: "Edit resume" }));
  await userEvent.type(screen.getByRole("textbox", { name: "Description for experience" }), "Every job");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(update).toHaveBeenCalledWith("resume", expect.objectContaining({ document_type: "resume" })));
  expect(reload).toHaveBeenCalled();
});

it("duplicates into a new saved name", async () => {
  create.mockResolvedValue({ name: "resume_copy" });
  render(<Schemas />);
  await userEvent.click(screen.getByRole("button", { name: "Duplicate resume" }));
  expect(screen.getByPlaceholderText("purchase_order")).toHaveValue("resume_copy");
  await userEvent.click(screen.getByRole("button", { name: "Save schema" }));
  expect(create).toHaveBeenCalledWith(expect.objectContaining({ document_type: "resume_copy" }));
  expect(update).not.toHaveBeenCalled();
});

it("confirms deletion and leaves failures visible for retry", async () => {
  remove.mockRejectedValueOnce(new Error("Schema is unavailable")).mockResolvedValueOnce({ deleted: "resume" });
  render(<Schemas />);
  await userEvent.click(screen.getByRole("button", { name: "Delete resume" }));
  expect(remove).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Delete schema" }));
  expect(await screen.findByText("Schema is unavailable")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Delete schema" }));
  await waitFor(() => expect(reload).toHaveBeenCalled());
  expect(remove).toHaveBeenLastCalledWith("resume");
});
