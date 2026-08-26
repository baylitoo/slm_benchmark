import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { createDynamicSchema } = vi.hoisted(() => ({
  createDynamicSchema: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  createDynamicSchema,
}));

import { SchemaBuilderSheet } from "./SchemaBuilderSheet";

describe("SchemaBuilderSheet", () => {
  beforeEach(() => {
    createDynamicSchema.mockReset();
  });

  it("creates a reusable schema and returns its name", async () => {
    createDynamicSchema.mockResolvedValue({ name: "purchase_order" });
    const onCreated = vi.fn();
    const onClose = vi.fn();

    render(
      <SchemaBuilderSheet open onClose={onClose} onCreated={onCreated} />,
    );

    await userEvent.type(screen.getByPlaceholderText("purchase_order"), "purchase_order");
    await userEvent.type(screen.getByPlaceholderText("field_name"), "supplier_name");
    await userEvent.click(screen.getByRole("button", { name: "Save schema" }));

    expect(createDynamicSchema).toHaveBeenCalledWith({
      document_type: "purchase_order",
      fields: [{ name: "supplier_name", type: "string" }],
    });
    expect(onCreated).toHaveBeenCalledWith("purchase_order");
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("rejects a non-snake_case document type before calling the API", async () => {
    render(<SchemaBuilderSheet open onClose={vi.fn()} onCreated={vi.fn()} />);

    await userEvent.type(screen.getByPlaceholderText("purchase_order"), "Purchase Order");
    await userEvent.type(screen.getByPlaceholderText("field_name"), "total");
    await userEvent.click(screen.getByRole("button", { name: "Save schema" }));

    expect(await screen.findByText(/lower snake_case/)).toBeInTheDocument();
    expect(createDynamicSchema).not.toHaveBeenCalled();
  });

  it("rejects a reserved field name before calling the API", async () => {
    render(<SchemaBuilderSheet open onClose={vi.fn()} onCreated={vi.fn()} />);

    await userEvent.type(screen.getByPlaceholderText("purchase_order"), "purchase_order");
    await userEvent.type(screen.getByPlaceholderText("field_name"), "document_type");
    await userEvent.click(screen.getByRole("button", { name: "Save schema" }));

    expect(await screen.findByText(/reserved name/)).toBeInTheDocument();
    expect(createDynamicSchema).not.toHaveBeenCalled();
  });
});
