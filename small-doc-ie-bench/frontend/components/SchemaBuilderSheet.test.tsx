import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const { createDynamicSchema } = vi.hoisted(() => ({
  createDynamicSchema: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  createDynamicSchema,
}));

import { SchemaBuilderSheet } from "./SchemaBuilderSheet";

describe("SchemaBuilderSheet", () => {
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

  it("does not nest its <form> inside an ancestor <form> (Sheet portals to document.body)", () => {
    // Reproduces the real bug: Playground wraps its whole page in its own
    // <form>, and opens this sheet from inside it. HTML forbids a <form>
    // descendant of a <form> -- without Sheet's portal, this sheet's own
    // <form> would land right inside the outer one.
    const { container } = render(
      <form data-testid="outer-form">
        <SchemaBuilderSheet open onClose={() => {}} onCreated={() => {}} />
      </form>,
    );
    const outerForm = container.querySelector('[data-testid="outer-form"]');
    const innerForm = screen.getByRole("button", { name: "Save schema" }).closest("form");
    expect(innerForm).not.toBeNull();
    expect(outerForm?.contains(innerForm)).toBe(false);
  });
});
