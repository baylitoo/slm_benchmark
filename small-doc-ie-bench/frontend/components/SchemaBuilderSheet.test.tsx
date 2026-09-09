import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { createDynamicSchema, updateDynamicSchema } = vi.hoisted(() => ({
  createDynamicSchema: vi.fn(),
  updateDynamicSchema: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  createDynamicSchema,
  updateDynamicSchema,
}));

import { SchemaBuilderSheet } from "./SchemaBuilderSheet";

describe("SchemaBuilderSheet", () => {
  beforeEach(() => {
    createDynamicSchema.mockReset();
    updateDynamicSchema.mockReset();
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


it("creates the complete ADBI resume starter with variable-length nested lists", async () => {
  createDynamicSchema.mockResolvedValue({ name: "adbi_resume" });
  render(<SchemaBuilderSheet open onClose={vi.fn()} onCreated={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: "Use resume starter" }));
  expect(screen.getByPlaceholderText("purchase_order")).toHaveValue("adbi_resume");
  await userEvent.click(screen.getByRole("button", { name: "Save schema" }));
  const spec = createDynamicSchema.mock.calls.at(-1)![0];
  expect(spec.document_type).toBe("adbi_resume");
  // Strip optional guidance to verify the exact field/type contract on the wire.
  const structure = (fields: import("@/lib/api").DynamicFieldSpec[]): unknown => fields.map((field) => ({
    name: field.name, type: field.type,
    ...(field.fields ? { fields: structure(field.fields) } : {}),
  }));
  expect(structure(spec.fields)).toEqual([
    { name: "name", type: "string" },
    { name: "title", type: "string" },
    { name: "years_experience", type: "string" },
    { name: "contact", type: "object", fields: [
      { name: "email", type: "string" }, { name: "phone", type: "string" },
      { name: "linkedin", type: "string" }, { name: "github", type: "string" },
      { name: "location", type: "string" },
    ] },
    { name: "experience", type: "list", fields: [
      { name: "company", type: "string" }, { name: "title", type: "string" },
      { name: "start_date", type: "date" }, { name: "end_date", type: "date" },
      { name: "location", type: "string" }, { name: "description", type: "string" },
      { name: "env_technique", type: "string" },
    ] },
    { name: "education", type: "list", fields: [
      { name: "degree", type: "string" }, { name: "institution", type: "string" },
      { name: "year", type: "string" },
    ] },
    { name: "skills", type: "list", fields: [
      { name: "category", type: "string" },
      { name: "items", type: "list", fields: [{ name: "item", type: "string" }] },
    ] },
    { name: "languages", type: "list", fields: [
      { name: "language", type: "string" }, { name: "level", type: "string" },
    ] },
    { name: "certifications", type: "list", fields: [
      { name: "name", type: "string" }, { name: "issuer", type: "string" },
      { name: "year", type: "string" },
    ] },
    { name: "interests", type: "list", fields: [{ name: "interest", type: "string" }] },
  ]);
});


it("edits nested objects and lists without flattening their item structure", async () => {
  updateDynamicSchema.mockResolvedValue({ name: "resume" });
  const initialSpec = { document_type: "resume", fields: [{ name: "profile", type: "object" as const, fields: [
    { name: "experience", type: "list" as const, description: "All jobs", fields: [
      { name: "company", type: "string" as const, description: "Employer" },
    ] },
  ] }] };
  render(<SchemaBuilderSheet open initialSpec={initialSpec} editName="resume" onClose={vi.fn()} onCreated={vi.fn()} />);
  expect(screen.getByPlaceholderText("purchase_order")).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "Add field to profile.experience" }));
  await userEvent.type(screen.getByRole("textbox", { name: "Name for profile.experience.field 2" }), "projects");
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "Type for profile.experience.projects" }), "list");
  await userEvent.type(screen.getByRole("textbox", { name: "Name for profile.experience.projects.field 1" }), "title");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(updateDynamicSchema).toHaveBeenCalledWith("resume", { document_type: "resume", fields: [
    { name: "profile", type: "object", fields: [
      { name: "experience", type: "list", description: "All jobs", fields: [
        { name: "company", type: "string", description: "Employer" },
        { name: "projects", type: "list", fields: [{ name: "title", type: "string" }] },
      ] },
    ] },
  ] });
});

it("rejects duplicate names inside a nested item", async () => {
  createDynamicSchema.mockClear();
  render(<SchemaBuilderSheet open onClose={vi.fn()} onCreated={vi.fn()} initialSpec={{
    document_type: "resume", fields: [{ name: "experience", type: "list", fields: [
      { name: "company", type: "string" }, { name: "company", type: "string" },
    ] }],
  }} />);
  await userEvent.click(screen.getByRole("button", { name: "Save schema" }));
  expect(await screen.findByText(/must be unique/)).toBeInTheDocument();
  expect(createDynamicSchema).not.toHaveBeenCalled();
});
