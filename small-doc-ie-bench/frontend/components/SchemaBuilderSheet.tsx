"use client";

import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import {
  ApiError,
  ApiUnavailable,
  createDynamicSchema,
  type DynamicFieldSpec,
  type DynamicFieldType,
  type DynamicSchemaSpec,
} from "@/lib/api";
import { Alert, Button, Field, Select, Sheet, TextInput } from "./ui";

interface FieldRow {
  name: string;
  type: DynamicFieldType;
  description: string;
  subFields: FieldRow[];
}

function emptyFieldRow(): FieldRow {
  return { name: "", type: "string", description: "", subFields: [] };
}

const SCALAR_FIELD_TYPES: DynamicFieldType[] = ["string", "date", "number", "money"];

export function SchemaBuilderSheet({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (name: string) => void;
}) {
  const [documentType, setDocumentType] = useState("");
  const [fields, setFields] = useState<FieldRow[]>([emptyFieldRow()]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDocumentType("");
      setFields([emptyFieldRow()]);
      setError(null);
    }
  }, [open]);

  function updateField(index: number, patch: Partial<FieldRow>) {
    setFields((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  function updateSubField(index: number, subIndex: number, patch: Partial<FieldRow>) {
    setFields((rows) =>
      rows.map((row, i) =>
        i === index
          ? {
              ...row,
              subFields: row.subFields.map((sub, j) =>
                j === subIndex ? { ...sub, ...patch } : sub,
              ),
            }
          : row,
      ),
    );
  }

  function toSpec(): DynamicSchemaSpec {
    const toFieldSpec = (row: FieldRow): DynamicFieldSpec => ({
      name: row.name.trim(),
      type: row.type,
      ...(row.description.trim() ? { description: row.description.trim() } : {}),
      ...(row.type === "list"
        ? {
            fields: row.subFields
              .filter((sub) => sub.name.trim())
              .map((sub) => ({
                name: sub.name.trim(),
                type: sub.type,
                ...(sub.description.trim()
                  ? { description: sub.description.trim() }
                  : {}),
              })),
          }
        : {}),
    });
    return {
      document_type: documentType.trim(),
      fields: fields.filter((field) => field.name.trim()).map(toFieldSpec),
    };
  }

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    if (!documentType.trim()) {
      setError("Document type is required.");
      return;
    }
    const namedFields = fields.filter((field) => field.name.trim());
    if (namedFields.length === 0) {
      setError("At least one named field is required.");
      return;
    }
    const emptyListField = namedFields.find(
      (field) =>
        field.type === "list" && !field.subFields.some((sub) => sub.name.trim()),
    );
    if (emptyListField) {
      setError(
        `"${emptyListField.name.trim()}" is a list field and needs at least one named sub-field.`,
      );
      return;
    }

    setSubmitting(true);
    try {
      const saved = await createDynamicSchema(toSpec());
      onCreated(saved.name);
      onClose();
    } catch (err) {
      const message =
        err instanceof ApiUnavailable
          ? "Dynamic schemas aren't available on this server."
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Failed to save schema.";
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Sheet open={open} onClose={onClose}>
      <form onSubmit={onSubmit} className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-foreground">New schema</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Define a reusable extraction schema once, then select it anywhere structured
            output is configured. A list field supports one flat level of sub-fields;
            nested objects are not exposed by this form.
          </p>
        </div>

        <Field label="Document type" required hint="Lower snake_case, e.g. purchase_order.">
          <TextInput
            value={documentType}
            onChange={(event) => setDocumentType(event.target.value)}
            placeholder="purchase_order"
          />
        </Field>

        <div className="space-y-3">
          {fields.map((field, index) => (
            <div key={index} className="space-y-2 rounded-md border border-border p-3">
              <div className="flex items-center gap-2">
                <TextInput
                  className="flex-1"
                  value={field.name}
                  onChange={(event) => updateField(index, { name: event.target.value })}
                  placeholder="field_name"
                />
                <Select
                  className="w-32"
                  value={field.type}
                  aria-label={`Type for ${field.name || `field ${index + 1}`}`}
                  onChange={(event) => {
                    const type = event.target.value as DynamicFieldType;
                    updateField(index, {
                      type,
                      subFields:
                        type === "list" && field.subFields.length === 0
                          ? [emptyFieldRow()]
                          : field.subFields,
                    });
                  }}
                >
                  <option value="string">string</option>
                  <option value="date">date</option>
                  <option value="number">number</option>
                  <option value="money">money</option>
                  <option value="list">list</option>
                </Select>
                <button
                  type="button"
                  onClick={() =>
                    setFields((rows) => rows.filter((_, rowIndex) => rowIndex !== index))
                  }
                  className="shrink-0 text-muted-foreground hover:text-red-500"
                  aria-label="Remove field"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
              <TextInput
                value={field.description}
                onChange={(event) =>
                  updateField(index, { description: event.target.value })
                }
                placeholder="Description (optional)"
              />
              {field.type === "list" && (
                <div className="ml-4 space-y-1.5 border-l border-border pl-3">
                  <p className="text-xs text-muted-foreground">
                    Sub-fields (one flat level):
                  </p>
                  {field.subFields.map((sub, subIndex) => (
                    <div key={subIndex} className="flex items-center gap-2">
                      <TextInput
                        className="flex-1"
                        value={sub.name}
                        onChange={(event) =>
                          updateSubField(index, subIndex, { name: event.target.value })
                        }
                        placeholder="sub_field_name"
                      />
                      <Select
                        className="w-28"
                        value={sub.type}
                        aria-label={`Type for ${sub.name || `sub-field ${subIndex + 1}`}`}
                        onChange={(event) =>
                          updateSubField(index, subIndex, {
                            type: event.target.value as DynamicFieldType,
                          })
                        }
                      >
                        {SCALAR_FIELD_TYPES.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </Select>
                      <button
                        type="button"
                        onClick={() =>
                          updateField(index, {
                            subFields: field.subFields.filter(
                              (_, rowIndex) => rowIndex !== subIndex,
                            ),
                          })
                        }
                        className="shrink-0 text-muted-foreground hover:text-red-500"
                        aria-label="Remove sub-field"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      updateField(index, {
                        subFields: [...field.subFields, emptyFieldRow()],
                      })
                    }
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                  >
                    <Plus className="h-3 w-3" /> Add sub-field
                  </button>
                </div>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={() => setFields((rows) => [...rows, emptyFieldRow()])}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            <Plus className="h-3.5 w-3.5" /> Add field
          </button>
        </div>

        {error && <Alert tone="err">{error}</Alert>}

        <Button type="submit" loading={submitting}>
          Save schema
        </Button>
      </form>
    </Sheet>
  );
}
