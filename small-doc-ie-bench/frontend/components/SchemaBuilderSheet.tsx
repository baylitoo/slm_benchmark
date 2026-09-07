"use client";

import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import {
  createDynamicSchema, updateDynamicSchema,
  type DynamicFieldSpec, type DynamicFieldType, type DynamicSchemaSpec,
} from "@/lib/api";
import { Alert, Button, Field, Select, Sheet, TextInput } from "./ui";
import { T, useI18n } from "@/lib/i18n";

const emptyField = (): DynamicFieldSpec => ({ name: "", type: "string" });
const NAME_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;
const RESERVED = new Set(["document_type", "extraction_notes"]);
const TYPES: { value: DynamicFieldType; label: string }[] = [
  { value: "string", label: "Text" }, { value: "date", label: "Date" },
  { value: "number", label: "Number" }, { value: "money", label: "Money" },
  { value: "object", label: "Object (group of fields)" },
  { value: "list", label: "List of objects" },
];
const isContainer = (field: DynamicFieldSpec) => field.type === "object" || field.type === "list";

export const RESUME_SCHEMA: DynamicSchemaSpec = {
  document_type: "adbi_resume",
  fields: [
    { name: "name", type: "string" },
    { name: "title", type: "string" },
    { name: "years_experience", type: "number" },
    { name: "contact", type: "object", fields: [
      { name: "email", type: "string" }, { name: "phone", type: "string" },
      { name: "linkedin", type: "string" }, { name: "github", type: "string" },
      { name: "location", type: "string" },
    ] },
    { name: "experience", type: "list", description: "All work experiences in the document.", fields: [
      { name: "company", type: "string" }, { name: "title", type: "string" },
      { name: "start_date", type: "date" }, { name: "end_date", type: "date" },
      { name: "location", type: "string" },
      { name: "description", type: "string", description: "Complete responsibilities and achievements for this experience, including text spanning multiple lines." },
      { name: "env_technique", type: "string", description: "Stack technique / environnement technologique de la mission" },
    ] },
    { name: "education", type: "list", description: "All education entries in the document.", fields: [
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
  ],
};

function cleanFields(fields: DynamicFieldSpec[]): DynamicFieldSpec[] {
  return fields.map((field) => ({
    name: field.name.trim(), type: field.type,
    ...(field.description?.trim() ? { description: field.description.trim() } : {}),
    ...(isContainer(field) ? { fields: cleanFields(field.fields ?? []) } : {}),
  }));
}

function validateFields(fields: DynamicFieldSpec[], parent = "Schema"): string | null {
  if (!fields.length) return `${parent} needs at least one named field.`;
  if (fields.length > 40) return `${parent} supports at most 40 fields.`;
  const names = new Set<string>();
  for (const field of fields) {
    const path = `${parent}.${field.name || "(unnamed)"}`;
    if (!NAME_PATTERN.test(field.name)) return `${path}: names must be lower snake_case, at most 64 characters.`;
    if (RESERVED.has(field.name)) return `${path}: "${field.name}" is a reserved name.`;
    if (names.has(field.name)) return `${path}: field names must be unique within their object.`;
    names.add(field.name);
    if ((field.description?.length ?? 0) > 300) return `${path}: descriptions must be at most 300 characters.`;
    if (isContainer(field)) {
      const error = validateFields(field.fields ?? [], path);
      if (error) return error;
    }
  }
  return null;
}

/** A type declaration, not a sample result with a fixed number of items. */
export function schemaShape(fields: DynamicFieldSpec[], indent = 0): string {
  const pad = "  ".repeat(indent);
  return fields.map((field) => {
    const name = field.name || "field_name";
    if (!isContainer(field)) return `${pad}${name}: ${field.type}`;
    const shape = schemaShape(field.fields ?? [], indent + 1);
    return `${pad}${name}: ${field.type === "list" ? "list<" : ""}{\n${shape}\n${pad}}${field.type === "list" ? ">" : ""}`;
  }).join("\n");
}

function FieldEditor({ fields, onChange, path = "", depth = 0 }: {
  fields: DynamicFieldSpec[]; onChange: (fields: DynamicFieldSpec[]) => void;
  path?: string; depth?: number;
}) {
  const { t } = useI18n();
  const patch = (index: number, change: Partial<DynamicFieldSpec>) =>
    onChange(fields.map((field, i) => i === index ? { ...field, ...change } : field));
  return <div className="space-y-3">
    {fields.map((field, index) => {
      const fieldPath = [path, field.name || `field ${index + 1}`].filter(Boolean).join(".");
      return <div key={index} className="space-y-2 rounded-md border border-border bg-card p-3">
        <div className="flex flex-wrap items-center gap-2">
          <TextInput className="min-w-0 flex-1" value={field.name}
            aria-label={`Name for ${fieldPath}`} placeholder={depth ? "sub_field_name" : "field_name"}
            onChange={(event) => patch(index, { name: event.target.value })} />
          <Select className="w-44" value={field.type} aria-label={`Type for ${fieldPath}`}
            onChange={(event) => {
              const type = event.target.value as DynamicFieldType;
              patch(index, { type, fields: type === "list" || type === "object"
                ? (field.fields?.length ? field.fields : [emptyField()]) : undefined });
            }}>
            {TYPES.map(({ value, label }) => <option key={value} value={value}>{t(label)}</option>)}
          </Select>
          <button type="button" onClick={() => onChange(fields.filter((_, i) => i !== index))}
            className="text-muted-foreground hover:text-red-500" aria-label={`Remove ${fieldPath}`}>
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
        <TextInput value={field.description ?? ""} maxLength={300}
          aria-label={`Description for ${fieldPath}`} placeholder="Description (optional)"
          onChange={(event) => patch(index, { description: event.target.value })} />
        {isContainer(field) && <div className="space-y-2 border-l-2 border-border pl-3">
          <p className="text-xs text-muted-foreground"><T>{field.type === "list"
            ? "Fields in each item. Define these once; the model returns zero or more items from the document."
            : "Fields in this object."}</T></p>
          <FieldEditor fields={field.fields ?? []} path={fieldPath} depth={depth + 1}
            onChange={(children) => patch(index, { fields: children })} />
        </div>}
      </div>;
    })}
    <button type="button" disabled={fields.length >= 40}
      onClick={() => onChange([...fields, emptyField()])}
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
      aria-label={path ? `Add field to ${path}` : "Add field"}>
      <Plus className="h-3.5 w-3.5" /><T>{path ? "Add child field" : "Add field"}</T>
    </button>
  </div>;
}

export function SchemaBuilderSheet({ open, onClose, onCreated, initialSpec, editName }: {
  open: boolean; onClose: () => void; onCreated: (name: string) => void;
  initialSpec?: DynamicSchemaSpec; editName?: string;
}) {
  const [documentType, setDocumentType] = useState("");
  const [fields, setFields] = useState<DynamicFieldSpec[]>([emptyField()]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (open) {
      setDocumentType(initialSpec?.document_type ?? "");
      setFields(initialSpec ? structuredClone(initialSpec.fields) : [emptyField()]);
      setError(null);
    }
  }, [open, initialSpec]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const name = documentType.trim();
    const cleaned = cleanFields(fields);
    const validation = !NAME_PATTERN.test(name)
      ? "Document type must be lower snake_case, start with a letter, and be at most 64 characters."
      : validateFields(cleaned);
    setError(validation);
    if (validation) return;
    setSubmitting(true);
    try {
      const spec = { document_type: name, fields: cleaned };
      const saved = editName ? await updateDynamicSchema(editName, spec) : await createDynamicSchema(spec);
      onCreated(saved.name);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save schema.");
    } finally { setSubmitting(false); }
  }

  return <Sheet open={open} onClose={() => { if (!submitting) onClose(); }}>
    <form onSubmit={onSubmit} className="space-y-5">
      <div>
        <h2 className="text-sm font-semibold text-foreground"><T>{editName ? "Edit schema" : "New schema"}</T></h2>
        <p className="mt-1 text-xs text-muted-foreground"><T>
          Define what to extract. Objects group fields; lists repeat an item structure as many times as needed.
        </T></p>
      </div>
      <fieldset disabled={submitting} className="space-y-5">
        {!editName && !initialSpec && <Button type="button" variant="secondary" size="sm"
          onClick={() => { setDocumentType(RESUME_SCHEMA.document_type); setFields(structuredClone(RESUME_SCHEMA.fields)); setError(null); }}>
          <T>Use resume starter</T>
        </Button>}
        <Field label="Document type" required hint={editName
          ? "The saved name stays the same so existing references continue to work."
          : "Lower snake_case, e.g. purchase_order."}>
          <TextInput value={documentType} disabled={!!editName} placeholder="purchase_order"
            onChange={(event) => setDocumentType(event.target.value)} />
        </Field>
        <FieldEditor fields={fields} onChange={setFields} />
        <details className="rounded-md border border-border p-3">
          <summary className="cursor-pointer text-xs font-medium"><T>Structure preview</T></summary>
          <pre className="mt-2 overflow-x-auto text-xs text-muted-foreground">{`${documentType || "document"}: {\n${schemaShape(fields, 1)}\n}`}</pre>
        </details>
        {editName && <p className="text-xs text-muted-foreground"><T>
          Changes apply the next time this saved schema is used. Past extraction results are unchanged.
        </T></p>}
        {error && <Alert tone="err">{error}</Alert>}
        <Button type="submit" loading={submitting}><T>{editName ? "Save changes" : "Save schema"}</T></Button>
      </fieldset>
    </form>
  </Sheet>;
}
