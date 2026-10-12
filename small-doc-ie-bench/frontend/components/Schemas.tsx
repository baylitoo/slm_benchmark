"use client";

import { useState } from "react";
import { Copy, Pencil, Plus, Trash2 } from "lucide-react";
import {
  deleteDynamicSchema, getSchemaDefinition, listDynamicSchemas, listSchemas,
  type DynamicSchemaSpec, type DynamicSchemaSummary,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { T, useI18n } from "@/lib/i18n";
import { Alert, Badge, Button, Dialog, TextInput } from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { SchemaBuilderSheet, schemaShape } from "./SchemaBuilderSheet";

function message(error: unknown): string {
  return error instanceof Error ? error.message : "Could not load schemas.";
}

function BuiltinDefinition({ name, onClose }: { name: string; onClose: () => void }) {
  const schema = useAsync(`schema-definition:${name}`, () => getSchemaDefinition(name));
  return <Dialog open onClose={onClose} title={name} subtitle="Built-in schema (read-only)">
    {schema.loading && <p><T>Loading…</T></p>}
    {!!schema.error && <Alert tone="err">{message(schema.error)}</Alert>}
    {schema.data && <pre className="max-h-96 overflow-auto text-xs">{JSON.stringify(schema.data, null, 2)}</pre>}
  </Dialog>;
}

export function Schemas() {
  const { t } = useI18n();
  const saved = useAsync("dynamic-schemas", listDynamicSchemas);
  const builtin = useAsync("schemas", listSchemas);
  const [query, setQuery] = useState("");
  const [editor, setEditor] = useState<{ spec?: DynamicSchemaSpec; name?: string } | null>(null);
  const [deleting, setDeleting] = useState<DynamicSchemaSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [builtinName, setBuiltinName] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const matches = (name: string) => name.toLowerCase().includes(query.trim().toLowerCase());
  const rows = (saved.data ?? []).filter((schema) => matches(schema.name));

  async function remove() {
    if (!deleting) return;
    setBusy(true);
    setDeleteError(null);
    try {
      await deleteDynamicSchema(deleting.name);
      setNotice(t("Schema deleted"));
      setDeleting(null);
      saved.reload();
    } catch (error) { setDeleteError(message(error)); }
    finally { setBusy(false); }
  }

  return <div className="space-y-5">
    <PageHeader title="Schemas"
      subtitle="Create and manage the structures your models extract. Define each list item once; the model determines how many items the document contains."
      actions={<Button onClick={() => setEditor({})}><Plus className="h-4 w-4" /><T>New schema</T></Button>} />
    <div className="flex gap-2">
      <TextInput value={query} onChange={(event) => setQuery(event.target.value)}
        placeholder="Search schemas" aria-label={t("Search schemas")} className="max-w-sm" />
      <Button variant="secondary" onClick={() => { saved.reload(); builtin.reload(); }}><T>Refresh</T></Button>
    </div>
    {notice && <p role="status" className="text-sm text-muted-foreground">{notice}</p>}
    <section className="space-y-3" aria-label={t("Saved schemas")}>
      <h2 className="text-sm font-semibold"><T>Saved schemas</T></h2>
      {!!saved.error && <Alert tone="err">{message(saved.error)}</Alert>}
      {saved.loading && <p className="text-sm text-muted-foreground"><T>Loading…</T></p>}
      {!saved.loading && !saved.error && !rows.length && <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
        <T>{query ? "No schemas match your search." : "No saved schemas yet. Create one from scratch or use the resume starter."}</T>
      </div>}
      {rows.map((schema) => <article key={schema.name} className="rounded-md border border-border bg-card p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="font-mono text-sm font-semibold">{schema.name}</h3>
            <p className="mt-1 text-xs text-muted-foreground"><T>Updated</T> {new Date(schema.updated_at).toLocaleString()}</p>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="secondary" aria-label={`Edit ${schema.name}`}
              onClick={() => setEditor({ spec: schema.spec, name: schema.name })}>
              <Pencil className="h-3.5 w-3.5" /><T>Edit</T>
            </Button>
            <Button size="sm" variant="secondary" aria-label={`Duplicate ${schema.name}`}
              onClick={() => setEditor({ spec: { ...schema.spec, document_type: `${schema.name.slice(0, 59)}_copy` } })}>
              <Copy className="h-3.5 w-3.5" /><T>Duplicate</T>
            </Button>
            <Button size="sm" variant="danger" aria-label={`Delete ${schema.name}`}
              onClick={() => { setDeleting(schema); setDeleteError(null); }}>
              <Trash2 className="h-3.5 w-3.5" /><T>Delete</T>
            </Button>
          </div>
        </div>
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-muted-foreground"><T>View structure</T></summary>
          <pre className="mt-2 overflow-x-auto text-xs">{schemaShape(schema.spec.fields)}</pre>
        </details>
      </article>)}
    </section>
    <section className="space-y-3" aria-label={t("Built-in schemas")}>
      <h2 className="text-sm font-semibold"><T>Built-in schemas</T></h2>
      <p className="text-xs text-muted-foreground"><T>Bundled definitions are read-only. Create a saved schema for a custom structure.</T></p>
      {!!builtin.error && <Alert tone="err">{message(builtin.error)}</Alert>}
      <div className="flex flex-wrap gap-2">
        {(builtin.data ?? []).filter(matches).map((name) => <Button key={name} variant="secondary" size="sm" onClick={() => setBuiltinName(name)}>
          {name}<Badge tone="neutral"><T>Built-in</T></Badge>
        </Button>)}
      </div>
    </section>
    <SchemaBuilderSheet open={editor !== null} initialSpec={editor?.spec} editName={editor?.name}
      onClose={() => setEditor(null)} onCreated={(name) => { setNotice(`${name}: ${t("Schema saved")}`); saved.reload(); }} />
    {builtinName && <BuiltinDefinition name={builtinName} onClose={() => setBuiltinName(null)} />}
    {deleting && <Dialog open onClose={() => { if (!busy) setDeleting(null); }} title={`Delete ${deleting.name}?`}
      footer={<><Button variant="secondary" disabled={busy} onClick={() => setDeleting(null)}><T>Cancel</T></Button>
        <Button variant="danger" loading={busy} onClick={() => void remove()}><T>Delete schema</T></Button></>}>
      <p className="text-sm"><T>Agents and requests referencing this saved name will need another schema. Past extraction results will remain available.</T></p>
      {deleteError && <Alert tone="err">{deleteError}</Alert>}
    </Dialog>}
  </div>;
}
