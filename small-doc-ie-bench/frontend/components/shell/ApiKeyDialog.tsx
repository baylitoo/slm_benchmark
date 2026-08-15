"use client";

// The operator's way to supply the X-API-Key the backend already checks
// (TenantQuotaManager.authenticate, security.py) once AUTH_REQUIRED=true is
// set for a shared/networked deployment. No login route, no session — this
// is a config value stored in THIS browser (lib/apiKey.ts), edited from a
// small top-bar affordance since the Studio has no Settings section to host
// a whole page for it.

import { useState } from "react";
import { KeyRound } from "lucide-react";
import { getApiKey, setApiKey, useHasApiKey } from "@/lib/apiKey";
import { Button, Dialog, Field, StatusDot, TextInput } from "../ui";

export function ApiKeyButton() {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const hasKey = useHasApiKey();

  function openDialog() {
    setDraft(getApiKey() ?? "");
    setOpen(true);
  }

  function save() {
    setApiKey(draft);
    setOpen(false);
  }

  function clear() {
    setApiKey(null);
    setDraft("");
    setOpen(false);
  }

  return (
    <>
      <button
        type="button"
        onClick={openDialog}
        aria-label="API key"
        title={hasKey ? "API key set" : "No API key set"}
        className="relative grid h-9 w-9 place-items-center rounded-md border border-border bg-card text-muted-foreground transition hover:bg-muted hover:text-foreground"
      >
        <KeyRound className="h-4 w-4" />
        <StatusDot
          tone={hasKey ? "ok" : "neutral"}
          className="absolute -right-0.5 -top-0.5 ring-2 ring-card"
        />
      </button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="API key"
        subtitle="Sent as X-API-Key on every request to this backend."
        footer={
          <>
            <Button variant="secondary" size="sm" onClick={clear} disabled={!hasKey && !draft}>
              Clear
            </Button>
            <Button size="sm" onClick={save}>
              Save
            </Button>
          </>
        }
      >
        <Field
          label="API key"
          htmlFor="api-key-input"
          hint="Stored only in this browser (localStorage), never sent anywhere but this backend. Leave empty for a server running with AUTH_REQUIRED=false."
        >
          <ApiKeyInput id="api-key-input" value={draft} onChange={setDraft} onSave={save} />
        </Field>
      </Dialog>
    </>
  );
}

/** Password-masked input that submits on Enter. */
function ApiKeyInput({
  id,
  value,
  onChange,
  onSave,
}: {
  id: string;
  value: string;
  onChange: (v: string) => void;
  onSave: () => void;
}) {
  return (
    <TextInput
      id={id}
      type="password"
      autoComplete="off"
      spellCheck={false}
      placeholder="sk-..."
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSave();
      }}
    />
  );
}

