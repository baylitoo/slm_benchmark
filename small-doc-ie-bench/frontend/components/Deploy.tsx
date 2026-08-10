"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Rocket, PackagePlus, X } from "lucide-react";
import {
  getStore,
  getFamilies,
  getDeployments,
  embeddingDeploymentNames,
  type StoreEntry,
  type DeploymentRecord,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { Button } from "./ui";
import { Sizing } from "./Sizing";
import { Catalog } from "./Catalog";
import { PageHeader } from "./patterns/PageHeader";
import { POLL_MS } from "./deploy/shared";
import { DeploymentsView } from "./deploy/DeploymentsView";
import { ModelsView } from "./deploy/ModelsView";
import { PortsView } from "./deploy/PortsView";
import { DeployForm } from "./deploy/DeployForm";
import { AddModelForm } from "./deploy/seed/AddModelForm";

type SlideOver = null | "deploy" | "seed";

/**
 * Deploy = a table-first serving console with three nav-driven sub-views
 * ("models" / "deployments" / "ports"). The Deploy + Seed forms live in
 * persistently-mounted slide-overs (visibility toggled, never unmounted) so an
 * in-flight deploy/seed and its ResultPanel survive closing the panel or
 * switching views. All pollers, handlers, and API calls are unchanged.
 */
export function Deploy({
  active = true,
  view = "deployments",
}: {
  active?: boolean;
  view?: string;
}) {
  // Auto-refreshing lists — paused when the tab is hidden OR Deploy isn't the
  // active section (every section stays mounted in the shell). Held at the top
  // level so switching sub-views never remounts a poller.
  const store = usePolling<StoreEntry[]>(getStore, POLL_MS, active);
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, POLL_MS, active);
  const families = useAsync("families", getFamilies); // shared SWR key with the Playground

  const [slideOver, setSlideOver] = useState<SlideOver>(null);

  // Deployment names that are embedding models (store family flagged embedding).
  const embeddingNames = useMemo(
    () => embeddingDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );

  const heading =
    view === "catalog"
      ? {
          title: "Catalog",
          subtitle:
            "Browse the Hugging Face Hub — segmented by serving family and support tier, ready to seed.",
        }
      : view === "models"
        ? { title: "Models", subtitle: "The model store you can deploy — GGUFs, encoders, embeddings." }
        : view === "ports"
          ? { title: "Ports", subtitle: "Live port allocation across running deployments." }
          : view === "sizing"
            ? {
                title: "Sizing",
                subtitle:
                  "GET /v1/serving/sizing — how many more instances fit in RAM right now.",
              }
            : { title: "Deployments", subtitle: "GET /v1/serving/deployments — live serving runtimes." };

  return (
    <div>
      <PageHeader
        title={heading.title}
        subtitle={heading.subtitle}
        actions={
          <>
            <Button variant="secondary" size="sm" onClick={() => setSlideOver("seed")}>
              <PackagePlus className="h-4 w-4" />
              Add model
            </Button>
            <Button size="sm" onClick={() => setSlideOver("deploy")}>
              <Rocket className="h-4 w-4" />
              Deploy model
            </Button>
          </>
        }
      />

      {view === "catalog" ? (
        <Catalog families={families.data} onSeeded={() => store.refresh()} />
      ) : view === "models" ? (
        <ModelsView store={store} onDeploy={() => setSlideOver("deploy")} />
      ) : view === "ports" ? (
        <PortsView deployments={deployments} />
      ) : view === "sizing" ? (
        <Sizing active={active && view === "sizing"} />
      ) : (
        <DeploymentsView
          deployments={deployments}
          embeddingNames={embeddingNames}
          store={store}
        />
      )}

      {/* Slide-overs: both forms stay mounted; only visibility toggles. */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 transition-opacity duration-200",
          slideOver ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={() => setSlideOver(null)}
        aria-hidden
      />
      <SlideOverPanel open={slideOver === "deploy"} onClose={() => setSlideOver(null)}>
        <DeployForm store={store} active={active} onDeployed={() => deployments.refresh()} />
      </SlideOverPanel>
      <SlideOverPanel open={slideOver === "seed"} onClose={() => setSlideOver(null)}>
        <AddModelForm families={families.data} onSeeded={() => store.refresh()} />
      </SlideOverPanel>
    </div>
  );
}

/**
 * Right-hand slide-over. Persistently mounted; slides off-screen when closed so
 * its children (a form + any in-flight ResultPanel) keep their state.
 *
 * A11y: while closed the panel is off-screen but still in the DOM, so without
 * `inert` its buttons/inputs would stay in the tab order and reachable by
 * screen readers. `inert={!open}` makes the whole subtree non-focusable and
 * a11y-hidden when closed while preserving the translate-x slide animation and
 * the persistent mount (form state / in-flight ResultPanel survive). On open we
 * move focus into the panel (standard dialog behavior).
 */
function SlideOverPanel({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(open);

  useEffect(() => {
    if (open && !wasOpen.current) {
      // Opening: remember the trigger, move focus into the panel.
      openerRef.current = document.activeElement as HTMLElement | null;
      closeRef.current?.focus();
    } else if (!open && wasOpen.current) {
      // Closing: return focus to the trigger that opened it, not <body>.
      openerRef.current?.focus();
      openerRef.current = null;
    }
    wasOpen.current = open;
  }, [open]);

  return (
    <aside
      inert={!open}
      aria-hidden={!open}
      className={cn(
        "fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-background shadow-elevated transition-transform duration-200",
        open ? "translate-x-0" : "translate-x-full",
      )}
    >
      <div className="flex items-center justify-end border-b border-border px-3 py-2">
        <button
          ref={closeRef}
          type="button"
          onClick={onClose}
          aria-label="Close panel"
          className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto p-4">{children}</div>
    </aside>
  );
}
