"use client";

import { useMemo, useState } from "react";
import { Rocket, PackagePlus } from "lucide-react";
import {
  getStore,
  getFamilies,
  getDeployments,
  embeddingDeploymentNames,
  rerankerDeploymentNames,
  type StoreEntry,
  type DeploymentRecord,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { Button, Sheet } from "./ui";
import { Sizing } from "./Sizing";
import { Catalog } from "./Catalog";
import { PageHeader } from "./patterns/PageHeader";
import { POLL_MS } from "./deploy/shared";
import { DeploymentsView } from "./deploy/DeploymentsView";
import { ModelsView } from "./deploy/ModelsView";
import { DownloadsView } from "./deploy/DownloadsView";
import { DeployForm } from "./deploy/DeployForm";
import { AddModelForm } from "./deploy/seed/AddModelForm";

type SlideOver = null | "deploy" | "seed";

/**
 * Deploy = a table-first serving console with nav-driven sub-views
 * ("catalog" / "models" / "deployments" / "sizing"). The Deploy + Seed forms live in
 * persistently-mounted slide-overs (visibility toggled, never unmounted) so an
 * in-flight deploy/seed and its ResultPanel survive closing the panel or
 * switching views. All pollers, handlers, and API calls are unchanged.
 */
export function Deploy({
  active = true,
  view = "deployments",
  filterQuery,
}: {
  active?: boolean;
  view?: string;
  /** Deep-link seed for the Deployments table's text filter (e.g. from
   * Observability's activity tile). Only relevant when view="deployments";
   * DeploymentsView ignores it otherwise. */
  filterQuery?: string;
}) {
  // Auto-refreshing lists — paused when the tab is hidden OR Deploy isn't the
  // active section (every section stays mounted in the shell). Held at the top
  // level so switching sub-views never remounts a poller.
  const store = usePolling<StoreEntry[]>(getStore, POLL_MS, active);
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, POLL_MS, active);
  const families = useAsync("families", getFamilies); // shared SWR key with the Playground

  const [slideOver, setSlideOver] = useState<SlideOver>(null);

  // Deployment names that are embedding/reranker models (store family flagged
  // accordingly) — the Deployments table's "Type" column source.
  const embeddingNames = useMemo(
    () => embeddingDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );
  const rerankerNames = useMemo(
    () => rerankerDeploymentNames(store.data, families.data),
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
        : view === "sizing"
          ? {
              title: "Sizing",
              subtitle:
                "GET /v1/serving/sizing — how many more instances fit in RAM right now.",
            }
          : view === "downloads"
            ? {
                title: "Downloads",
                subtitle:
                  "GET /v1/studio/seeds — ongoing and past seed jobs, with error detail for failures.",
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
        <Catalog
          families={families.data}
          existingStoreNames={(store.data ?? []).map((entry) => entry.name)}
          onSeeded={() => store.refresh()}
        />
      ) : view === "models" ? (
        <ModelsView store={store} onDeploy={() => setSlideOver("deploy")} />
      ) : view === "sizing" ? (
        <Sizing active={active && view === "sizing"} />
      ) : view === "downloads" ? (
        <DownloadsView active={active && view === "downloads"} />
      ) : (
        <DeploymentsView
          deployments={deployments}
          embeddingNames={embeddingNames}
          rerankerNames={rerankerNames}
          store={store}
          initialFilter={filterQuery}
        />
      )}

      {/* Slide-overs: both forms stay mounted; only visibility toggles. Each
          Sheet carries its own backdrop -- harmless with only one ever open
          at a time (single `slideOver` state), see ui.tsx's own note on this
          tradeoff versus the single shared backdrop this used to hand-roll. */}
      <Sheet open={slideOver === "deploy"} onClose={() => setSlideOver(null)}>
        <DeployForm store={store} active={active} onDeployed={() => deployments.refresh()} />
      </Sheet>
      <Sheet open={slideOver === "seed"} onClose={() => setSlideOver(null)}>
        <AddModelForm
          families={families.data}
          existingStoreNames={(store.data ?? []).map((entry) => entry.name)}
          onSeeded={() => store.refresh()}
        />
      </Sheet>
    </div>
  );
}
