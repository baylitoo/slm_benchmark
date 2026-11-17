import { describe, expect, it } from "vitest";

import { modelPickerOptions } from "./helpers";
import type { DeploymentRecord } from "./serving";

function record(
  name: string,
  group: string,
  node: string | null,
  live = true,
): DeploymentRecord {
  return {
    spec: { name },
    replica_group: group,
    node_id: node,
    state: live ? "ready" : "stopped",
    endpoint: live ? `http://${node}:8090/v1` : "",
    activation: live ? undefined : "managed",
  } as DeploymentRecord;
}

describe("modelPickerOptions", () => {
  it("offers a scaled model once, load-balanced, before its node-labelled replicas", () => {
    const options = modelPickerOptions([
      record("lfm2.5-350m", "lfm2.5-350m", "rocky-llm"),
      record("lfm2.5-350m-2", "lfm2.5-350m", "ash-llm", false),
      record("nuextract3", "nuextract3", "rocky-llm"),
    ]);
    expect(options).toEqual([
      { value: "store:lfm2.5-350m", label: "lfm2.5-350m · 2 replicas" },
      { value: "lfm2.5-350m", label: "lfm2.5-350m · on rocky-llm" },
      { value: "lfm2.5-350m-2", label: "lfm2.5-350m-2 · on ash-llm" },
      { value: "nuextract3", label: "nuextract3" },
    ]);
  });

  it("names the model ahead of a single minted record", () => {
    expect(modelPickerOptions([record("nuextract3-k2f9", "nuextract3", null)])).toEqual([
      { value: "nuextract3", label: "nuextract3" },
      { value: "nuextract3-k2f9", label: "nuextract3-k2f9 · on unknown node" },
    ]);
  });

  it("offers only the reloadable records while every replica is evicted", () => {
    const options = modelPickerOptions([
      record("lfm2.5-350m-a1", "lfm2.5-350m", "rocky-llm", false),
      record("lfm2.5-350m-b2", "lfm2.5-350m", "ash-llm", false),
    ]);
    expect(options.map((o) => o.value)).toEqual(["lfm2.5-350m-a1", "lfm2.5-350m-b2"]);
  });
});
