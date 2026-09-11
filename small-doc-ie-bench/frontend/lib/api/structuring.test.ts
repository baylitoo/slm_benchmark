import { describe, expect, it } from "vitest";

import { structuringDeploymentNames } from "./helpers";
import type { ModelFamily, StoreEntry } from "./serving";

const families = [
  { name: "encoder_gliformer", structured_extraction: true },
  { name: "encoder_gliner2" },
  { name: "lfm2" },
] as ModelFamily[];

const store = [
  { name: "gliformer-large-v1", family: "encoder_gliformer" },
  { name: "guard-pii", family: "encoder_gliner2" },
  { name: "lfm2.5-2.6b", family: "lfm2" },
] as StoreEntry[];

describe("structuringDeploymentNames", () => {
  it("names the encoders that answer a schema", () => {
    expect(structuringDeploymentNames(store, families)).toEqual(
      new Set(["gliformer-large-v1"]),
    );
  });

  it("leaves out an encoder with no structuring head", () => {
    expect(structuringDeploymentNames(store, families).has("guard-pii")).toBe(false);
  });

  it("is empty before the families or the store have loaded", () => {
    expect(structuringDeploymentNames(null, families).size).toBe(0);
    expect(structuringDeploymentNames(store, null).size).toBe(0);
    expect(structuringDeploymentNames(undefined, undefined).size).toBe(0);
  });

  it("ignores a store entry with no family", () => {
    expect(structuringDeploymentNames([{ name: "orphan" } as StoreEntry], families).size).toBe(0);
  });
});
