/**
 * The theme contract, which nothing checked.
 *
 * A token declared in `:root` but not in `.dark` does not fail — it inherits
 * the LIGHT value, so the component using it keeps a white surface in dark
 * mode. Nothing surfaces that until somebody opens dark mode on that one
 * screen, and with no theming test at all the parity was holding by hand.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const ROOT = path.resolve(import.meta.dirname, "..");
const css = readFileSync(path.join(ROOT, "app", "globals.css"), "utf8");
const tailwind = readFileSync(path.join(ROOT, "tailwind.config.ts"), "utf8");
const providers = readFileSync(
  path.join(ROOT, "components", "Providers.tsx"),
  "utf8",
);

/** The custom properties declared in one selector's block. */
function tokensIn(selector: string): string[] {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\n\\s*\\}`).exec(css);
  if (match === null) throw new Error(`no ${selector} block in globals.css`);
  return [...match[1].matchAll(/(--[a-z-]+)\s*:/g)].map((m) => m[1]).sort();
}

describe("theme tokens", () => {
  it("declares every light token in dark too", () => {
    // Not just the same COUNT: a renamed token would keep the count equal
    // while leaving the old name unthemed.
    expect(tokensIn(".dark")).toEqual(tokensIn(":root"));
  });

  it("themes every token tailwind resolves as a colour", () => {
    // `--font-sans` / `--font-mono` come from next/font and are deliberately
    // not themed, so only the rgb() colour tokens are required here.
    const colours = [
      ...tailwind.matchAll(/rgb\(var\((--[a-z-]+)\)/g),
    ].map((m) => m[1]);
    expect(colours.length).toBeGreaterThan(0);
    const dark = new Set(tokensIn(".dark"));
    expect(colours.filter((token) => !dark.has(token))).toEqual([]);
  });

  it("documents the default theme the provider actually sets", () => {
    // globals.css claimed "Dark is the default" while Providers.tsx passed
    // defaultTheme="light" -- a comment that said the opposite of the
    // behaviour, which is worse than no comment for whoever touches theming
    // next. Asserting both keeps them from drifting apart again.
    const configured = /defaultTheme="(\w+)"/.exec(providers)?.[1];
    expect(configured).toBe("light");
    expect(css).toMatch(/Light is the default/);
    expect(css).not.toMatch(/Dark is the default/);
  });
});
