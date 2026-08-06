"use client";

import { createContext, useContext } from "react";

/**
 * Whether the subtree belongs to the currently VISIBLE section. All five
 * sections stay mounted (in-flight work survives navigation), so periodic
 * cosmetic work — e.g. LiveIndicator's relative-time tick — must not run in
 * hidden subtrees. Defaults to true so components used outside the shell
 * behave unchanged.
 */
const SectionActivityContext = createContext(true);

export const SectionActivityProvider = SectionActivityContext.Provider;

export function useSectionActive(): boolean {
  return useContext(SectionActivityContext);
}
