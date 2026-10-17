"use client";

import { getCapabilities, type OcrBackendCapability } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";

// Used until /v1/capabilities answers, or when it can't: every backend is
// offered and a missing one fails at run time, as before this hook existed.
const FALLBACK: OcrBackendCapability[] = ["liteparse", "tesseract", "paddleocr", "indra"].map(
  (name) => ({ name, aliases: [], available: true, extra: null }),
);

const NOTES: Record<string, string> = {
  liteparse: "PDF text + OCR fallback (light)",
  indra: "CPU ONNX OCR",
};

/** The OCR backends this deployment accepts, with whether each is installed. */
export function useOcrBackends(): OcrBackendCapability[] {
  const { data } = useAsync("capabilities", getCapabilities);
  return data?.ocr_backends ?? FALLBACK;
}

export function ocrBackendLabel(backend: OcrBackendCapability): string {
  const note = NOTES[backend.name];
  const label = note ? `${backend.name} — ${note}` : backend.name;
  if (backend.available) return label;
  return `${label} (not installed${backend.extra ? `: needs the ${backend.extra} extra` : ""})`;
}
