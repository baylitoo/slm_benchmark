// Browser-only helpers with no server round-trip.

import type { InngestRun } from "./extract";

/** Read a File as base64 (without the `data:...;base64,` prefix). */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Unexpected FileReader result"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

export function statusIs(run: InngestRun, ...want: string[]): boolean {
  const s = (run.status ?? "").toString().toLowerCase();
  return want.some((w) => w.toLowerCase() === s);
}
