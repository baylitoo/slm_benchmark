// GET /v1/capabilities -- what this deployment accepts, as configured.

import { request } from "./core";

export interface OcrBackendCapability {
  name: string;
  aliases: string[];
  /** The backend's import is present in the API image. */
  available: boolean;
  /** Package extra that installs it; null for a built-in. */
  extra: string | null;
}

export interface Capabilities {
  upload: {
    max_bytes: number;
    max_request_body_bytes: number;
    allowed_mime_types: string[];
  };
  text: { max_chars: number; max_ocr_blocks: number; max_ocr_block_chars: number };
  metadata: { max_entries: number };
  ocr_backends: OcrBackendCapability[];
}

export function getCapabilities(): Promise<Capabilities> {
  return request<Capabilities>("/v1/capabilities");
}
