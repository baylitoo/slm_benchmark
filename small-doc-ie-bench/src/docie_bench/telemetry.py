from __future__ import annotations

from prometheus_client import Counter, Histogram

EXTRACTION_REQUESTS = Counter(
    "docie_extraction_requests_total",
    "Total extraction requests",
    ["schema_name", "model_profile", "valid"],
)

EXTRACTION_LATENCY = Histogram(
    "docie_extraction_latency_seconds",
    "Extraction latency in seconds",
    ["schema_name", "model_profile"],
    buckets=(0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

OCR_BLOCKS = Histogram(
    "docie_ocr_blocks",
    "Number of OCR blocks per extraction",
    ["schema_name"],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)
