from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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

MODEL_GATEWAY_REQUESTS = Counter(
    "docie_model_gateway_requests_total",
    "Model gateway attempts and rejected requests",
    ["model_profile", "model", "outcome"],
)

MODEL_GATEWAY_RETRIES = Counter(
    "docie_model_gateway_retries_total",
    "Model gateway retries",
    ["model_profile", "model", "classification"],
)

MODEL_GATEWAY_WAIT = Histogram(
    "docie_model_gateway_queue_wait_seconds",
    "Time spent waiting for a model execution slot",
    ["model_profile", "model"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 15, 30, 60),
)

MODEL_GATEWAY_QUEUE_DEPTH = Gauge(
    "docie_model_gateway_queue_depth",
    "Current queued model requests",
    ["model_profile", "model"],
)

MODEL_GATEWAY_IN_FLIGHT = Gauge(
    "docie_model_gateway_in_flight",
    "Current model requests holding an execution slot",
    ["model_profile", "model"],
)

MODEL_GATEWAY_CIRCUIT_OPEN = Gauge(
    "docie_model_gateway_circuit_open",
    "Whether the model gateway circuit is open",
    ["model_profile", "model"],
)

REVIEW_ACTIONS = Counter(
    "docie_review_actions_total",
    "Total human review actions",
    ["action"],
)

REVIEW_QUEUE_DEPTH = Gauge(
    "docie_review_queue_depth",
    "Current review queue depth by status",
    ["status"],
)
