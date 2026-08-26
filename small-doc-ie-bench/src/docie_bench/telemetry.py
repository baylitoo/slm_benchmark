from __future__ import annotations

import os
import warnings
from pathlib import Path


def _prepare_multiprocess_dir() -> Path | None:
    """Create Prometheus' mmap directory before importing the client.

    ``prometheus_client`` selects its value implementation at import time. If
    multiprocess mode points at a missing directory, the first labelled metric
    raises ``FileNotFoundError`` and can turn an otherwise successful request
    into a 500. A bad observability path must not break application traffic, so
    an unusable directory disables multiprocess mode and falls back to the
    normal in-process registry.
    """
    raw = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.warn(
            f"Cannot prepare PROMETHEUS_MULTIPROC_DIR={raw!r}; "
            f"using single-process metrics instead: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        return None
    return path


PROMETHEUS_MULTIPROC_PATH = _prepare_multiprocess_dir()

# Import after the directory bootstrap: the client chooses multiprocess vs
# in-process storage while importing prometheus_client.values.
from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST,
)
from prometheus_client import (  # noqa: E402
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

CONTENT_TYPE_LATEST = _CONTENT_TYPE_LATEST


def generate_metrics() -> bytes:
    """Render metrics from the correct registry for the configured mode."""
    if PROMETHEUS_MULTIPROC_PATH is None:
        return generate_latest()
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry)

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

AGENT_REQUESTS = Counter(
    "docie_agent_requests_total",
    "Agent chat-completion requests by outcome (ok / pii_blocked / "
    "guard_unavailable / upstream errors / ...)",
    ["agent", "kind", "outcome"],
)

AGENT_LATENCY = Histogram(
    "docie_agent_latency_seconds",
    "End-to-end agent request latency (analysis + backing model)",
    ["agent", "kind"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

AGENT_PII_DETECTED = Counter(
    "docie_agent_pii_detected_total",
    "PII entities detected by security-proxy agents, by placeholder type",
    ["agent", "entity_type"],
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

ASR_JOB_ITEMS = Counter(
    "docie_asr_job_items_total",
    "Durable ASR job items by deployment and terminal outcome",
    ["deployment", "outcome"],
)

ASR_JOB_ITEM_LATENCY = Histogram(
    "docie_asr_job_item_latency_seconds",
    "Processing time for durable ASR job items",
    ["deployment"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

ASR_JOB_REAL_TIME_FACTOR = Histogram(
    "docie_asr_job_real_time_factor",
    "Processing seconds divided by audio seconds for durable ASR items",
    ["deployment"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10),
)
