from __future__ import annotations

from docie_bench.review import enqueue_review
from docie_bench.schemas.common import ExtractionResponse
from docie_bench.schemas.review import ReviewTaskCreate
from docie_bench.security import redact_fields
from docie_bench.settings import get_settings
from docie_bench.storage.db import ExtractionAudit, database_enabled, session_scope
from docie_bench.telemetry import EXTRACTION_LATENCY, EXTRACTION_REQUESTS


def save_extraction_audit(response: ExtractionResponse, *, tenant_id: str | None = None) -> None:
    settings = get_settings()
    with session_scope() as session:
        if session is None:
            return
        session.add(
            ExtractionAudit(
                request_id=response.request_id,
                tenant_id=tenant_id,
                schema_name=response.schema_name,
                model_profile=response.model_profile,
                document_hash=response.document_hash,
                valid=1 if response.validation.valid else 0,
                latency_ms=response.latency_ms,
                result_json=redact_fields(response.result, settings.audit_redaction_fields),
                warnings_json=response.validation.warnings,
                errors_text="\n".join(response.validation.errors),
            )
        )
    if database_enabled():
        enqueue_review(
            ReviewTaskCreate(
                source_request_id=response.request_id,
                schema_name=response.schema_name,
                model_profile=response.model_profile,
                document_hash=response.document_hash,
                original_prediction=response.result,
                validation_valid=response.validation.valid,
                validation_errors=response.validation.errors,
                dynamic_schema=response.dynamic_schema,
            )
        )


def record_extraction(response: ExtractionResponse, *, tenant_id: str | None = None) -> None:
    """Emit the observability side effects for one extraction: Prometheus metrics
    plus the durable audit row.

    Shared by the sync API handlers and the async Studio (Inngest) worker path so
    both surface identically in the Observability tab / Grafana. Increments the
    request counter, observes latency, then persists the audit row (same order and
    labels/fields the sync path has always used).
    """
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(
        response.latency_ms / 1000
    )
    save_extraction_audit(response, tenant_id=tenant_id)
