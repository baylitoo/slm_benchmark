from __future__ import annotations

from docie_bench.schemas.common import ExtractionResponse
from docie_bench.storage.db import ExtractionAudit, session_scope


def save_extraction_audit(response: ExtractionResponse) -> None:
    with session_scope() as session:
        if session is None:
            return
        session.add(
            ExtractionAudit(
                request_id=response.request_id,
                schema_name=response.schema_name,
                model_profile=response.model_profile,
                document_hash=response.document_hash,
                valid=1 if response.validation.valid else 0,
                latency_ms=response.latency_ms,
                result_json=response.result,
                warnings_json=response.validation.warnings,
                errors_text="\n".join(response.validation.errors),
            )
        )
