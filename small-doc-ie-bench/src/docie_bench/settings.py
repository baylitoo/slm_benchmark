from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hermetic-test escape hatch: the suite sets this in tests/conftest.py before any
# docie_bench import so a developer's local .env can never change test behavior.
_ENV_FILE = None if os.environ.get("DOCIE_IGNORE_ENV_FILE") == "1" else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore", case_sensitive=False)

    app_env: str = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - container service bind default
    api_port: int = 8080
    max_request_body_mb: int = Field(default=26, ge=1, le=1_025)
    max_upload_mb: int = Field(default=25, ge=1, le=1_024)
    max_text_chars: int = Field(default=1_000_000, ge=1, le=100_000_000)
    max_ocr_blocks: int = Field(default=1_000, ge=1, le=100_000)
    max_ocr_block_chars: int = Field(default=20_000, ge=1, le=10_000_000)
    max_metadata_entries: int = Field(default=50, ge=0, le=10_000)
    raw_document_storage: Literal["disabled", "enabled"] = "disabled"
    allowed_upload_mime_types: str = (
        "application/pdf,text/plain,image/png,image/jpeg,image/tiff"
    )

    # Fail closed by default (B3). Local dev sets AUTH_REQUIRED=false in .env;
    # any networked deployment must populate API_KEYS and leave this on.
    auth_required: bool = True
    api_keys: SecretStr = Field(default=SecretStr(""))
    # GET/HEAD traffic has its own budget because the authenticated Studio
    # continuously polls several control-plane views. Counting those reads
    # against inference/mutations makes the default 60/min quota self-deny
    # during ordinary use (Deployments alone can issue ~45 reads/minute).
    rate_limit_requests: int = Field(default=60, ge=0)
    tenant_read_rate_limit_requests: int = Field(default=600, ge=0)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    tenant_max_concurrent_requests: int = Field(default=4, ge=0)
    # Limits for UNAUTHENTICATED callers (auth_required=false), bucketed per
    # client IP. Deliberately generous — the Studio UI is chatty (auto-refresh,
    # realtime tokens, polling) — but never zero: turning auth off must not also
    # turn off request bounding. 0 disables the respective check.
    anonymous_rate_limit_requests: int = Field(default=600, ge=0)
    anonymous_max_concurrent_requests: int = Field(default=16, ge=0)
    enable_benchmark_api: bool = False

    redacted_response_fields: str = ""
    redacted_audit_fields: str = ""
    log_document_content: bool = False

    database_url: str | None = None
    review_claim_lease_seconds: int = Field(default=900, ge=30, le=86400)
    review_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    review_evidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # Whether a review task's OCR blocks (text + bounding box + page, no raster
    # pixels) are persisted alongside it for the Evidence panel. "disabled"
    # keeps today's behavior (no evidence, never persisted); "ocr_text" writes
    # the blocks that were already grounded against the prediction. Page
    # images are deliberately not an option here -- the OCR pipeline never
    # rasterizes pages today, and adding that is a separate, larger change.
    review_evidence_retention: Literal["disabled", "ocr_text"] = "ocr_text"
    annotation_export_dir: Path = Path("annotations")

    default_schema_name: str = "invoice"
    # Repair UTF-8 mojibake in model output ("universitÃ©" -> "université"): small
    # OCR/vision models emit double-encoded UTF-8 on accented (esp. French) text.
    # ftfy's encoding-only fix, applied to completion + extraction content. Set
    # False for a byte-faithful proxy (the transport itself is already clean).
    fix_mojibake: bool = True
    # Studio-friendly default: strongest structured style declared, with the
    # serving negotiation ladder auto-downgrading per runtime so the Playground
    # returns valid JSON out-of-box even on small models.
    default_model_profile: str = "studio_default"
    default_ocr_backend: str = "pdf_text"
    # Validity gate: fail a benchmark run loudly when a profile's valid_rate is
    # below this threshold instead of silently scoring zeros. 0.0 disables it.
    valid_rate_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # Constrained-decode downgrade view: flag (report-only) a profile whose
    # constrained_rate — the fraction of rows decoded with the requested strong
    # style rather than silently downgraded — is below this threshold. 0.0
    # disables it; unlike valid_rate_threshold it never fails a run.
    constrained_rate_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_cache_dir: Path = Path(".cache/ocr")
    ocr_cache_max_mb: int = Field(default=2048, ge=0)
    ocr_cache_enabled: bool = True
    # PDF/image ingestion via liteparse (PDFium spatial text + pluggable OCR).
    ocr_dpi: int = Field(default=150, ge=72, le=600)
    # Optional VLM-backed OCR server implementing the liteparse OCR API spec. When
    # set, the pdf_text backend routes text-poor/scanned pages through it instead
    # of the built-in Tesseract. Relevant only for text-only extraction models that
    # cannot read the page image themselves; vision-capable profiles receive page
    # images directly and never hit this path.
    ocr_server_url: str | None = None
    ocr_language: str | None = None
    runs_dir: Path = Path("runs")

    # MCP tool sources for served chat models (docie_bench.mcp_tools): the
    # operator-owned registry of reachable MCP servers. Callers of
    # /v1/chat/completions pick servers from this file BY NAME via the
    # request's "mcp_servers" field — a request can never supply its own URL
    # or command line. A missing file simply means "no servers registered".
    mcp_servers_config: Path = Path("configs/mcp-servers.json")
    # Upper bound on model<->tool rounds per chat request: a model stuck
    # re-calling tools forever must terminate deterministically (502) instead
    # of burning upstream tokens unbounded.
    mcp_max_tool_iterations: int = Field(default=8, ge=1, le=64)
    # Per-response read timeout for MCP client sessions (list_tools/call_tool).
    mcp_tool_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    # Durable, addressable artifact store for Studio benchmark runs — the worker
    # writes here and the api/web replicas read back by artifact id, never by a
    # worker-local path. Metrics summaries live in Postgres (small);
    # report.html / predictions.jsonl live only in this store.
    #
    # "filesystem" (default): blobs under artifact_store_dir, which must resolve
    # to the SAME location on every replica that reads it (a shared volume).
    # "s3": blobs as objects in an S3-compatible bucket (AWS S3 or MinIO) — no
    # shared volume needed, the right choice for multi-replica deployments.
    artifact_store_backend: Literal["filesystem", "s3"] = "filesystem"
    artifact_store_dir: Path = Path("artifacts")
    # S3 backend knobs (only read when artifact_store_backend="s3"). Credentials
    # and region come from the standard AWS chain (AWS_ACCESS_KEY_ID,
    # AWS_SECRET_ACCESS_KEY, AWS_REGION / config files / instance roles) —
    # deliberately never from docie settings. endpoint_url None targets AWS;
    # set it for MinIO or any other S3-compatible server. The optional prefix
    # namespaces this store's objects inside a shared bucket.
    artifact_store_s3_bucket: str | None = None
    artifact_store_s3_endpoint_url: str | None = None
    artifact_store_s3_prefix: str = ""
    # Retention/GC for the Studio run index (see docie_bench.studio.store.RunStore.gc
    # and docs/docie-studio.md). Age wins first, then a hard cap on run count.
    studio_run_retention_days: int = Field(default=30, ge=1, le=3650)
    studio_run_retention_max: int = Field(default=500, ge=1, le=1_000_000)
    # Grace window for the orphan-blob mark-and-sweep: a blob physically present
    # in the store but referenced by no artifact row is only reclaimed once it is
    # older than this, so a blob an in-flight job just ``put()`` (before its run
    # ``complete()``-commits the artifact row) is never swept out from under it.
    studio_orphan_grace_hours: int = Field(default=24, ge=0, le=8760)

    # Cross-container serving reachability (PR-1). A deployed runtime's process
    # binds ``serving_bind_host`` (all interfaces inside its container) while the
    # DeploymentRecord advertises ``serving_advertise_host`` — a name every replica
    # resolves to the node that runs the runtime. The two are split so the recorded
    # endpoint (read by the api/other replicas via profile_resolver) is
    # cross-container reachable instead of a worker-local loopback. Local CLI keeps
    # 127.0.0.1 (same host); Docker sets DOCIE_SERVING_ADVERTISE_HOST to the deploy
    # service name (see docker-compose.yml). DOCIE_-prefixed aliases mirror
    # DOCIE_SERVING_HOME so all serving knobs share one namespace.
    #
    # Both default to the SAFE local value 127.0.0.1 (loopback). `docie up` /
    # `docie serve` run same-host, so a loopback bind never exposes the unauth
    # runtime on the LAN. The Docker path deliberately opts INTO an all-interfaces
    # bind by setting DOCIE_SERVING_BIND_HOST=0.0.0.0 in compose (paired with the
    # advertise service name) so sibling containers can reach it over the compose
    # network — see docker-compose.yml / .env.example.
    serving_advertise_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DOCIE_SERVING_ADVERTISE_HOST", "serving_advertise_host"),
    )
    serving_bind_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DOCIE_SERVING_BIND_HOST", "serving_bind_host"),
    )

    # Per-deploy port allocation window. When a deploy supplies no explicit port
    # the control plane picks the first free port in [start, end] that is neither
    # held by a non-removed deployment record nor bound by a live socket, so two
    # concurrent store deploys land on distinct ports with no operator input.
    # 8088 stays the first pick, so single-deploy behavior is unchanged. Widen
    # the window (or stop a deployment) if it exhausts. DOCIE_-prefixed aliases
    # mirror the other serving knobs.
    serving_port_range_start: int = Field(
        default=8088,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "DOCIE_SERVING_PORT_RANGE_START", "serving_port_range_start"
        ),
    )
    serving_port_range_end: int = Field(
        default=8188,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "DOCIE_SERVING_PORT_RANGE_END", "serving_port_range_end"
        ),
    )

    # Sizing safety margin (PR-3): the slice of node TOTAL RAM the fit table
    # holds back before pricing prospective instances — explicit and surfaced
    # in the API/UI, never a hidden fudge factor. Design doc §3 brackets it at
    # 10-15%; default the low end. DOCIE_-prefixed alias mirrors the other
    # serving knobs.
    serving_sizing_margin_fraction: float = Field(
        default=0.10,
        ge=0.0,
        lt=1.0,
        validation_alias=AliasChoices(
            "DOCIE_SIZING_MARGIN_FRACTION", "serving_sizing_margin_fraction"
        ),
    )

    # Dynamic lifecycle knobs (PR-4, design doc §4). All DOCIE_-prefixed like
    # the other serving knobs.
    #
    # Idle-TTL unload: a hot, unpinned deployment that served nothing for this
    # long is unloaded by the reconciler (record + port kept, phase=evicted,
    # auto-reloadable on the next request). 0 disables idle unload entirely.
    serving_idle_ttl_seconds: float = Field(
        default=900.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "DOCIE_SERVING_IDLE_TTL_SECONDS", "serving_idle_ttl_seconds"
        ),
    )
    # Minimum hot time before a deployment may be idle-unloaded or chosen as an
    # eviction victim — the anti-thrash guard: a just-loaded model must not be
    # immediately re-evicted by the load that follows it.
    serving_min_hot_seconds: float = Field(
        default=120.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "DOCIE_SERVING_MIN_HOT_SECONDS", "serving_min_hot_seconds"
        ),
    )
    # Eviction rate limit (storm guard): at most this many victims per load
    # attempt / reconcile cycle. If the allowed victims cannot make the
    # candidate fit, NOTHING is evicted (never evict-to-not-fit).
    serving_max_evictions_per_cycle: int = Field(
        default=2,
        ge=1,
        le=100,
        validation_alias=AliasChoices(
            "DOCIE_SERVING_MAX_EVICTIONS_PER_CYCLE", "serving_max_evictions_per_cycle"
        ),
    )

    openai_compat_base_url: str = "http://llm-llamacpp:8000/v1"
    openai_compat_api_key: SecretStr = Field(default=SecretStr("local-not-used"))
    openai_compat_model: str = "local-model"
    openai_compat_timeout_seconds: float = 180.0
    openai_compat_response_format_style: str = "openai_json_schema"

    prometheus_multiproc_dir: str | None = None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_request_body_bytes(self) -> int:
        return self.max_request_body_mb * 1024 * 1024

    @property
    def allowed_mime_types(self) -> set[str]:
        return {
            value.strip().lower()
            for value in self.allowed_upload_mime_types.split(",")
            if value.strip()
        }

    @property
    def response_redaction_fields(self) -> set[str]:
        return {
            value.strip()
            for value in self.redacted_response_fields.split(",")
            if value.strip()
        }

    @property
    def audit_redaction_fields(self) -> set[str]:
        return {value.strip() for value in self.redacted_audit_fields.split(",") if value.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
