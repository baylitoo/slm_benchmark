from __future__ import annotations

import hmac
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, Header, HTTPException, Request, UploadFile

from docie_bench.settings import get_settings

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    authenticated: bool


def parse_api_keys(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): str(tenant) for key, tenant in parsed.items() if key and tenant}

    result: dict[str, str] = {}
    for item in raw.split(","):
        key, separator, tenant = item.strip().partition(":")
        if separator and key and tenant:
            result[key] = tenant
    return result


# Ceiling on distinct throttle buckets kept in memory. Anonymous buckets are
# keyed per client IP, and an attacker who can spoof source addresses (or churn
# through proxies) must not grow the dicts without bound.
_MAX_TRACKED_BUCKETS = 10_000


class TenantQuotaManager:
    def __init__(
        self,
        *,
        api_keys: dict[str, str],
        auth_required: bool,
        requests_per_window: int,
        read_requests_per_window: int | None = None,
        window_seconds: int,
        max_concurrent: int,
        anonymous_requests_per_window: int = 0,
        anonymous_max_concurrent: int = 0,
    ) -> None:
        self.api_keys = api_keys
        self.auth_required = auth_required
        self.requests_per_window = requests_per_window
        # ``None`` preserves the old one-budget behaviour for direct callers;
        # the application factory always supplies the dedicated read limit.
        self.read_requests_per_window = (
            requests_per_window
            if read_requests_per_window is None
            else read_requests_per_window
        )
        self.window_seconds = window_seconds
        self.max_concurrent = max_concurrent
        # Anonymous (auth-off) callers get their OWN limits, bucketed per client
        # IP. Previously AUTH_REQUIRED=false zeroed the quotas entirely — one
        # env var silently disabled BOTH authentication AND throttling, so a
        # LAN-exposed dev instance had no request bounding at all.
        self.anonymous_requests_per_window = anonymous_requests_per_window
        self.anonymous_max_concurrent = anonymous_max_concurrent
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._concurrent: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def authenticate(self, api_key: str | None, client_host: str | None = None) -> TenantContext:
        if api_key:
            for configured_key, tenant_id in self.api_keys.items():
                if hmac.compare_digest(api_key, configured_key):
                    return TenantContext(tenant_id=tenant_id, authenticated=True)
        if self.auth_required:
            raise HTTPException(
                status_code=401,
                detail="A valid API key is required",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        # Per-IP bucket so one noisy client cannot starve the others and the
        # anonymous limits actually bound *someone* rather than one global pool.
        bucket = f"anon:{client_host}" if client_host else "anonymous"
        return TenantContext(tenant_id=bucket, authenticated=False)

    def _limits_for(
        self,
        context: TenantContext,
        quota: Literal["request", "read"],
    ) -> tuple[int, int]:
        if context.authenticated:
            rate = (
                self.read_requests_per_window
                if quota == "read"
                else self.requests_per_window
            )
            return rate, self.max_concurrent
        return self.anonymous_requests_per_window, self.anonymous_max_concurrent

    def acquire(
        self,
        context: TenantContext,
        *,
        quota: Literal["request", "read"] = "request",
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else now
        requests_per_window, max_concurrent = self._limits_for(context, quota)
        with self._lock:
            self._prune_locked()
            # Separate histories are the essential boundary: Studio polling
            # must not consume (or be blocked by) the inference/mutation budget.
            rate_bucket = f"{context.tenant_id}:{quota}"
            requests = self._requests[rate_bucket]
            cutoff = current - self.window_seconds
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if (
                max_concurrent > 0
                and self._concurrent[context.tenant_id] >= max_concurrent
            ):
                raise HTTPException(status_code=429, detail="Tenant concurrency limit exceeded")
            if requests_per_window > 0 and len(requests) >= requests_per_window:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Tenant read request rate limit exceeded"
                        if quota == "read"
                        else "Tenant request rate limit exceeded"
                    ),
                    headers={"Retry-After": str(self.window_seconds)},
                )
            requests.append(current)
            self._concurrent[context.tenant_id] += 1

    def _prune_locked(self) -> None:
        """Bound the bucket dicts (caller holds the lock).

        Drops empty/idle buckets first; a pathological flood of DISTINCT
        source addresses then falls back to clearing request history (never
        the in-flight concurrency counts, which must stay balanced for
        ``release``).
        """
        if len(self._requests) < _MAX_TRACKED_BUCKETS:
            return
        for key in [k for k, q in self._requests.items() if not q]:
            del self._requests[key]
        while len(self._requests) >= _MAX_TRACKED_BUCKETS:
            self._requests.pop(next(iter(self._requests)))

    def release(self, context: TenantContext) -> None:
        with self._lock:
            self._concurrent[context.tenant_id] = max(
                0, self._concurrent[context.tenant_id] - 1
            )


def detect_mime_type(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if b"\x00" not in data[:8192]:
        try:
            data[:8192].decode("utf-8")
        except UnicodeDecodeError:
            return None
        return "text/plain"
    return None


async def read_validated_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    allowed_mime_types: set[str],
) -> tuple[bytes, str, str]:
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    expected_mime = MIME_BY_SUFFIX.get(suffix)
    if expected_mime is None:
        raise HTTPException(status_code=415, detail=f"Unsupported file suffix: {suffix}")
    if expected_mime not in allowed_mime_types:
        raise HTTPException(status_code=415, detail=f"File type is disabled: {expected_mime}")

    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(min(1024 * 1024, max_bytes + 1)):
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Max {max_bytes} bytes")
        chunks.append(chunk)
    data = b"".join(chunks)
    detected_mime = detect_mime_type(data)
    if detected_mime != expected_mime:
        raise HTTPException(
            status_code=415,
            detail=f"File content does not match its suffix; detected {detected_mime or 'unknown'}",
        )
    claimed_mime = (file.content_type or "").lower()
    if claimed_mime not in GENERIC_MIME_TYPES and claimed_mime != detected_mime:
        raise HTTPException(
            status_code=415, detail="Declared content type does not match file content"
        )
    return data, suffix, detected_mime


def redact_fields(value: Any, field_names: set[str], replacement: str = "[REDACTED]") -> Any:
    if not field_names:
        return value
    if isinstance(value, dict):
        return {
            key: (
                replacement
                if key in field_names
                else redact_fields(item, field_names, replacement)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_fields(item, field_names, replacement) for item in value]
    return value


@lru_cache(maxsize=1)
def get_quota_manager() -> TenantQuotaManager:
    """Process-wide tenant quota manager, built once from settings.

    Single source of truth so every router enforces the same auth + rate limit.
    NOTE: state is per-process; horizontal scale needs a shared store (review H4).
    Tests that toggle auth_required/api_keys must call get_quota_manager.cache_clear()
    (and get_settings.cache_clear()).
    """
    settings = get_settings()
    # Auth and throttling are DECOUPLED. AUTH_REQUIRED=false used to zero the
    # quotas too — one env var silently disabled both. Anonymous callers now
    # keep their own (generous — the Studio UI is chatty: auto-refresh +
    # realtime-token + polling) per-client-IP limits, so a LAN-exposed dev
    # instance is still bounded. Authenticated tenants keep the configured
    # per-tenant quotas exactly as before.
    return TenantQuotaManager(
        api_keys=parse_api_keys(settings.api_keys.get_secret_value()),
        auth_required=settings.auth_required,
        requests_per_window=settings.rate_limit_requests,
        read_requests_per_window=settings.tenant_read_rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        max_concurrent=settings.tenant_max_concurrent_requests,
        anonymous_requests_per_window=settings.anonymous_rate_limit_requests,
        anonymous_max_concurrent=settings.anonymous_max_concurrent_requests,
    )


async def tenant_guard(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> AsyncIterator[TenantContext]:
    """FastAPI dependency: authenticate the caller, then bound per-tenant quota."""
    manager = get_quota_manager()
    client_host = request.client.host if request.client else None
    context = manager.authenticate(x_api_key, client_host)
    # Safe control-plane reads and polling use a deliberately larger, separate
    # budget. Authentication and concurrency limits still apply to all methods.
    quota = "read" if request.method in {"GET", "HEAD", "OPTIONS"} else "request"
    manager.acquire(context, quota=quota)
    try:
        yield context
    finally:
        manager.release(context)


TenantDependency = Annotated[TenantContext, Depends(tenant_guard)]
