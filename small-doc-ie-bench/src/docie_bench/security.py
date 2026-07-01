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
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, UploadFile

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


class TenantQuotaManager:
    def __init__(
        self,
        *,
        api_keys: dict[str, str],
        auth_required: bool,
        requests_per_window: int,
        window_seconds: int,
        max_concurrent: int,
    ) -> None:
        self.api_keys = api_keys
        self.auth_required = auth_required
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.max_concurrent = max_concurrent
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._concurrent: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def authenticate(self, api_key: str | None) -> TenantContext:
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
        return TenantContext(tenant_id="anonymous", authenticated=False)

    def acquire(self, context: TenantContext, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            requests = self._requests[context.tenant_id]
            cutoff = current - self.window_seconds
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if (
                self.max_concurrent > 0
                and self._concurrent[context.tenant_id] >= self.max_concurrent
            ):
                raise HTTPException(status_code=429, detail="Tenant concurrency limit exceeded")
            if self.requests_per_window > 0 and len(requests) >= self.requests_per_window:
                raise HTTPException(
                    status_code=429,
                    detail="Tenant request rate limit exceeded",
                    headers={"Retry-After": str(self.window_seconds)},
                )
            requests.append(current)
            self._concurrent[context.tenant_id] += 1

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
    auth_required = settings.auth_required
    # With auth off (local single-operator dev) every caller collapses into the
    # one "anonymous" tenant, so per-tenant quotas are pure friction — the chatty
    # Studio UI (auto-refresh + realtime-token + polling) trips the default
    # 60/window. Disable them by passing 0 (acquire()'s `> 0` guards then skip the
    # checks). Networked runs keep auth on and the configured quotas apply.
    return TenantQuotaManager(
        api_keys=parse_api_keys(settings.api_keys.get_secret_value()),
        auth_required=auth_required,
        requests_per_window=settings.rate_limit_requests if auth_required else 0,
        window_seconds=settings.rate_limit_window_seconds,
        max_concurrent=settings.tenant_max_concurrent_requests if auth_required else 0,
    )


async def tenant_guard(
    x_api_key: Annotated[str | None, Header()] = None,
) -> AsyncIterator[TenantContext]:
    """FastAPI dependency: authenticate the caller, then bound per-tenant quota."""
    manager = get_quota_manager()
    context = manager.authenticate(x_api_key)
    manager.acquire(context)
    try:
        yield context
    finally:
        manager.release(context)


TenantDependency = Annotated[TenantContext, Depends(tenant_guard)]
