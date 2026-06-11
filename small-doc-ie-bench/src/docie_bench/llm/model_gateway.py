from __future__ import annotations

import asyncio
import email.utils
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Never, TypeVar

import httpx

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.telemetry import (
    MODEL_GATEWAY_CIRCUIT_OPEN,
    MODEL_GATEWAY_IN_FLIGHT,
    MODEL_GATEWAY_QUEUE_DEPTH,
    MODEL_GATEWAY_REQUESTS,
    MODEL_GATEWAY_RETRIES,
    MODEL_GATEWAY_WAIT,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorClassification(StrEnum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    PERMANENT = "permanent"
    INVALID_RESPONSE = "invalid_response"
    CAPABILITY = "capability"
    CIRCUIT_OPEN = "circuit_open"
    QUEUE_FULL = "queue_full"


class ModelGatewayError(RuntimeError):
    classification = ErrorClassification.PERMANENT
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class TransientModelError(ModelGatewayError):
    classification = ErrorClassification.TRANSIENT
    retryable = True


class RateLimitedModelError(ModelGatewayError):
    classification = ErrorClassification.RATE_LIMITED
    retryable = True


class InvalidModelResponseError(ModelGatewayError):
    classification = ErrorClassification.INVALID_RESPONSE
    retryable = True


class ModelCapabilityError(ModelGatewayError):
    classification = ErrorClassification.CAPABILITY


class CircuitOpenError(ModelGatewayError):
    classification = ErrorClassification.CIRCUIT_OPEN


class ModelQueueFullError(ModelGatewayError):
    classification = ErrorClassification.QUEUE_FULL


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    vision: bool | None = None
    response_format_styles: frozenset[str] | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class _GatewayState:
    semaphore: asyncio.Semaphore
    queue_limit: int
    waiting: int = 0
    failures: int = 0
    circuit_opened_at: float | None = None
    half_open_in_flight: bool = False


_STATES: dict[tuple[str, str], _GatewayState] = {}


def reset_gateway_state() -> None:
    """Clear shared scheduler/circuit state. Intended for tests and process reconfiguration."""
    _STATES.clear()


def _state_for(profile: ModelProfile) -> _GatewayState:
    key = (profile.base_url, profile.model)
    state = _STATES.get(key)
    if state is None:
        state = _GatewayState(
            semaphore=asyncio.Semaphore(profile.max_concurrency),
            queue_limit=profile.queue_limit,
        )
        _STATES[key] = state
    return state


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def classify_response_error(response: httpx.Response) -> ModelGatewayError:
    status = response.status_code
    body = response.text[:500]
    message = f"Model endpoint returned {status}: {body}"
    if status == 429:
        return RateLimitedModelError(
            message,
            status_code=status,
            retry_after=_retry_after(response),
        )
    if status in {408, 409, 425} or status >= 500:
        return TransientModelError(message, status_code=status)
    return ModelGatewayError(message, status_code=status)


class ModelGateway:
    def __init__(
        self,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile = profile
        self.client = client
        self._sleep = sleep
        self._monotonic = monotonic
        self._state = _state_for(profile)
        self._capabilities: ModelCapabilities | None = None
        self._capability_lock = asyncio.Lock()

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        await self._acquire()
        MODEL_GATEWAY_IN_FLIGHT.labels(self.profile.name, self.profile.model).inc()
        try:
            self._check_circuit()
            for attempt in range(1, self.profile.retry_max_attempts + 1):
                try:
                    result = await operation()
                except Exception as exc:
                    error = self._classify_exception(exc)
                    self._record_failure(error)
                    MODEL_GATEWAY_REQUESTS.labels(
                        self.profile.name,
                        self.profile.model,
                        error.classification,
                    ).inc()
                    if not error.retryable or attempt >= self.profile.retry_max_attempts:
                        if error is exc:
                            raise
                        raise error from exc
                    MODEL_GATEWAY_RETRIES.labels(
                        self.profile.name,
                        self.profile.model,
                        error.classification,
                    ).inc()
                    await self._sleep(self._retry_delay(error, attempt))
                else:
                    self._record_success()
                    MODEL_GATEWAY_REQUESTS.labels(
                        self.profile.name,
                        self.profile.model,
                        "success",
                    ).inc()
                    return result
            raise AssertionError("retry loop exited unexpectedly")
        finally:
            MODEL_GATEWAY_IN_FLIGHT.labels(self.profile.name, self.profile.model).dec()
            self._release()

    async def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            response = await self.client.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise classify_response_error(response)
            try:
                data = response.json()
            except ValueError as exc:
                raise InvalidModelResponseError("Model endpoint returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise InvalidModelResponseError(
                    "Model endpoint returned a non-object JSON response"
                )
            return data

        return await self.execute(operation)

    async def discover_capabilities(self, *, force: bool = False) -> ModelCapabilities:
        if self._capabilities is not None and not force:
            return self._capabilities
        async with self._capability_lock:
            if self._capabilities is not None and not force:
                return self._capabilities
            data = await self.request_json("GET", "/models")
            models = data.get("data")
            if not isinstance(models, list):
                self._raise_capability_error(
                    "Capability discovery response has no model data list"
                )
            model_data = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("id") == self.profile.model
                ),
                None,
            )
            if model_data is None:
                self._raise_capability_error(
                    f"Configured model {self.profile.model!r} was not returned by /models"
                )
            capabilities = self._parse_capabilities(model_data)
            self._validate_profile(capabilities)
            self._capabilities = capabilities
            return capabilities

    async def validate_request(self, *, needs_vision: bool) -> ModelCapabilities | None:
        mode = self.profile.capability_discovery
        if mode == "disabled":
            return None
        try:
            capabilities = await self.discover_capabilities()
        except ModelCapabilityError:
            raise
        except ModelGatewayError:
            if mode == "required":
                raise
            logger.warning(
                "Optional model capability discovery failed",
                extra={
                    "docie_model_profile": self.profile.name,
                    "docie_model": self.profile.model,
                },
                exc_info=True,
            )
            return None
        if needs_vision and capabilities.vision is False:
            self._raise_capability_error(
                f"Model {self.profile.model!r} reports that it does not support vision"
            )
        return capabilities

    def _parse_capabilities(self, model_data: dict[str, Any]) -> ModelCapabilities:
        raw_capabilities = model_data.get("capabilities")
        capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
        modalities = model_data.get("modalities", capabilities.get("modalities"))
        vision: bool | None = None
        if isinstance(model_data.get("vision"), bool):
            vision = model_data["vision"]
        elif isinstance(capabilities.get("vision"), bool):
            vision = capabilities["vision"]
        elif isinstance(modalities, list):
            vision = any(str(item).lower() in {"image", "vision"} for item in modalities)

        formats = model_data.get(
            "response_format_styles",
            capabilities.get("response_format_styles"),
        )
        response_formats = (
            frozenset(str(value) for value in formats) if isinstance(formats, list) else None
        )
        return ModelCapabilities(
            model=self.profile.model,
            vision=vision,
            response_format_styles=response_formats,
            raw=model_data,
        )

    def _validate_profile(self, capabilities: ModelCapabilities) -> None:
        formats = capabilities.response_format_styles
        if formats is not None and self.profile.response_format_style not in formats:
            self._raise_capability_error(
                f"Model {self.profile.model!r} does not report support for "
                f"response_format_style={self.profile.response_format_style!r}"
            )
        if self.profile.vision and capabilities.vision is False:
            self._raise_capability_error(
                f"Vision profile {self.profile.name!r} targets a model without vision support"
            )

    def _raise_capability_error(self, message: str) -> Never:
        MODEL_GATEWAY_REQUESTS.labels(
            self.profile.name,
            self.profile.model,
            ErrorClassification.CAPABILITY,
        ).inc()
        raise ModelCapabilityError(message)

    async def _acquire(self) -> None:
        if self._state.semaphore.locked() and self._state.waiting >= self._state.queue_limit:
            error = ModelQueueFullError(
                f"Queue limit reached for model profile {self.profile.name!r}"
            )
            MODEL_GATEWAY_REQUESTS.labels(
                self.profile.name,
                self.profile.model,
                error.classification,
            ).inc()
            raise error
        started = self._monotonic()
        self._state.waiting += 1
        MODEL_GATEWAY_QUEUE_DEPTH.labels(self.profile.name, self.profile.model).set(
            self._state.waiting
        )
        try:
            if self._state.semaphore.locked():
                await asyncio.wait_for(
                    self._state.semaphore.acquire(),
                    timeout=self.profile.queue_timeout_seconds,
                )
            else:
                await self._state.semaphore.acquire()
        except TimeoutError as exc:
            error = ModelQueueFullError(
                f"Queue wait timed out for model profile {self.profile.name!r}"
            )
            MODEL_GATEWAY_REQUESTS.labels(
                self.profile.name,
                self.profile.model,
                error.classification,
            ).inc()
            raise error from exc
        finally:
            self._state.waiting -= 1
            MODEL_GATEWAY_QUEUE_DEPTH.labels(self.profile.name, self.profile.model).set(
                self._state.waiting
            )
            MODEL_GATEWAY_WAIT.labels(self.profile.name, self.profile.model).observe(
                self._monotonic() - started
            )

    def _release(self) -> None:
        self._state.semaphore.release()

    def _check_circuit(self) -> None:
        opened_at = self._state.circuit_opened_at
        if opened_at is None:
            return
        if self._monotonic() - opened_at < self.profile.circuit_breaker_reset_seconds:
            MODEL_GATEWAY_CIRCUIT_OPEN.labels(self.profile.name, self.profile.model).set(1)
            MODEL_GATEWAY_REQUESTS.labels(
                self.profile.name,
                self.profile.model,
                ErrorClassification.CIRCUIT_OPEN,
            ).inc()
            raise CircuitOpenError(f"Circuit is open for model profile {self.profile.name!r}")
        if self._state.half_open_in_flight:
            MODEL_GATEWAY_REQUESTS.labels(
                self.profile.name,
                self.profile.model,
                ErrorClassification.CIRCUIT_OPEN,
            ).inc()
            raise CircuitOpenError(
                f"Circuit recovery probe is already running for model profile {self.profile.name!r}"
            )
        self._state.half_open_in_flight = True

    def _record_failure(self, error: ModelGatewayError) -> None:
        if error.classification not in {
            ErrorClassification.TRANSIENT,
            ErrorClassification.RATE_LIMITED,
            ErrorClassification.INVALID_RESPONSE,
        }:
            return
        self._state.half_open_in_flight = False
        self._state.failures += 1
        if self._state.failures >= self.profile.circuit_breaker_failure_threshold:
            self._state.circuit_opened_at = self._monotonic()
            MODEL_GATEWAY_CIRCUIT_OPEN.labels(self.profile.name, self.profile.model).set(1)

    def _record_success(self) -> None:
        self._state.failures = 0
        self._state.circuit_opened_at = None
        self._state.half_open_in_flight = False
        MODEL_GATEWAY_CIRCUIT_OPEN.labels(self.profile.name, self.profile.model).set(0)

    def _classify_exception(self, exc: Exception) -> ModelGatewayError:
        if isinstance(exc, ModelGatewayError):
            return exc
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
            return TransientModelError(f"Model endpoint transport failure: {exc}")
        return ModelGatewayError(f"Model gateway operation failed: {exc}")

    def _retry_delay(self, error: ModelGatewayError, attempt: int) -> float:
        if error.retry_after is not None:
            return min(error.retry_after, self.profile.retry_backoff_max_seconds)
        delay = min(
            self.profile.retry_backoff_base_seconds * (2 ** (attempt - 1)),
            self.profile.retry_backoff_max_seconds,
        )
        if self.profile.retry_jitter_seconds:
            delay += secrets.SystemRandom().uniform(0, self.profile.retry_jitter_seconds)
        return float(delay)
