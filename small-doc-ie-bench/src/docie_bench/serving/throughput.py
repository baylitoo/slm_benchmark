"""Measured per-deployment inference throughput for the ``serving`` service.

The Studio wants to say "≈47 tok/s · TTFT 380 ms" next to a hot deployment.
There is exactly one honest way to get that number: send a request and time it.
It is NOT derivable from the CPU model name. Two machines reporting the same
``model name`` in ``/proc/cpuinfo`` differ by memory bandwidth, core count,
thread pinning, cgroup CPU quota, AVX-512/AMX availability, whether another
runtime is co-resident, and the quantization of the GGUF being served — a
lookup table keyed on the CPU string would produce a confident number that is
wrong by a factor of several. This module therefore measures, and publishes
nothing at all when it has not measured (the same discipline
``serving.sizing`` applies with "size unknown" / ``unpriceable``).

What is measured (one fixed probe, so numbers are comparable ACROSS MODELS):

* one non-streaming ``POST /v1/chat/completions`` against the deployment's own
  endpoint — never through the gateway, because the gateway stamps the recency
  sidecars the idle-TTL unload keys on, and a background probe that counts as
  "served" would silently pin every probed deployment hot forever;
* a fixed synthetic prompt (:data:`PROBE_PROMPT`, ~512 tokens for a typical
  BPE tokenizer), a fixed :data:`PROBE_MAX_TOKENS` cap and temperature 0;
* the ACTUAL prompt/completion token counts are stored alongside the numbers —
  "~512 tokens" is a property of a tokenizer, not of a string, so the record
  carries what the server actually counted rather than asserting the intent.

Where the numbers come from, in preference order:

``timings``
    llama.cpp's server returns a ``timings`` object on every completion.
    ``predicted_per_second`` is the server's own generation rate, measured
    without the HTTP round-trip we would otherwise be timing, and
    ``prompt_ms`` is the server-reported PREFILL duration. Prefill is what a
    streaming client would experience as time-to-first-token, but it is a
    server-reported number, not a client-observed one — hence the explicit
    ``source`` on every sample so nobody later mistakes the two.
``wall-clock``
    No ``timings`` (any non-llama.cpp OpenAI-compatible runtime): tokens/sec is
    ``usage.completion_tokens / elapsed``. ``ttft_ms`` stays **None** here.
    A non-streamed round-trip is not a time-to-first-token, and recording it as
    one would be precisely the fabricated metric this codebase refuses to ship.
``unmeasured``
    The response arrived but carried no usable token count, or generation
    stopped below :data:`MIN_COMPLETION_TOKENS` (temperature 0 hits EOS early on
    some models, and a rate computed over three tokens is noise). No number is
    published.

Not every deployment is chat. :func:`probe_applicability` refuses to probe
encoder/analyzer runtimes (they answer entity analysis, not token generation),
embedding deployments (``/v1/embeddings``: there are no generated tokens to
time) and remote deployments (a third-party endpoint's latency describes the
provider's queueing, not this node — and a background loop must not send
billable requests). Those record "not applicable", never a zero.

Samples are persisted as one tiny JSON sidecar per MODEL PATH under
``<serving-home>/throughput/``, mirroring :class:`~.resources.FootprintStore`
down to the key scheme and the never-raise discipline: a redeploy under a new
deployment name, or a second replica of the same model, inherits the
measurement instead of re-probing.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docie_bench.serving.resources import _serving_home
from docie_bench.serving.resources import footprint_key as model_key
from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec

logger = logging.getLogger(__name__)

# ------------------------------------------------------------ probe constants

# The fixed probe prompt. An instruction plus filler, not bare filler: a model
# handed only repeated text at temperature 0 will often emit EOS immediately,
# and a probe that generates nothing measures nothing. Repeated deterministic
# filler keeps the constant readable while landing near 512 tokens on a typical
# BPE tokenizer — the ACTUAL count is recorded per sample, never assumed.
_PROBE_FILLER_SENTENCE = (
    "The invoice line item was recorded on the ledger and the running total "
    "was recomputed from the unit price and the quantity. "
)
PROBE_FILLER_REPEATS = 30
PROBE_PROMPT = (
    "Summarize the following text in one short paragraph.\n\n"
    + (_PROBE_FILLER_SENTENCE * PROBE_FILLER_REPEATS).strip()
)

# Fixed generation cap and sampling for every probe: comparability across
# models depends on the request being identical, not on it being large.
PROBE_MAX_TOKENS = 64
PROBE_TEMPERATURE = 0.0

# Below this many generated tokens a rate is noise, not a measurement (an early
# EOS, or a runtime that reports usage counts of 0 — the transformers server
# does exactly that). Such a probe records "unmeasured", never a number.
MIN_COMPLETION_TOKENS = 8

# How long a sample stays current. Past it the sample is STALE: a hot
# deployment gets re-probed, and until a fresh number lands every reader is
# told the age rather than served the old number as if it were current.
THROUGHPUT_TTL_S = 6 * 3600.0

# Probe request timeout. Generous: a busy CPU-only runtime can take tens of
# seconds for 64 tokens, and a timeout that fires early would record nothing
# forever on exactly the slow nodes the number matters most for.
PROBE_TIMEOUT_S = 120.0

# Sample sources (also the published ``throughput_source`` value).
SOURCE_TIMINGS = "timings"
SOURCE_WALL_CLOCK = "wall-clock"
SOURCE_UNMEASURED = "unmeasured"
SOURCE_NOT_APPLICABLE = "not-applicable"


# ------------------------------------------------------------------- samples


@dataclass(frozen=True)
class ThroughputSample:
    """One measurement of one model path, with its provenance attached.

    ``tokens_per_second``/``ttft_ms`` are ``None`` whenever they were not
    measured — never 0, which a reader would render as a real (terrible)
    number. ``ttft_ms`` is additionally ``None`` on the ``wall-clock`` source
    by construction (see the module docstring).
    """

    measured_at: float
    source: str
    tokens_per_second: float | None = None
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    max_tokens: int = PROBE_MAX_TOKENS
    temperature: float = PROBE_TEMPERATURE
    elapsed_ms: float | None = None
    detail: str | None = None

    @property
    def measured(self) -> bool:
        """True when this sample carries a real generation rate."""
        return self.tokens_per_second is not None and self.source in {
            SOURCE_TIMINGS,
            SOURCE_WALL_CLOCK,
        }

    def age_s(self, now: float) -> float:
        return max(now - self.measured_at, 0.0)

    def is_fresh(self, now: float, *, ttl_s: float = THROUGHPUT_TTL_S) -> bool:
        """Within the TTL — i.e. safe to publish as a current measurement."""
        return self.age_s(now) <= ttl_s

    def to_json(self) -> dict[str, Any]:
        return {
            "measured_at": self.measured_at,
            "source": self.source,
            "tokens_per_second": self.tokens_per_second,
            "ttft_ms": self.ttft_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "elapsed_ms": self.elapsed_ms,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, payload: Any) -> ThroughputSample | None:
        """Parse a sidecar payload; ``None`` for anything unusable."""
        if not isinstance(payload, dict):
            return None
        measured_at = _as_float(payload.get("measured_at"))
        source = payload.get("source")
        if measured_at is None or not isinstance(source, str) or not source:
            return None
        return cls(
            measured_at=measured_at,
            source=source,
            tokens_per_second=_as_positive_float(payload.get("tokens_per_second")),
            ttft_ms=_as_non_negative_float(payload.get("ttft_ms")),
            prompt_tokens=_as_non_negative_int(payload.get("prompt_tokens")),
            completion_tokens=_as_non_negative_int(payload.get("completion_tokens")),
            max_tokens=_as_non_negative_int(payload.get("max_tokens")) or PROBE_MAX_TOKENS,
            temperature=_as_float(payload.get("temperature")) or 0.0,
            elapsed_ms=_as_non_negative_float(payload.get("elapsed_ms")),
            detail=payload.get("detail") if isinstance(payload.get("detail"), str) else None,
        )


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN/inf
        return None
    return number


def _as_positive_float(value: Any) -> float | None:
    number = _as_float(value)
    return number if number is not None and number > 0 else None


def _as_non_negative_float(value: Any) -> float | None:
    number = _as_float(value)
    return number if number is not None and number >= 0 else None


def _as_non_negative_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None and number >= 0 else None


# ---------------------------------------------------------------- parse a probe


def parse_probe_payload(
    payload: Any, *, elapsed_s: float, now: float
) -> ThroughputSample:
    """Turn one chat-completion response into a sample. Never raises.

    Prefers llama.cpp's ``timings`` (server-measured, so the HTTP round-trip is
    excluded), falls back to wall-clock over ``usage.completion_tokens``, and
    degrades to ``unmeasured`` when neither yields a defensible rate. A
    malformed ``timings`` (wrong type, missing or non-numeric fields) is not an
    error — it simply falls through to the next source.
    """
    elapsed_ms = max(elapsed_s, 0.0) * 1000.0
    if not isinstance(payload, dict):
        return ThroughputSample(
            measured_at=now,
            source=SOURCE_UNMEASURED,
            elapsed_ms=elapsed_ms,
            detail="response was not a JSON object",
        )

    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    usage_prompt = _as_non_negative_int(usage.get("prompt_tokens"))
    usage_completion = _as_non_negative_int(usage.get("completion_tokens"))

    timings = payload.get("timings")
    if isinstance(timings, dict):
        rate = _as_positive_float(timings.get("predicted_per_second"))
        predicted_n = _as_non_negative_int(timings.get("predicted_n"))
        prompt_n = _as_non_negative_int(timings.get("prompt_n"))
        generated = predicted_n if predicted_n is not None else usage_completion
        if rate is not None and generated is not None and generated >= MIN_COMPLETION_TOKENS:
            return ThroughputSample(
                measured_at=now,
                source=SOURCE_TIMINGS,
                tokens_per_second=rate,
                # Server-reported PREFILL duration, not a client-observed TTFT.
                ttft_ms=_as_non_negative_float(timings.get("prompt_ms")),
                prompt_tokens=prompt_n if prompt_n is not None else usage_prompt,
                completion_tokens=generated,
                elapsed_ms=elapsed_ms,
            )

    if (
        usage_completion is not None
        and usage_completion >= MIN_COMPLETION_TOKENS
        and elapsed_s > 0
    ):
        return ThroughputSample(
            measured_at=now,
            source=SOURCE_WALL_CLOCK,
            tokens_per_second=usage_completion / elapsed_s,
            # Deliberately None: a non-streamed round-trip is not a TTFT.
            ttft_ms=None,
            prompt_tokens=usage_prompt,
            completion_tokens=usage_completion,
            elapsed_ms=elapsed_ms,
        )

    generated = usage_completion if usage_completion is not None else 0
    return ThroughputSample(
        measured_at=now,
        source=SOURCE_UNMEASURED,
        prompt_tokens=usage_prompt,
        completion_tokens=usage_completion,
        elapsed_ms=elapsed_ms,
        detail=(
            f"generated {generated} tokens (< {MIN_COMPLETION_TOKENS}): "
            "too few to measure a rate"
        ),
    )


# ------------------------------------------------------------- applicability


def probe_applicability(launch: RuntimeLaunchSpec) -> tuple[bool, str]:
    """``(probeable, reason)`` for one launch — the per-kind honesty gate.

    A generation rate only means something for a runtime that generates tokens
    on ``/v1/chat/completions``. Everything else records "not applicable"
    rather than a zero that reads like a measurement.
    """
    if launch.runtime == RuntimeKind.ENCODER:
        return False, "encoder deployments analyze text; they do not generate tokens"
    if launch.runtime == RuntimeKind.REMOTE:
        return False, (
            "remote endpoint: a measurement would describe the provider's "
            "queueing, not this node"
        )
    if "--embedding" in launch.extra_args:
        return False, "embedding deployments answer /v1/embeddings; no tokens are generated"
    return True, ""


# ------------------------------------------------------------------- the probe


@dataclass(frozen=True)
class ProbeRequest:
    """Everything one probe needs, captured under the supervisor lock.

    Frozen and self-contained on purpose: the probe executes AFTER the lock is
    released (it blocks on HTTP), so it must never reach back into the
    supervisor for a record that may have changed underneath it.
    """

    name: str
    model: str  # launch model path — the sidecar key
    endpoint: str  # advertised base URL including /v1
    alias: str  # served model id


# (url, json body, timeout) -> decoded JSON payload. Injectable for tests.
ProbeTransport = Callable[[str, dict[str, Any], float], Any]


def _urllib_transport(url: str, body: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(  # noqa: S310 - fixed http(s) deployment endpoint
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def probe_deployment(
    request: ProbeRequest,
    *,
    transport: ProbeTransport = _urllib_transport,
    timeout_s: float = PROBE_TIMEOUT_S,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
) -> ThroughputSample | None:
    """Run one probe. ``None`` when the request itself failed (never raises).

    A transport failure returns ``None`` rather than an ``unmeasured`` sample
    so a transient hiccup is not persisted as this model's verdict for the
    whole TTL — the caller's per-deployment throttle spaces the retry instead.
    """
    url = request.endpoint.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": request.alias,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": PROBE_TEMPERATURE,
        "stream": False,
    }
    started = monotonic()
    try:
        payload = transport(url, body, timeout_s)
    except (OSError, urllib.error.URLError, ValueError, TimeoutError) as exc:
        logger.info("throughput probe of %s failed: %s", request.name, exc)
        return None
    except Exception:  # noqa: BLE001 - an exotic transport error is not a cycle failure
        logger.warning("throughput probe of %s failed", request.name, exc_info=True)
        return None
    return parse_probe_payload(payload, elapsed_s=monotonic() - started, now=clock())


# -------------------------------------------------------------- persisted store


class ThroughputStore:
    """Per-MODEL-PATH throughput samples, persisted on the serving volume.

    One tiny JSON sidecar per model under ``<serving-home>/throughput/``,
    written atomically via ``os.replace`` — the same pattern, directory
    conventions and never-raise discipline as
    :class:`~.resources.FootprintStore`, and the same key scheme (a digest of
    the FULL model path, so two store models whose weights are both named
    ``model.gguf`` never collide and a redeploy under a new deployment name
    inherits the measurement).

    One deliberate difference from ``FootprintStore.record_steady``: writes are
    LAST-WRITE-WINS, not monotonic-max. A footprint must stay conservative
    because under-counting over-commits the node; a throughput number must stay
    CURRENT, because the honest answer after a node gets busier is the lower
    new number, not the best one ever seen.
    """

    def __init__(self, home: Path | None = None) -> None:
        self._directory = (home if home is not None else _serving_home()) / "throughput"

    @property
    def directory(self) -> Path:
        return self._directory

    def get(self, model: str) -> ThroughputSample | None:
        """Last recorded sample for ``model`` (``None`` = never measured)."""
        path = self._directory / model_key(model)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return ThroughputSample.from_json(payload)

    def record(self, model: str, sample: ThroughputSample) -> None:
        """Persist one sample (last-write-wins; best-effort, never raises)."""
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            target = self._directory / model_key(model)
            temporary = self._directory / f".{target.name}.tmp"
            temporary.write_text(
                json.dumps(sample.to_json(), sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, target)
        except OSError:  # pragma: no cover - a disk hiccup must not fail a cycle
            logger.warning("could not persist throughput for %r", model, exc_info=True)


__all__ = [
    "MIN_COMPLETION_TOKENS",
    "PROBE_MAX_TOKENS",
    "PROBE_PROMPT",
    "PROBE_TEMPERATURE",
    "PROBE_TIMEOUT_S",
    "SOURCE_NOT_APPLICABLE",
    "SOURCE_TIMINGS",
    "SOURCE_UNMEASURED",
    "SOURCE_WALL_CLOCK",
    "THROUGHPUT_TTL_S",
    "ProbeRequest",
    "ProbeTransport",
    "ThroughputSample",
    "ThroughputStore",
    "parse_probe_payload",
    "probe_applicability",
    "probe_deployment",
]
