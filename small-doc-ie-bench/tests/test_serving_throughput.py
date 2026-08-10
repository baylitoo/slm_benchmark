"""Measured per-deployment throughput: the probe, the store, and the trigger.

Same discipline as ``test_serving_reconciler`` / ``test_serving_resources``:
scripted adapters, injected clocks and transports — no real processes, no
network, no Postgres. The point of every test here is that the system either
publishes a number it actually measured or publishes nothing at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docie_bench.serving.reconciler import ObservedDeployment, ServingReconciler
from docie_bench.serving.runtime import (
    HealthResult,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor
from docie_bench.serving.throughput import (
    MIN_COMPLETION_TOKENS,
    PROBE_MAX_TOKENS,
    PROBE_PROMPT,
    PROBE_TEMPERATURE,
    SOURCE_NOT_APPLICABLE,
    SOURCE_TIMINGS,
    SOURCE_UNMEASURED,
    SOURCE_WALL_CLOCK,
    THROUGHPUT_TTL_S,
    ProbeRequest,
    ThroughputSample,
    ThroughputStore,
    parse_probe_payload,
    probe_applicability,
    probe_deployment,
)

SPAWN_CREATE_TIME = 1000.0


# --------------------------------------------------------------- the parser


def _timings_response(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576},
        "timings": {
            "prompt_n": 512,
            "prompt_ms": 380.0,
            "prompt_per_second": 1347.4,
            "predicted_n": 64,
            "predicted_ms": 1361.7,
            "predicted_per_second": 47.0,
        },
    }
    payload.update(overrides)
    return payload


def test_parser_prefers_the_runtimes_own_timings() -> None:
    """llama.cpp measures generation server-side; that beats timing the HTTP
    round-trip ourselves, and its prompt_ms is the prefill/TTFT number."""
    sample = parse_probe_payload(_timings_response(), elapsed_s=9.9, now=100.0)

    assert sample.source == SOURCE_TIMINGS
    assert sample.tokens_per_second == pytest.approx(47.0)
    assert sample.ttft_ms == pytest.approx(380.0)
    assert sample.prompt_tokens == 512
    assert sample.completion_tokens == 64
    assert sample.measured is True
    # The wall-clock elapsed is recorded for transparency but NOT used as the rate.
    assert sample.elapsed_ms == pytest.approx(9900.0)


def test_parser_falls_back_to_wall_clock_and_refuses_to_invent_a_ttft() -> None:
    """No timings (any non-llama.cpp OpenAI-compatible runtime): the rate comes
    from usage counts over elapsed time, and ttft stays None — a non-streamed
    round-trip is not a time-to-first-token."""
    payload = {"usage": {"prompt_tokens": 500, "completion_tokens": 64}}

    sample = parse_probe_payload(payload, elapsed_s=2.0, now=100.0)

    assert sample.source == SOURCE_WALL_CLOCK
    assert sample.tokens_per_second == pytest.approx(32.0)
    assert sample.ttft_ms is None
    assert sample.measured is True


@pytest.mark.parametrize(
    "timings",
    [
        "not-an-object",
        [],
        {},
        {"predicted_per_second": "fast"},
        {"predicted_per_second": None, "predicted_n": 64},
        {"predicted_per_second": 0, "predicted_n": 64},
        {"predicted_per_second": -12.0, "predicted_n": 64},
    ],
    ids=["string", "list", "empty", "text-rate", "null-rate", "zero-rate", "negative-rate"],
)
def test_parser_treats_malformed_timings_as_absent(timings: Any) -> None:
    """A malformed timings block is not an error — it falls through to the next
    source. A runtime that garbles its own telemetry must not take the whole
    measurement down with it."""
    payload = _timings_response(timings=timings)

    sample = parse_probe_payload(payload, elapsed_s=2.0, now=100.0)

    assert sample.source == SOURCE_WALL_CLOCK
    assert sample.tokens_per_second == pytest.approx(32.0)


def test_parser_records_unmeasured_when_generation_stopped_too_early() -> None:
    """Temperature 0 hits EOS early on some models. A rate computed over three
    tokens is noise, so no number is published at all."""
    payload = {"usage": {"prompt_tokens": 500, "completion_tokens": 3}}

    sample = parse_probe_payload(payload, elapsed_s=2.0, now=100.0)

    assert sample.source == SOURCE_UNMEASURED
    assert sample.tokens_per_second is None
    assert sample.measured is False
    assert str(MIN_COMPLETION_TOKENS) in (sample.detail or "")


def test_parser_rejects_a_runtime_that_reports_zero_usage() -> None:
    """The transformers server answers chat completions but reports usage
    counts of 0. Zero tokens over any elapsed time is not 0 tok/s — it is no
    measurement, and must be published as such."""
    payload = {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    sample = parse_probe_payload(payload, elapsed_s=4.0, now=100.0)

    assert sample.source == SOURCE_UNMEASURED
    assert sample.tokens_per_second is None


@pytest.mark.parametrize("payload", ["", None, [], 42], ids=["str", "none", "list", "int"])
def test_parser_survives_a_response_that_is_not_an_object(payload: Any) -> None:
    sample = parse_probe_payload(payload, elapsed_s=1.0, now=100.0)

    assert sample.source == SOURCE_UNMEASURED
    assert sample.tokens_per_second is None


def test_parser_ignores_a_timings_rate_whose_token_count_is_too_small() -> None:
    """A server-reported rate over 2 generated tokens is as meaningless as a
    client-timed one; it falls through rather than being trusted for its
    provenance alone."""
    payload = _timings_response(
        timings={"predicted_per_second": 900.0, "predicted_n": 2, "prompt_ms": 10.0},
        usage={"prompt_tokens": 500, "completion_tokens": 2},
    )

    sample = parse_probe_payload(payload, elapsed_s=1.0, now=100.0)

    assert sample.source == SOURCE_UNMEASURED


# ------------------------------------------------------------ the probe call


def test_probe_sends_one_fixed_deterministic_request() -> None:
    """Comparability across models depends on every model getting the SAME
    request: same prompt, same cap, temperature 0, non-streaming."""
    seen: list[tuple[str, dict[str, Any], float]] = []

    def transport(url: str, body: dict[str, Any], timeout: float) -> Any:
        seen.append((url, body, timeout))
        return _timings_response()

    sample = probe_deployment(
        ProbeRequest(
            name="invoice",
            model="/models/invoice/model.gguf",
            endpoint="http://serving:8090/v1",
            alias="invoice",
        ),
        transport=transport,
        clock=lambda: 500.0,
        monotonic=iter([0.0, 2.0]).__next__,
    )

    url, body, _timeout = seen[0]
    assert url == "http://serving:8090/v1/chat/completions"
    assert body["model"] == "invoice"
    assert body["messages"] == [{"role": "user", "content": PROBE_PROMPT}]
    assert body["max_tokens"] == PROBE_MAX_TOKENS
    assert body["temperature"] == PROBE_TEMPERATURE
    assert body["stream"] is False
    assert sample is not None
    assert sample.measured_at == 500.0
    assert sample.source == SOURCE_TIMINGS


def test_probe_returns_none_when_the_request_itself_fails() -> None:
    """A transport failure records NOTHING (not an 'unmeasured' verdict): a
    transient hiccup must not become this model's answer for the whole TTL."""

    def transport(url: str, body: dict[str, Any], timeout: float) -> Any:
        raise OSError("connection refused")

    assert (
        probe_deployment(
            ProbeRequest("invoice", "/models/m.gguf", "http://h:1/v1", "invoice"),
            transport=transport,
        )
        is None
    )


def test_probe_returns_none_on_an_unexpected_transport_error() -> None:
    def transport(url: str, body: dict[str, Any], timeout: float) -> Any:
        raise RuntimeError("something exotic")

    assert (
        probe_deployment(
            ProbeRequest("invoice", "/models/m.gguf", "http://h:1/v1", "invoice"),
            transport=transport,
        )
        is None
    )


# ------------------------------------------------------------ applicability


def _launch(runtime: RuntimeKind, **overrides: Any) -> RuntimeLaunchSpec:
    values: dict[str, Any] = {
        "runtime": runtime,
        "model": "/models/x/model.gguf",
        "alias": "x",
        "port": 8090,
    }
    values.update(overrides)
    return RuntimeLaunchSpec(**values)


def test_chat_and_vision_runtimes_are_probeable() -> None:
    for runtime in (RuntimeKind.LLAMACPP, RuntimeKind.OLLAMA, RuntimeKind.VLLM):
        probeable, reason = probe_applicability(_launch(runtime))
        assert probeable is True
        assert reason == ""


def test_encoder_deployments_are_never_probed() -> None:
    probeable, reason = probe_applicability(_launch(RuntimeKind.ENCODER))

    assert probeable is False
    assert "generate" in reason


def test_embedding_deployments_are_never_probed() -> None:
    """An embedding launch is a llama-server with --embedding: it answers
    /v1/embeddings and generates no tokens, so tok/s does not exist for it."""
    launch = _launch(RuntimeKind.LLAMACPP, extra_args=("--embedding", "--pooling", "mean"))

    probeable, reason = probe_applicability(launch)

    assert probeable is False
    assert "embedding" in reason.lower()


def test_remote_deployments_are_never_probed() -> None:
    """A third-party endpoint's speed describes the provider, not this node —
    and a background loop must not send billable requests."""
    probeable, reason = probe_applicability(_launch(RuntimeKind.REMOTE))

    assert probeable is False
    assert "provider" in reason


# ------------------------------------------------------------------- storage


def test_store_round_trips_a_sample_keyed_by_model_path(tmp_path: Path) -> None:
    store = ThroughputStore(home=tmp_path)
    sample = ThroughputSample(
        measured_at=1000.0,
        source=SOURCE_TIMINGS,
        tokens_per_second=47.0,
        ttft_ms=380.0,
        prompt_tokens=512,
        completion_tokens=64,
    )

    store.record("/models/invoice/model.gguf", sample)

    restored = store.get("/models/invoice/model.gguf")
    assert restored == sample


def test_store_keys_on_the_full_path_not_the_basename(tmp_path: Path) -> None:
    """The canonical store names EVERY model's weights ``model.gguf``; keying
    on the basename would make one model's speed answer for all of them."""
    store = ThroughputStore(home=tmp_path)
    store.record(
        "/models/a/model.gguf",
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=47.0),
    )

    assert store.get("/models/b/model.gguf") is None


def test_store_is_last_write_wins_not_max(tmp_path: Path) -> None:
    """Unlike the footprint sidecars (conservative => keep the max), a speed
    must stay CURRENT: after the node gets busier the honest answer is the
    lower new number, not the best one ever seen."""
    store = ThroughputStore(home=tmp_path)
    model = "/models/invoice/model.gguf"
    store.record(
        model,
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=90.0),
    )

    store.record(
        model,
        ThroughputSample(measured_at=2.0, source=SOURCE_TIMINGS, tokens_per_second=12.0),
    )

    restored = store.get(model)
    assert restored is not None
    assert restored.tokens_per_second == pytest.approx(12.0)


def test_store_reads_a_corrupt_sidecar_as_never_measured(tmp_path: Path) -> None:
    store = ThroughputStore(home=tmp_path)
    store.directory.mkdir(parents=True, exist_ok=True)
    store.record(
        "/models/invoice/model.gguf",
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=47.0),
    )
    next(store.directory.iterdir()).write_text("{not json", encoding="utf-8")

    assert store.get("/models/invoice/model.gguf") is None


def test_store_read_is_missing_directory_safe(tmp_path: Path) -> None:
    assert ThroughputStore(home=tmp_path).get("/models/x/model.gguf") is None


# ----------------------------------------------------------------- staleness


def test_a_sample_is_fresh_until_the_ttl_then_stale() -> None:
    sample = ThroughputSample(
        measured_at=1000.0, source=SOURCE_TIMINGS, tokens_per_second=47.0
    )

    assert sample.is_fresh(1000.0 + THROUGHPUT_TTL_S - 1) is True
    assert sample.is_fresh(1000.0 + THROUGHPUT_TTL_S) is True
    assert sample.is_fresh(1000.0 + THROUGHPUT_TTL_S + 1) is False
    assert sample.age_s(1000.0 + 60) == pytest.approx(60.0)


# ------------------------------------------------------- reconciler wiring


class ScriptedAdapter:
    """The reconciler test harness's fake runtime (no processes, no sockets)."""

    def __init__(self) -> None:
        self.next_pid = 100
        self.running: set[int] = set()

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        del log_path
        self.next_pid += 1
        self.running.add(self.next_pid)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        if pid is not None:
            self.running.discard(pid)

    def crash(self, pid: int | None) -> None:
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del spec, timeout
        return HealthResult(True, 200)


class RecordingProbe:
    """A probe that counts calls and returns a scripted sample (or raises)."""

    def __init__(self, sample: ThroughputSample | None = None, *, boom: bool = False) -> None:
        self.sample = sample
        self.boom = boom
        self.calls: list[ProbeRequest] = []

    def __call__(self, request: ProbeRequest) -> ThroughputSample | None:
        self.calls.append(request)
        if self.boom:
            raise RuntimeError("probe exploded")
        return self.sample


def _spec(name: str = "invoice", **launch_overrides: Any) -> DeploymentSpec:
    values: dict[str, Any] = {
        "runtime": RuntimeKind.LLAMACPP,
        "model": f"/models/{name}/model.gguf",
        "alias": name,
        "port": 8090,
    }
    values.update(launch_overrides)
    return DeploymentSpec(name=name, launch=RuntimeLaunchSpec(**values))


class FakeClock:
    def __init__(self, now: float = 10_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _build(
    tmp_path: Path,
    *,
    probe: Any,
    clock: FakeClock | None = None,
    probe_hot_cycles: int = 3,
) -> tuple[PersistentSupervisor, ServingReconciler, list[list[ObservedDeployment]]]:
    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter, RuntimeKind.ENCODER: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    published: list[list[ObservedDeployment]] = []
    reconciler = ServingReconciler(
        supervisor,
        interval_s=0.01,
        ready_miss_threshold=2,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 123_000_000,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=published.append,
        recency_home=tmp_path,
        clock=clock or FakeClock(),
        probe=probe,
        probe_hot_cycles=probe_hot_cycles,
        throughput=ThroughputStore(home=tmp_path),
    )
    return supervisor, reconciler, published


def _only(observations: list[ObservedDeployment], name: str) -> ObservedDeployment:
    matches = [obs for obs in observations if obs.name == name]
    assert len(matches) == 1
    return matches[0]


def test_a_hot_deployment_is_probed_then_publishes_the_measurement(
    tmp_path: Path,
) -> None:
    """The whole point: after a few hot cycles the deployment is measured, and
    the measurement rides out on the next observation."""
    clock = FakeClock()
    probe = RecordingProbe(
        ThroughputSample(
            measured_at=clock.now,
            source=SOURCE_TIMINGS,
            tokens_per_second=47.0,
            ttft_ms=380.0,
        )
    )
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, clock=clock)
    supervisor.deploy(_spec())

    # Nothing published before the model has been hot long enough.
    first = _only(reconciler.run_cycle(), "invoice")
    assert first.tokens_per_second is None
    assert probe.calls == []

    reconciler.run_cycle()
    reconciler.run_cycle()  # third hot cycle: the probe runs after the publish
    assert [request.name for request in probe.calls] == ["invoice"]
    assert probe.calls[0].model == "/models/invoice/model.gguf"
    assert probe.calls[0].endpoint == supervisor.get("invoice").endpoint

    observed = _only(reconciler.run_cycle(), "invoice")
    assert observed.tokens_per_second == pytest.approx(47.0)
    assert observed.ttft_ms == pytest.approx(380.0)
    assert observed.throughput_source == SOURCE_TIMINGS
    assert observed.throughput_measured_at is not None


def test_the_probe_waits_out_the_mmap_ramp(tmp_path: Path) -> None:
    """llama.cpp mmaps the GGUF and pages fault in after /health starts
    passing. A probe on the first hot cycle measures a cold cache — and that
    pessimistic number would be persisted per model path and inherited by every
    later redeploy."""
    probe = RecordingProbe(
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=47.0)
    )
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, probe_hot_cycles=6)
    supervisor.deploy(_spec())

    for _ in range(5):
        reconciler.run_cycle()
    assert probe.calls == []

    reconciler.run_cycle()
    assert len(probe.calls) == 1


def test_a_health_miss_restarts_the_hot_streak(tmp_path: Path) -> None:
    """Anything that is not a clean hot observation resets the wait, so a
    deployment that flapped is not measured mid-recovery."""
    probe = RecordingProbe(
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=47.0)
    )
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, probe_hot_cycles=3)
    record = supervisor.deploy(_spec())
    reconciler.run_cycle()
    reconciler.run_cycle()

    supervisor.adapters[RuntimeKind.LLAMACPP].crash(record.pid)  # type: ignore[attr-defined]
    reconciler.run_cycle()  # dead -> respawned; streak restarts
    reconciler.run_cycle()
    assert probe.calls == []

    reconciler.run_cycle()
    reconciler.run_cycle()
    assert len(probe.calls) == 1


def test_a_fresh_sample_is_reused_instead_of_reprobing(tmp_path: Path) -> None:
    """Keyed by MODEL PATH, so a redeploy under a new name — or a second
    replica of the same model — inherits the measurement for free."""
    clock = FakeClock()
    store = ThroughputStore(home=tmp_path)
    store.record(
        "/models/invoice/model.gguf",
        ThroughputSample(
            measured_at=clock.now - 60, source=SOURCE_TIMINGS, tokens_per_second=47.0
        ),
    )
    probe = RecordingProbe()
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, clock=clock)
    supervisor.deploy(_spec("replica-2", model="/models/invoice/model.gguf", port=8091))

    for _ in range(6):
        observations = reconciler.run_cycle()

    assert probe.calls == []
    assert _only(observations, "replica-2").tokens_per_second == pytest.approx(47.0)


def test_a_stale_sample_is_published_with_its_own_age_and_reprobed(
    tmp_path: Path,
) -> None:
    """Stale is surfaced, not laundered: the old number keeps its original
    measured_at (so readers can say how old it is) while a fresh probe runs."""
    clock = FakeClock()
    store = ThroughputStore(home=tmp_path)
    stale_at = clock.now - THROUGHPUT_TTL_S - 3600
    store.record(
        "/models/invoice/model.gguf",
        ThroughputSample(measured_at=stale_at, source=SOURCE_TIMINGS, tokens_per_second=47.0),
    )
    probe = RecordingProbe()
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, clock=clock)
    supervisor.deploy(_spec())

    for _ in range(3):
        observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.tokens_per_second == pytest.approx(47.0)
    assert observed.throughput_measured_at is not None
    assert observed.throughput_measured_at.timestamp() == pytest.approx(stale_at)
    assert len(probe.calls) == 1  # re-measured because it went stale


def test_an_encoder_deployment_reports_not_applicable_never_a_zero(
    tmp_path: Path,
) -> None:
    probe = RecordingProbe()
    supervisor, reconciler, _ = _build(tmp_path, probe=probe)
    supervisor.deploy(_spec("guard", runtime=RuntimeKind.ENCODER))

    for _ in range(5):
        observations = reconciler.run_cycle()

    observed = _only(observations, "guard")
    assert observed.throughput_source == SOURCE_NOT_APPLICABLE
    assert observed.tokens_per_second is None
    assert observed.ttft_ms is None
    assert probe.calls == []


def test_an_embedding_deployment_reports_not_applicable(tmp_path: Path) -> None:
    probe = RecordingProbe()
    supervisor, reconciler, _ = _build(tmp_path, probe=probe)
    supervisor.deploy(_spec("embed", extra_args=("--embedding", "--pooling", "mean")))

    for _ in range(5):
        observations = reconciler.run_cycle()

    assert _only(observations, "embed").throughput_source == SOURCE_NOT_APPLICABLE
    assert probe.calls == []


def test_a_failed_probe_is_throttled_not_retried_every_cycle(tmp_path: Path) -> None:
    """A probe that measured nothing writes no sidecar (so a transient bad
    reading is not persisted as the answer) — which means only the per-
    deployment throttle stands between it and a generation request per cycle.
    The attempt budget (tested separately) then stops the retries entirely."""
    clock = FakeClock()
    probe = RecordingProbe(None)  # transport failure
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, clock=clock)
    supervisor.deploy(_spec())

    for _ in range(12):
        reconciler.run_cycle()
    assert len(probe.calls) == 1

    clock.now += 299
    reconciler.run_cycle()
    assert len(probe.calls) == 1

    clock.now += 2
    reconciler.run_cycle()
    assert len(probe.calls) == 2


def test_a_runtime_that_can_never_be_measured_is_left_alone(tmp_path: Path) -> None:
    """The transformers server answers chat completions but reports usage
    counts of 0 by construction, so its probe can NEVER yield a number. Without
    an attempt budget the retry timer would fire a 64-token generation against
    it every five minutes forever, on this node, producing nothing."""
    clock = FakeClock()
    probe = RecordingProbe(None)
    supervisor, reconciler, _ = _build(
        tmp_path, probe=probe, clock=clock, probe_hot_cycles=1
    )
    supervisor.deploy(_spec())

    for _ in range(10):
        reconciler.run_cycle()
        clock.now += 301
    assert len(probe.calls) == 3  # three attempts, then quiet

    # A respawn is a NEW process — maybe the one that can be measured. The
    # attempt budget resets with it rather than condemning the deployment.
    supervisor.adapters[RuntimeKind.LLAMACPP].crash(  # type: ignore[attr-defined]
        supervisor.get("invoice").pid
    )
    reconciler.run_cycle()  # dead -> respawned, budget reset, probed again

    assert len(probe.calls) == 4


def test_at_most_one_probe_runs_per_cycle(tmp_path: Path) -> None:
    """A probe blocks on a real generation; several per cycle would stretch the
    reconcile interval and put avoidable load on the node."""
    probe = RecordingProbe(None)
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, probe_hot_cycles=1)
    supervisor.deploy(_spec("a", port=8090))
    supervisor.deploy(_spec("b", port=8091))
    supervisor.deploy(_spec("c", port=8092))

    reconciler.run_cycle()

    assert len(probe.calls) == 1


def test_a_probe_exception_never_breaks_the_cycle(tmp_path: Path) -> None:
    """The cycle's job is liveness and repair. A measurement is a nice-to-have
    and must never be able to take that down or change a phase."""
    probe = RecordingProbe(boom=True)
    supervisor, reconciler, published = _build(tmp_path, probe=probe, probe_hot_cycles=1)
    supervisor.deploy(_spec())

    for _ in range(3):
        observations = reconciler.run_cycle()

    assert len(probe.calls) >= 1
    observed = _only(observations, "invoice")
    assert observed.phase == "hot"
    assert observed.health_ok is True
    assert observed.tokens_per_second is None
    assert len(published) == 3


def test_an_unmeasurable_probe_result_is_not_persisted(tmp_path: Path) -> None:
    """An 'unmeasured' verdict is a fact about one attempt, not about the
    model. Persisting it would blank the model for the whole TTL."""
    probe = RecordingProbe(
        ThroughputSample(measured_at=1.0, source=SOURCE_UNMEASURED, detail="EOS at 3 tokens")
    )
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, probe_hot_cycles=1)
    supervisor.deploy(_spec())

    observations = reconciler.run_cycle()
    reconciler.run_cycle()

    assert ThroughputStore(home=tmp_path).get("/models/invoice/model.gguf") is None
    assert _only(observations, "invoice").tokens_per_second is None


def test_probing_does_not_count_as_serving_the_deployment(tmp_path: Path) -> None:
    """The probe must never look like traffic. It goes straight to the
    deployment's endpoint, not through the gateway that stamps the recency
    sidecars — otherwise every probed deployment would look permanently
    'just served' and the idle-TTL unload would silently stop working."""
    probe = RecordingProbe(
        ThroughputSample(measured_at=1.0, source=SOURCE_TIMINGS, tokens_per_second=47.0)
    )
    supervisor, reconciler, _ = _build(tmp_path, probe=probe, probe_hot_cycles=1)
    supervisor.deploy(_spec())
    assert supervisor.get("invoice").last_served is None

    for _ in range(4):
        reconciler.run_cycle()

    assert len(probe.calls) >= 1
    assert supervisor.get("invoice").last_served is None
