from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from typer.testing import CliRunner

import docie_bench.api as main_api
import docie_bench.asr.api as asr_api
import docie_bench.cli as cli
from docie_bench import security
from docie_bench.asr.backend import ASRDependencyUnavailableError, FasterWhisperBackend
from docie_bench.asr.benchmark import load_asr_manifest, run_asr_benchmark
from docie_bench.asr.formats import (
    TranscriptionFormat,
    render_srt,
    render_transcription,
    render_vtt,
)
from docie_bench.asr.metrics import edit_distance, normalize_transcript, score_transcript
from docie_bench.asr.models import (
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)
from docie_bench.asr.routing import ASRRoute
from docie_bench.asr.uploads import (
    detect_audio_mime_type,
    store_validated_audio_upload,
)


def _upload(name: str, data: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=name,
        headers=Headers({"content-type": content_type}),
    )


def _wav(payload: bytes = b"audio") -> bytes:
    size = (36 + len(payload)).to_bytes(4, "little")
    return b"RIFF" + size + b"WAVEfmt " + b"\x00" * 24 + b"data" + payload


class FakeBackend:
    model_name = "small"
    backend_name = "fake-asr"

    def __init__(self, transcripts: dict[str, str] | None = None) -> None:
        self.transcripts = transcripts or {}
        self.paths: list[Path] = []
        self.options: list[TranscriptionOptions] = []

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        self.paths.append(audio_path)
        self.options.append(options)
        text = self.transcripts.get(audio_path.name, "hello world")
        return TranscriptionResult(
            text=text,
            language=options.language or "en",
            duration=2.0,
            processing_seconds=0.5,
            model=self.model_name,
            backend=self.backend_name,
            segments=(
                TranscriptionSegment(id=0, start=0.0, end=2.0, text=f" {text}"),
            ),
        )


class FakeASRClient:
    def __init__(self, *, fail: bool = False, **_kwargs: object) -> None:
        self.fail = fail

    async def __aenter__(self) -> FakeASRClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("runtime offline")
        files = kwargs["files"]
        assert isinstance(files, dict)
        assert files["file"][1].read().startswith(b"RIFF")
        assert url == "http://asr-runtime:8093/v1/audio/transcriptions"
        return httpx.Response(
            200,
            json={"text": "hello world", "backend": "fake-asr"},
            headers={"content-type": "application/json"},
        )


def _install_managed_route(
    monkeypatch: pytest.MonkeyPatch, *, fail: bool = False
) -> None:
    monkeypatch.setattr(
        asr_api,
        "resolve_asr_route",
        lambda _model: ASRRoute("asr-default", "http://asr-runtime:8093/v1", "whisper-small"),
    )
    monkeypatch.setattr(
        asr_api.httpx,
        "AsyncClient",
        lambda **kwargs: FakeASRClient(fail=fail, **kwargs),
    )
    monkeypatch.setattr(asr_api, "stamp", lambda _name: None)


def test_transcript_normalization_and_edit_distance() -> None:
    assert normalize_transcript("  HéLLo—WORLD!  ") == "héllo world"
    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    score = score_transcript("hello brave world", "hello world")
    assert score.reference_words == 3
    assert score.word_errors == 1
    assert score.wer == pytest.approx(1 / 3)
    assert score.cer > 0


def test_empty_reference_rates_are_stable() -> None:
    assert score_transcript("", "").wer == 0.0
    assert score_transcript("", "hallucination").wer == 1.0


def test_openai_formats_include_segments_and_precise_timestamps() -> None:
    result = TranscriptionResult(
        text="hello world",
        language="en",
        duration=3661.234,
        processing_seconds=4.0,
        model="small",
        backend="fake",
        segments=(
            TranscriptionSegment(
                id=3,
                seek=12,
                start=0.125,
                end=3661.234,
                text=" hello world",
                tokens=(1, 2),
                avg_logprob=-0.2,
            ),
        ),
    )
    assert "00:00:00,125 --> 01:01:01,234" in render_srt(result)
    assert render_vtt(result).startswith("WEBVTT\n\n00:00:00.125")
    payload, media_type = render_transcription(result, TranscriptionFormat.VERBOSE_JSON)
    assert media_type == "application/json"
    assert isinstance(payload, dict)
    assert payload["segments"][0]["tokens"] == [1, 2]
    assert payload["real_time_factor"] == pytest.approx(4.0 / 3661.234)


@pytest.mark.parametrize(
    ("header", "mime_type"),
    [
        (_wav(), "audio/wav"),
        (b"fLaCpayload", "audio/flac"),
        (b"OggSpayload", "audio/ogg"),
        (b"ID3payload", "audio/mpeg"),
        (b"\xff\xfbpayload", "audio/mpeg"),
        (b"\x00\x00\x00\x18ftypM4A payload", "audio/mp4"),
        (b"\x1aE\xdf\xa3payload", "audio/webm"),
    ],
)
def test_audio_magic_detection(header: bytes, mime_type: str) -> None:
    assert detect_audio_mime_type(header) == mime_type


@pytest.mark.asyncio
async def test_audio_upload_is_streamed_and_validated() -> None:
    stored = await store_validated_audio_upload(
        _upload("clip.wav", _wav(), "audio/x-wav"),
        max_bytes=1_000,
        allowed_mime_types={"audio/wav"},
    )
    try:
        assert stored.path.read_bytes() == _wav()
        assert stored.mime_type == "audio/wav"
        assert stored.size_bytes == len(_wav())
    finally:
        stored.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_audio_upload_rejects_disguised_and_oversized_files() -> None:
    with pytest.raises(HTTPException) as disguised:
        await store_validated_audio_upload(
            _upload("clip.wav", b"not wave audio", "audio/wav"),
            max_bytes=1_000,
            allowed_mime_types={"audio/wav"},
        )
    assert disguised.value.status_code == 415

    with pytest.raises(HTTPException) as oversized:
        await store_validated_audio_upload(
            _upload("clip.wav", _wav(b"x" * 100), "audio/wav"),
            max_bytes=32,
            allowed_mime_types={"audio/wav"},
        )
    assert oversized.value.status_code == 413


def test_faster_whisper_adapter_is_lazy_and_maps_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class WhisperModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            calls.append((model, kwargs))

        def transcribe(self, path: str, **kwargs: object):
            segment = SimpleNamespace(
                id=0,
                seek=0,
                start=0.0,
                end=1.5,
                text=" hello",
                tokens=[1, 2],
                temperature=0.0,
                avg_logprob=-0.1,
                compression_ratio=1.2,
                no_speech_prob=0.01,
            )
            return iter([segment]), SimpleNamespace(language="en", duration=1.5)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=WhisperModel))
    backend = FasterWhisperBackend(model="tiny", beam_size=3, vad_filter=True)
    assert calls == []
    result = backend.transcribe(Path("clip.wav"), TranscriptionOptions(language="en"))
    assert calls[0][0] == "tiny"
    assert result.text == "hello"
    assert result.segments[0].tokens == (1, 2)
    assert result.duration == 1.5


def test_faster_whisper_missing_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    backend = FasterWhisperBackend(model="tiny")
    with pytest.raises(ASRDependencyUnavailableError, match=r"\[asr\]"):
        backend.transcribe(Path("clip.wav"), TranscriptionOptions())


def test_transcription_endpoint_returns_verbose_json_and_cleans_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_managed_route(monkeypatch)
    response = TestClient(main_api.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", _wav(), "audio/wav")},
        data={
            "model": "asr-default",
            "language": "fr",
            "prompt": "meeting",
            "temperature": "0.25",
            "response_format": "verbose_json",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hello world"
    assert response.json()["backend"] == "fake-asr"


def test_transcription_endpoint_rejects_unknown_model_before_reading_audio() -> None:
    response = TestClient(main_api.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.exe", b"", "application/octet-stream")},
        data={"model": "download-anything"},
    )
    assert response.status_code == 404
    assert "Unknown ASR" in response.json()["detail"]


def test_transcription_endpoint_uses_platform_api_key_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = security.TenantQuotaManager(
        api_keys={"secret": "tenant-a"},
        auth_required=True,
        requests_per_window=10,
        window_seconds=60,
        max_concurrent=2,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    _install_managed_route(monkeypatch)
    client = TestClient(main_api.app)
    request = {
        "files": {"file": ("clip.wav", _wav(), "audio/wav")},
        "data": {"model": "asr-default"},
    }
    assert client.post("/v1/audio/transcriptions", **request).status_code == 401
    response = client.post(
        "/v1/audio/transcriptions",
        headers={"X-API-Key": "secret"},
        **request,
    )
    assert response.status_code == 200


def test_transcription_endpoint_maps_runtime_unavailability_and_cleans_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_managed_route(monkeypatch, fail=True)
    response = TestClient(main_api.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", _wav(), "audio/wav")},
        data={"model": "asr-default"},
    )
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


def test_asr_manifest_and_corpus_benchmark_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(_wav())
    second.write_bytes(_wav(b"other"))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "one", "audio": "first.wav", "reference": "hello world"}
                ),
                json.dumps(
                    {"id": "two", "audio": "second.wav", "reference": "good night"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    backend = FakeBackend({"first.wav": "hello world", "second.wav": "good day"})
    output = tmp_path / "run"
    result = run_asr_benchmark(
        manifest_path=dataset,
        backend=backend,
        output_dir=output,
    )
    assert result.metrics["items"] == 2
    assert result.metrics["word_errors"] == 1
    assert result.metrics["reference_words"] == 4
    assert result.metrics["wer"] == 0.25
    assert result.metrics["real_time_factor"] == 0.25
    assert result.predictions_path.read_text(encoding="utf-8").count("\n") == 2
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["kind"] == "asr"
    assert "Corpus WER: 0.2500" in result.report_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_asr_benchmark(manifest_path=dataset, backend=backend, output_dir=output)


def test_asr_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(_wav())
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        '{"id":"same","audio":"clip.wav","reference":"a"}\n'
        '{"id":"same","audio":"clip.wav","reference":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate ASR manifest id"):
        load_asr_manifest(manifest)


def test_asr_cli_transcribe_supports_machine_readable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(_wav())
    monkeypatch.setattr(cli, "_configured_asr_backend", lambda _model=None: FakeBackend())
    result = CliRunner().invoke(cli.app, ["asr", "transcribe", str(audio), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"text": "hello world"}
