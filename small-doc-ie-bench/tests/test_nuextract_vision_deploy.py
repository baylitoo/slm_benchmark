"""PR-2 — NuExtract vision deploy path (stacked on PR-1).

Pure unit tests, no live stack: the live-deployment source is INJECTED
(fabricated ``DeploymentRecord``s), the Postgres catalog is STUBBED, and the
Ollama store is a fake on-disk manifest. They prove the contract PR-2 owns:

* the resolved ``ModelProfile`` for a NuExtract vision deployment carries
  ``vision=True`` (via the yaml-match path OR the catalog-family path),
  ``model`` == the served alias, and ``base_url`` == the deployment's own
  cross-container endpoint (PR-1's advertise URL, passed through verbatim);
* a ``needs_mmproj`` family can be seeded even when the pulled GGUF ships no
  projector layer, by supplying an explicit ``mmproj_source``;
* the family launch args emit ``--jinja`` + ``--mmproj`` for the vision family;
* a resolved vision profile drives the page-image extraction branch, not OCR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.serving.catalog import ModelStoreEntry, _to_view
from docie_bench.serving.model_store import ModelStore, ModelStoreError
from docie_bench.serving.profile_resolver import resolve_extraction_profile
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentRecord, DeploymentSpec
from docie_bench.vision import DocumentImage

# The cross-container advertise endpoint PR-1 records on the DeploymentRecord.
# It is deliberately NOT a loopback: the resolver's job is to pass it through
# verbatim so the api container reaches the runtime.
_ADVERTISE_ENDPOINT = "http://serving:8088/v1"

# models.yaml with a full nuextract3 vision profile (paging + non-default tuning,
# so inheritance is provably the yaml, not ModelProfile defaults) and NO profile
# whose upstream id equals an arbitrary deployment alias (forces the catalog path).
_MODELS_YAML = """\
profiles:
  studio_default:
    model: qwen3:4b
    base_url: http://localhost:11434/v1
    api_key: local-not-used

  nuextract3:
    model: nuextract3
    base_url: http://localhost:8088/v1
    api_key: local-not-used
    response_format_style: nuextract3
    prompt_profile: nuextract3
    temperature: 0.2
    max_tokens: 4096
    vision: true
    vision_max_pages: 8
    vision_pdf_dpi: 150
"""


@pytest.fixture
def models_config(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_YAML, encoding="utf-8")
    return path


def _live_record(
    *,
    name: str,
    alias: str,
    model: str = "/app/.serving/models/x/model.gguf",
    endpoint: str = _ADVERTISE_ENDPOINT,
) -> DeploymentRecord:
    """A READY llamacpp store-deploy record with a cross-container endpoint."""
    return DeploymentRecord(
        spec=DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP, model=model, alias=alias
            ),
        ),
        state=LifecycleState.READY,
        endpoint=endpoint,
    )


class _FakeCatalog:
    """Stub for ``ModelCatalog`` — returns a fixed store-entry view, no DB."""

    def __init__(self, view: dict[str, object] | None) -> None:
        self._view = view

    def get(self, name: str) -> dict[str, object] | None:
        return self._view


# ── (1) catalog-family path: vision inheritance is name-independent ─────────────


def test_catalog_family_yields_vision_for_arbitrary_alias(
    models_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A store deploy named anything (alias != 'nuextract3') still resolves vision
    # from the catalog family, so vision is NOT coupled to the yaml profile name.
    monkeypatch.setattr(
        "docie_bench.serving.catalog.ModelCatalog",
        lambda: _FakeCatalog({"family": "nuextract3"}),
    )
    record = _live_record(name="my-invoices", alias="my-invoices")

    profile = resolve_extraction_profile(
        deployment="my-invoices", models_config_path=models_config, deployments=[record]
    )

    assert profile.vision is True
    assert profile.model == "my-invoices"  # served alias (llama-server --alias)
    assert profile.base_url == _ADVERTISE_ENDPOINT  # PR-1 cross-container endpoint
    assert profile.response_format_style == "nuextract3"
    assert profile.prompt_profile == "nuextract3"
    # Paging is safe on the family-traits path: ModelProfile defaults (8/150).
    assert profile.vision_max_pages == 8
    assert profile.vision_pdf_dpi == 150


def test_catalog_openai_chat_family_is_not_vision(
    models_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-vision family must resolve vision=False (OCR path), never leak vision.
    monkeypatch.setattr(
        "docie_bench.serving.catalog.ModelCatalog",
        lambda: _FakeCatalog({"family": "openai_chat"}),
    )
    record = _live_record(name="plain", alias="plain")

    profile = resolve_extraction_profile(
        deployment="plain", models_config_path=models_config, deployments=[record]
    )

    assert profile.vision is False
    assert profile.response_format_style == "openai_json_schema"


# ── (2) yaml-match path: alias == upstream id inherits the FULL vision profile ──


def test_alias_nuextract3_inherits_full_yaml_vision_profile(models_config: Path) -> None:
    record = _live_record(
        name="nux-live", alias="nuextract3", model="/p/nux.gguf"
    )

    profile = resolve_extraction_profile(
        deployment="nux-live", models_config_path=models_config, deployments=[record]
    )

    assert profile.name == "nux-live"  # honest deployment label
    assert profile.model == "nuextract3"  # served alias
    assert profile.base_url == _ADVERTISE_ENDPOINT
    assert profile.vision is True
    assert profile.vision_pdf_dpi == 150
    assert profile.vision_max_pages == 8
    # Non-default tuning proves the WHOLE yaml profile was inherited (not defaults).
    assert profile.temperature == pytest.approx(0.2)
    assert profile.max_tokens == 4096
    assert profile.kind == "passthrough"


# ── (3) explicit mmproj seed path (gap: pulled GGUF ships no projector layer) ───


def _fake_ollama_home(
    tmp_path: Path,
    reference_parts: tuple[str, ...],
    tag: str,
    *,
    with_projector: bool,
) -> Path:
    home = tmp_path / "ollama"
    blobs = home / "blobs"
    blobs.mkdir(parents=True)

    def _blob(content: bytes) -> str:
        # Ollama blobs are content-addressed by their real sha256; the store now
        # verifies transferred blobs against the manifest digest, so the fixture
        # must use the true content hash (not a placeholder) to pass that check.
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        (blobs / digest.replace(":", "-")).write_bytes(content)
        return digest

    model_digest = _blob(b"GGUF-model-weights")
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}]
    if with_projector:
        proj_digest = _blob(b"MANIFEST-mmproj")
        layers.append(
            {"mediaType": "application/vnd.ollama.image.projector", "digest": proj_digest}
        )
    manifest_dir = home.joinpath("manifests", *reference_parts)
    manifest_dir.mkdir(parents=True)
    (manifest_dir / tag).write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return home


def test_seed_uses_explicit_mmproj_when_manifest_lacks_projector(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=False
    )
    mmproj = tmp_path / "external-mmproj.gguf"
    mmproj.write_bytes(b"EXTERNAL-mmproj")
    store = ModelStore(tmp_path / "models")

    entry = store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
        mmproj_source=mmproj,
    )

    assert entry.mmproj_path is not None
    assert entry.mmproj_path.is_file()
    assert entry.mmproj_path.read_bytes() == b"EXTERNAL-mmproj"
    # Round-trips + surfaces has_mmproj to the catalog view.
    assert store.entry("nuextract3").mmproj_path == entry.mmproj_path


def test_seed_explicit_mmproj_overrides_manifest_projector(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=True
    )
    mmproj = tmp_path / "external-mmproj.gguf"
    mmproj.write_bytes(b"EXTERNAL-mmproj")
    store = ModelStore(tmp_path / "models")

    entry = store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
        mmproj_source=mmproj,
    )

    assert entry.mmproj_path is not None
    assert entry.mmproj_path.read_bytes() == b"EXTERNAL-mmproj"  # explicit wins


def test_seed_missing_explicit_mmproj_file_raises(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=False
    )
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="mmproj not found"):
        store.seed_from_ollama(
            "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
            name="nuextract3",
            family="nuextract3",
            ollama_home=home,
            mmproj_source=tmp_path / "does-not-exist.gguf",
        )


def test_seed_vision_family_without_any_projector_still_refuses(tmp_path: Path) -> None:
    # No manifest projector AND no explicit mmproj -> the needs_mmproj guard holds.
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=False
    )
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="requires a vision projector"):
        store.seed_from_ollama(
            "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
            name="nuextract3",
            family="nuextract3",
            ollama_home=home,
        )


# ── (4) family launch args emit the vision flags ───────────────────────────────


def test_family_launch_args_emit_jinja_and_mmproj(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=True
    )
    store = ModelStore(tmp_path / "models")
    store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
    )

    args = store.family_launch_args("nuextract3")

    assert "--jinja" in args
    assert "--mmproj" in args
    # --mmproj is followed by the projector path.
    assert args[args.index("--mmproj") + 1].endswith("mmproj.gguf")


# ── (5) catalog view surfaces vision=True without a DB ─────────────────────────


def test_catalog_view_reports_vision_true_for_nuextract3() -> None:
    row = ModelStoreEntry(
        name="nuextract3",
        family="nuextract3",
        model_path="/models/nuextract3/model.gguf",
        mmproj_path="/models/nuextract3/mmproj.gguf",
    )
    view = _to_view(row)
    assert view["vision"] is True
    assert view["has_mmproj"] is True
    # nuextract3 is ollama_faithful=False -> llama-server only, never Ollama.
    assert view["available_backends"] == ["llama-server"]


# ── (6) a resolved vision profile drives the page-image branch, not OCR ────────


@pytest.mark.asyncio
async def test_resolved_vision_profile_takes_page_image_branch(
    models_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # End-to-end (in-process): resolve a NuExtract vision profile from a live
    # record + catalog family, then confirm ExtractionService routes it to the
    # page-image branch (blocks=[], images passed) — never the OCR branch.
    monkeypatch.setattr(
        "docie_bench.serving.catalog.ModelCatalog",
        lambda: _FakeCatalog({"family": "nuextract3"}),
    )
    record = _live_record(name="nux-dep", alias="nux-dep")
    profile = resolve_extraction_profile(
        deployment="nux-dep", models_config_path=models_config, deployments=[record]
    )
    assert profile.vision is True

    path = tmp_path / "invoice.png"
    path.write_bytes(b"document")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images",
        lambda *a, **k: [DocumentImage(page=1, media_type="image/png", data=b"png")],
    )

    async def fake_extract_blocks(self: ExtractionService, **kwargs: object) -> str:
        captured.update(kwargs)
        return "vision-response"

    monkeypatch.setattr(ExtractionService, "_extract_blocks", fake_extract_blocks)

    response = await ExtractionService(profile).extract_from_file(
        path=path, ocr_backend_name="pdf_text", schema_name="invoice"
    )

    # The service passes the (faked) _extract_blocks result straight through; the
    # sentinel proves the vision branch ran. Cast to object so the comparison is
    # type-legal under --strict (declared return is ExtractionResponse).
    assert cast(object, response) == "vision-response"
    assert captured["blocks"] == []  # OCR blocks NOT produced
    images = cast(list[DocumentImage], captured["images"])
    assert images[0].data == b"png"  # page image WAS sent
