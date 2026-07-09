"""Unit tests for the LiquidAI LFM2.5 integration (families + profiles + resolver).

Covers the four things the LFM2.5 design promises to prove without a live model:

* the ``lfm2`` / ``lfm2_vl`` FamilyContracts (template delivery, --jinja, vision);
* ``configs/models.yaml`` loads the LFM2.5 profiles with the right vision flags;
* ``family_launch_args`` emits ``--jinja`` (+ ``--mmproj`` only for VL);
* the resolver inherits ``vision=True`` for a ``lfm2_vl`` store deploy end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docie_bench.llm.model_profiles import load_model_profiles
from docie_bench.serving.model_store import (
    FAMILIES,
    ModelStore,
    ModelStoreError,
    TemplateDelivery,
)
from docie_bench.serving.profile_resolver import resolve_extraction_profile
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentRecord, DeploymentSpec

# Path to the shipped config, so the profile-load test asserts against the real
# LFM2.5 entries rather than a fixture that could drift from them.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_YAML = _REPO_ROOT / "configs" / "models.yaml"


# ── family contracts ───────────────────────────────────────────────────────────


def test_lfm2_text_contract() -> None:
    contract = FAMILIES["lfm2"]
    assert contract.template_delivery is TemplateDelivery.OPENAI_JSON_SCHEMA
    assert contract.response_format_style == "openai_json_schema"
    assert contract.prompt_profile == "strict_extraction_v1"
    # --jinja for template fidelity, and NOT the chat_template_kwargs path.
    assert contract.llama_server_args == ("--jinja",)
    assert not contract.needs_mmproj
    assert not contract.vision
    assert contract.ollama_faithful
    assert contract.default_temperature == pytest.approx(0.0)


def test_lfm2_vl_contract() -> None:
    contract = FAMILIES["lfm2_vl"]
    assert contract.needs_mmproj
    assert contract.vision
    # KEY diff vs nuextract3: schema rides response_format, not a bespoke style.
    assert contract.response_format_style == "openai_json_schema"
    assert contract.template_delivery is TemplateDelivery.OPENAI_JSON_SCHEMA
    assert contract.prompt_profile == "strict_extraction_v1"
    assert "--jinja" in contract.llama_server_args
    assert contract.default_max_tokens == 4096
    assert contract.default_timeout_seconds == pytest.approx(600.0)
    # Ollama mmproj/ADAPTER for lfm2-vl is unverified → llama-server only.
    assert contract.ollama_faithful is False


# ── models.yaml profile load ────────────────────────────────────────────────────


def test_profile_load_vision_flag() -> None:
    profiles = load_model_profiles(_MODELS_YAML)
    # All four LFM2.5 store profiles are present.
    for name in ("lfm25_230m", "lfm25_350m", "lfm25_1_2b", "lfm25_vl_1_6b"):
        assert name in profiles, f"missing LFM2.5 profile {name!r}"
    # Only the VL profile carries vision; the text ones do not.
    assert profiles["lfm25_vl_1_6b"].vision is True
    assert profiles["lfm25_350m"].vision is False
    assert profiles["lfm25_1_2b"].vision is False
    # They route to the store/llama-server endpoint via openai_json_schema.
    assert profiles["lfm25_1_2b"].response_format_style == "openai_json_schema"
    assert profiles["lfm25_1_2b"].base_url == "http://localhost:8088/v1"


def test_ollama_lfm25_350m_profile_untouched() -> None:
    # The pre-existing Ollama-runtime profile coexists with the new store profile.
    profiles = load_model_profiles(_MODELS_YAML)
    ollama = profiles["ollama_lfm25_350m"]
    assert ollama.model == "lfm2.5:350m"
    assert ollama.base_url == "http://localhost:11434/v1"
    assert ollama.response_format_style == "json_object"


# ── family_launch_args (store serving flags) ────────────────────────────────────


def _fake_gguf(tmp_path: Path, name: str) -> Path:
    # add_gguf requires a real file on disk (it stats + verifies copy-fidelity).
    path = tmp_path / name
    path.write_bytes(b"GGUF-fake-weights")
    return path


def test_family_launch_args_vl_emits_jinja_and_mmproj(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    store.add_gguf(
        name="vl",
        family="lfm2_vl",
        model_gguf=_fake_gguf(tmp_path, "vl.gguf"),
        mmproj=_fake_gguf(tmp_path, "mmproj.gguf"),
        link=False,
    )
    args = store.family_launch_args("vl")
    assert args[0] == "--jinja"
    assert "--mmproj" in args
    assert args[args.index("--mmproj") + 1].endswith("mmproj.gguf")


def test_family_launch_args_text_is_jinja_only(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    store.add_gguf(
        name="t",
        family="lfm2",
        model_gguf=_fake_gguf(tmp_path, "t.gguf"),
        link=False,
    )
    assert store.family_launch_args("t") == ("--jinja",)  # no --mmproj for text


def test_llama_server_command_vl_includes_jinja_and_mmproj(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "models")
    store.add_gguf(
        name="lfm25_vl_1_6b",
        family="lfm2_vl",
        model_gguf=_fake_gguf(tmp_path, "vl.gguf"),
        mmproj=_fake_gguf(tmp_path, "mmproj.gguf"),
        link=False,
    )
    command = store.llama_server_command("lfm25_vl_1_6b", port=8088)
    assert command[0] == "llama-server"
    assert "--jinja" in command
    assert "--mmproj" in command
    assert "--alias" in command
    assert "lfm25_vl_1_6b" in command


def test_lfm2_vl_not_ollama_faithful_served_via_llama_server(tmp_path: Path) -> None:
    # Ollama mmproj/ADAPTER support for lfm2-vl is unverified, so lfm2_vl is
    # ollama_faithful=False and refuses a Modelfile (like nuextract3) — the VL
    # path is llama-server only.
    store = ModelStore(tmp_path / "models")
    store.add_gguf(
        name="vl",
        family="lfm2_vl",
        model_gguf=_fake_gguf(tmp_path, "vl.gguf"),
        mmproj=_fake_gguf(tmp_path, "mmproj.gguf"),
        link=False,
    )
    with pytest.raises(ModelStoreError):
        store.ollama_modelfile("vl")


# ── resolver: VL store deploy inherits vision=True end-to-end ────────────────────


def _install_store_index(home: Path, name: str, family: str) -> None:
    """Write ``<home>/models/index.json`` so the resolver's on-disk family lookup
    is hermetic (never the machine's real store)."""
    models_dir = home / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "index.json").write_text(
        json.dumps(
            {
                name: {
                    "name": name,
                    "family": family,
                    "model_path": f"/store/{name}/model.gguf",
                    "mmproj_path": f"/store/{name}/mmproj.gguf",
                    "source": "test",
                }
            }
        ),
        encoding="utf-8",
    )


def _record(name: str, alias: str) -> DeploymentRecord:
    return DeploymentRecord(
        spec=DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP, model="/p/x.gguf", alias=alias
            ),
        ),
        state=LifecycleState.READY,
        endpoint="http://127.0.0.1:8088/v1",
    )


def test_resolver_yields_vision_true_for_vl_via_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A store deploy whose alias matches no models.yaml profile is synthesized from
    # its store-entry family (lfm2_vl) -> vision True, with NO Postgres catalog.
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    _install_store_index(tmp_path, "vl-store", "lfm2_vl")

    class _Boom:
        def __init__(self) -> None:
            raise RuntimeError("catalog must not be consulted when the store entry exists")

    monkeypatch.setattr("docie_bench.serving.catalog.ModelCatalog", _Boom)

    profile = resolve_extraction_profile(
        deployment="vl-store",
        models_config_path=tmp_path / "absent-models.yaml",
        deployments=[_record("vl-store", "vl-store")],
    )
    assert profile.vision is True
    assert profile.response_format_style == "openai_json_schema"
    assert profile.prompt_profile == "strict_extraction_v1"
    # VL family generation defaults propagate (not the bare 900 / 180 defaults).
    assert profile.max_tokens == 4096
    assert profile.timeout_seconds == pytest.approx(600.0)


def test_resolver_yields_vision_false_for_text_lfm2_via_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    _install_store_index(tmp_path, "text-store", "lfm2")

    profile = resolve_extraction_profile(
        deployment="text-store",
        models_config_path=tmp_path / "absent-models.yaml",
        deployments=[_record("text-store", "text-store")],
    )
    assert profile.vision is False
    assert profile.response_format_style == "openai_json_schema"
