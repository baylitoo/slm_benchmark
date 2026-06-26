"""Canonical GGUF model store shared by llama.cpp (llama-server) and Ollama.

The goal is a single `models/` directory that *both* runtimes can serve from,
where each model **family** declares how its prompt template must be delivered so
the model is served correctly. The same GGUF is never downloaded twice: a store
entry can be *seeded* from a model already pulled by Ollama by hard-linking its
content-addressed blobs into the store under stable `*.gguf` filenames.

Layout::

    <root>/<name>/model.gguf      # weights
    <root>/<name>/mmproj.gguf     # optional vision projector
    <root>/index.json             # name -> entry metadata

Why a canonical store and not the raw Ollama blob? Two reasons surfaced while
integrating NuExtract3 (verified 2026-06-17):

* Ollama stores GGUFs as extension-less ``blobs/sha256-<hex>`` files; the
  llama.cpp runtime adapter requires a ``.gguf`` path, so it cannot point at a
  blob directly. Stable ``*.gguf`` names fix that.
* Ollama silently drops ``chat_template_kwargs`` over GGUF, so families that
  deliver their template that way (NuExtract3) cannot be served *faithfully* by
  Ollama at all — they must go through ``llama-server --jinja``. The family
  contract records this so the store refuses to emit a misleading Ollama
  Modelfile for such families.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class TemplateDelivery(StrEnum):
    """How a model family receives the extraction template/schema."""

    # Template passed via extra_body.chat_template_kwargs; requires the GGUF's
    # embedded jinja template, i.e. `llama-server --jinja`. (NuExtract3.)
    CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
    # Template baked into the prompt text, e.g. NuExtract v1's
    # `<|input|>### Template: ... ### Text: ...<|output|>` format.
    INPROMPT_NUEXTRACT_V1 = "inprompt_nuextract_v1"
    # Standard chat models: schema delivered via OpenAI `response_format`.
    OPENAI_JSON_SCHEMA = "openai_json_schema"


@dataclass(frozen=True)
class FamilyContract:
    """How one model *family* must be launched and prompted to respect its template.

    A contract is runtime-agnostic intent; the store/runtime translate it into
    concrete launch flags and extraction-client settings.
    """

    name: str
    template_delivery: TemplateDelivery
    # Extraction-client wiring (see docie_bench.llm).
    response_format_style: str
    prompt_profile: str
    # Extra llama-server flags required to honour the template (e.g. --jinja).
    llama_server_args: tuple[str, ...] = ()
    needs_mmproj: bool = False
    vision: bool = False
    stop_sequences: tuple[str, ...] = ()
    default_temperature: float = 0.0
    # Can Ollama serve this family *faithfully* (respecting its template)?
    # False for CHAT_TEMPLATE_KWARGS families: Ollama drops chat_template_kwargs.
    ollama_faithful: bool = True


# Known families. Adding a new model is "drop the GGUF + pick (or add) a family".
FAMILIES: dict[str, FamilyContract] = {
    "nuextract3": FamilyContract(
        name="nuextract3",
        template_delivery=TemplateDelivery.CHAT_TEMPLATE_KWARGS,
        response_format_style="nuextract3",
        prompt_profile="nuextract3",
        llama_server_args=("--jinja",),
        needs_mmproj=True,
        vision=True,
        default_temperature=0.2,
        ollama_faithful=False,
    ),
    "nuextract_v1": FamilyContract(
        name="nuextract_v1",
        template_delivery=TemplateDelivery.INPROMPT_NUEXTRACT_V1,
        response_format_style="none",
        prompt_profile="nuextract_v1",
        stop_sequences=("<|end-output|>",),
        default_temperature=0.0,
        ollama_faithful=True,
    ),
    "openai_chat": FamilyContract(
        name="openai_chat",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        default_temperature=0.0,
        ollama_faithful=True,
    ),
}


def get_family(name: str) -> FamilyContract:
    try:
        return FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(FAMILIES))
        raise ValueError(f"Unknown model family {name!r}. Known families: {known}") from None


class ModelStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoreEntry:
    name: str
    family: str
    model_path: Path
    mmproj_path: Path | None = None
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "model_path": self.model_path.as_posix(),
            "mmproj_path": self.mmproj_path.as_posix() if self.mmproj_path else None,
            "source": self.source,
        }


def default_ollama_home() -> Path:
    """Resolve Ollama's models directory ($OLLAMA_MODELS or ~/.ollama/models)."""
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ollama" / "models"


def _ollama_manifest_path(ollama_home: Path, reference: str) -> Path:
    """Map an Ollama model reference to its on-disk manifest path.

    Examples:
        ``hf.co/numind/NuExtract3-GGUF:Q4_K_M``
            -> manifests/hf.co/numind/NuExtract3-GGUF/Q4_K_M
        ``nuextract:3.8b``
            -> manifests/registry.ollama.ai/library/nuextract/3.8b
    """
    name_part, _, tag = reference.partition(":")
    tag = tag or "latest"
    parts = [segment for segment in name_part.split("/") if segment]
    if not parts:
        raise ModelStoreError(f"Invalid Ollama reference: {reference!r}")
    if len(parts) == 1:  # bare library model, e.g. "nuextract"
        parts = ["registry.ollama.ai", "library", *parts]
    elif "." not in parts[0]:  # "namespace/model" without an explicit registry host
        parts = ["registry.ollama.ai", *parts]
    return ollama_home.joinpath("manifests", *parts, tag)


def _blob_path(ollama_home: Path, digest: str) -> Path:
    return ollama_home / "blobs" / digest.replace(":", "-")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)  # hard link: zero extra disk, instant
    except OSError:
        shutil.copy2(source, destination)  # cross-device or unsupported FS


class ModelStore:
    """A canonical on-disk GGUF store served by both llama.cpp and Ollama."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"

    # ------------------------------------------------------------------ seeding
    def seed_from_ollama(
        self,
        reference: str,
        *,
        name: str,
        family: str,
        ollama_home: Path | None = None,
        link: bool = True,
    ) -> StoreEntry:
        """Register a model already pulled by Ollama, without re-downloading.

        Reads Ollama's manifest for ``reference`` and hard-links (or copies) its
        weights — and vision projector, if present — into the store under stable
        ``*.gguf`` names.
        """
        contract = get_family(family)
        home = ollama_home or default_ollama_home()
        manifest_path = _ollama_manifest_path(home, reference)
        if not manifest_path.is_file():
            raise ModelStoreError(
                f"Ollama manifest not found for {reference!r} at {manifest_path}. "
                f"Pull it first (e.g. `ollama pull {reference}`)."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layers = manifest.get("layers", [])
        model_digest = _select_layer(layers, "model")
        if model_digest is None:
            raise ModelStoreError(f"No model layer found in Ollama manifest for {reference!r}")
        projector_digest = _select_layer(layers, "projector")

        model_blob = _blob_path(home, model_digest)
        if not model_blob.is_file():
            raise ModelStoreError(f"Ollama model blob is missing: {model_blob}")
        mmproj_blob = _blob_path(home, projector_digest) if projector_digest else None
        if contract.needs_mmproj and mmproj_blob is None:
            raise ModelStoreError(
                f"Family {family!r} requires a vision projector but {reference!r} has none"
            )

        destination = self.root / name
        model_path = destination / "model.gguf"
        _transfer(model_blob, model_path, link=link)
        mmproj_path: Path | None = None
        if mmproj_blob is not None and mmproj_blob.is_file():
            mmproj_path = destination / "mmproj.gguf"
            _transfer(mmproj_blob, mmproj_path, link=link)

        entry = StoreEntry(
            name=name,
            family=family,
            model_path=model_path,
            mmproj_path=mmproj_path,
            source=f"ollama:{reference}",
        )
        self._write_entry(entry)
        return entry

    def add_gguf(
        self,
        *,
        name: str,
        family: str,
        model_gguf: str | Path,
        mmproj: str | Path | None = None,
        source: str | None = None,
        link: bool = True,
    ) -> StoreEntry:
        """Register a GGUF already on disk (e.g. downloaded from Hugging Face)."""
        contract = get_family(family)
        model_gguf = Path(model_gguf)
        if not model_gguf.is_file():
            raise ModelStoreError(f"GGUF not found: {model_gguf}")
        if contract.needs_mmproj and mmproj is None:
            raise ModelStoreError(f"Family {family!r} requires an mmproj (vision projector)")
        destination = self.root / name
        model_path = destination / "model.gguf"
        _transfer(model_gguf, model_path, link=link)
        mmproj_path: Path | None = None
        if mmproj is not None:
            mmproj_path = destination / "mmproj.gguf"
            _transfer(Path(mmproj), mmproj_path, link=link)
        entry = StoreEntry(
            name=name,
            family=family,
            model_path=model_path,
            mmproj_path=mmproj_path,
            source=source,
        )
        self._write_entry(entry)
        return entry

    # ------------------------------------------------------------------- query
    def entry(self, name: str) -> StoreEntry:
        index = self._read_index()
        if name not in index:
            raise ModelStoreError(f"Unknown model {name!r} in store {self.root}")
        return _entry_from_json(index[name])

    def list(self) -> list[StoreEntry]:
        return [_entry_from_json(value) for _, value in sorted(self._read_index().items())]

    # ----------------------------------------------------------------- serving
    def family_launch_args(self, name: str) -> tuple[str, ...]:
        """Family-specific llama-server flags for ``name`` (e.g. ``--jinja``, ``--mmproj``).

        This is the single source of truth for the template/vision flags a family
        needs; both ``llama_server_command`` and the background serving bridge
        (``docie up``) derive their invocation from it.
        """
        entry = self.entry(name)
        contract = get_family(entry.family)
        extra = list(contract.llama_server_args)
        if contract.needs_mmproj and entry.mmproj_path is not None:
            extra.extend(["--mmproj", entry.mmproj_path.as_posix()])
        return tuple(extra)

    def llama_server_command(
        self,
        name: str,
        *,
        executable: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8088,
        context_length: int = 8192,
    ) -> tuple[str, ...]:
        """Build the llama-server command that serves ``name`` per its family contract.

        Includes the family's required flags (e.g. ``--jinja``) and ``--mmproj``
        for vision families — exactly what the raw ``llama-server`` invocation in
        the docs needs, sourced from the canonical store.
        """
        entry = self.entry(name)
        command = [
            executable,
            "--model",
            entry.model_path.as_posix(),
            "--alias",
            name,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(context_length),
        ]
        command.extend(self.family_launch_args(name))
        return tuple(command)

    def ollama_modelfile(self, name: str) -> str:
        """Generate an Ollama Modelfile so Ollama can serve ``name`` from the store.

        Raises for families Ollama cannot serve faithfully (those whose template is
        delivered via ``chat_template_kwargs``), because the resulting deployment
        would silently ignore the template and produce wrong extractions.
        """
        entry = self.entry(name)
        contract = get_family(entry.family)
        if not contract.ollama_faithful:
            raise ModelStoreError(
                f"Family {entry.family!r} cannot be served faithfully by Ollama "
                f"(template is delivered via {contract.template_delivery}; Ollama drops it). "
                f"Serve {name!r} with `llama-server` instead — see "
                f"`ModelStore.llama_server_command`."
            )
        lines = [f'FROM {entry.model_path.as_posix()}']
        if entry.mmproj_path is not None:
            lines.append(f'ADAPTER {entry.mmproj_path.as_posix()}')
        for stop in contract.stop_sequences:
            lines.append(f'PARAMETER stop "{stop}"')
        lines.append(f"PARAMETER temperature {contract.default_temperature}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ private
    def _read_index(self) -> dict[str, Any]:
        if not self._index_path.is_file():
            return {}
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ModelStoreError(f"Corrupt store index: {self._index_path}")
        return data

    def _write_entry(self, entry: StoreEntry) -> None:
        index = self._read_index()
        index[entry.name] = entry.to_json()
        temporary = self._index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._index_path)


def _select_layer(layers: list[dict[str, Any]], needle: str) -> str | None:
    for layer in layers:
        media_type = str(layer.get("mediaType", ""))
        if needle in media_type and layer.get("digest"):
            return str(layer["digest"])
    return None


def _transfer(source: Path, destination: Path, *, link: bool) -> None:
    if link:
        _link_or_copy(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _entry_from_json(value: dict[str, Any]) -> StoreEntry:
    mmproj = value.get("mmproj_path")
    return StoreEntry(
        name=value["name"],
        family=value["family"],
        model_path=Path(value["model_path"]),
        mmproj_path=Path(mmproj) if mmproj else None,
        source=value.get("source"),
    )
