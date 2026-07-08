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

import hashlib
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
    # Generation defaults inherited by a family-synthesized profile (a store deploy
    # whose served id matches no models.yaml profile). These are the single source
    # of truth for a family's tuning, so such a deployment runs with the family's
    # intended params — NOT the bare ModelProfile defaults (900 / 0.0 / 180), which
    # would silently degrade a model like NuExtract3 that needs 4096 tokens.
    default_temperature: float = 0.0
    default_max_tokens: int = 900
    default_timeout_seconds: float = 180.0
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
        default_max_tokens=4096,
        default_timeout_seconds=600.0,
        ollama_faithful=False,
    ),
    "nuextract_v1": FamilyContract(
        name="nuextract_v1",
        template_delivery=TemplateDelivery.INPROMPT_NUEXTRACT_V1,
        response_format_style="none",
        prompt_profile="nuextract_v1",
        stop_sequences=("<|end-output|>",),
        default_temperature=0.0,
        default_max_tokens=2000,
        default_timeout_seconds=300.0,
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


def _sha256_file(path: Path) -> str:
    """Chunked sha256 of ``path`` as bare lowercase hex (no ``sha256:`` prefix).

    Mirrors ``extract.service.hash_file`` but keeps the serving layer
    self-contained (no cross-layer import) and returns bare hex so it compares
    directly against an Ollama manifest digest's hex tail.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hex(digest: str | None) -> str | None:
    """Hex tail of a canonical ``sha256:<hex>`` digest, else ``None``.

    ``None`` means "no canonical digest to verify against" — the caller falls
    back to a best-effort copy-fidelity check. Guarding on the ``sha256:`` prefix
    means a non-sha256 manifest digest (should Ollama ever emit one) skips the
    canonical comparison rather than false-failing an otherwise valid transfer.
    """
    prefix = "sha256:"
    if digest and digest.startswith(prefix):
        return digest[len(prefix) :]
    return None


def _assert_within(path: Path, root: Path, *, label: str) -> Path:
    """Resolve ``path`` and require it to stay within ``root`` (block traversal).

    Both the Ollama ``reference`` (-> manifest path) and the store ``name`` (->
    destination dir) are attacker-influenced strings joined into filesystem
    paths. Legitimate references legitimately contain '/' and ':' (e.g.
    ``hf.co/numind/NuExtract3-GGUF:Q4_K_M``) and resolve *inside* the root, so we
    can't blanket-reject separators — instead resolve and enforce containment,
    which rejects only ``..``/absolute escapes.
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ModelStoreError(
            f"Refusing {label}: resolved path escapes {root_resolved} "
            f"(possible path traversal)"
        )
    return resolved


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
        mmproj_source: str | Path | None = None,
    ) -> StoreEntry:
        """Register a model already pulled by Ollama, without re-downloading.

        Reads Ollama's manifest for ``reference`` and hard-links (or copies) its
        weights — and vision projector, if present — into the store under stable
        ``*.gguf`` names.

        ``mmproj_source`` supplies a vision projector explicitly (an on-disk GGUF,
        e.g. one downloaded separately from Hugging Face). It takes precedence over
        the projector embedded in the Ollama manifest and is the ONLY way to make a
        ``needs_mmproj`` family (NuExtract3) deployable when the pulled GGUF ships
        no projector layer — the exact gap that otherwise leaves vision families
        un-servable via the seed path.
        """
        if not reference.strip() or not name.strip():
            raise ModelStoreError("seed requires a non-empty reference and name")
        contract = get_family(family)
        home = ollama_home or default_ollama_home()
        manifest_path = _ollama_manifest_path(home, reference)
        # Containment: a crafted reference ("../../..") must not read manifests
        # outside Ollama's manifests root, and a crafted store name must not write
        # blobs outside the store root.
        _assert_within(manifest_path, home / "manifests", label=f"Ollama reference {reference!r}")
        _assert_within(self.root / name, self.root, label=f"store name {name!r}")
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

        # Resolve the vision projector: an explicit ``mmproj_source`` wins over the
        # manifest's projector layer (some GGUF pulls ship none). ``.is_file()``
        # rather than a mere digest presence check so a dangling projector digest
        # cannot masquerade as a usable projector for a ``needs_mmproj`` family.
        explicit_mmproj: Path | None = None
        if mmproj_source is not None:
            explicit_mmproj = Path(mmproj_source)
            if not explicit_mmproj.is_file():
                raise ModelStoreError(f"mmproj not found: {explicit_mmproj}")
        manifest_mmproj = _blob_path(home, projector_digest) if projector_digest else None
        if manifest_mmproj is not None and not manifest_mmproj.is_file():
            manifest_mmproj = None
        mmproj_blob = explicit_mmproj or manifest_mmproj
        if contract.needs_mmproj and mmproj_blob is None:
            raise ModelStoreError(
                f"Family {family!r} requires a vision projector but {reference!r} has none. "
                f"Pass mmproj_source=<projector.gguf> (or pull a GGUF that includes one)."
            )

        # An explicit mmproj_source is an HF download with no manifest digest, so
        # only copy-fidelity is checkable there; the manifest projector layer has
        # its canonical digest. Verify BOTH blobs before writing the index so a
        # canonical model.gguf / index entry only ever appears fully verified.
        mmproj_digest = None if explicit_mmproj is not None else projector_digest
        destination = self.root / name
        model_path = destination / "model.gguf"
        mmproj_path: Path | None = None
        try:
            _transfer_verified(model_blob, model_path, link=link, expected_digest=model_digest)
            if mmproj_blob is not None:
                mmproj_path = destination / "mmproj.gguf"
                _transfer_verified(
                    mmproj_blob, mmproj_path, link=link, expected_digest=mmproj_digest
                )
        except BaseException:
            # A verify failure on either blob must leave zero partial state (no
            # stray model.gguf when mmproj fails). Bounded by ``name``.
            shutil.rmtree(destination, ignore_errors=True)
            raise

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
        # HF/on-disk GGUFs carry no canonical manifest digest, so only
        # copy-fidelity is verifiable (expected_digest=None) — best-effort, per
        # the HF trust boundary. Still routed through _transfer_verified so no
        # path writes the canonical name un-verified.
        destination = self.root / name
        model_path = destination / "model.gguf"
        mmproj_path: Path | None = None
        try:
            _transfer_verified(model_gguf, model_path, link=link, expected_digest=None)
            if mmproj is not None:
                mmproj_path = destination / "mmproj.gguf"
                _transfer_verified(Path(mmproj), mmproj_path, link=link, expected_digest=None)
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
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
        self._write_index(index)

    def _write_index(self, index: dict[str, Any]) -> None:
        temporary = self._index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._index_path)

    def remove_entry(self, name: str) -> None:
        """Remove store entry ``name``: drop its index key, then delete its dir.

        Compensation for a seed whose catalog upsert failed, making the seed
        all-or-nothing. Ordering matters: rewrite the index WITHOUT the key first
        (atomic temp+replace) so a failure can never leave the index pointing at
        deleted files — an unreferenced directory is harmless, a dangling index
        reference is exactly the error-producing partial state this PR removes.
        Bounded by ``name`` (containment-checked) so it can only ever touch
        ``root/name`` and that one index key, never a pre-existing unrelated entry.
        """
        destination = _assert_within(self.root / name, self.root, label=f"store name {name!r}")
        index = self._read_index()
        if name in index:
            del index[name]
            self._write_index(index)
        shutil.rmtree(destination, ignore_errors=True)


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


def _transfer_verified(
    source: Path,
    destination: Path,
    *,
    link: bool,
    expected_digest: str | None,
) -> None:
    """Transfer ``source`` to ``destination`` and refuse a mislabeled/corrupt result.

    Writes into a sibling ``*.tmp`` first, sha256s it, and ``os.replace`` moves it
    onto the canonical ``destination`` name ONLY when the hash matches. So the
    canonical file never appears unless it is fully written and verified; on any
    mismatch or error the tmp is removed and ``ModelStoreError`` is raised,
    leaving no partial/mislabeled file behind.

    Verification target:
      * ``expected_digest`` is a canonical ``sha256:<hex>`` (an Ollama manifest
        layer digest) -> the dest is checked against the CANONICAL content hash.
        This catches a wrong-sha even for hard links, i.e. a corrupt SOURCE blob,
        because a same-inode tmp still hashes to the source's real (wrong) bytes.
      * ``expected_digest`` is ``None`` (or non-sha256) -> best-effort
        copy-fidelity only (source hash == dest hash). For a hard link this is
        trivially true (same inode); for a copy it catches a truncated/altered
        copy. There is no canonical digest to assert here (HF trust boundary).
    """
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        _transfer(source, tmp, link=link)
        got = _sha256_file(tmp)
        want = _canonical_hex(expected_digest)
        canonical = want is not None
        if want is None:
            want = _sha256_file(source)  # copy-fidelity fallback
        if got.lower() != want.lower():
            kind = "manifest digest" if canonical else "source copy"
            raise ModelStoreError(
                f"blob integrity check failed for {destination.name}: "
                f"got sha256:{got} != want sha256:{want} ({kind}; source={source})"
            )
        os.replace(tmp, destination)
    except BaseException:
        # Never leave a half-written or mislabeled tmp behind, even on
        # KeyboardInterrupt / a mid-transfer crash.
        tmp.unlink(missing_ok=True)
        raise


def _entry_from_json(value: dict[str, Any]) -> StoreEntry:
    mmproj = value.get("mmproj_path")
    return StoreEntry(
        name=value["name"],
        family=value["family"],
        model_path=Path(value["model_path"]),
        mmproj_path=Path(mmproj) if mmproj else None,
        source=value.get("source"),
    )
