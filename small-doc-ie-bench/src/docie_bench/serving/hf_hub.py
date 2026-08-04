"""Hugging Face Hub client for direct GGUF seeding — no Ollama in the path.

The Ollama seed constrained the catalog to what Ollama's own hub mirrors
(namespaced refs like ``LiquidAI/lfm2.5-350m``, a subset of the provider's
actual HF collection). This module talks to the Hub directly, the same way
``llama-server -hf owner/Repo-GGUF:Q4_K_M`` does — but downloads into the
canonical model store so sizing, families, calibration and placement all keep
working unchanged.

Surface:

* :func:`list_repo_ggufs` — the repo's GGUF files with parsed quant labels,
  sizes, and mmproj detection (backs the Studio's quant picker).
* :func:`pick_gguf` — choose one file for a requested quant (or a sensible
  default), with an error that lists what IS available.
* :func:`download_file` — streaming download with a progress callback (the
  seed job forwards it to the realtime ``progress`` topic).
* :func:`list_collection` — a provider-curated HF collection's model repos
  (backs "seed a whole collection" in the Studio).

Auth: ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is attached when present so
gated/private repos work; anonymous access covers the common public case.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

HF_BASE = "https://huggingface.co"

# Preferred default quants when the caller doesn't specify one: a pragmatic
# CPU-quality ladder (best first).
DEFAULT_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "F16")

_QUANT_RE = re.compile(r"(?i)\b(iq\d[a-z0-9_]*|q\d[a-z0-9_]*|f16|f32|bf16)\b")
_MULTIPART_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)

ProgressCallback = Callable[[int, int | None], Awaitable[None]]


class HfHubError(Exception):
    """A Hub request failed or a repo has no usable GGUF."""


@dataclass(frozen=True)
class HfGgufFile:
    filename: str
    size_bytes: int | None
    quant: str | None
    is_mmproj: bool
    is_multipart: bool


def hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_quant(filename: str) -> str | None:
    """The quant label embedded in a GGUF filename (``...Q4_K_M.gguf`` -> Q4_K_M)."""
    stem = filename.rsplit("/", 1)[-1]
    stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
    matches = _QUANT_RE.findall(stem)
    return matches[-1].upper() if matches else None


def default_store_name(repo: str, quant: str | None = None) -> str:
    """A store-name suggestion from a repo id (``LiquidAI/LFM2.5-350M-GGUF`` ->
    ``lfm2.5-350m``); the UI prefills it, the user can override."""
    tail = repo.rsplit("/", 1)[-1].lower()
    tail = re.sub(r"(?i)[-_.]?gguf$", "", tail)
    if quant:
        tail = f"{tail}-{quant.lower()}"
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", tail).strip("-.")
    return cleaned or "model"


def _gguf_from_sibling(sibling: dict[str, Any]) -> HfGgufFile | None:
    filename = str(sibling.get("rfilename") or "")
    if not filename.lower().endswith(".gguf"):
        return None
    size = sibling.get("size")
    return HfGgufFile(
        filename=filename,
        size_bytes=int(size) if isinstance(size, (int, float)) else None,
        quant=parse_quant(filename),
        is_mmproj="mmproj" in filename.lower(),
        is_multipart=bool(_MULTIPART_RE.search(filename)),
    )


async def list_repo_ggufs(repo: str, *, client: httpx.AsyncClient) -> list[HfGgufFile]:
    """The repo's GGUF files (``?blobs=true`` so sizes come back too)."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise HfHubError(f"invalid Hugging Face repo id {repo!r} (expected owner/name)")
    url = f"{HF_BASE}/api/models/{repo}?blobs=true"
    try:
        response = await client.get(url, headers=hf_headers(), timeout=20.0)
    except httpx.RequestError as exc:
        raise HfHubError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if response.status_code == 404:
        raise HfHubError(f"repo {repo!r} does not exist on the Hugging Face Hub")
    if response.status_code in (401, 403):
        # The Hub answers 401 for a NONEXISTENT repo when unauthenticated (it
        # won't leak existence), so this is ambiguous by design — say so.
        raise HfHubError(
            f"repo {repo!r} was refused by the Hub: either the id has a typo "
            "(check the exact repo name), or it is gated/private — in that "
            "case set HF_TOKEN on the api and serving services"
        )
    if response.status_code >= 400:
        raise HfHubError(f"Hub returned HTTP {response.status_code} for {repo!r}")
    payload = response.json()
    siblings = [s for s in payload.get("siblings", []) if isinstance(s, dict)]
    files = [
        gguf for sibling in siblings if (gguf := _gguf_from_sibling(sibling)) is not None
    ]
    if not files:
        # A safetensors checkpoint is usually an ENCODER (GLiNER/GLiNER2,
        # classifiers) — those are not llama.cpp-servable at all and belong to
        # the encoder runtime, not the GGUF store. Route the user there
        # instead of sending them hunting for a -GGUF repo that can't exist.
        has_safetensors = any(
            str(s.get("rfilename", "")).endswith(".safetensors") for s in siblings
        )
        if has_safetensors:
            raise HfHubError(
                f"repo {repo!r} is a transformers/encoder checkpoint (safetensors), "
                "not a GGUF model. If it is a GLiNER/GLiNER2 analyzer, add it as "
                "an encoder instead (or deploy with runtime=\"encoder\")."
            )
        raise HfHubError(
            f"repo {repo!r} ships no GGUF files — pick that model's -GGUF conversion repo"
        )
    return files


def pick_gguf(files: list[HfGgufFile], quant: str | None, *, prefer: bool = False) -> HfGgufFile:
    """Choose the model GGUF for ``quant`` (or the best default available).

    ``prefer=False`` (single-repo, explicit pick from the quant chips): an
    unavailable ``quant`` is a hard error listing what IS there. ``prefer=True``
    (a batch/collection applying one quant across heterogeneous repos): the
    ``quant`` is a PREFERENCE — if a repo lacks it, fall back to the default
    quant ladder / smallest file instead of failing the whole repo.
    """
    candidates = [f for f in files if not f.is_mmproj and not f.is_multipart]
    multipart_only = [f for f in files if not f.is_mmproj and f.is_multipart]
    if not candidates:
        if multipart_only:
            raise HfHubError(
                "this repo ships only multi-part GGUFs (…-00001-of-000NN), which are "
                "not supported — pick a single-file quant or another repo"
            )
        raise HfHubError(
            "no model GGUF found in this repo — only a vision projector (mmproj) file"
        )
    if quant:
        wanted = quant.strip().upper()
        for f in candidates:
            if f.quant == wanted:
                return f
        if not prefer:
            available = sorted({f.quant or f.filename for f in candidates})
            raise HfHubError(
                f"quant {quant!r} not found; available: {', '.join(available)}"
            )
        # prefer: fall through to the default ladder below.
    for preferred in DEFAULT_QUANT_PREFERENCE:
        for f in candidates:
            if f.quant == preferred:
                return f
    # No preferred label matched (exotic naming) — smallest file is the safest
    # CPU default.
    return sorted(candidates, key=lambda f: (f.size_bytes is None, f.size_bytes or 0))[0]


def pick_mmproj(files: list[HfGgufFile]) -> HfGgufFile | None:
    """The repo's vision projector, when it ships one (largest wins on ties)."""
    mmprojs = [f for f in files if f.is_mmproj and not f.is_multipart]
    if not mmprojs:
        return None
    return sorted(mmprojs, key=lambda f: f.size_bytes or 0, reverse=True)[0]


async def download_file(
    repo: str,
    filename: str,
    destination: Path,
    *,
    client: httpx.AsyncClient,
    revision: str = "main",
    progress: ProgressCallback | None = None,
) -> Path:
    """Stream one repo file to ``destination`` (``.part`` then atomic rename).

    ``progress(received_bytes, total_bytes)`` is awaited as chunks land; the
    caller owns throttling. Redirects (Hub -> CDN) are followed.
    """
    url = f"{HF_BASE}/{repo}/resolve/{revision}/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    try:
        async with client.stream(
            "GET", url, headers=hf_headers(), follow_redirects=True, timeout=None
        ) as response:
            if response.status_code >= 400:
                raise HfHubError(
                    f"download of {filename!r} failed: HTTP {response.status_code}"
                )
            total_raw = response.headers.get("content-length")
            total = int(total_raw) if total_raw and total_raw.isdigit() else None
            received = 0
            with part.open("wb") as handle:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    received += len(chunk)
                    if progress is not None:
                        await progress(received, total)
        part.replace(destination)
        return destination
    except httpx.RequestError as exc:
        part.unlink(missing_ok=True)
        raise HfHubError(f"download of {filename!r} failed: {exc}") from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise


# Snapshot download (analyzer/encoder checkpoints — safetensors + configs).
# Alternate weight formats and hardware-specific exports are skipped: the
# encoder runtime loads safetensors, so pulling .bin/.onnx/.h5 doubles the
# download for nothing.
_SNAPSHOT_SKIP_SUFFIXES = frozenset(
    {".gguf", ".bin", ".pt", ".pth", ".h5", ".onnx", ".msgpack", ".tflite", ".ot"}
)
_SNAPSHOT_SKIP_DIRS = ("onnx/", "openvino/", "coreml/", "tflite/")


def _is_snapshot_file(filename: str) -> bool:
    lower = filename.lower()
    if any(lower.startswith(prefix) or f"/{prefix}" in lower for prefix in _SNAPSHOT_SKIP_DIRS):
        return False
    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    return suffix not in _SNAPSHOT_SKIP_SUFFIXES


async def list_snapshot_files(repo: str, *, client: httpx.AsyncClient) -> list[HfGgufFile]:
    """The repo's safetensors-checkpoint files (weights + config + tokenizer).

    Reuses the same ``?blobs=true`` metadata as :func:`list_repo_ggufs` but keeps
    the transformers snapshot instead of the GGUFs — for analyzer/encoder
    families served by the encoder runtime. Refuses a repo with no safetensors
    (that is a GGUF-only or an incompatible repo).
    """
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise HfHubError(f"invalid Hugging Face repo id {repo!r} (expected owner/name)")
    url = f"{HF_BASE}/api/models/{repo}?blobs=true"
    try:
        response = await client.get(url, headers=hf_headers(), timeout=20.0)
    except httpx.RequestError as exc:
        raise HfHubError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if response.status_code == 404:
        raise HfHubError(f"repo {repo!r} does not exist on the Hugging Face Hub")
    if response.status_code in (401, 403):
        raise HfHubError(
            f"repo {repo!r} was refused by the Hub — check the id or set HF_TOKEN"
        )
    if response.status_code >= 400:
        raise HfHubError(f"Hub returned HTTP {response.status_code} for {repo!r}")
    payload = response.json()
    files: list[HfGgufFile] = []
    has_safetensors = False
    for sibling in payload.get("siblings", []):
        if not isinstance(sibling, dict):
            continue
        filename = str(sibling.get("rfilename") or "")
        if not filename or not _is_snapshot_file(filename):
            continue
        if filename.lower().endswith(".safetensors"):
            has_safetensors = True
        size = sibling.get("size")
        files.append(
            HfGgufFile(
                filename=filename,
                size_bytes=int(size) if isinstance(size, (int, float)) else None,
                quant=None,
                is_mmproj=False,
                is_multipart=False,
            )
        )
    if not has_safetensors:
        raise HfHubError(
            f"repo {repo!r} ships no safetensors weights — not an encoder checkpoint"
        )
    return files


async def download_snapshot(
    repo: str,
    files: list[HfGgufFile],
    destination: Path,
    *,
    client: httpx.AsyncClient,
    revision: str = "main",
    progress: ProgressCallback | None = None,
) -> Path:
    """Download every snapshot file under ``destination`` (preserving repo paths).

    ``progress(received, total)`` reports CUMULATIVE bytes across all files
    against the snapshot's known total, so a single bar tracks the whole pull.
    """
    total = sum(f.size_bytes or 0 for f in files) or None
    done_before = 0
    for file in files:
        target = destination / file.filename
        if progress is not None:
            base = done_before

            async def file_progress(received: int, _t: int | None, _base: int = base) -> None:
                await progress(_base + received, total)  # type: ignore[misc]

            await download_file(
                repo, file.filename, target, client=client, revision=revision, progress=file_progress
            )
        else:
            await download_file(repo, file.filename, target, client=client, revision=revision)
        done_before += file.size_bytes or 0
    return destination


async def search_models(
    query: str, *, client: httpx.AsyncClient, limit: int = 25, gguf_only: bool = True
) -> list[dict[str, Any]]:
    """Search the Hub for models (server-side proxy, HF_TOKEN aware).

    ``gguf_only`` filters to GGUF repos (the primary deploy path); drop it to
    also surface safetensors (encoders). Sorted by downloads. Returns light
    cards — verdicts are fetched per-repo on demand via :func:`inspect_repo`.
    """
    params: dict[str, str] = {
        "search": query,
        "limit": str(max(1, min(limit, 50))),
        "sort": "downloads",
        "direction": "-1",
    }
    if gguf_only:
        params["filter"] = "gguf"
    try:
        response = await client.get(
            f"{HF_BASE}/api/models", params=params, headers=hf_headers(), timeout=20.0
        )
    except httpx.RequestError as exc:
        raise HfHubError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if response.status_code >= 400:
        raise HfHubError(f"Hub search returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, list):
        return []
    cards: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = item.get("id") or item.get("modelId")  # HF uses either
        if not repo_id:
            continue
        cards.append(
            {
                "id": str(repo_id),
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "gated": bool(item.get("gated")),
                "tags": [str(t) for t in (item.get("tags") or []) if isinstance(t, str)][:12],
            }
        )
    return cards


async def inspect_repo(repo: str, *, client: httpx.AsyncClient) -> dict[str, Any]:
    """Pre-flight support detection for a repo — WITHOUT downloading weights.

    Reads the architecture from the Hub's parsed metadata (the ``gguf`` block
    for GGUF repos: ``general.architecture`` + chat template; else the model
    ``config``'s ``model_type``/``architectures``; else the raw ``config.json``),
    detects modality signals (GGUF / safetensors / mmproj), and resolves them to
    a family + a verdict via :mod:`docie_bench.serving.arch_registry`.

    Returns a dict the Studio renders into a Deploy button state:
    ``{repo, architecture, verdict, family, reason, has_gguf, has_safetensors,
    has_mmproj, quants, suggested_name}``.
    """
    from docie_bench.serving.arch_registry import resolve_family

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise HfHubError(f"invalid Hugging Face repo id {repo!r} (expected owner/name)")
    url = f"{HF_BASE}/api/models/{repo}?blobs=true"
    try:
        response = await client.get(url, headers=hf_headers(), timeout=20.0)
    except httpx.RequestError as exc:
        raise HfHubError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if response.status_code == 404:
        raise HfHubError(f"repo {repo!r} does not exist on the Hugging Face Hub")
    if response.status_code in (401, 403):
        raise HfHubError(f"repo {repo!r} was refused by the Hub — check the id or set HF_TOKEN")
    if response.status_code >= 400:
        raise HfHubError(f"Hub returned HTTP {response.status_code} for {repo!r}")
    payload = response.json()

    siblings = [s for s in payload.get("siblings", []) if isinstance(s, dict)]
    ggufs = [g for s in siblings if (g := _gguf_from_sibling(s)) is not None]
    has_gguf = bool(ggufs)
    has_safetensors = any(
        str(s.get("rfilename", "")).endswith(".safetensors") for s in siblings
    )
    has_mmproj = any(g.is_mmproj for g in ggufs)

    architecture = _extract_architecture(payload)
    if architecture is None and has_safetensors:
        architecture = await _config_architecture(repo, client)

    result = resolve_family(
        architecture,
        has_gguf=has_gguf,
        has_safetensors=has_safetensors,
        has_mmproj=has_mmproj,
    )
    return {
        "repo": repo,
        "architecture": architecture,
        "verdict": result.verdict,
        "family": result.family,
        "reason": result.reason,
        "runtime_note": result.runtime_note,
        "has_gguf": has_gguf,
        "has_safetensors": has_safetensors,
        "has_mmproj": has_mmproj,
        "quants": sorted(
            {g.quant for g in ggufs if g.quant and not g.is_mmproj and not g.is_multipart}
        ),
        "suggested_name": default_store_name(repo),
    }


def _extract_architecture(payload: dict[str, Any]) -> str | None:
    """Architecture from the Hub model-info payload (gguf block, then config)."""
    gguf = payload.get("gguf")
    if isinstance(gguf, dict) and gguf.get("architecture"):
        return str(gguf["architecture"])
    config = payload.get("config")
    if isinstance(config, dict):
        if config.get("model_type"):
            return str(config["model_type"])
        archs = config.get("architectures")
        if isinstance(archs, list) and archs:
            return str(archs[0])
    return None


async def _config_architecture(repo: str, client: httpx.AsyncClient) -> str | None:
    """Fallback: read model_type/architectures from the raw config.json."""
    url = f"{HF_BASE}/{repo}/resolve/main/config.json"
    try:
        response = await client.get(
            url, headers=hf_headers(), follow_redirects=True, timeout=15.0
        )
    except httpx.RequestError:
        return None
    if response.status_code >= 400:
        return None
    try:
        config = response.json()
    except ValueError:
        return None
    if not isinstance(config, dict):
        return None
    if config.get("model_type"):
        return str(config["model_type"])
    archs = config.get("architectures")
    if isinstance(archs, list) and archs:
        return str(archs[0])
    return None


async def list_collection(slug: str, *, client: httpx.AsyncClient) -> dict[str, Any]:
    """A HF collection's model repos (``owner/slug-hash`` or its full URL).

    Returns ``{"slug", "title", "models": [repo ids]}`` in the collection's
    curated order — the provider's own grouping, which is exactly the nesting
    the Studio's collection picker surfaces.
    """
    cleaned = slug.strip()
    if cleaned.startswith("http"):
        cleaned = cleaned.split("/collections/", 1)[-1]
    cleaned = cleaned.strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", cleaned):
        raise HfHubError(
            f"invalid collection {slug!r} (expected owner/slug-hash or its URL)"
        )
    url = f"{HF_BASE}/api/collections/{cleaned}"
    try:
        response = await client.get(url, headers=hf_headers(), timeout=20.0)
    except httpx.RequestError as exc:
        raise HfHubError(f"Hugging Face Hub is unreachable: {exc}") from exc
    if response.status_code == 404:
        raise HfHubError(f"collection {cleaned!r} does not exist")
    if response.status_code >= 400:
        raise HfHubError(f"Hub returned HTTP {response.status_code} for {cleaned!r}")
    payload = response.json()
    models = [
        str(item.get("id"))
        for item in payload.get("items", [])
        if isinstance(item, dict) and item.get("type") == "model" and item.get("id")
    ]
    return {"slug": cleaned, "title": payload.get("title") or cleaned, "models": models}
