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
        raise HfHubError(
            f"repo {repo!r} is gated/private — set HF_TOKEN on the serving service"
        )
    if response.status_code >= 400:
        raise HfHubError(f"Hub returned HTTP {response.status_code} for {repo!r}")
    payload = response.json()
    files = [
        gguf
        for sibling in payload.get("siblings", [])
        if isinstance(sibling, dict) and (gguf := _gguf_from_sibling(sibling)) is not None
    ]
    if not files:
        raise HfHubError(
            f"repo {repo!r} ships no GGUF files — pick that model's -GGUF conversion repo"
        )
    return files


def pick_gguf(files: list[HfGgufFile], quant: str | None) -> HfGgufFile:
    """Choose the model GGUF for ``quant`` (or the best default available)."""
    candidates = [f for f in files if not f.is_mmproj and not f.is_multipart]
    multipart_only = [f for f in files if not f.is_mmproj and f.is_multipart]
    if not candidates:
        if multipart_only:
            raise HfHubError(
                "this repo ships only multi-part GGUFs (…-00001-of-000NN), which the "
                "store does not assemble yet — pick a single-file quant or another repo"
            )
        raise HfHubError("no model GGUF found in this repo (mmproj-only?)")
    if quant:
        wanted = quant.strip().upper()
        for f in candidates:
            if f.quant == wanted:
                return f
        available = sorted({f.quant or f.filename for f in candidates})
        raise HfHubError(
            f"quant {quant!r} not found; available: {', '.join(available)}"
        )
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
