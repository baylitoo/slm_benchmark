"""What makes a model deployable from the shared repository (#595 stage 2).

A bucket full of weights is not a repository. Two of the decided constraints
say so from opposite ends -- *publish artifacts first and the manifest last*,
and *arbitrary files uploaded to a bucket must not automatically become
deployable models* -- and both reduce to one rule:

    **The manifest, not the presence of bytes, is what makes a model exist.**

Everything here follows from that.

Layout
------

::

    <prefix>/blobs/sha256/<hex>                    immutable, content-addressed
    <prefix>/models/<name>/<manifest-digest>.json  immutable, one per publication
    <prefix>/models/<name>/current.json            mutable pointer

* **Blobs are shared and immutable.** Two models with the same projector, or a
  re-publication that changes one file, upload only what is new; a node that
  already holds a digest skips it entirely.
* **A manifest is immutable.** Re-publishing a name writes a NEW manifest, so a
  pull in flight keeps resolving the bytes it started with.
* **``current.json`` is the only mutable object, and it is written last.**
  Before that write the publication is invisible; after it, complete. The
  "artifacts first, manifest last" rule reduces to a single atomic PUT, and an
  interrupted publish leaves orphaned blobs and no visible model -- garbage,
  never corruption.

Why a pointer AND an immutable manifest
---------------------------------------

The pointer names a digest rather than describing the model, so a deployment
resolves the mutable name ONCE when its intent is accepted and then pins that
digest. Materialisation and every retry use the pinned value, which is what
stops a later publication silently changing a running deployment. Rollback
becomes a one-field change rather than a re-publication.

This module is pure: building, validating and serialising. It touches no
bucket and no disk, so every rule about what is deployable is a table test.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Bumped when a reader could MISREAD an older document, never for additions a
#: reader may ignore. A reader refuses what it does not understand rather than
#: guessing, which is the only safe default for something that decides what
#: gets served.
SCHEMA_VERSION = 1

#: Object-key prefixes. Kept here because the manifest and the layout are one
#: design: a reader that knows the format knows where to look.
BLOB_PREFIX = "blobs/sha256"
MODEL_PREFIX = "models"
POINTER_NAME = "current.json"

_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
#: A file's path inside the model directory. Relative, forward-slashed, and
#: with no traversal -- it is joined onto a local directory when materialising.
_SAFE_PATH = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\Z")

#: What a file is FOR. Carried explicitly so a reader never has to infer a
#: role from a filename.
ROLES = frozenset({"model", "mmproj", "snapshot"})


class InvalidManifestError(ValueError):
    """A document is not a manifest this version can act on."""


@dataclass(frozen=True)
class ManifestFile:
    """One artifact: what it is, where it goes, and how to know it arrived."""

    role: str
    path: str
    digest: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }

    @property
    def blob_key(self) -> str:
        """Where this artifact's bytes live, independent of any model name."""
        return f"{BLOB_PREFIX}/{self.digest.split(':', 1)[1]}"


@dataclass(frozen=True)
class Manifest:
    """A complete, immutable description of one publication of one model."""

    name: str
    family: str
    kind: str  # "gguf" | "snapshot"
    files: tuple[ManifestFile, ...]
    created_at: str
    source: str | None = None
    published_by: str | None = None
    schema: int = SCHEMA_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        """Summed here, and also written into the document.

        Redundant on purpose: the disk-headroom check runs BEFORE a download
        starts and should be able to refuse without parsing every entry.
        """
        return sum(f.size_bytes for f in self.files)

    def file_for(self, role: str) -> ManifestFile | None:
        for entry in self.files:
            if entry.role == role:
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": self.schema,
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "created_at": self.created_at,
            "total_bytes": self.total_bytes,
            "files": [f.as_dict() for f in self.files],
        }
        if self.source is not None:
            document["source"] = self.source
        if self.published_by is not None:
            document["published_by"] = self.published_by
        document.update(self.extra)
        return document

    def to_json(self) -> bytes:
        """Canonical bytes: sorted keys, compact, UTF-8.

        The digest of these bytes NAMES the manifest, so two publications of
        identical content must serialise identically or the same model would
        land under two keys.
        """
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def manifest_key(self, digest: str) -> str:
        return f"{MODEL_PREFIX}/{self.name}/{digest.split(':', 1)[1]}.json"


def pointer_key(name: str) -> str:
    return f"{MODEL_PREFIX}/{name}/{POINTER_NAME}"


def pointer_document(manifest_digest: str) -> bytes:
    """The single mutable object, naming the manifest it points at.

    Carries the schema so a reader can refuse a pointer written by a newer
    publisher instead of fetching a manifest it cannot act on.
    """
    return json.dumps(
        {"schema": SCHEMA_VERSION, "manifest": manifest_digest},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def parse_pointer(raw: bytes | str) -> str:
    """The manifest digest a pointer names, or raise."""
    document = _load(raw, what="pointer")
    _require_schema(document, what="pointer")
    digest = document.get("manifest")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise InvalidManifestError(
            f"pointer names manifest {digest!r}, which is not a sha256 digest"
        )
    return digest


def build(
    *,
    name: str,
    family: str,
    kind: str,
    files: Sequence[ManifestFile],
    source: str | None = None,
    published_by: str | None = None,
    now: dt.datetime | None = None,
) -> Manifest:
    """Assemble a manifest and refuse one that could not be served.

    Validated at construction rather than at publish time: an invalid manifest
    should never reach a bucket, because the whole point of the format is that
    its presence is a promise.
    """
    stamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    manifest = Manifest(
        name=name,
        family=family,
        kind=kind,
        files=tuple(files),
        created_at=stamp.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        source=source,
        published_by=published_by,
    )
    validate(manifest)
    return manifest


def validate(manifest: Manifest) -> Manifest:
    """Raise unless ``manifest`` describes something actually servable."""
    if manifest.schema != SCHEMA_VERSION:
        raise InvalidManifestError(
            f"manifest schema {manifest.schema} is not {SCHEMA_VERSION}"
        )
    if not manifest.name or not manifest.family:
        raise InvalidManifestError("a manifest needs both a name and a family")
    if manifest.kind not in {"gguf", "snapshot"}:
        raise InvalidManifestError(
            f"unknown manifest kind {manifest.kind!r} (expected 'gguf' or 'snapshot')"
        )
    if not manifest.files:
        raise InvalidManifestError(
            f"manifest for {manifest.name!r} lists no files, so it describes "
            f"nothing that can be served"
        )

    seen: set[str] = set()
    for entry in manifest.files:
        if entry.role not in ROLES:
            raise InvalidManifestError(
                f"unknown role {entry.role!r} (expected one of {sorted(ROLES)})"
            )
        if not _DIGEST.match(entry.digest):
            raise InvalidManifestError(
                f"{entry.path!r} has digest {entry.digest!r}, which is not "
                f"'sha256:<64 hex>'"
            )
        if entry.size_bytes < 0:
            raise InvalidManifestError(f"{entry.path!r} has a negative size")
        # The path is joined onto a local directory when materialising, so it
        # is restricted the same way a store name is -- a manifest must not be
        # able to write outside the model it describes.
        if not _SAFE_PATH.match(entry.path) or ".." in entry.path.split("/"):
            raise InvalidManifestError(
                f"{entry.path!r} is not a safe relative path inside the model "
                f"directory"
            )
        if entry.path in seen:
            raise InvalidManifestError(f"{entry.path!r} is listed twice")
        seen.add(entry.path)

    # Kind-specific completeness. A GGUF model without its weights, or a
    # snapshot whose tree is described as a single GGUF, would publish cleanly
    # and fail to load -- which is exactly the class of failure the manifest
    # exists to prevent.
    if manifest.kind == "gguf":
        if manifest.file_for("model") is None:
            raise InvalidManifestError(
                f"gguf manifest for {manifest.name!r} has no 'model' file"
            )
        wrong = [f.path for f in manifest.files if f.role == "snapshot"]
        if wrong:
            raise InvalidManifestError(
                f"gguf manifest for {manifest.name!r} lists snapshot files: {wrong}"
            )
    else:
        wrong = [f.path for f in manifest.files if f.role != "snapshot"]
        if wrong:
            raise InvalidManifestError(
                f"snapshot manifest for {manifest.name!r} lists non-snapshot "
                f"files: {wrong}"
            )
    return manifest


def parse(raw: bytes | str) -> Manifest:
    """Read a manifest document, validating it. Never returns a partial one."""
    document = _load(raw, what="manifest")
    _require_schema(document, what="manifest")

    raw_files = document.get("files")
    if not isinstance(raw_files, list):
        raise InvalidManifestError("manifest 'files' must be a list")
    files = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise InvalidManifestError("each entry in 'files' must be an object")
        try:
            files.append(
                ManifestFile(
                    role=str(item["role"]),
                    path=str(item["path"]),
                    digest=str(item["digest"]),
                    size_bytes=int(item["size_bytes"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidManifestError(f"malformed file entry: {item!r}") from exc

    known = {
        "schema",
        "name",
        "family",
        "kind",
        "created_at",
        "total_bytes",
        "files",
        "source",
        "published_by",
    }
    manifest = Manifest(
        name=str(document.get("name") or ""),
        family=str(document.get("family") or ""),
        kind=str(document.get("kind") or ""),
        files=tuple(files),
        created_at=str(document.get("created_at") or ""),
        source=document.get("source"),
        published_by=document.get("published_by"),
        schema=int(document.get("schema", 0)),
        # Forward fields a newer publisher added are preserved rather than
        # dropped, so a round-trip through an older reader does not quietly
        # destroy them.
        extra={k: v for k, v in document.items() if k not in known},
    )
    return validate(manifest)


def _load(raw: bytes | str, *, what: str) -> dict[str, Any]:
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise InvalidManifestError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise InvalidManifestError(f"{what} must be a JSON object")
    return document


def _require_schema(document: Mapping[str, Any], *, what: str) -> None:
    schema = document.get("schema")
    if schema != SCHEMA_VERSION:
        raise InvalidManifestError(
            f"{what} declares schema {schema!r}; this build understands "
            f"{SCHEMA_VERSION}. Refusing rather than guessing at its meaning."
        )


__all__ = [
    "BLOB_PREFIX",
    "MODEL_PREFIX",
    "POINTER_NAME",
    "ROLES",
    "SCHEMA_VERSION",
    "InvalidManifestError",
    "Manifest",
    "ManifestFile",
    "build",
    "parse",
    "parse_pointer",
    "pointer_document",
    "pointer_key",
    "validate",
]
