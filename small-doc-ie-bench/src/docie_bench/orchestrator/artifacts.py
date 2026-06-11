from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredArtifact:
    name: str
    uri: str
    sha256: str
    size_bytes: int
    media_type: str


class ArtifactStore(Protocol):
    def put(
        self,
        *,
        run_id: str,
        task_id: str | None,
        name: str,
        content: bytes,
        media_type: str = "application/octet-stream",
    ) -> StoredArtifact: ...


class LocalArtifactStore:
    """Atomic content-addressed storage scoped by run and task."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(
        self,
        *,
        run_id: str,
        task_id: str | None,
        name: str,
        content: bytes,
        media_type: str = "application/octet-stream",
    ) -> StoredArtifact:
        safe_name = Path(name).name
        if safe_name != name or not safe_name:
            raise ValueError("Artifact name must be a plain file name")
        digest = hashlib.sha256(content).hexdigest()
        # Keep paths short enough for Windows while isolating competing content.
        directory = self.root / digest[:16]
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / safe_name
        fd, temporary = tempfile.mkstemp(prefix=f".{safe_name}.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return StoredArtifact(
            name=name,
            uri=destination.resolve().as_uri(),
            sha256=digest,
            size_bytes=len(content),
            media_type=media_type,
        )
