from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from rapidfuzz.fuzz import ratio

from docie_bench.benchmark.dataset import DatasetItem, load_dataset

REGISTRY_FORMAT_VERSION = 1
DEFAULT_REGISTRY_PATH = Path("data/datasets.yaml")
SUPPORTED_DOCUMENT_SUFFIXES = {
    ".txt",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetVersion(BaseModel):
    manifest: str
    dataset_hash: str
    created_at: str
    statistics: dict[str, Any] = Field(default_factory=dict)


class DatasetRecord(BaseModel):
    description: str | None = None
    latest: str | None = None
    versions: dict[str, DatasetVersion] = Field(default_factory=dict)


class DatasetRegistry(BaseModel):
    registry_version: int = REGISTRY_FORMAT_VERSION
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)


class ResolvedDataset(BaseModel):
    reference: str
    manifest_path: Path
    dataset_hash: str
    items: list[DatasetItem]
    version: str | None = None


def sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def document_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _canonical_item(item: DatasetItem) -> dict[str, Any]:
    value = item.model_dump(mode="json", exclude_none=True)
    # An empty provenance sidecar must not perturb dataset identity: pre-provenance
    # manifests keep their pinned dataset_hash. A *populated* sidecar is a real
    # labeling change and is intentionally hashed. (exclude_none keeps {} — an
    # empty dict is not None — so drop it explicitly.)
    if not value.get("label_provenance"):
        value.pop("label_provenance", None)
    value["file_hash"] = document_hash(Path(item.file_path))
    del value["file_path"]
    return value


def dataset_hash(items: list[DatasetItem]) -> str:
    canonical = sorted((_canonical_item(item) for item in items), key=lambda row: row["doc_id"])
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in canonical
    )
    return sha256_bytes(payload.encode("utf-8"))


def dataset_statistics(items: list[DatasetItem]) -> dict[str, Any]:
    ground_truth_fields = Counter(key for item in items for key in item.ground_truth)
    sizes = [
        Path(item.file_path).stat().st_size
        for item in items
        if Path(item.file_path).is_file()
    ]
    return {
        "documents": len(items),
        "total_bytes": sum(sizes),
        "schemas": dict(sorted(Counter(item.schema_name for item in items).items())),
        "languages": dict(
            sorted(Counter(item.language or "unspecified" for item in items).items())
        ),
        "splits": dict(sorted(Counter(item.split for item in items).items())),
        "ground_truth_fields": dict(sorted(ground_truth_fields.items())),
        "labeled_documents": sum(bool(item.ground_truth) for item in items),
    }


def _normalized_text(path: Path) -> str | None:
    if path.suffix.lower() != ".txt":
        return None
    text = path.read_text(encoding="utf-8", errors="replace").casefold()
    return " ".join(text.split())


def detect_leakage(
    items: list[DatasetItem],
    near_duplicate_threshold: float = 0.92,
) -> dict[str, Any]:
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    documents = [
        {
            "doc_id": item.doc_id,
            "split": item.split,
            "hash": document_hash(Path(item.file_path)),
            "text": _normalized_text(Path(item.file_path)),
        }
        for item in items
    ]
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for index, left in enumerate(documents):
        for right in documents[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            pair = {
                "left_doc_id": left["doc_id"],
                "left_split": left["split"],
                "right_doc_id": right["doc_id"],
                "right_split": right["split"],
            }
            if left["hash"] == right["hash"]:
                exact.append(pair)
                continue
            if left["text"] is None or right["text"] is None:
                continue
            similarity = ratio(left["text"], right["text"]) / 100
            if similarity >= near_duplicate_threshold:
                near.append({**pair, "similarity": round(similarity, 4)})
    return {
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "leakage_pairs": len(exact) + len(near),
    }


def validate_dataset(
    manifest_path: Path,
    *,
    near_duplicate_threshold: float = 0.92,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        items = load_dataset(manifest_path)
    except (OSError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}

    ids = Counter(item.doc_id for item in items)
    if not items:
        errors.append("Dataset must contain at least one document")
    errors.extend(f"Duplicate doc_id: {doc_id}" for doc_id, count in ids.items() if count > 1)
    for item in items:
        path = Path(item.file_path)
        if not path.is_file():
            errors.append(f"{item.doc_id}: file does not exist: {path}")
        elif path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            errors.append(f"{item.doc_id}: unsupported file suffix: {path.suffix}")
        if not item.split.strip():
            errors.append(f"{item.doc_id}: split must not be empty")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    current_hash = dataset_hash(items)
    leakage = detect_leakage(items, near_duplicate_threshold)
    if leakage["leakage_pairs"]:
        errors.append(f"Detected {leakage['leakage_pairs']} cross-split leakage pair(s)")
    if expected_hash is not None and current_hash != expected_hash:
        errors.append(f"Dataset hash mismatch: expected {expected_hash}, got {current_hash}")
    if all(item.split == "unspecified" for item in items):
        warnings.append("All documents use the unspecified split")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset_hash": current_hash,
        "statistics": dataset_statistics(items),
        "leakage": leakage,
    }


def load_registry(path: Path) -> DatasetRegistry:
    if not path.exists():
        return DatasetRegistry()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    registry = DatasetRegistry.model_validate(raw)
    if registry.registry_version != REGISTRY_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported registry_version={registry.registry_version}; "
            f"expected {REGISTRY_FORMAT_VERSION}"
        )
    return registry


def save_registry(path: Path, registry: DatasetRegistry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(
        registry.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def parse_reference(reference: str) -> tuple[str, str | None]:
    name, separator, version = reference.partition("@")
    if not name or (separator and not version):
        raise ValueError(f"Invalid dataset reference: {reference!r}")
    return name, version or None


def resolve_dataset(
    source: str | Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    verify_hash: bool = True,
) -> ResolvedDataset:
    path = Path(source)
    if path.is_file():
        items = load_dataset(path)
        return ResolvedDataset(
            reference=str(source),
            manifest_path=path.resolve(),
            dataset_hash=dataset_hash(items),
            items=items,
        )

    reference = str(source)
    name, requested_version = parse_reference(reference)
    registry = load_registry(registry_path)
    if name not in registry.datasets:
        raise ValueError(f"Unknown dataset {name!r} in registry {registry_path}")
    record = registry.datasets[name]
    version = requested_version or record.latest
    if version is None or version not in record.versions:
        raise ValueError(f"Unknown version {version!r} for dataset {name!r}")
    entry = record.versions[version]
    manifest_path = (registry_path.parent / entry.manifest).resolve()
    items = load_dataset(manifest_path)
    current_hash = dataset_hash(items)
    if verify_hash and current_hash != entry.dataset_hash:
        raise ValueError(
            f"Dataset hash mismatch for {name}@{version}: "
            f"expected {entry.dataset_hash}, got {current_hash}"
        )
    return ResolvedDataset(
        reference=f"{name}@{version}",
        manifest_path=manifest_path,
        dataset_hash=current_hash,
        items=items,
        version=version,
    )


def register_dataset_version(
    *,
    registry_path: Path,
    name: str,
    version: str,
    manifest_path: Path,
    description: str | None = None,
) -> DatasetVersion:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            "Dataset name may contain only letters, numbers, dots, underscores, and dashes"
        )
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("Dataset version must be semantic, for example 1.0.0")
    report = validate_dataset(manifest_path)
    if not report["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
    registry = load_registry(registry_path)
    record = registry.datasets.setdefault(name, DatasetRecord(description=description))
    if version in record.versions:
        raise ValueError(f"Dataset version already exists: {name}@{version}")
    if description is not None:
        record.description = description
    try:
        relative_manifest = manifest_path.resolve().relative_to(registry_path.parent.resolve())
    except ValueError:
        relative_manifest = Path(
            os.path.relpath(manifest_path.resolve(), registry_path.parent.resolve())
        )
    entry = DatasetVersion(
        manifest=relative_manifest.as_posix(),
        dataset_hash=report["dataset_hash"],
        created_at=datetime.now(UTC).isoformat(),
        statistics=report["statistics"],
    )
    record.versions[version] = entry
    record.latest = version
    save_registry(registry_path, registry)
    return entry


def migrate_manifest(
    source: Path,
    destination: Path,
    *,
    default_split: str = "test",
    split_map: dict[str, str] | None = None,
) -> Path:
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if split_map is not None and not isinstance(split_map, dict):
        raise ValueError("split_map must be a JSON object mapping doc_id to split")
    items = load_dataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for item in items:
        item.split = (split_map or {}).get(item.doc_id, default_split)
        value = item.model_dump(mode="json", exclude_none=True)
        value["file_path"] = Path(
            os.path.relpath(Path(item.file_path), destination.parent.resolve())
        ).as_posix()
        rows.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination
