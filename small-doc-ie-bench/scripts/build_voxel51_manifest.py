"""Build a benchmark manifest for the Voxel51 invoice dataset (thin caller).

All label logic lives in the declarative spec
``scripts/label_mapping/voxel51_invoice.yaml`` and the generic engine
``docie_bench.benchmark.label_mapping.apply_mapping``. This script only:

  1. loads the raw per-document annotations,
  2. calls ``apply_mapping`` to produce ``(ground_truth, label_provenance)``,
  3. writes a JSONL manifest of ``DatasetItem`` rows, and
  4. writes a ``label_audit.json`` sidecar (findings only; labels untouched).

Because the total (TTC) is not printed on Voxel51 invoices, ``total_ttc.amount``
is emitted with ``derived`` provenance and the audit records
``total_ttc_reconciled: null`` — see the YAML header for the rationale.

NOTE: end-to-end execution requires the actual Voxel51 export (raw annotations +
document files), which is NOT vendored in this repo. The mapping engine, spec,
and audit are unit-tested against synthetic annotations; running this script
against a real export is deferred to live verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from docie_bench.benchmark.dataset import DatasetItem, load_dataset
from docie_bench.benchmark.label_audit import write_label_audit
from docie_bench.benchmark.label_mapping import apply_mapping

DEFAULT_SPEC = Path(__file__).parent / "label_mapping" / "voxel51_invoice.yaml"


def load_spec(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return spec


def build_items(
    annotations: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for annotation in annotations:
        ground_truth, label_provenance = apply_mapping(annotation, spec)
        items.append(
            DatasetItem(
                doc_id=str(annotation["doc_id"]),
                file_path=str(annotation["file_path"]),
                schema_name=spec.get("schema_name", "invoice"),
                split=str(annotation.get("split", "unspecified")),
                language=annotation.get("language"),
                ground_truth=ground_truth,
                label_provenance=label_provenance,
            )
        )
    return items


def write_manifest(path: Path, items: list[DatasetItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            item.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in items
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="JSON list of raw annotations")
    parser.add_argument("manifest", type=Path, help="Output manifest.jsonl path")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Mapping YAML spec")
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Output label_audit.json path (defaults next to the manifest)",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    items = build_items(annotations, spec)
    write_manifest(args.manifest, items)

    audit_path = args.audit or args.manifest.with_name("label_audit.json")
    # Re-load through load_dataset so audit runs on exactly what benchmarks consume.
    write_label_audit(audit_path, load_dataset(args.manifest))
    print(f"Wrote {len(items)} items to {args.manifest} and audit to {audit_path}")  # noqa: T201


if __name__ == "__main__":
    main()
