"""POST /v1/studio/datasets/{name}/validate: the CLI's dataset validate/inspect
logic (duplicate doc_id, missing files, cross-split leakage, statistics),
reachable from the Studio instead of `docie-bench dataset validate` only.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import docie_bench.api as api
import docie_bench.benchmark.registry as registry


def _write_manifest(root: Path, rows: list[tuple[str, str, str]]) -> Path:
    files = root / "files"
    files.mkdir(parents=True)
    manifest_rows = []
    for doc_id, split, text in rows:
        path = files / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "doc_id": doc_id,
                "file_path": f"files/{doc_id}.txt",
                "schema_name": "invoice",
                "language": "en",
                "split": split,
                "ground_truth": {"invoice_number": doc_id},
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8"
    )
    return manifest


def _client() -> TestClient:
    return TestClient(api.app)


def test_validate_endpoint_reports_a_clean_registered_dataset(tmp_path, monkeypatch) -> None:
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest
    )
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)

    resp = _client().post("/v1/studio/datasets/invoices/validate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["reference"] == "invoices@1.0.0"
    assert body["version"] == "1.0.0"
    assert body["statistics"]["documents"] == 1
    assert body["leakage"]["leakage_pairs"] == 0


def test_validate_endpoint_reports_corruption_introduced_after_registration(
    tmp_path, monkeypatch
) -> None:
    # register_dataset_version gate-keeps validity AT REGISTRATION time (it runs
    # validate_dataset itself and refuses to register an invalid manifest) -- so the
    # only way an already-registered dataset can be invalid later is corruption
    # introduced to the manifest on disk afterward (duplicated here, pointing at the
    # SAME still-existing file -- see the next test for the missing-file case, which
    # takes a different code path). Appending the row ALSO drifts the dataset's hash
    # away from the registered entry -- this is deliberately both a structural error
    # AND real drift at once, to pin validate_dataset's own short-circuit: it returns
    # on the structural error before ever reaching the hash comparison, so a caller
    # only sees "Duplicate doc_id", never a hash-mismatch alongside it. That's
    # validate_dataset's existing behavior (registry.py), not something the route
    # changes -- see the route's own docstring.
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest
    )
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + json.dumps(row) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)

    resp = _client().post("/v1/studio/datasets/invoices/validate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert any("Duplicate doc_id" in e for e in body["errors"])
    assert not any("hash mismatch" in e for e in body["errors"])
    assert "statistics" not in body


def test_validate_endpoint_surfaces_a_missing_referenced_file_as_422_not_500(
    tmp_path, monkeypatch
) -> None:
    # resolve_dataset unconditionally hashes every referenced file, even with
    # verify_hash=False -- a manifest referencing a file that no longer exists at all
    # raises OSError there, before validate_dataset's own graceful per-item existence
    # check ever runs (same gap the CLI's dataset_validate has, cli.py:363). Must not
    # leak as a raw 500.
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest
    )
    (manifest.parent / "files" / "a.txt").unlink()
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)

    resp = _client().post("/v1/studio/datasets/invoices/validate")

    assert resp.status_code == 422
    # validate_dataset returns early on structural errors (duplicate/missing-file),
    # before it ever reaches the hash comparison -- so hash drift is NOT reported
    # here even though it's also present; that's existing validate_dataset behavior
    # (see test_validation_reports_duplicates_missing_files_and_statistics), not
    # something this endpoint changes.


def test_validate_endpoint_targets_a_specific_version(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "datasets.yaml"
    manifest_v1 = _write_manifest(tmp_path / "v1", [("a", "test", "Invoice A")])
    manifest_v2 = _write_manifest(
        tmp_path / "v2", [("a", "test", "Invoice A"), ("b", "train", "Invoice B")]
    )
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest_v1
    )
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="2.0.0", manifest_path=manifest_v2
    )
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)

    resp = _client().post("/v1/studio/datasets/invoices/validate", params={"version": "1.0.0"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["reference"] == "invoices@1.0.0"
    assert body["statistics"]["documents"] == 1


def test_validate_endpoint_rejects_unknown_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", tmp_path / "datasets.yaml")

    resp = _client().post("/v1/studio/datasets/does-not-exist/validate")

    assert resp.status_code == 404


def test_validate_endpoint_reports_drift_within_one_report_not_a_bare_404(
    tmp_path, monkeypatch
) -> None:
    # Files changing after registration is the one case register_dataset_version's own
    # gate can't prevent -- verified NOT to raise/stop early (unlike duplicate/missing
    # errors above): expected_hash flows into the SAME report as statistics/leakage.
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest
    )
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)
    (manifest.parent / "files" / "a.txt").write_text("Changed", encoding="utf-8")

    resp = _client().post("/v1/studio/datasets/invoices/validate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert any("hash mismatch" in e for e in body["errors"])
    assert body["statistics"]["documents"] == 1


def test_validate_endpoint_honors_near_duplicate_threshold(tmp_path, monkeypatch) -> None:
    # Below the default 0.92 registration gate (~0.68 similarity) so register_dataset_version
    # accepts it cleanly; a strict query threshold still catches it as near-duplicate.
    manifest = _write_manifest(
        tmp_path / "dataset",
        [
            ("a", "train", "Customer Acme invoice 456 total 1000 euros due tomorrow"),
            ("b", "test", "Customer Beta invoice 789 total 500 dollars due next week"),
        ],
    )
    registry_path = tmp_path / "datasets.yaml"
    registry.register_dataset_version(
        registry_path=registry_path, name="invoices", version="1.0.0", manifest_path=manifest
    )
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_path)

    lenient = _client().post(
        "/v1/studio/datasets/invoices/validate", params={"near_duplicate_threshold": 0.9}
    )
    strict = _client().post(
        "/v1/studio/datasets/invoices/validate", params={"near_duplicate_threshold": 0.5}
    )

    assert lenient.json()["leakage"]["leakage_pairs"] == 0
    assert strict.json()["leakage"]["leakage_pairs"] == 1
