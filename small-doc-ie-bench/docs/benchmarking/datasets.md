# Datasets

A dataset is a `manifest.jsonl` plus the documents it points at. The registry
gives it a stable, versioned name and pins its contents by hash, so a benchmark
result can always say exactly what it was measured on.

## The manifest

One JSON object per document:

```json
{"doc_id": "invoice-001", "file_path": "files/invoice-001.txt", "schema_name": "invoice", "language": "fr", "split": "test", "ground_truth": {"invoice_number": "INV-2026-0001", "vendor_name": "ACME SAS", "total_ttc.amount": "1245.30", "total_ttc.currency": "EUR", "issue_date": "2026-05-21"}}
```

Ground-truth keys are dotted field paths. A row may carry `dynamic_schema` to
benchmark against a schema that is not built in.

| File | Ingestion |
|---|---|
| `.txt` | used as-is, already OCR'd |
| `.pdf` | the PDF text layer via liteparse, with OCR for scanned pages |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | an OCR backend, or a vision profile |

A vision profile rasterises PDFs and images and rejects `.txt`. Every result is
labelled `vision` or `ocr:<backend>`, so one manifest can compare both paths.

### Tables

Put a list under the matching key to evaluate a table:

```json
{"ground_truth": {"line_items": [{"description": "Keyboard", "quantity": "2", "line_total.amount": "150.00"}]}}
```

Predicted rows are aligned to expected rows by best cell similarity, then cell
accuracy and row precision, recall and F1 are computed. Validation also checks
each `quantity × unit_price` against its line total, and the line totals
against the subtotal, reporting mismatches as warnings.

## The registry

`data/datasets.yaml` maps references such as `sample@1.0.0` to manifests. A
version pins a SHA-256 and stored statistics; a run verifies the hash before it
starts and records the resolved reference, version and hash in `metrics.json`.
A reference without `@version` resolves to `latest`. A bare manifest path still
works for one-off runs.

```bash
docie-bench dataset version invoices 1.0.0 --manifest data/invoices/v1/manifest.jsonl
docie-bench dataset inspect invoices@1.0.0
docie-bench dataset validate invoices@1.0.0
docie-bench dataset leakage invoices@1.0.0 --near-duplicate-threshold 0.92
```

The hash covers every manifest row plus each referenced document's own hash,
and ignores row order and where the manifest lives. Any change to a label,
split, schema or document is a new dataset identity.

### Validation

`validate` checks that rows parse, document ids are unique, files exist and are
supported, no split is empty, the pinned hash still matches, and no document
leaks across splits.

Exact leakage compares document bytes. Near-duplicate detection compares
normalised text and is deliberately conservative; to near-match PDFs or images,
materialise their OCR as `.txt` first.

### Older manifests

A manifest without splits loads as split `unspecified`. Migrate it with one
default split, or a JSON map from document id to split:

```bash
docie-bench dataset migrate data/legacy/manifest.jsonl data/invoices/v1/manifest.jsonl --default-split test
docie-bench dataset migrate data/legacy/manifest.jsonl data/invoices/v1/manifest.jsonl --split-map data/invoices/splits.json
```

## Sample data

`data/sample_dataset/` has seven invoices and two identity cards in French,
English, German and Spanish, with ground truth for each.
