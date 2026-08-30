"""Convert the Voxel51 high-quality-invoice-images dataset into a docie-bench manifest.

The Voxel51 dataset stores per-sample ground truth as a JSON string in the
`json_annotation` field of `samples.json`. This script maps that schema onto the
docie-bench `invoice` schema and emits a `manifest.jsonl` covering whichever
`batch1-*.jpg` images are present locally under `<root>/data/`.

Mapping (Voxel51 -> docie-bench invoice ground truth):
    invoice.invoice_number  -> invoice_number
    invoice.seller_name     -> vendor_name
    invoice.client_name     -> customer_name
    invoice.invoice_date    -> issue_date.value   (MM/DD/YYYY -> ISO)
    invoice.due_date        -> due_date.value     (when present)
    subtotal.total          -> subtotal.amount    (sum of line items, pre-tax)
    subtotal.tax            -> vat_amount.amount
    subtotal.total + tax    -> total_ttc.amount    (computed; Voxel51 has no TTC field)
    items[]                 -> line_items[] (description, quantity, line_total.amount)

Amounts are locale-normalized (handles both `232.95` and `6 204,19` / `3,00`).
Currency is intentionally omitted: the source has no currency symbol and mixes
US/EU formatting, so asserting one would create wrong ground truth.

Usage:
    python scripts/build_voxel51_manifest.py [--root data/voxel51_invoices]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


def normalize_amount(raw: str | None) -> str | None:
    """Return a canonical 'NNNN.NN' string from a US- or EU-formatted amount."""
    if not raw:
        return None
    s = raw.strip().replace(" ", " ").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        # Whichever separator comes last is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # EU: . thousands, , decimal
        else:
            s = s.replace(",", "")  # US: , thousands, . decimal
    elif "," in s:
        frac = s.split(",")[-1]
        s = s.replace(",", ".") if len(frac) in (1, 2) else s.replace(",", "")
    try:
        return f"{Decimal(s):.2f}"
    except InvalidOperation:
        return None


def normalize_quantity(raw: str | None) -> str | None:
    """Normalize a quantity; drop a trailing .00 so '3,00' -> '3'."""
    amount = normalize_amount(raw)
    if amount is None:
        return None
    value = Decimal(amount)
    return str(int(value)) if value == value.to_integral_value() else str(value.normalize())


def to_iso_date(raw: str | None) -> str | None:
    """Convert MM/DD/YYYY to ISO YYYY-MM-DD, or None if empty/unparseable."""
    if not raw or not raw.strip():
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def build_ground_truth(annotation: dict) -> dict[str, object]:
    invoice = annotation.get("invoice", {})
    subtotal = annotation.get("subtotal", {})
    gt: dict[str, object] = {}

    def put(key: str, value: object | None) -> None:
        if value is not None and value != "":
            gt[key] = value

    put("invoice_number", (invoice.get("invoice_number") or "").strip() or None)
    put("vendor_name", (invoice.get("seller_name") or "").strip() or None)
    put("customer_name", (invoice.get("client_name") or "").strip() or None)
    put("issue_date.value", to_iso_date(invoice.get("invoice_date")))
    put("due_date.value", to_iso_date(invoice.get("due_date")))

    sub = normalize_amount(subtotal.get("total"))
    vat = normalize_amount(subtotal.get("tax"))
    put("subtotal.amount", sub)
    put("vat_amount.amount", vat)
    if sub is not None and vat is not None:
        put("total_ttc.amount", f"{Decimal(sub) + Decimal(vat):.2f}")
    elif sub is not None:
        # No tax line: the pre-tax total is the only total we can assert.
        put("total_ttc.amount", sub)

    line_items = []
    for item in annotation.get("items", []):
        row: dict[str, object] = {}
        desc = (item.get("description") or "").replace("\n", " ").strip()
        if desc:
            row["description"] = desc
        qty = normalize_quantity(item.get("quantity"))
        if qty is not None:
            row["quantity"] = qty
        total = normalize_amount(item.get("total_price"))
        if total is not None:
            row["line_total.amount"] = total
        if row:
            line_items.append(row)
    if line_items:
        gt["line_items"] = line_items
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/voxel51_invoices"))
    args = parser.parse_args()

    samples_path = args.root / "samples.json"
    images_dir = args.root / "data"
    samples = json.load(samples_path.open(encoding="utf-8"))["samples"]
    by_name = {
        Path(s.get("filepath", "").replace("\\", "/")).name: s for s in samples
    }

    local_images = sorted(p.name for p in images_dir.glob("*.jpg"))
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for image_name in local_images:
        sample = by_name.get(image_name)
        if sample is None:
            missing.append(image_name)
            continue
        annotation = json.loads(sample["json_annotation"])
        rows.append(
            {
                "doc_id": Path(image_name).stem,
                "file_path": f"data/{image_name}",
                "schema_name": "invoice",
                "language": "en",
                "ground_truth": build_ground_truth(annotation),
            }
        )

    manifest_path = args.root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {manifest_path}")
    if missing:
        print(f"WARNING: {len(missing)} local images had no annotation: {missing}")


if __name__ == "__main__":
    main()
