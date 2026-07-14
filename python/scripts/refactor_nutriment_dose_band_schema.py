from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


BUILDER_PATH = Path(__file__).resolve().parent / "build_nutriment_dose_band_table.py"
SPEC = importlib.util.spec_from_file_location("build_nutriment_dose_band_table", BUILDER_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["build_nutriment_dose_band_table"] = builder
SPEC.loader.exec_module(builder)


OLD_FIELDS = {
    "threshold_basis",
    "source",
    "review_status",
}


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=builder.FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def meaningful_basis(row: dict[str, str], reference: dict[str, str]) -> str:
    reference_type = reference.get("reference_type", "")
    threshold_notes = reference.get("threshold_notes", "")
    previous_basis = row.get("threshold_basis", "")
    if row.get("meaningful_threshold_amount") and row.get("meaningful_threshold_unit"):
        if reference_type or threshold_notes:
            return f"Draft meaningful threshold from existing table; reference context: {reference_type}; {threshold_notes}".strip("; ")
        return f"Draft meaningful threshold from existing table; previous basis: {previous_basis}".strip("; ")
    if reference_type or threshold_notes:
        return f"No meaningful threshold set; reference context: {reference_type}; {threshold_notes}".strip("; ")
    return "No meaningful threshold set; needs reference-backed meaningful-dose review."


def meaningful_source(row: dict[str, str], reference: dict[str, str]) -> str:
    source = reference.get("source", "")
    if source:
        return source
    return row.get("source", "")


def meaningful_status(row: dict[str, str]) -> str:
    name = row["canonical_bioactive_name"]
    meaningful_amount = row.get("meaningful_threshold_amount", "")
    meaningful_unit = row.get("meaningful_threshold_unit", "")
    supplement_amount = row.get("supplement_threshold_amount", "")
    supplement_unit = row.get("supplement_threshold_unit", "")
    if name == "Ph":
        return "not_applicable"
    if not meaningful_amount or not meaningful_unit:
        return "needs_meaningful_review"
    if builder.is_non_monotonic_threshold(meaningful_amount, meaningful_unit, supplement_amount, supplement_unit):
        return "needs_meaningful_review_non_monotonic"
    return "draft_meaningful_baseline"


def supplement_status(row: dict[str, str]) -> str:
    if row.get("supplement_review_status"):
        return row["supplement_review_status"]
    status = row.get("review_status", "")
    if not row.get("supplement_threshold_amount") or not row.get("supplement_threshold_unit"):
        return "needs_supplement_dose_review"
    return status or "needs_supplement_dose_review"


def supplement_basis(row: dict[str, str]) -> str:
    if not row.get("supplement_threshold_amount") or not row.get("supplement_threshold_unit"):
        return "Not yet reviewed; populate after supplement-dose reference review."
    return row.get("supplement_threshold_basis") or row.get("threshold_basis", "")


def supplement_source(row: dict[str, str]) -> str:
    if not row.get("supplement_threshold_amount") or not row.get("supplement_threshold_unit"):
        return ""
    return row.get("supplement_threshold_source") or row.get("source", "")


def migrate_row(row: dict[str, str], reference_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    reference = reference_rows.get(row["canonical_bioactive_name"], {})
    meaningful_amount = row.get("meaningful_threshold_amount", "")
    meaningful_unit = row.get("meaningful_threshold_unit", "")
    supplement_amount = row.get("supplement_threshold_amount", "")
    supplement_unit = row.get("supplement_threshold_unit", "")

    migrated = {
        "canonical_bioactive_name": row["canonical_bioactive_name"],
        "meaningful_threshold_amount": meaningful_amount,
        "meaningful_threshold_unit": meaningful_unit,
        "meaningful_threshold_basis": row.get("meaningful_threshold_basis") or meaningful_basis(row, reference),
        "meaningful_threshold_source": row.get("meaningful_threshold_source") or meaningful_source(row, reference),
        "meaningful_review_status": row.get("meaningful_review_status") or meaningful_status(row),
        "supplement_threshold_amount": supplement_amount,
        "supplement_threshold_unit": supplement_unit,
        "supplement_threshold_basis": supplement_basis(row),
        "supplement_threshold_source": supplement_source(row),
        "supplement_review_status": supplement_status(row),
        "trace_definition": builder.build_trace_definition(meaningful_amount, meaningful_unit),
        "meaningful_definition": builder.build_meaningful_definition(
            meaningful_amount,
            meaningful_unit,
            supplement_amount,
            supplement_unit,
        ),
        "supplement_definition": builder.build_supplement_definition(supplement_amount, supplement_unit),
        "notes": row.get("notes", ""),
    }
    return migrated


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    meaningful_counts: dict[str, int] = {}
    supplement_counts: dict[str, int] = {}
    for row in rows:
        meaningful_counts[row["meaningful_review_status"]] = meaningful_counts.get(row["meaningful_review_status"], 0) + 1
        supplement_counts[row["supplement_review_status"]] = supplement_counts.get(row["supplement_review_status"], 0) + 1
    summary = {
        "row_count": len(rows),
        "meaningful_review_status_counts": dict(sorted(meaningful_counts.items())),
        "supplement_review_status_counts": dict(sorted(supplement_counts.items())),
        "output_csv": str(path.with_name("nutriment_dose_band_table.csv")),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactor nutriment dose-band CSV to split meaningful/supplement metadata.")
    parser.add_argument("--table", type=Path, required=True, help="Existing nutriment_dose_band_table.csv path.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference-intake table CSV.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    args = parser.parse_args()

    fieldnames, rows = read_csv_rows(args.table)
    if not rows:
        raise ValueError(f"no_rows:{args.table}")
    if not OLD_FIELDS.intersection(fieldnames) and set(fieldnames) == set(builder.FIELDNAMES):
        print("schema_already_refactored=true")

    reference_rows = builder.read_reference_rows(args.reference)
    migrated_rows = [migrate_row(row, reference_rows) for row in rows]
    write_csv(args.table, migrated_rows)
    summary_path = args.summary or args.table.with_name("nutriment_dose_band_table_summary.json")
    write_summary(summary_path, migrated_rows)

    print(f"rows={len(migrated_rows)}")
    print(f"wrote={args.table}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
