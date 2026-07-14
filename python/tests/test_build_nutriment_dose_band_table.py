from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_nutriment_dose_band_table.py"
SPEC = importlib.util.spec_from_file_location("build_nutriment_dose_band_table", SCRIPT_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["build_nutriment_dose_band_table"] = builder
SPEC.loader.exec_module(builder)

MIGRATION_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refactor_nutriment_dose_band_schema.py"
MIGRATION_SPEC = importlib.util.spec_from_file_location("refactor_nutriment_dose_band_schema", MIGRATION_PATH)
assert MIGRATION_SPEC is not None
migration = importlib.util.module_from_spec(MIGRATION_SPEC)
assert MIGRATION_SPEC.loader is not None
sys.modules["refactor_nutriment_dose_band_schema"] = migration
MIGRATION_SPEC.loader.exec_module(migration)


FULL_SUMMARIES_PATH = (
    Path(__file__).resolve().parents[1]
    / "composition_health_runs"
    / "2026-07-01T01-18-13-386907+00-00_step1_join"
    / "significance_full"
    / "nutriment_summaries_full_flash"
    / "nutriment_summaries.cleaned.jsonl"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_reference_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_bioactive_name",
                "reference_amount",
                "reference_unit",
                "reference_type",
                "caution_limit",
                "caution_limit_unit",
                "threshold_notes",
                "source",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_bioactive_name": "Vitamin C",
                "reference_amount": "90",
                "reference_unit": "mg/day",
                "reference_type": "beneficial_daily_value",
                "caution_limit": "2000",
                "caution_limit_unit": "mg/day",
                "threshold_notes": "Adult daily value context.",
                "source": "FDA Daily Value",
                "review_status": "reviewed",
            }
        )
        writer.writerow(
            {
                "canonical_bioactive_name": "Sodium",
                "reference_amount": "",
                "reference_unit": "",
                "reference_type": "limit_or_caution",
                "caution_limit": "2300",
                "caution_limit_unit": "mg/day",
                "threshold_notes": "Adult sodium limit context.",
                "source": "FDA Daily Value",
                "review_status": "reviewed",
            }
        )


def test_build_rows_has_expected_schema_and_spot_thresholds(tmp_path: Path) -> None:
    summaries_path = tmp_path / "summaries.jsonl"
    reference_path = tmp_path / "reference.csv"
    write_jsonl(
        summaries_path,
        [
            {"canonical_bioactive_name": "Vitamin C"},
            {"canonical_bioactive_name": "Sodium"},
            {"canonical_bioactive_name": "Cocoa flavanols"},
        ],
    )
    write_reference_csv(reference_path)

    nutriments = builder.read_unique_nutriments(summaries_path)
    rows = builder.build_dose_band_rows(nutriments, builder.read_reference_rows(reference_path))
    rows_by_name = {row["canonical_bioactive_name"]: row for row in rows}

    assert len(rows) == 3
    assert set(rows[0]) == set(builder.FIELDNAMES)
    assert rows_by_name["Vitamin C"]["meaningful_threshold_amount"] == "10"
    assert rows_by_name["Vitamin C"]["meaningful_threshold_unit"] == "mg/serving"
    assert rows_by_name["Vitamin C"]["supplement_threshold_amount"] == "200"
    assert "do not surface direct health-effect claims" in rows_by_name["Vitamin C"]["trace_definition"]
    assert rows_by_name["Sodium"]["supplement_review_status"] == "draft_caution_context"
    assert rows_by_name["Sodium"]["meaningful_threshold_amount"] == "230"
    assert rows_by_name["Cocoa flavanols"]["supplement_review_status"] == "needs_review"


def test_reference_only_rows_are_added_as_supplement_review_placeholders(tmp_path: Path) -> None:
    summaries_path = tmp_path / "summaries.jsonl"
    reference_path = tmp_path / "reference.csv"
    write_jsonl(summaries_path, [{"canonical_bioactive_name": "Vitamin C"}])
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_bioactive_name",
                "reference_amount",
                "reference_unit",
                "reference_type",
                "caution_limit",
                "caution_limit_unit",
                "threshold_notes",
                "source",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_bioactive_name": "Vitamin C",
                "reference_amount": "90",
                "reference_unit": "mg/day",
                "reference_type": "beneficial_daily_value",
                "caution_limit": "",
                "caution_limit_unit": "",
                "threshold_notes": "Adult daily value context.",
                "source": "FDA Daily Value",
                "review_status": "reviewed",
            }
        )
        writer.writerow(
            {
                "canonical_bioactive_name": "Choline",
                "reference_amount": "550",
                "reference_unit": "mg/day",
                "reference_type": "beneficial_daily_value",
                "caution_limit": "",
                "caution_limit_unit": "",
                "threshold_notes": "Adult daily value context.",
                "source": "FDA Daily Value",
                "review_status": "reviewed",
            }
        )
        writer.writerow(
            {
                "canonical_bioactive_name": "Ph",
                "reference_amount": "",
                "reference_unit": "",
                "reference_type": "not_intake_reference",
                "caution_limit": "",
                "caution_limit_unit": "",
                "threshold_notes": "pH is not an intake amount.",
                "source": "No daily value",
                "review_status": "reviewed",
            }
        )

    reference_rows = builder.read_reference_rows(reference_path)
    nutriments = set(builder.read_unique_nutriments(summaries_path))
    nutriments.update(reference_rows)
    rows = builder.build_dose_band_rows(sorted(nutriments), reference_rows)
    rows_by_name = {row["canonical_bioactive_name"]: row for row in rows}

    assert rows_by_name["Vitamin C"]["supplement_threshold_amount"] == "200"
    assert rows_by_name["Choline"]["meaningful_threshold_amount"] == "55"
    assert rows_by_name["Choline"]["meaningful_threshold_unit"] == "mg/serving"
    assert rows_by_name["Choline"]["meaningful_review_status"] == "draft_reference_baseline"
    assert rows_by_name["Choline"]["supplement_threshold_amount"] == ""
    assert rows_by_name["Choline"]["supplement_review_status"] == "needs_supplement_dose_review"
    assert rows_by_name["Ph"]["meaningful_threshold_amount"] == ""
    assert rows_by_name["Ph"]["meaningful_review_status"] == "needs_meaningful_review"
    assert rows_by_name["Ph"]["supplement_definition"].startswith("Not yet reviewed")


def test_full_cleaned_nutriment_summaries_have_dose_band_overrides() -> None:
    if not FULL_SUMMARIES_PATH.exists():
        pytest.skip(f"full cleaned summary artifact missing: {FULL_SUMMARIES_PATH}")

    nutriments = set(builder.read_unique_nutriments(FULL_SUMMARIES_PATH))
    missing = nutriments - set(builder.DOSE_BAND_OVERRIDES)
    extra = set(builder.DOSE_BAND_OVERRIDES) - nutriments

    assert missing == set()
    assert extra == set()


def test_numeric_same_unit_supplement_thresholds_exceed_meaningful_thresholds() -> None:
    for name, override in builder.DOSE_BAND_OVERRIDES.items():
        if override.meaningful_unit != override.supplement_unit:
            continue
        assert float(override.supplement_amount) > float(override.meaningful_amount), name


def test_generated_definitions_match_structured_thresholds() -> None:
    row = builder.build_row("Vitamin C", {})

    assert "10 mg/serving" in row["trace_definition"]
    assert "10 mg/serving" in row["meaningful_definition"]
    assert "200 mg/serving" in row["meaningful_definition"]
    assert "200 mg/serving" in row["supplement_definition"]


def test_schema_migration_splits_meaningful_and_supplement_metadata() -> None:
    row = {
        "canonical_bioactive_name": "Vitamin C",
        "meaningful_threshold_amount": "10",
        "meaningful_threshold_unit": "mg/serving",
        "supplement_threshold_amount": "500",
        "supplement_threshold_unit": "mg/serving",
        "trace_definition": "< 10 mg/serving",
        "meaningful_definition": "stale below 200 mg/serving",
        "supplement_definition": ">= 500 mg/serving",
        "threshold_basis": "common standalone vitamin C tablet/caplet dose",
        "source": "supplement label source",
        "review_status": "supplement_dose_reviewed",
        "notes": "Raised from food-level 200 mg.",
    }
    reference_rows = {
        "Vitamin C": {
            "reference_type": "beneficial_daily_value",
            "threshold_notes": "FDA Daily Value.",
            "source": "FDA source",
        }
    }

    migrated = migration.migrate_row(row, reference_rows)

    assert set(migrated) == set(builder.FIELDNAMES)
    assert migrated["meaningful_threshold_source"] == "FDA source"
    assert "FDA Daily Value" in migrated["meaningful_threshold_basis"]
    assert migrated["meaningful_review_status"] == "draft_meaningful_baseline"
    assert migrated["supplement_threshold_basis"] == "common standalone vitamin C tablet/caplet dose"
    assert migrated["supplement_threshold_source"] == "supplement label source"
    assert migrated["supplement_review_status"] == "supplement_dose_reviewed"
    assert "500 mg/serving" in migrated["meaningful_definition"]

    row_without_supplement = {
        **row,
        "supplement_threshold_amount": "",
        "supplement_threshold_unit": "",
        "review_status": "needs_supplement_dose_review",
    }
    migrated_without_supplement = migration.migrate_row(row_without_supplement, reference_rows)

    assert migrated_without_supplement["supplement_threshold_basis"].startswith("Not yet reviewed")
    assert migrated_without_supplement["supplement_threshold_source"] == ""
