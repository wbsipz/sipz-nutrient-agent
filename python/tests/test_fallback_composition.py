from __future__ import annotations

import csv
from pathlib import Path

import orjson

from sipz_agent.core.fallback_composition import (
    FALLBACK_INGREDIENT_PROFILES_JSONL,
    FALLBACK_UNKNOWN_PLACEHOLDERS_JSONL,
    build_fallback_composition_inputs,
    combine_composition_summaries,
    complete_final_health_summary_merges,
)
from sipz_agent.core.final_main_csv import write_final_main_csv


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows)
    )


def read_jsonl(path: Path) -> list[dict]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def test_build_fallback_composition_inputs_splits_profiles_and_unknowns(
    tmp_path: Path,
) -> None:
    skipped = tmp_path / "skipped.csv"
    canonical = tmp_path / "canonical.csv"
    alias = tmp_path / "alias.csv"
    reference = tmp_path / "reference.csv"
    out_dir = tmp_path / "out"
    write_csv(
        skipped,
        [
            {
                "canonical_beverage_id": "usable",
                "canonical_beverage_name": "Usable",
                "reason": "missing_direct_literature_claims",
                "details": "",
            },
            {
                "canonical_beverage_id": "empty",
                "canonical_beverage_name": "Empty",
                "reason": "no_new_summary_source",
                "details": "",
            },
        ],
    )
    write_csv(
        canonical,
        [
            {
                "id": "usable",
                "slug": "usable",
                "category": "usable",
                "beverage_family": "test",
                "dataset_version": "test-version",
                "avg_nutriments": '{"sodium_100g":0.04,"sugars_100g":12,"nova-group_100g":3}',
            },
            {
                "id": "empty",
                "slug": "empty",
                "category": "empty",
                "beverage_family": "test",
                "dataset_version": "test-version",
                "avg_nutriments": "{}",
            },
        ],
    )
    write_csv(
        alias,
        [
            {
                "source_key": "sodium_100g",
                "include": "true",
                "canonical_bioactive_name": "Sodium",
                "display_name": "Sodium",
                "source_unit": "g/100g",
                "display_unit": "mg/100g",
                "conversion_notes": "convert g to mg",
                "exclude_reason": "",
                "review_status": "reviewed",
                "match_method": "manual",
            },
            {
                "source_key": "sugars_100g",
                "include": "true",
                "canonical_bioactive_name": "Sugars",
                "display_name": "Sugars",
                "source_unit": "g/100g",
                "display_unit": "g/100g",
                "conversion_notes": "already grams",
                "exclude_reason": "",
                "review_status": "reviewed",
                "match_method": "manual",
            },
            {
                "source_key": "nova-group_100g",
                "include": "false",
                "canonical_bioactive_name": "",
                "display_name": "",
                "source_unit": "g/100g",
                "display_unit": "g/100g",
                "conversion_notes": "",
                "exclude_reason": "not nutriment",
                "review_status": "excluded",
                "match_method": "",
            },
            {
                "source_key": "alanine_100g",
                "include": "true",
                "canonical_bioactive_name": "",
                "display_name": "Alanine",
                "source_unit": "g/100g",
                "display_unit": "g/100g",
                "conversion_notes": "",
                "exclude_reason": "",
                "review_status": "unmatched_review",
                "match_method": "unmatched",
            },
        ],
    )
    write_csv(
        reference,
        [
            {
                "canonical_bioactive_name": "Sodium",
                "reference_amount": "",
                "reference_unit": "",
                "reference_type": "limit_or_caution",
                "caution_limit": "2300",
                "caution_limit_unit": "mg/day",
                "threshold_notes": "",
                "source": "test",
                "review_status": "reviewed",
            },
            {
                "canonical_bioactive_name": "Sugars",
                "reference_amount": "",
                "reference_unit": "",
                "reference_type": "limit_or_caution",
                "caution_limit": "50",
                "caution_limit_unit": "g/day",
                "threshold_notes": "",
                "source": "test",
                "review_status": "reviewed",
            },
        ],
    )

    result = build_fallback_composition_inputs(
        skipped_path=skipped,
        canonical_path=canonical,
        alias_path=alias,
        reference_path=reference,
        out_dir=out_dir,
    )

    assert result.summary["fallback_profile_rows"] == 1
    assert result.summary["unknown_placeholder_rows"] == 1
    profile = read_jsonl(out_dir / FALLBACK_INGREDIENT_PROFILES_JSONL)[0]
    by_key = {nutriment["source_key"]: nutriment for nutriment in profile["nutriments"]}
    assert by_key["sodium_100g"]["display_amount"] == 40
    assert by_key["sodium_100g"]["display_unit"] == "mg/100g"
    assert by_key["sugars_100g"]["display_amount"] == 12
    placeholders = read_jsonl(out_dir / FALLBACK_UNKNOWN_PLACEHOLDERS_JSONL)
    assert placeholders[0]["reason"] == "empty_avg_nutriments"


def test_combine_composition_summaries_rejects_duplicates(tmp_path: Path) -> None:
    existing = tmp_path / "existing.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    out = tmp_path / "combined.jsonl"
    write_jsonl(existing, [{"canonical_beverage_id": "a", "canonical_beverage_name": "A"}])
    write_jsonl(fallback, [{"canonical_beverage_id": "b", "canonical_beverage_name": "B"}])

    result = combine_composition_summaries(
        existing_path=existing,
        fallback_path=fallback,
        out_path=out,
    )

    assert result.summary["combined_rows"] == 2
    assert [row["canonical_beverage_id"] for row in read_jsonl(out)] == ["a", "b"]


def test_complete_final_merges_appends_unknown_placeholders_and_final_csv_accepts_them(
    tmp_path: Path,
) -> None:
    merges = tmp_path / "merges.jsonl"
    placeholders = tmp_path / "placeholders.jsonl"
    complete = tmp_path / "complete.jsonl"
    final_csv = tmp_path / "final.csv"
    write_jsonl(
        merges,
        [
            {
                "canonical_beverage_id": "known",
                "canonical_beverage_name": "Known",
                "summary_source_type": "composition_based",
                "summary_confidence_status": "ready",
                "used_llm": True,
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "final_positive_summary": "Known positive.",
                "final_negative_summary": "Known negative.",
                "strong_evidence": [],
                "medium_evidence": [],
                "low_evidence": [],
                "negative_or_cautionary_effects": [],
                "supplement_level_only_effects": [],
                "caveats": [],
                "legacy_claims_used": [],
                "legacy_claims_excluded": [],
                "merge_warnings": [],
            }
        ],
    )
    write_jsonl(
        placeholders,
        [
            {
                "canonical_beverage_id": "unknown",
                "canonical_beverage_name": "Unknown",
                "reason": "empty_avg_nutriments",
            }
        ],
    )

    result = complete_final_health_summary_merges(
        merges_path=merges,
        placeholders_path=placeholders,
        out_path=complete,
    )

    assert result.summary["complete_rows"] == 2
    unknown = {row["canonical_beverage_id"]: row for row in read_jsonl(complete)}["unknown"]
    assert unknown["summary_source_type"] == "unknown"
    assert unknown["summary_confidence_status"] == "unknown"
    assert unknown["final_positive_summary"] == "Current health information is unknown."
    csv_result = write_final_main_csv(
        merges_path=complete,
        out_path=final_csv,
        timestamp="2026-07-05T00:00:00+00:00",
    )
    assert csv_result.unknown_rows == 1
