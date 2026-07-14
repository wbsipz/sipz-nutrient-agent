from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from sipz_agent.core.artifacts import model_dump_jsonable, write_json
from sipz_agent.core.final_export_inputs import atomic_write, read_jsonl, write_jsonl
from sipz_agent.core.final_export_merge import FinalHealthSummaryMergeRecord


FALLBACK_INGREDIENT_PROFILES_JSONL = "fallback_ingredient_profiles.jsonl"
FALLBACK_UNKNOWN_PLACEHOLDERS_JSONL = "fallback_unknown_placeholders.jsonl"
FALLBACK_INPUT_SUMMARY_JSON = "fallback_composition_input_summary.json"
COMBINED_COMPOSITION_SUMMARIES_JSONL = "combined_composition_summaries.jsonl"
COMBINED_COMPOSITION_SUMMARY_JSON = "combined_composition_summaries_summary.json"
COMPLETE_FINAL_MERGES_JSONL = "final_health_summary_merges.complete.jsonl"
COMPLETE_FINAL_MERGES_SUMMARY_JSON = "final_health_summary_merges.complete_summary.json"

PLACEHOLDER_SUMMARY = "Current health information is unknown."


@dataclass(frozen=True)
class FallbackCompositionInputResult:
    out_dir: Path
    profile_rows: list[dict[str, Any]]
    placeholder_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class CombinedCompositionSummaryResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class CompleteFinalMergeResult:
    out_path: Path
    summary: dict[str, Any]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def parse_json_object(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    parsed = orjson.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def is_true(value: str) -> bool:
    return value.strip().casefold() == "true"


def parse_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_nonzero_numeric_nutriments(avg_nutriments: dict[str, Any]) -> bool:
    return any((amount := parse_float(value)) is not None and amount != 0 for value in avg_nutriments.values())


def read_canonical_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    required = {"id", "avg_nutriments"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("canonical_columns_missing:" + ",".join(missing))
    return {row["id"]: row for row in rows if row.get("id")}


def read_alias_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    required = {
        "source_key",
        "include",
        "canonical_bioactive_name",
        "display_name",
        "source_unit",
        "display_unit",
        "conversion_notes",
        "review_status",
        "match_method",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("alias_columns_missing:" + ",".join(missing))
    return {
        row["source_key"]: row
        for row in rows
        if row.get("source_key")
        and is_true(row.get("include", ""))
        and row.get("canonical_bioactive_name", "").strip()
    }


def read_reference_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    required = {
        "canonical_bioactive_name",
        "reference_amount",
        "reference_unit",
        "reference_type",
        "caution_limit",
        "caution_limit_unit",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("reference_columns_missing:" + ",".join(missing))
    return {row["canonical_bioactive_name"].casefold(): row for row in rows}


def convert_amount(raw_amount: float, source_unit: str, display_unit: str) -> float:
    source = source_unit.strip().casefold()
    display = display_unit.strip().casefold()
    if source == display:
        return raw_amount
    conversions = {
        ("g/100g", "mg/100g"): 1000.0,
        ("g/100g", "ug/100g"): 1_000_000.0,
        ("g/100g", "µg/100g"): 1_000_000.0,
        ("mg/100g", "g/100g"): 0.001,
        ("mg/100g", "ug/100g"): 1000.0,
        ("ug/100g", "mg/100g"): 0.001,
        ("µg/100g", "mg/100g"): 0.001,
    }
    factor = conversions.get((source, display))
    if factor is None:
        return raw_amount
    return raw_amount * factor


def amount_quality_flags(raw_amount: float, source_unit: str) -> list[str]:
    flags: list[str] = []
    source = source_unit.strip().casefold()
    if raw_amount < 0:
        flags.append("raw_amount_negative")
    if source == "g/100g" and raw_amount > 100:
        flags.append("raw_g_per_100g_gt_100_impossible")
    return flags


def nutriment_profile(
    *,
    source_key: str,
    raw_amount: float,
    alias: dict[str, str],
    reference: dict[str, str] | None,
) -> dict[str, Any]:
    display_amount = convert_amount(raw_amount, alias["source_unit"], alias["display_unit"])
    flags = amount_quality_flags(raw_amount, alias["source_unit"])
    reference = reference or {}
    return {
        "source_key": source_key,
        "canonical_bioactive_name": alias["canonical_bioactive_name"],
        "raw_amount": raw_amount,
        "raw_unit": alias["source_unit"],
        "display_amount": display_amount,
        "display_unit": alias["display_unit"],
        "display_name": alias["display_name"],
        "reference_amount": reference.get("reference_amount", ""),
        "reference_unit": reference.get("reference_unit", ""),
        "reference_type": reference.get("reference_type", ""),
        "caution_limit": reference.get("caution_limit", ""),
        "caution_limit_unit": reference.get("caution_limit_unit", ""),
        "amount_quality_status": "review_source_data" if flags else "ok",
        "amount_quality_flags": flags,
        "alias_review_status": alias.get("review_status", ""),
        "alias_match_method": alias.get("match_method", ""),
        "conversion_notes": alias.get("conversion_notes", ""),
    }


def placeholder_row(
    *,
    canonical_beverage_id: str,
    canonical_beverage_name: str,
    reason: str,
    source_skip_reason: str = "",
) -> dict[str, Any]:
    return {
        "canonical_beverage_id": canonical_beverage_id,
        "canonical_beverage_name": canonical_beverage_name,
        "reason": reason,
        "source_skip_reason": source_skip_reason,
    }


def build_fallback_profile(
    *,
    skipped_row: dict[str, str],
    canonical_row: dict[str, Any],
    aliases_by_source_key: dict[str, dict[str, str]],
    references_by_name: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    avg_nutriments = parse_json_object(str(canonical_row.get("avg_nutriments", "")))
    nutriments: list[dict[str, Any]] = []
    for source_key, value in sorted(avg_nutriments.items()):
        raw_amount = parse_float(value)
        if raw_amount is None or raw_amount == 0:
            continue
        alias = aliases_by_source_key.get(source_key)
        if alias is None:
            continue
        canonical_name = alias["canonical_bioactive_name"]
        nutriments.append(
            nutriment_profile(
                source_key=source_key,
                raw_amount=raw_amount,
                alias=alias,
                reference=references_by_name.get(canonical_name.casefold()),
            )
        )
    if not nutriments:
        return None
    return {
        "canonical_beverage_id": skipped_row["canonical_beverage_id"],
        "canonical_beverage_name": skipped_row["canonical_beverage_name"],
        "canonical_category": canonical_row.get("category", ""),
        "canonical_slug": canonical_row.get("slug", ""),
        "beverage_family": canonical_row.get("beverage_family", ""),
        "serving_basis": "100g",
        "nutriments": nutriments,
        "dataset_version": canonical_row.get("dataset_version", ""),
        "fallback_source_skip_reason": skipped_row.get("reason", ""),
        "normalization_counts": {
            "raw_nutriment_keys": len(avg_nutriments),
            "included_nonzero_nutriments": len(nutriments),
            "nutriments_with_quality_flags": sum(
                bool(nutriment["amount_quality_flags"]) for nutriment in nutriments
            ),
        },
    }


def build_fallback_composition_inputs(
    *,
    skipped_path: Path,
    canonical_path: Path,
    alias_path: Path,
    reference_path: Path,
    out_dir: Path,
) -> FallbackCompositionInputResult:
    skipped_rows = read_csv_rows(skipped_path)
    canonical_by_id = read_canonical_rows(canonical_path)
    aliases_by_source_key = read_alias_rows(alias_path)
    references_by_name = read_reference_rows(reference_path)

    profile_rows: list[dict[str, Any]] = []
    placeholder_rows: list[dict[str, Any]] = []
    placeholder_reason_counts: dict[str, int] = {}

    for skipped in skipped_rows:
        canonical_id = skipped.get("canonical_beverage_id", "").strip()
        canonical_name = skipped.get("canonical_beverage_name", "").strip()
        if not canonical_id:
            continue
        canonical = canonical_by_id.get(canonical_id)
        if canonical is None:
            reason = "missing_canonical_row"
            placeholder_rows.append(
                placeholder_row(
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    reason=reason,
                    source_skip_reason=skipped.get("reason", ""),
                )
            )
            placeholder_reason_counts[reason] = placeholder_reason_counts.get(reason, 0) + 1
            continue
        avg_nutriments = parse_json_object(str(canonical.get("avg_nutriments", "")))
        if not avg_nutriments:
            reason = "empty_avg_nutriments"
            placeholder_rows.append(
                placeholder_row(
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    reason=reason,
                    source_skip_reason=skipped.get("reason", ""),
                )
            )
            placeholder_reason_counts[reason] = placeholder_reason_counts.get(reason, 0) + 1
            continue
        if not has_nonzero_numeric_nutriments(avg_nutriments):
            reason = "no_nonzero_avg_nutriments"
            placeholder_rows.append(
                placeholder_row(
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    reason=reason,
                    source_skip_reason=skipped.get("reason", ""),
                )
            )
            placeholder_reason_counts[reason] = placeholder_reason_counts.get(reason, 0) + 1
            continue
        profile = build_fallback_profile(
            skipped_row=skipped,
            canonical_row=canonical,
            aliases_by_source_key=aliases_by_source_key,
            references_by_name=references_by_name,
        )
        if profile is None:
            reason = "no_mapped_nonzero_nutriments"
            placeholder_rows.append(
                placeholder_row(
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    reason=reason,
                    source_skip_reason=skipped.get("reason", ""),
                )
            )
            placeholder_reason_counts[reason] = placeholder_reason_counts.get(reason, 0) + 1
            continue
        profile_rows.append(profile)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / FALLBACK_INGREDIENT_PROFILES_JSONL, profile_rows)
    write_jsonl(out_dir / FALLBACK_UNKNOWN_PLACEHOLDERS_JSONL, placeholder_rows)
    summary = {
        "skipped_input_rows": len(skipped_rows),
        "fallback_profile_rows": len(profile_rows),
        "unknown_placeholder_rows": len(placeholder_rows),
        "placeholder_reason_counts": placeholder_reason_counts,
        "nutriments_in_profiles": sum(len(row["nutriments"]) for row in profile_rows),
        "profiles_with_quality_flags": sum(
            any(nutriment["amount_quality_flags"] for nutriment in row["nutriments"])
            for row in profile_rows
        ),
    }
    write_json(out_dir / FALLBACK_INPUT_SUMMARY_JSON, summary)
    return FallbackCompositionInputResult(
        out_dir=out_dir,
        profile_rows=profile_rows,
        placeholder_rows=placeholder_rows,
        summary=summary,
    )


def combine_composition_summaries(
    *,
    existing_path: Path,
    fallback_path: Path,
    out_path: Path,
) -> CombinedCompositionSummaryResult:
    combined_by_id: dict[str, dict[str, Any]] = {}
    source_counts = {"existing_rows": 0, "fallback_rows": 0}
    for source, path in [("existing_rows", existing_path), ("fallback_rows", fallback_path)]:
        for row in read_jsonl_rows(path):
            canonical_id = str(row.get("canonical_beverage_id", "")).strip()
            if not canonical_id:
                raise ValueError(f"composition_summary_missing_canonical_id:{path}")
            if canonical_id in combined_by_id:
                raise ValueError(f"duplicate_composition_summary:{canonical_id}")
            combined_by_id[canonical_id] = row
            source_counts[source] += 1
    rows = sorted(
        combined_by_id.values(),
        key=lambda row: (
            str(row.get("canonical_beverage_name", "")).casefold(),
            str(row.get("canonical_beverage_id", "")),
        ),
    )
    write_jsonl(out_path, rows)
    summary = {
        **source_counts,
        "combined_rows": len(rows),
        "out_path": str(out_path),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    return CombinedCompositionSummaryResult(out_path=out_path, summary=summary)


def placeholder_final_merge_record(row: dict[str, Any]) -> dict[str, Any]:
    return FinalHealthSummaryMergeRecord(
        canonical_beverage_id=row["canonical_beverage_id"],
        canonical_beverage_name=row["canonical_beverage_name"],
        summary_source_type="unknown",
        summary_confidence_status="unknown",
        used_llm=False,
        model_provider="",
        model_name="",
        final_positive_summary=PLACEHOLDER_SUMMARY,
        final_negative_summary=PLACEHOLDER_SUMMARY,
        strong_evidence=[],
        medium_evidence=[],
        low_evidence=[],
        negative_or_cautionary_effects=[],
        supplement_level_only_effects=[],
        caveats=[],
        legacy_claims_used=[],
        legacy_claims_excluded=[],
        merge_warnings=[f"unknown_placeholder:{row.get('reason', '')}"],
    ).model_dump(mode="json")


def complete_final_health_summary_merges(
    *,
    merges_path: Path,
    placeholders_path: Path,
    out_path: Path,
) -> CompleteFinalMergeResult:
    merge_rows = [
        FinalHealthSummaryMergeRecord.model_validate(row).model_dump(mode="json")
        for row in read_jsonl_rows(merges_path)
    ]
    placeholder_rows = [placeholder_final_merge_record(row) for row in read_jsonl_rows(placeholders_path)]
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in [*merge_rows, *placeholder_rows]:
        canonical_id = str(row["canonical_beverage_id"])
        if canonical_id in rows_by_id:
            duplicate_ids.append(canonical_id)
            continue
        rows_by_id[canonical_id] = row
    if duplicate_ids:
        raise ValueError("duplicate_final_merge_ids:" + ",".join(sorted(set(duplicate_ids))))
    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            str(row.get("canonical_beverage_name", "")).casefold(),
            str(row.get("canonical_beverage_id", "")),
        ),
    )
    write_jsonl(out_path, rows)
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("summary_source_type", ""))
        source_counts[source] = source_counts.get(source, 0) + 1
    summary = {
        "input_merge_rows": len(merge_rows),
        "placeholder_rows": len(placeholder_rows),
        "complete_rows": len(rows),
        "source_type_counts": source_counts,
        "out_path": str(out_path),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    return CompleteFinalMergeResult(out_path=out_path, summary=summary)
