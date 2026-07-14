from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from sipz_agent.core.artifacts import model_dump_jsonable
from sipz_agent.core.final_export_inputs import atomic_write, read_jsonl
from sipz_agent.core.final_export_merge import FINAL_MERGE_RECORD_ADAPTER


FINAL_MAIN_CSV_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "summary_source_type",
    "summary_confidence_status",
    "final_positive_summary",
    "final_negative_summary",
    "strong_evidence_json",
    "medium_evidence_json",
    "low_evidence_json",
    "negative_or_cautionary_effects_json",
    "supplement_level_only_effects_json",
    "caveats_json",
    "model_provider",
    "model_name",
    "created_at",
    "updated_at",
]

FINAL_EFFECTS_CSV_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "effect_slug",
    "effect_label",
    "effect_summary",
    "effect_bucket",
    "evidence_level",
    "score",
    "supporting_nutrients_json",
    "summary_source_type",
    "model_provider",
    "model_name",
    "created_at",
    "updated_at",
]

FINAL_TAGS_CSV_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "tag",
    "tag_type",
    "source_bucket",
    "score",
    "created_at",
    "updated_at",
]

JSON_BUCKETS = [
    "strong_evidence",
    "medium_evidence",
    "low_evidence",
    "negative_or_cautionary_effects",
    "supplement_level_only_effects",
    "caveats",
]

EFFECT_BUCKETS = [
    "strong_evidence",
    "medium_evidence",
    "low_evidence",
    "negative_or_cautionary_effects",
    "supplement_level_only_effects",
]


@dataclass(frozen=True)
class FinalMainCsvResult:
    out_path: Path
    rows_written: int
    direct_literature_rows: int
    composition_based_rows: int
    unknown_rows: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class FinalSearchCompanionCsvResult:
    effects_path: Path
    tags_path: Path
    effect_rows_written: int
    tag_rows_written: int
    ingredient_rows: int
    created_at: str
    updated_at: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def compact_json_cell(value: Any) -> str:
    return orjson.dumps(model_dump_jsonable(value)).decode("utf-8")


def validate_final_main_row(row: dict[str, Any]) -> None:
    canonical_id = str(row.get("canonical_beverage_id") or "").strip()
    canonical_name = str(row.get("canonical_beverage_name") or "").strip()
    positive = str(row.get("final_positive_summary") or "").strip()
    negative = str(row.get("final_negative_summary") or "").strip()
    source_type = str(row.get("summary_source_type") or "").strip()
    confidence = str(row.get("summary_confidence_status") or "").strip()

    if not canonical_id:
        raise ValueError("missing_canonical_beverage_id")
    if not canonical_name:
        raise ValueError(f"missing_canonical_beverage_name:{canonical_id}")
    if source_type not in {"direct_literature", "composition_based", "unknown"}:
        raise ValueError(f"invalid_summary_source_type:{canonical_id}:{source_type}")
    if not confidence:
        raise ValueError(f"missing_summary_confidence_status:{canonical_id}")
    if not positive:
        raise ValueError(f"missing_final_positive_summary:{canonical_id}")
    if not negative:
        raise ValueError(f"missing_final_negative_summary:{canonical_id}")
    for bucket in JSON_BUCKETS:
        if not isinstance(row.get(bucket), list):
            raise ValueError(f"final_main_json_bucket_not_list:{canonical_id}:{bucket}")


def load_validated_merge_rows(merges_path: Path) -> list[dict[str, Any]]:
    rows = [FINAL_MERGE_RECORD_ADAPTER.validate_python(row) for row in read_jsonl(merges_path)]
    jsonable_rows = [row.model_dump(mode="json") for row in rows]
    seen_ids: set[str] = set()
    for row in jsonable_rows:
        validate_final_main_row(row)
        canonical_id = str(row["canonical_beverage_id"])
        if canonical_id in seen_ids:
            raise ValueError(f"duplicate_canonical_beverage_id:{canonical_id}")
        seen_ids.add(canonical_id)
    return sorted(
        jsonable_rows,
        key=lambda row: (
            str(row["canonical_beverage_name"]).casefold(),
            str(row["canonical_beverage_id"]),
        ),
    )


def final_main_csv_row(
    row: dict[str, Any],
    *,
    created_at: str,
    updated_at: str,
) -> dict[str, str]:
    return {
        "canonical_beverage_id": str(row["canonical_beverage_id"]),
        "canonical_beverage_name": str(row["canonical_beverage_name"]),
        "summary_source_type": str(row["summary_source_type"]),
        "summary_confidence_status": str(row["summary_confidence_status"]),
        "final_positive_summary": str(row["final_positive_summary"]),
        "final_negative_summary": str(row["final_negative_summary"]),
        "strong_evidence_json": compact_json_cell(row["strong_evidence"]),
        "medium_evidence_json": compact_json_cell(row["medium_evidence"]),
        "low_evidence_json": compact_json_cell(row["low_evidence"]),
        "negative_or_cautionary_effects_json": compact_json_cell(
            row["negative_or_cautionary_effects"]
        ),
        "supplement_level_only_effects_json": compact_json_cell(
            row["supplement_level_only_effects"]
        ),
        "caveats_json": compact_json_cell(row["caveats"]),
        "model_provider": str(row.get("model_provider") or ""),
        "model_name": str(row.get("model_name") or ""),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def write_final_main_csv(
    *,
    merges_path: Path,
    out_path: Path,
    timestamp: str | None = None,
) -> FinalMainCsvResult:
    rows = load_validated_merge_rows(merges_path)
    run_timestamp = timestamp or utc_now()
    csv_rows = [
        final_main_csv_row(
            row,
            created_at=run_timestamp,
            updated_at=run_timestamp,
        )
        for row in rows
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FINAL_MAIN_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out_path, output.getvalue().encode("utf-8"))

    return FinalMainCsvResult(
        out_path=out_path,
        rows_written=len(csv_rows),
        direct_literature_rows=sum(
            row["summary_source_type"] == "direct_literature" for row in rows
        ),
        composition_based_rows=sum(
            row["summary_source_type"] == "composition_based" for row in rows
        ),
        unknown_rows=sum(row["summary_source_type"] == "unknown" for row in rows),
        created_at=run_timestamp,
        updated_at=run_timestamp,
    )


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, output.getvalue().encode("utf-8"))


def normalized_tag(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def tag_row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["canonical_beverage_id"],
        row["tag"],
        row["tag_type"],
        row["source_bucket"],
    )


def score_text(value: object) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def add_tag(
    tags_by_key: dict[tuple[str, str, str, str], dict[str, str]],
    *,
    canonical_beverage_id: str,
    canonical_beverage_name: str,
    tag: object,
    tag_type: str,
    source_bucket: str,
    score: object,
    created_at: str,
    updated_at: str,
) -> None:
    normalized = normalized_tag(tag)
    if not normalized:
        return
    row = {
        "canonical_beverage_id": canonical_beverage_id,
        "canonical_beverage_name": canonical_beverage_name,
        "tag": normalized,
        "tag_type": tag_type,
        "source_bucket": source_bucket,
        "score": score_text(score),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    key = tag_row_key(row)
    existing = tags_by_key.get(key)
    if existing is None:
        tags_by_key[key] = row
        return
    try:
        existing_score = float(existing["score"]) if existing["score"] else -1.0
        new_score = float(row["score"]) if row["score"] else -1.0
    except ValueError:
        return
    if new_score > existing_score:
        existing["score"] = row["score"]


def effect_csv_row(
    *,
    ingredient: dict[str, Any],
    effect: dict[str, Any],
    bucket: str,
    created_at: str,
    updated_at: str,
) -> dict[str, str]:
    effect_slug = str(effect.get("effect_slug") or "").strip()
    if not effect_slug:
        raise ValueError(
            f"missing_effect_slug:{ingredient['canonical_beverage_id']}:{bucket}"
        )
    return {
        "canonical_beverage_id": str(ingredient["canonical_beverage_id"]),
        "canonical_beverage_name": str(ingredient["canonical_beverage_name"]),
        "effect_slug": effect_slug,
        "effect_label": str(effect.get("effect_label") or ""),
        "effect_summary": str(effect.get("summary") or ""),
        "effect_bucket": bucket,
        "evidence_level": str(effect.get("evidence_level") or ""),
        "score": score_text(effect.get("score")),
        "supporting_nutrients_json": compact_json_cell(effect.get("supporting_nutrients") or []),
        "summary_source_type": str(ingredient["summary_source_type"]),
        "model_provider": str(ingredient.get("model_provider") or ""),
        "model_name": str(ingredient.get("model_name") or ""),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def write_final_search_companion_csvs(
    *,
    merges_path: Path,
    effects_out_path: Path,
    tags_out_path: Path,
    timestamp: str | None = None,
) -> FinalSearchCompanionCsvResult:
    rows = load_validated_merge_rows(merges_path)
    run_timestamp = timestamp or utc_now()
    effect_rows: list[dict[str, str]] = []
    tags_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for ingredient in rows:
        canonical_id = str(ingredient["canonical_beverage_id"])
        canonical_name = str(ingredient["canonical_beverage_name"])
        source_type = str(ingredient["summary_source_type"])
        add_tag(
            tags_by_key,
            canonical_beverage_id=canonical_id,
            canonical_beverage_name=canonical_name,
            tag=source_type,
            tag_type="source_type",
            source_bucket="ingredient",
            score="",
            created_at=run_timestamp,
            updated_at=run_timestamp,
        )
        for bucket in EFFECT_BUCKETS:
            effects = ingredient[bucket]
            if effects:
                add_tag(
                    tags_by_key,
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    tag=bucket,
                    tag_type="effect_bucket",
                    source_bucket=bucket,
                    score="",
                    created_at=run_timestamp,
                    updated_at=run_timestamp,
                )
            if bucket == "negative_or_cautionary_effects" and effects:
                add_tag(
                    tags_by_key,
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    tag="has_cautionary_effects",
                    tag_type="routing_flag",
                    source_bucket=bucket,
                    score="",
                    created_at=run_timestamp,
                    updated_at=run_timestamp,
                )
            if bucket == "supplement_level_only_effects" and effects:
                add_tag(
                    tags_by_key,
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    tag="has_supplement_level_only_effects",
                    tag_type="routing_flag",
                    source_bucket=bucket,
                    score="",
                    created_at=run_timestamp,
                    updated_at=run_timestamp,
                )
            for effect in effects:
                effect_rows.append(
                    effect_csv_row(
                        ingredient=ingredient,
                        effect=effect,
                        bucket=bucket,
                        created_at=run_timestamp,
                        updated_at=run_timestamp,
                    )
                )
                add_tag(
                    tags_by_key,
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    tag=effect.get("effect_slug"),
                    tag_type="effect_slug",
                    source_bucket=bucket,
                    score=effect.get("score"),
                    created_at=run_timestamp,
                    updated_at=run_timestamp,
                )
                add_tag(
                    tags_by_key,
                    canonical_beverage_id=canonical_id,
                    canonical_beverage_name=canonical_name,
                    tag=effect.get("effect_label"),
                    tag_type="effect_label",
                    source_bucket=bucket,
                    score=effect.get("score"),
                    created_at=run_timestamp,
                    updated_at=run_timestamp,
                )
                for nutrient in effect.get("supporting_nutrients") or []:
                    add_tag(
                        tags_by_key,
                        canonical_beverage_id=canonical_id,
                        canonical_beverage_name=canonical_name,
                        tag=nutrient,
                        tag_type="nutrient",
                        source_bucket=bucket,
                        score=effect.get("score"),
                        created_at=run_timestamp,
                        updated_at=run_timestamp,
                    )

    effect_rows.sort(
        key=lambda row: (
            row["canonical_beverage_name"].casefold(),
            row["canonical_beverage_id"],
            row["effect_bucket"],
            row["effect_slug"],
        )
    )
    tag_rows = sorted(
        tags_by_key.values(),
        key=lambda row: (
            row["canonical_beverage_name"].casefold(),
            row["canonical_beverage_id"],
            row["tag_type"],
            row["tag"],
            row["source_bucket"],
        ),
    )

    write_csv_rows(effects_out_path, effect_rows, FINAL_EFFECTS_CSV_COLUMNS)
    write_csv_rows(tags_out_path, tag_rows, FINAL_TAGS_CSV_COLUMNS)

    return FinalSearchCompanionCsvResult(
        effects_path=effects_out_path,
        tags_path=tags_out_path,
        effect_rows_written=len(effect_rows),
        tag_rows_written=len(tag_rows),
        ingredient_rows=len(rows),
        created_at=run_timestamp,
        updated_at=run_timestamp,
    )
