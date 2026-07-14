from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pytest

from sipz_agent.core.final_main_csv import (
    FINAL_EFFECTS_CSV_COLUMNS,
    FINAL_MAIN_CSV_COLUMNS,
    FINAL_TAGS_CSV_COLUMNS,
    write_final_main_csv,
    write_final_search_companion_csvs,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows)
    )


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def merge_row(
    *,
    canonical_id: str = "ingredient-1",
    name: str = "Test Ingredient",
    source_type: str = "composition_based",
    positive: str = "Positive summary.",
    negative: str = "Negative summary.",
) -> dict:
    return {
        "canonical_beverage_id": canonical_id,
        "canonical_beverage_name": name,
        "summary_source_type": source_type,
        "summary_confidence_status": "ready",
        "final_positive_summary": positive,
        "final_negative_summary": negative,
        "strong_evidence": [
            {
                "effect_slug": "hydration",
                "effect_label": "Hydration",
                "summary": "Supports hydration.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": ["Water"],
            }
        ],
        "medium_evidence": [],
        "low_evidence": [],
        "negative_or_cautionary_effects": [],
        "supplement_level_only_effects": [
            {
                "effect_slug": "supplement_only",
                "effect_label": "Supplement only",
                "summary": "Requires supplemental dose.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": ["Vitamin C"],
            }
        ],
        "caveats": ["Test caveat."],
        "legacy_claims_used": ["Legacy claim."],
        "legacy_claims_excluded": ["Rejected legacy claim."],
        "merge_warnings": ["Test warning."],
        "used_llm": True,
        "model_provider": "deepseek",
        "model_name": "deepseek-v4-flash",
    }


def test_write_final_main_csv_writes_stable_schema_and_json_cells(tmp_path: Path) -> None:
    merges = tmp_path / "merges.jsonl"
    out = tmp_path / "health_reports_ingredients_final_rows.csv"
    write_jsonl(
        merges,
        [
            merge_row(canonical_id="b", name="Zeta", source_type="direct_literature"),
            merge_row(canonical_id="a", name="Alpha", source_type="composition_based"),
        ],
    )

    result = write_final_main_csv(
        merges_path=merges,
        out_path=out,
        timestamp="2026-07-05T00:00:00+00:00",
    )

    assert result.rows_written == 2
    assert result.direct_literature_rows == 1
    assert result.composition_based_rows == 1
    headers, rows = read_csv_rows(out)
    assert headers == FINAL_MAIN_CSV_COLUMNS
    assert [row["canonical_beverage_name"] for row in rows] == ["Alpha", "Zeta"]
    assert rows[0]["created_at"] == "2026-07-05T00:00:00+00:00"
    assert rows[0]["updated_at"] == "2026-07-05T00:00:00+00:00"
    assert rows[0]["final_positive_summary"] == "Positive summary."
    assert rows[0]["model_provider"] == "deepseek"
    assert orjson.loads(rows[0]["strong_evidence_json"])[0]["effect_slug"] == "hydration"
    assert orjson.loads(rows[0]["medium_evidence_json"]) == []
    assert orjson.loads(rows[0]["caveats_json"]) == ["Test caveat."]


def test_write_final_main_csv_excludes_legacy_warnings_embeddings_and_tags(tmp_path: Path) -> None:
    merges = tmp_path / "merges.jsonl"
    out = tmp_path / "final.csv"
    write_jsonl(merges, [merge_row()])

    write_final_main_csv(
        merges_path=merges,
        out_path=out,
        timestamp="2026-07-05T00:00:00+00:00",
    )

    headers, _ = read_csv_rows(out)
    excluded = {
        "legacy_claims_used",
        "legacy_claims_excluded",
        "merge_warnings",
        "health_effect_positive_embedding",
        "health_effect_negative_embedding",
        "health_effect_positive_tags",
        "health_effect_negative_tags",
    }
    assert not excluded.intersection(headers)


def test_write_final_main_csv_rejects_duplicate_canonical_ids(tmp_path: Path) -> None:
    merges = tmp_path / "merges.jsonl"
    out = tmp_path / "final.csv"
    write_jsonl(
        merges,
        [
            merge_row(canonical_id="same", name="First"),
            merge_row(canonical_id="same", name="Second"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate_canonical_beverage_id:same"):
        write_final_main_csv(
            merges_path=merges,
            out_path=out,
            timestamp="2026-07-05T00:00:00+00:00",
        )


def test_write_final_main_csv_rejects_missing_required_summary(tmp_path: Path) -> None:
    merges = tmp_path / "merges.jsonl"
    out = tmp_path / "final.csv"
    write_jsonl(merges, [merge_row(positive="")])

    with pytest.raises(ValueError):
        write_final_main_csv(
            merges_path=merges,
            out_path=out,
            timestamp="2026-07-05T00:00:00+00:00",
        )


def test_write_final_search_companion_csvs_flattens_effects_and_tags(tmp_path: Path) -> None:
    merges = tmp_path / "merges.jsonl"
    effects_out = tmp_path / "health_reports_ingredient_effects_rows.csv"
    tags_out = tmp_path / "health_reports_ingredient_tags_rows.csv"
    write_jsonl(merges, [merge_row()])

    result = write_final_search_companion_csvs(
        merges_path=merges,
        effects_out_path=effects_out,
        tags_out_path=tags_out,
        timestamp="2026-07-05T00:00:00+00:00",
    )

    assert result.ingredient_rows == 1
    assert result.effect_rows_written == 2
    assert result.tag_rows_written > 0

    effect_headers, effect_rows = read_csv_rows(effects_out)
    assert effect_headers == FINAL_EFFECTS_CSV_COLUMNS
    assert [row["effect_slug"] for row in effect_rows] == ["hydration", "supplement_only"]
    hydration = effect_rows[0]
    assert hydration["effect_bucket"] == "strong_evidence"
    assert hydration["effect_summary"] == "Supports hydration."
    assert hydration["score"] == "0.9"
    assert orjson.loads(hydration["supporting_nutrients_json"]) == ["Water"]
    assert hydration["created_at"] == "2026-07-05T00:00:00+00:00"

    tag_headers, tag_rows = read_csv_rows(tags_out)
    assert tag_headers == FINAL_TAGS_CSV_COLUMNS
    tag_keys = {
        (row["tag"], row["tag_type"], row["source_bucket"])
        for row in tag_rows
    }
    assert ("composition_based", "source_type", "ingredient") in tag_keys
    assert ("hydration", "effect_slug", "strong_evidence") in tag_keys
    assert ("water", "nutrient", "strong_evidence") in tag_keys
    assert (
        "has_supplement_level_only_effects",
        "routing_flag",
        "supplement_level_only_effects",
    ) in tag_keys


def test_write_final_search_companion_csvs_deduplicates_tags_with_max_score(
    tmp_path: Path,
) -> None:
    merges = tmp_path / "merges.jsonl"
    effects_out = tmp_path / "effects.csv"
    tags_out = tmp_path / "tags.csv"
    row = merge_row()
    row["medium_evidence"] = [
        {
            "effect_slug": "hydration",
            "effect_label": "Hydration",
            "summary": "Duplicate lower-score hydration.",
            "evidence_level": "medium",
            "score": 0.4,
            "supporting_nutrients": ["Water"],
        }
    ]
    write_jsonl(merges, [row])

    write_final_search_companion_csvs(
        merges_path=merges,
        effects_out_path=effects_out,
        tags_out_path=tags_out,
        timestamp="2026-07-05T00:00:00+00:00",
    )

    _, tag_rows = read_csv_rows(tags_out)
    water_tags = [
        row for row in tag_rows
        if row["tag"] == "water" and row["tag_type"] == "nutrient"
    ]
    assert len(water_tags) == 2
    assert {row["source_bucket"] for row in water_tags} == {
        "strong_evidence",
        "medium_evidence",
    }
