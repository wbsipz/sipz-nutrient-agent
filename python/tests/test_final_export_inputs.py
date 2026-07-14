from __future__ import annotations

import csv
import json
from pathlib import Path

from sipz_agent.core.final_export_inputs import (
    FINAL_EXPORT_MERGE_INPUTS_JSONL,
    FINAL_EXPORT_MERGE_INPUT_SUMMARY_JSON,
    SKIPPED_OR_UNCOVERED_INGREDIENTS_CSV,
    assemble_final_export_inputs,
    slugify,
)


LEGACY_FIELDS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "health_effect_positive",
    "health_effect_negative",
    "health_effect_positive_embedding",
    "health_effect_negative_embedding",
    "health_effect_positive_tags",
    "health_effect_negative_tags",
    "payload_nutrients_count",
    "matched_nutrients_count",
    "skipped_keys_count",
    "missing_summary_count",
    "missing_amount_count",
    "source",
    "embedding_model",
    "embedded_at",
    "created_at",
    "updated_at",
]

MAP_FIELDS = [
    "source_set",
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "canonical_search_name",
    "group_role",
    "adaptation",
    "preparation_reason",
    "preparation_confidence",
    "final_mapping_status",
    "source_mapping_status",
    "mapped_target_source",
    "final_run_target_id",
    "final_run_target_name",
    "final_representative_canonical_beverage_id",
    "final_representative_canonical_beverage_name",
    "collapse_reason_code",
    "collapse_reason",
    "pubmed_exclusion_code",
    "pubmed_exclusion_reason",
    "final_exclusion_code",
    "final_exclusion_reason",
    "run_path",
    "claim_file",
    "web_audited_claim_count",
    "included_or_mapped",
    "include_in_export",
    "needs_manual_review",
    "exclusion_reason",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def legacy_row(canonical_id: str, name: str, positive: str, negative: str) -> dict[str, str]:
    return {
        "canonical_beverage_id": canonical_id,
        "canonical_beverage_name": name,
        "health_effect_positive": positive,
        "health_effect_negative": negative,
        "health_effect_positive_tags": '[{"tag":"positive_tag","score":0.8}]',
        "health_effect_negative_tags": '[{"tag":"negative_tag","score":0.7}]',
    }


def map_row(
    canonical_id: str,
    name: str,
    *,
    target_id: str = "",
    target_name: str = "",
    run_path: str = "",
    claim_file: str = "",
) -> dict[str, str]:
    return {
        "canonical_beverage_id": canonical_id,
        "canonical_beverage_name": name,
        "final_run_target_id": target_id,
        "final_run_target_name": target_name or target_id.replace("_", " "),
        "final_mapping_status": "mapped_to_run_target",
        "run_path": run_path,
        "claim_file": claim_file,
        "included_or_mapped": "true",
    }


def make_run(
    ingredient_runs: Path,
    slug: str,
    *,
    claims_by_file: dict[str, list[dict]],
) -> Path:
    run_dir = ingredient_runs / f"2026-01-01T00-00-00+00-00_{slug}"
    run_dir.mkdir(parents=True)
    (run_dir / "ingredient_packet.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "input": {
                    "canonical_beverage_id": f"id-{slug}",
                    "canonical_beverage_name": slug,
                },
            }
        ),
        encoding="utf-8",
    )
    for filename, claims in claims_by_file.items():
        (run_dir / filename).write_text(json.dumps(claims), encoding="utf-8")
    return run_dir


def claim(statement: str) -> dict:
    return {
        "effect_row_id": statement,
        "accepted": True,
        "support_level": "human_rct",
        "validated_statement": statement,
    }


def test_slugify_matches_run_directory_suffix_style() -> None:
    assert slugify("Tart Cherry Juice") == "tart-cherry-juice"
    assert slugify("beer_and_ale") == "beer-and-ale"


def test_assemble_final_export_inputs_attaches_sources_and_skips_uncovered(tmp_path: Path) -> None:
    legacy_v1 = tmp_path / "health_reports_ingredients_v1_rows.csv"
    legacy_v2 = tmp_path / "health_reports_ingredients_v2_rows.csv"
    claim_audit = tmp_path / "health_effect_claim_audit.csv"
    combined = tmp_path / "combined.csv"
    included = tmp_path / "included.csv"
    composition = tmp_path / "composition.jsonl"
    ingredient_runs = tmp_path / "ingredient_runs"
    out = tmp_path / "out"

    explicit_run = make_run(
        ingredient_runs,
        "explicit-target",
        claims_by_file={
            "validated_ingredient_claims.json": [claim("mapped validated claim")],
            "web_audited_ingredient_claims.json": [claim("mapped web claim")],
        },
    )
    make_run(
        ingredient_runs,
        "slug-target",
        claims_by_file={
            "validated_ingredient_claims.json": [claim("slug validated claim")],
            "web_audited_ingredient_claims.json": [claim("slug web claim")],
        },
    )

    write_csv(
        legacy_v1,
        LEGACY_FIELDS,
        [
            legacy_row("direct-explicit", "Direct Explicit", "v1 positive explicit", "v1 negative"),
            legacy_row("direct-slug", "Direct Slug", "v1 positive slug", "v1 negative"),
            legacy_row("composition-only", "Composition", "v1 positive comp", "v1 negative"),
            legacy_row("uncovered", "Uncovered", "v1 positive uncovered", "v1 negative"),
        ],
    )
    write_csv(
        legacy_v2,
        LEGACY_FIELDS,
        [
            legacy_row("direct-explicit", "Direct Explicit", "v2 positive explicit", "v2 negative"),
            legacy_row("direct-slug", "Direct Slug", "v2 positive slug", "v2 negative"),
            legacy_row("composition-only", "Composition", "v2 positive comp", "v2 negative"),
            legacy_row("uncovered", "Uncovered", "v2 positive uncovered", "v2 negative"),
        ],
    )
    write_csv(
        claim_audit,
        ["source_file", "row_number", "ingredient", "effect_type", "claim", "verdict", "sources"],
        [
            {
                "source_file": legacy_v1.name,
                "row_number": "2",
                "ingredient": "Direct Explicit",
                "effect_type": "positive",
                "claim": "legacy accepted claim",
                "verdict": "accept",
                "sources": "https://example.com/a; https://example.com/b",
            },
            {
                "source_file": legacy_v2.name,
                "row_number": "2",
                "ingredient": "Direct Explicit",
                "effect_type": "negative",
                "claim": "legacy rejected claim",
                "verdict": "reject",
                "sources": "",
            },
            {
                "source_file": legacy_v1.name,
                "row_number": "2",
                "ingredient": "Wrong Ingredient",
                "effect_type": "positive",
                "claim": "mismatched legacy claim",
                "verdict": "accept",
                "sources": "https://example.com/wrong",
            },
        ],
    )
    write_csv(
        combined,
        MAP_FIELDS,
        [
            map_row(
                "direct-explicit",
                "Direct Explicit",
                target_id="explicit_target",
                run_path=str(explicit_run),
                claim_file="validated_ingredient_claims.json",
            ),
            map_row("direct-slug", "Direct Slug", target_id="slug_target"),
            map_row("composition-only", "Composition"),
            map_row("uncovered", "Uncovered"),
        ],
    )
    write_csv(
        included,
        MAP_FIELDS,
        [
            map_row(
                "direct-explicit",
                "Direct Explicit",
                target_id="explicit_target",
                run_path=str(explicit_run),
                claim_file="validated_ingredient_claims.json",
            ),
            map_row("direct-slug", "Direct Slug", target_id="slug_target"),
        ],
    )
    write_jsonl(
        composition,
        [
            {
                "canonical_beverage_id": "composition-only",
                "canonical_beverage_name": "Composition",
                "serving_basis": "100g",
                "strong_evidence": [{"effect_slug": "strong"}],
                "medium_evidence": [],
                "low_evidence": [],
                "negative_or_cautionary_effects": [],
                "supplement_level_only_effects": [],
                "caveats": ["composition caveat"],
                "overall_summary": "composition summary",
                "summary_status": "summarized",
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
            }
        ],
    )

    result = assemble_final_export_inputs(
        legacy_v1_path=legacy_v1,
        legacy_v2_path=legacy_v2,
        claim_audit_path=claim_audit,
        combined_map_path=combined,
        included_map_path=included,
        composition_summaries_path=composition,
        ingredient_runs_dir=ingredient_runs,
        out_dir=out,
    )

    assert result.summary["written_rows"] == 3
    assert result.summary["direct_literature_rows"] == 2
    assert result.summary["composition_based_rows"] == 1
    assert result.summary["skipped_rows"] == 1
    assert result.summary["legacy_audit_claims_attached"] == 2
    assert result.summary["legacy_audit_claims_skipped_name_mismatch"] == 1

    rows = {row["canonical_beverage_id"]: row for row in read_jsonl(out / FINAL_EXPORT_MERGE_INPUTS_JSONL)}
    assert rows["direct-explicit"]["summary_source_type"] == "direct_literature"
    assert rows["direct-explicit"]["direct_literature"]["claim_file"] == "validated_ingredient_claims.json"
    assert rows["direct-explicit"]["direct_literature"]["claims"][0]["validated_statement"] == (
        "mapped validated claim"
    )
    assert rows["direct-explicit"]["legacy"]["v1"]["positive"] == "v1 positive explicit"
    assert rows["direct-explicit"]["legacy"]["audited_claims"][0]["sources"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert "mismatched legacy claim" not in [
        claim["claim"] for claim in rows["direct-explicit"]["legacy"]["audited_claims"]
    ]

    assert rows["direct-slug"]["direct_literature"]["claim_file"] == "web_audited_ingredient_claims.json"
    assert rows["direct-slug"]["direct_literature"]["claims"][0]["validated_statement"] == (
        "slug web claim"
    )

    assert rows["composition-only"]["summary_source_type"] == "composition_based"
    assert rows["composition-only"]["composition_summary"]["strong_evidence"][0]["effect_slug"] == "strong"
    assert rows["composition-only"]["legacy"]["v2"]["positive"] == "v2 positive comp"

    skipped = read_csv_rows(out / SKIPPED_OR_UNCOVERED_INGREDIENTS_CSV)
    assert skipped == [
        {
            "canonical_beverage_id": "uncovered",
            "canonical_beverage_name": "Uncovered",
            "reason": "no_new_summary_source",
            "details": "",
        }
    ]
    summary = json.loads((out / FINAL_EXPORT_MERGE_INPUT_SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary == result.summary
