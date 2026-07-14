from __future__ import annotations

import json
from pathlib import Path

from sipz_agent.core.final_export_normalization import (
    NORMALIZED_NEW_SUMMARIES_FAILURES_JSONL,
    NORMALIZED_NEW_SUMMARIES_JSONL,
    NORMALIZED_NEW_SUMMARIES_SUMMARY_JSON,
    normalize_final_export_summaries,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def direct_claim(
    *,
    effect_row_id: str,
    support_level: str,
    statement: str,
    accepted: bool = True,
) -> dict:
    return {
        "effect_row_id": effect_row_id,
        "proposed_ingredient_claim_id": f"claim-{effect_row_id}",
        "citation_id": f"pmid:{effect_row_id}",
        "verdict": "supported",
        "support_level": support_level,
        "claim_scope": "test claim scope",
        "validated_statement": statement,
        "validator_reasoning": "test validator reasoning",
        "supporting_quotes": [{"quote": "test quote"}],
        "limitations": ["test limitation"],
        "accepted": accepted,
        "web_audit_verdict": "confirmed",
    }


def base_row(*, canonical_id: str, name: str, source_type: str) -> dict:
    return {
        "canonical_beverage_id": canonical_id,
        "canonical_beverage_name": name,
        "summary_source_type": source_type,
        "summary_confidence_status": "ready",
        "legacy": {"audited_claims": [{"claim": "legacy claim"}]},
        "source_paths": {"inputs": "test"},
    }


def test_normalize_direct_literature_maps_support_levels_without_caution(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    out_dir = tmp_path / "out"
    row = base_row(canonical_id="direct-1", name="Direct", source_type="direct_literature")
    row["direct_literature"] = {
        "final_run_target_id": "direct",
        "final_run_target_name": "direct",
        "run_path": "ingredient_runs/direct",
        "claim_file": "web_audited_ingredient_claims.json",
        "claims": [
            direct_claim(
                effect_row_id="systematic",
                support_level="human_systematic_review",
                statement="Systematic review statement.",
            ),
            direct_claim(
                effect_row_id="rct",
                support_level="human_rct",
                statement="RCT statement.",
            ),
            direct_claim(
                effect_row_id="observational",
                support_level="human_observational",
                statement="Observational statement.",
            ),
            direct_claim(
                effect_row_id="author",
                support_level="review_author_interpretation",
                statement="Author interpretation statement.",
            ),
        ],
    }
    write_jsonl(inputs, [row])

    result = normalize_final_export_summaries(inputs_path=inputs, out_dir=out_dir)

    assert result.summary["normalized_rows"] == 1
    normalized = read_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_JSONL)[0]
    assert [effect["summary"] for effect in normalized["strong_evidence"]] == [
        "Systematic review statement.",
        "RCT statement.",
    ]
    assert normalized["strong_evidence"][0]["score"] == 0.84
    assert normalized["strong_evidence"][1]["score"] == 0.75
    assert [effect["summary"] for effect in normalized["medium_evidence"]] == [
        "Observational statement."
    ]
    assert normalized["low_evidence"][0]["summary"] == "Author interpretation statement."
    assert normalized["negative_or_cautionary_effects"] == []
    assert normalized["supplement_level_only_effects"] == []
    assert normalized["strong_evidence"][0]["source_claim"]["supporting_quotes_count"] == 1
    assert normalized["legacy"]["audited_claims"][0]["claim"] == "legacy claim"


def test_normalize_composition_copies_buckets_and_omits_supporting_nutrient_lists(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs.jsonl"
    out_dir = tmp_path / "out"
    row = base_row(canonical_id="composition-1", name="Composition", source_type="composition_based")
    row["composition_summary"] = {
        "serving_basis": "100g",
        "strong_evidence": [{"effect_slug": "strong", "summary": "strong summary"}],
        "medium_evidence": [{"effect_slug": "medium", "summary": "medium summary"}],
        "low_evidence": [],
        "negative_or_cautionary_effects": [
            {"effect_slug": "caution", "summary": "caution summary"}
        ],
        "supplement_level_only_effects": [
            {"effect_slug": "supplement", "summary": "supplement summary"}
        ],
        "dominant_nutrients": [{"canonical_bioactive_name": "Vitamin C"}],
        "ignored_trace_nutrients": [{"canonical_bioactive_name": "Sodium"}],
        "caveats": ["composition caveat"],
        "overall_summary": "composition overall",
    }
    write_jsonl(inputs, [row])

    normalize_final_export_summaries(inputs_path=inputs, out_dir=out_dir)

    normalized = read_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_JSONL)[0]
    assert normalized["strong_evidence"] == [{"effect_slug": "strong", "summary": "strong summary"}]
    assert normalized["medium_evidence"] == [{"effect_slug": "medium", "summary": "medium summary"}]
    assert normalized["negative_or_cautionary_effects"][0]["effect_slug"] == "caution"
    assert normalized["supplement_level_only_effects"][0]["effect_slug"] == "supplement"
    assert normalized["caveats"] == ["composition caveat"]
    assert normalized["source_summary_text"] == "composition overall"
    assert "dominant_nutrients" not in normalized
    assert "ignored_trace_nutrients" not in normalized


def test_normalize_final_export_summaries_records_failures_and_writes_summary(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs.jsonl"
    out_dir = tmp_path / "out"
    good = base_row(canonical_id="good", name="Good", source_type="direct_literature")
    good["direct_literature"] = {
        "final_run_target_id": "good",
        "final_run_target_name": "good",
        "run_path": "ingredient_runs/good",
        "claim_file": "validated_ingredient_claims.json",
        "claims": [
            direct_claim(
                effect_row_id="good-claim",
                support_level="unknown_level",
                statement="Unknown support level statement.",
            )
        ],
    }
    malformed = {
        "canonical_beverage_id": "",
        "canonical_beverage_name": "Missing ID",
        "summary_source_type": "composition_based",
        "composition_summary": {},
    }
    write_jsonl(inputs, [good, malformed])

    result = normalize_final_export_summaries(inputs_path=inputs, out_dir=out_dir)

    assert result.summary["input_rows"] == 2
    assert result.summary["normalized_rows"] == 1
    assert result.summary["failed_rows"] == 1
    normalized = read_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_JSONL)[0]
    assert normalized["low_evidence"][0]["score"] == 0.4
    failures = read_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_FAILURES_JSONL)
    assert failures[0]["error_message"] == "missing_canonical_beverage_id"
    summary = json.loads((out_dir / NORMALIZED_NEW_SUMMARIES_SUMMARY_JSON).read_text())
    assert summary == result.summary
