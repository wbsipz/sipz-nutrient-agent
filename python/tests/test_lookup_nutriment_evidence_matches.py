from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lookup_nutriment_evidence_matches.py"
SPEC = importlib.util.spec_from_file_location("lookup_nutriment_evidence_matches", SCRIPT_PATH)
assert SPEC is not None
lookup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(lookup)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_evidence_match_records_matches_significant_only(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.csv"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bioactive_name",
                "effect_slug",
                "effect_label",
                "description",
                "score",
                "evidence_level",
                "tags",
                "sources",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bioactive_name": "Vitamin C",
                "effect_slug": "immune_support",
                "effect_label": "Immune support",
                "description": "Supports normal immune function.",
                "score": "0.8",
                "evidence_level": "strong",
                "tags": '["immune"]',
                "sources": '[{"pmid":"1"}]',
            }
        )

    significance = [
        {
            "canonical_beverage_id": "ingredient-1",
            "ingredient_name": "orange juice",
            "serving_basis": "100g",
            "classifications": [
                {
                    "source_key": "vitamin-c_100g",
                    "canonical_bioactive_name": "Vitamin C",
                    "raw_amount": 0.03,
                    "raw_unit": "g/100g",
                    "display_amount": 30,
                    "display_unit": "mg/100g",
                    "amount_context": "30 mg/100g",
                    "reference_amount": "90",
                    "reference_unit": "mg/day",
                    "reference_type": "beneficial_daily_value",
                    "caution_limit": "",
                    "caution_limit_unit": "",
                    "significance": "significant",
                    "confidence": 0.9,
                    "reasoning": "Meaningful amount.",
                    "classification_status": "classified",
                },
                {
                    "source_key": "potassium_100g",
                    "canonical_bioactive_name": "Potassium",
                    "significance": "minor",
                },
                {
                    "source_key": "sugars_100g",
                    "canonical_bioactive_name": "Sugars",
                    "raw_amount": 12,
                    "raw_unit": "g/100g",
                    "display_amount": 12,
                    "display_unit": "g/100g",
                    "amount_context": "12 g/100g",
                    "significance": "significant",
                    "confidence": 0.8,
                    "reasoning": "Cautionary.",
                    "classification_status": "classified",
                },
            ],
        }
    ]

    evidence_by_bioactive = lookup.read_evidence_by_bioactive(evidence_path)
    records = lookup.build_evidence_match_records(
        significance_results=significance,
        evidence_by_bioactive=evidence_by_bioactive,
    )

    assert len(records) == 2
    vitamin_c = records[0]
    sugars = records[1]
    assert vitamin_c["matched"] is True
    assert vitamin_c["match_count"] == 1
    assert vitamin_c["evidence_matches"][0]["effect_slug"] == "immune_support"
    assert vitamin_c["evidence_matches"][0]["tags"] == ["immune"]
    assert vitamin_c["evidence_matches"][0]["sources"] == [{"pmid": "1"}]
    assert sugars["matched"] is False
    assert sugars["evidence_matches"] == []
