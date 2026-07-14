from __future__ import annotations

import csv
import json
from pathlib import Path

from sipz_agent.core.composition_ingredient_summary_inputs import (
    INGREDIENT_SUMMARY_INPUTS_JSONL,
    classify_dose_band,
    run_ingredient_summary_input_assembly,
)


DOSE_FIELDS = [
    "canonical_bioactive_name",
    "meaningful_threshold_amount",
    "meaningful_threshold_unit",
    "meaningful_threshold_basis",
    "meaningful_threshold_source",
    "meaningful_review_status",
    "supplement_threshold_amount",
    "supplement_threshold_unit",
    "supplement_threshold_basis",
    "supplement_threshold_source",
    "supplement_review_status",
    "trace_definition",
    "meaningful_definition",
    "supplement_definition",
    "notes",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def nutriment(
    *,
    source_key: str,
    name: str,
    amount: float,
    unit: str,
) -> dict:
    return {
        "source_key": source_key,
        "canonical_bioactive_name": name,
        "raw_amount": amount,
        "raw_unit": unit,
        "display_amount": amount,
        "display_unit": unit,
        "display_name": name,
        "amount_quality_status": "ok",
        "amount_quality_flags": [],
    }


def classification(
    *,
    source_key: str,
    name: str,
    amount: float,
    unit: str,
    significance: str = "significant",
) -> dict:
    return {
        "source_key": source_key,
        "canonical_bioactive_name": name,
        "raw_amount": amount,
        "raw_unit": unit,
        "display_amount": amount,
        "display_unit": unit,
        "amount_context": f"{amount:g} {unit}",
        "reference_amount": "",
        "reference_unit": "",
        "reference_type": "",
        "caution_limit": "",
        "caution_limit_unit": "",
        "significance": significance,
        "confidence": 0.9,
        "reasoning": "test classification",
        "classification_status": "classified",
        "used_llm": True,
        "amount_quality_flags": [],
    }


def summary(
    *,
    source_key: str,
    name: str,
    amount: float,
    unit: str,
) -> dict:
    return {
        "canonical_beverage_id": "ingredient-1",
        "ingredient_name": "test ingredient",
        "serving_basis": "100g",
        "source_key": source_key,
        "canonical_bioactive_name": name,
        "raw_amount": amount,
        "raw_unit": unit,
        "display_amount": amount,
        "display_unit": unit,
        "amount_context": f"{amount:g} {unit}",
        "reference_amount": "",
        "reference_unit": "",
        "reference_type": "",
        "caution_limit": "",
        "caution_limit_unit": "",
        "significance": "significant",
        "significance_confidence": 0.9,
        "significance_reasoning": "test classification",
        "evidence_match_count": 1,
        "food_level_relevance": f"{name} is relevant at this dose.",
        "strong_evidence_effects": [
            {
                "effect_slug": f"{source_key}_strong",
                "effect_label": "Strong effect",
                "summary": "Strong effect summary.",
                "evidence_level": "strong",
                "score": 0.9,
            }
        ],
        "medium_evidence_effects": [],
        "low_evidence_effects": [],
        "supplement_level_relevance": [
            {
                "effect_slug": f"{source_key}_supplement",
                "effect_label": "Supplement effect",
                "summary": "Supplement effect summary.",
                "evidence_level": "moderate",
                "score": 0.7,
            }
        ],
        "caveats": ["test caveat"],
        "summary_status": "summarized",
        "used_llm": True,
        "model_provider": "test",
        "model_name": "test",
    }


def dose_row(
    *,
    name: str,
    meaningful_amount: str,
    meaningful_unit: str,
    supplement_amount: str = "",
    supplement_unit: str = "",
    supplement_status: str = "supplement_dose_reviewed",
) -> dict[str, str]:
    return {
        "canonical_bioactive_name": name,
        "meaningful_threshold_amount": meaningful_amount,
        "meaningful_threshold_unit": meaningful_unit,
        "meaningful_threshold_basis": "meaningful basis",
        "meaningful_threshold_source": "https://example.com/meaningful",
        "meaningful_review_status": "meaningful_dose_reviewed",
        "supplement_threshold_amount": supplement_amount,
        "supplement_threshold_unit": supplement_unit,
        "supplement_threshold_basis": "supplement basis",
        "supplement_threshold_source": "https://example.com/supplement",
        "supplement_review_status": supplement_status,
        "trace_definition": "trace",
        "meaningful_definition": "meaningful",
        "supplement_definition": "supplement",
        "notes": "",
    }


def write_dose_table(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOSE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_classify_dose_band_routes_trace_meaningful_supplement_and_pH() -> None:
    vitamin_c = dose_row(
        name="Vitamin C",
        meaningful_amount="10",
        meaningful_unit="mg/serving",
        supplement_amount="500",
        supplement_unit="mg/serving",
    )
    assert (
        classify_dose_band(
            display_amount=5,
            display_unit="mg/100g",
            amount_context="5 mg/100g",
            dose_row=vitamin_c,
        )["dose_band"]
        == "trace"
    )
    assert (
        classify_dose_band(
            display_amount=30,
            display_unit="mg/100g",
            amount_context="30 mg/100g",
            dose_row=vitamin_c,
        )["dose_band"]
        == "meaningful"
    )
    assert (
        classify_dose_band(
            display_amount=0.6,
            display_unit="g/100g",
            amount_context="0.6 g/100g",
            dose_row=vitamin_c,
        )["dose_band"]
        == "supplement"
    )

    pH = dose_row(
        name="Ph",
        meaningful_amount="4.6",
        meaningful_unit="pH",
        supplement_amount="4.6",
        supplement_unit="pH",
        supplement_status="not_typical_supplement_reviewed",
    )
    routed = classify_dose_band(
        display_amount=4.2,
        display_unit="pH",
        amount_context="pH 4.2",
        dose_row=pH,
    )
    assert routed["dose_band"] == "meaningful"
    assert routed["allow_supplement_level_effects"] is False


def test_build_ingredient_summary_inputs_routes_effect_context(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.jsonl"
    significance = tmp_path / "significance.jsonl"
    summaries = tmp_path / "summaries.jsonl"
    dose_table = tmp_path / "dose.csv"
    failures = tmp_path / "failures.jsonl"
    out = tmp_path / "out"

    write_jsonl(
        profiles,
        [
            {
                "canonical_beverage_id": "ingredient-1",
                "canonical_beverage_name": "test ingredient",
                "serving_basis": "100g",
                "nutriments": [
                    nutriment(source_key="vitamin-c_100g", name="Vitamin C", amount=30, unit="mg/100g"),
                    nutriment(source_key="taurine_100g", name="Taurine", amount=1200, unit="mg/100g"),
                    nutriment(source_key="protein_100g", name="Proteins", amount=1, unit="g/100g"),
                    nutriment(source_key="sodium_100g", name="Sodium", amount=5, unit="mg/100g"),
                    nutriment(source_key="copper_100g", name="Copper", amount=0.2, unit="mg/100g"),
                ],
            },
            {
                "canonical_beverage_id": "failed-ingredient",
                "canonical_beverage_name": "failed ingredient",
                "serving_basis": "100g",
                "nutriments": [],
            },
        ],
    )
    write_jsonl(
        significance,
        [
            {
                "canonical_beverage_id": "ingredient-1",
                "ingredient_name": "test ingredient",
                "serving_basis": "100g",
                "classifications": [
                    classification(source_key="vitamin-c_100g", name="Vitamin C", amount=30, unit="mg/100g"),
                    classification(source_key="taurine_100g", name="Taurine", amount=1200, unit="mg/100g"),
                    classification(
                        source_key="protein_100g",
                        name="Proteins",
                        amount=1,
                        unit="g/100g",
                        significance="minor",
                    ),
                    classification(
                        source_key="sodium_100g",
                        name="Sodium",
                        amount=5,
                        unit="mg/100g",
                        significance="trace",
                    ),
                    classification(source_key="copper_100g", name="Copper", amount=0.2, unit="mg/100g"),
                ],
            }
        ],
    )
    write_jsonl(
        summaries,
        [
            summary(source_key="vitamin-c_100g", name="Vitamin C", amount=30, unit="mg/100g"),
            summary(source_key="taurine_100g", name="Taurine", amount=1200, unit="mg/100g"),
        ],
    )
    write_jsonl(
        failures,
        [
            {
                "canonical_beverage_id": "failed-ingredient",
                "ingredient_name": "failed ingredient",
                "error_type": "llm_error",
                "error_message": "failed",
            }
        ],
    )
    write_dose_table(
        dose_table,
        [
            dose_row(
                name="Vitamin C",
                meaningful_amount="10",
                meaningful_unit="mg/serving",
                supplement_amount="500",
                supplement_unit="mg/serving",
            ),
            dose_row(
                name="Taurine",
                meaningful_amount="50",
                meaningful_unit="mg taurine/serving",
                supplement_amount="1000",
                supplement_unit="mg taurine/serving",
            ),
            dose_row(
                name="Proteins",
                meaningful_amount="5",
                meaningful_unit="g protein/serving",
                supplement_amount="20",
                supplement_unit="g protein/serving",
            ),
            dose_row(
                name="Sodium",
                meaningful_amount="230",
                meaningful_unit="mg/serving",
                supplement_amount="394",
                supplement_unit="mg/serving",
                supplement_status="supplement_dose_reviewed_caution",
            ),
            dose_row(
                name="Copper",
                meaningful_amount="0.09",
                meaningful_unit="mg/serving",
                supplement_amount="",
                supplement_unit="",
                supplement_status="not_typical_supplement_reviewed",
            ),
        ],
    )

    result = run_ingredient_summary_input_assembly(
        profiles_path=profiles,
        significance_path=significance,
        nutriment_summaries_path=summaries,
        dose_band_table_path=dose_table,
        significance_failures_path=failures,
        out_dir=out,
    )

    rows = read_jsonl(out / INGREDIENT_SUMMARY_INPUTS_JSONL)
    assert result.summary["ingredient_rows"] == 2
    ingredient = rows[0]
    assert len(ingredient["food_level_summaries"]) == 2
    assert len(ingredient["supplement_level_only_effects"]) == 1
    assert ingredient["supplement_level_only_effects"][0]["canonical_bioactive_name"] == "Taurine"
    assert [item["canonical_bioactive_name"] for item in ingredient["minor_or_trace_background_nutrients"]] == [
        "Proteins"
    ]
    assert [item["canonical_bioactive_name"] for item in ingredient["ignored_trace_nutrients"]] == [
        "Sodium"
    ]
    assert [item["canonical_bioactive_name"] for item in ingredient["skipped_no_effect_summary_nutrients"]] == [
        "Copper"
    ]

    failed = rows[1]
    assert failed["input_warnings"] == [
        "significance_classification_failed:llm_error:failed",
        "missing_significance_result",
    ]


def test_unit_mismatch_routes_unknown_with_warning() -> None:
    routed = classify_dose_band(
        display_amount=500,
        display_unit="IU/100g",
        amount_context="500 IU/100g",
        dose_row=dose_row(
            name="Vitamin A",
            meaningful_amount="90",
            meaningful_unit="ug RAE/serving",
        ),
    )

    assert routed["dose_band"] == "unknown"
    assert routed["allow_food_level_effects"] is False
    assert routed["warnings"] == ["meaningful_threshold_compare_failed:unsupported_unit"]
