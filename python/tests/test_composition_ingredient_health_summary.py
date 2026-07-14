from __future__ import annotations

import orjson
import pytest
from pydantic import TypeAdapter

from sipz_agent.core.composition_ingredient_health_summary import (
    ingredient_health_summary_prompt,
    run_ingredient_health_summary_generation,
    summarize_ingredient_record,
)


class StaticProvider:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, adapter: TypeAdapter):
        self.prompts.append(prompt)
        return adapter.validate_python(self.payload)


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, prompt: str, adapter: TypeAdapter):
        _ = prompt
        _ = adapter
        self.calls += 1
        raise RuntimeError("provider_failed")


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows)
    )


def read_jsonl(path) -> list[dict]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def ingredient_summary_input_row(
    canonical_beverage_id: str = "ingredient-1",
    canonical_beverage_name: str = "Test Beverage",
) -> dict:
    return {
        "canonical_beverage_id": canonical_beverage_id,
        "canonical_beverage_name": canonical_beverage_name,
        "ingredient_name": canonical_beverage_name,
        "serving_basis": "100g",
        "summary_type": "composition_based_literature_derived",
        "full_normalized_nutriment_profile": [],
        "dose_banded_nutriments": [],
        "food_level_summaries": [
            {
                "source_key": "vitamin-c_100g",
                "canonical_bioactive_name": "Vitamin C",
                "amount_context": "30 mg/100g",
                "dose_band": "meaningful",
                "dose_band_basis": "Meaningful food-level dose.",
                "food_level_relevance": "Meaningful vitamin C contribution.",
                "strong_evidence_effects": [
                    {
                        "effect_slug": "immune_support",
                        "effect_label": "Immune support",
                        "summary": "Vitamin C supports normal immune function.",
                        "evidence_level": "strong",
                        "score": 0.9,
                    }
                ],
                "medium_evidence_effects": [],
                "low_evidence_effects": [],
                "caveats": [],
                "significance_reasoning": "30 mg is a meaningful fraction of daily intake.",
            },
            {
                "source_key": "sugars_100g",
                "canonical_bioactive_name": "Sugars",
                "amount_context": "12 g/100g",
                "dose_band": "meaningful",
                "dose_band_basis": "Meaningful sugar exposure.",
                "food_level_relevance": "Meaningful sugar contribution.",
                "strong_evidence_effects": [],
                "medium_evidence_effects": [
                    {
                        "effect_slug": "dental_caries",
                        "effect_label": "Dental caries risk",
                        "summary": "Sugars can contribute to dental caries risk.",
                        "evidence_level": "medium",
                        "score": 0.7,
                    }
                ],
                "low_evidence_effects": [],
                "caveats": [],
                "significance_reasoning": "12 g is meaningful in a 100g serving.",
            },
        ],
        "supplement_level_only_effects": [
            {
                "source_key": "taurine_100g",
                "canonical_bioactive_name": "Taurine",
                "amount_context": "1200 mg/100g",
                "dose_band": "supplement",
                "dose_band_basis": "Supplement-level dose.",
                "effects": [
                    {
                        "effect_slug": "exercise_performance",
                        "effect_label": "Exercise performance",
                        "summary": "Taurine has evidence at supplemental doses.",
                        "evidence_level": "medium",
                        "score": 0.6,
                    }
                ],
                "caveats": [],
            },
            {
                "source_key": "sugars_100g",
                "canonical_bioactive_name": "Sugars",
                "amount_context": "75 g/100g",
                "dose_band": "supplement",
                "dose_band_basis": "Generic total sugars should not inherit rare-sugar effects.",
                "effects": [
                    {
                        "effect_slug": "reduced_postprandial_glucose",
                        "effect_label": "Reduced postprandial glucose",
                        "summary": "Allulose, a rare sugar not present here, has this effect.",
                        "evidence_level": "medium",
                        "score": 0.6,
                    }
                ],
                "caveats": [],
            }
        ],
        "minor_or_trace_background_nutrients": [],
        "ignored_trace_nutrients": [
            {
                "source_key": "sodium_100g",
                "canonical_bioactive_name": "Sodium",
                "amount_context": "44 mg/100g",
                "significance": "trace",
                "dose_band": "trace",
                "dose_band_basis": "Below meaningful threshold.",
                "reasoning": "Too low to drive sodium-related claims.",
            }
        ],
        "unmatched_significant_nutrients": [],
        "skipped_no_effect_summary_nutrients": [],
        "input_warnings": [],
    }


def llm_payload() -> dict:
    return {
        "strong_evidence": [
            {
                "effect_slug": "immune_support",
                "effect_label": "Immune support",
                "summary": "The vitamin C content can meaningfully support normal immune function.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "dental_caries",
                "effect_label": "Dental caries risk",
                "summary": "The sugar content can increase dental caries risk.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            }
        ],
        "medium_evidence": [
            {
                "effect_slug": "immune_support",
                "effect_label": "Immune support duplicate",
                "summary": "Duplicate should be dropped because strong evidence already used it.",
                "evidence_level": "medium",
                "score": 0.8,
                "supporting_nutrients": ["Vitamin C"],
            },
            {
                "effect_slug": "hallucinated_effect",
                "effect_label": "Invented",
                "summary": "This is not in the allowed input.",
                "evidence_level": "medium",
                "score": 0.5,
                "supporting_nutrients": ["Vitamin C"],
            },
        ],
        "low_evidence": [],
        "negative_or_cautionary_effects": [
            {
                "effect_slug": "immune_support",
                "effect_label": "Immune support duplicate",
                "summary": "Duplicate should be dropped because strong evidence already used it.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "dental_caries",
                "effect_label": "Dental caries risk",
                "summary": "The sugar content is a plausible cautionary factor.",
                "evidence_level": "medium",
                "score": 0.7,
                "supporting_nutrients": ["Sugars"],
            }
        ],
        "dominant_nutrients": [
            {
                "canonical_bioactive_name": "Vitamin C",
                "amount_context": "30 mg/100g",
                "dose_band": "meaningful",
                "reason": "It is present at a meaningful food-level concentration.",
            },
            {
                "canonical_bioactive_name": "Invented Nutrient",
                "amount_context": "1 g/100g",
                "dose_band": "meaningful",
                "reason": "Should be removed.",
            },
        ],
        "supplement_level_only_effects": [
            {
                "effect_slug": "exercise_performance",
                "effect_label": "Exercise performance",
                "summary": "This belongs only in supplement-level context.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "reduced_postprandial_glucose",
                "effect_label": "Reduced postprandial glucose",
                "summary": "This generic sugar rare-sugar carryover should be dropped.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": ["Sugars"],
            }
        ],
        "ignored_trace_nutrients": [
            {
                "canonical_bioactive_name": "Sodium",
                "amount_context": "44 mg/100g",
                "reason": "Below the meaningful threshold, so it should not drive the summary.",
            },
            {
                "canonical_bioactive_name": "Invented Trace",
                "amount_context": "1 mg/100g",
                "reason": "Should be removed.",
            },
        ],
        "overall_summary": (
            "This ingredient has composition-based support from vitamin C, with sugar as a "
            "cautionary factor; supplement-only taurine effects should not be treated as "
            "food-level claims."
        ),
        "caveats": ["Composition-based summary, not ingredient-specific clinical evidence."],
    }


def test_prompt_separates_food_supplement_and_trace_inputs() -> None:
    prompt = ingredient_health_summary_prompt(ingredient_summary_input_row())

    assert "Use supplement_level_only_effects only in supplement_level_only_effects" in prompt
    assert "ignored_trace_nutrients means below meaningful threshold" in prompt
    assert "harmful, risk-increasing, or cautionary effects only in" in prompt
    assert "Keep the response compact" in prompt
    assert "immune_support" in prompt
    assert "exercise_performance" in prompt
    assert "reduced_postprandial_glucose" not in prompt
    assert "Allulose" not in prompt


def test_summarize_ingredient_record_sanitizes_model_output() -> None:
    provider = StaticProvider(llm_payload())

    record = summarize_ingredient_record(
        input_row=ingredient_summary_input_row(),
        provider=provider,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
    )

    assert record.summary_type == "composition_based_literature_derived"
    assert [effect.effect_slug for effect in record.strong_evidence] == ["immune_support"]
    assert record.strong_evidence[0].supporting_nutrients == ["Vitamin C"]
    assert record.medium_evidence == []
    assert [effect.effect_slug for effect in record.negative_or_cautionary_effects] == [
        "dental_caries"
    ]
    assert record.negative_or_cautionary_effects[0].supporting_nutrients == ["Sugars"]
    assert [effect.effect_slug for effect in record.supplement_level_only_effects] == [
        "exercise_performance"
    ]
    assert record.supplement_level_only_effects[0].supporting_nutrients == ["Taurine"]
    assert [nutrient.canonical_bioactive_name for nutrient in record.dominant_nutrients] == [
        "Vitamin C"
    ]
    assert [nutrient.canonical_bioactive_name for nutrient in record.ignored_trace_nutrients] == [
        "Sodium"
    ]


def test_run_ingredient_health_summary_generation_writes_artifacts_and_resumes(tmp_path) -> None:
    inputs = tmp_path / "ingredient_health_summary_inputs.jsonl"
    out_dir = tmp_path / "ingredient_health_summaries"
    write_jsonl(
        inputs,
        [
            ingredient_summary_input_row("ingredient-1", "First"),
            ingredient_summary_input_row("ingredient-2", "Second"),
        ],
    )

    first = run_ingredient_health_summary_generation(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(llm_payload()),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
        limit=1,
    )

    assert first.summary.selected_input_rows == 1
    assert first.summary.not_processed_due_to_limit == 1
    assert first.summary.processed_rows == 1
    assert (out_dir / "ingredient_health_summaries.jsonl").exists()
    assert (out_dir / "ingredient_health_summaries_summary.json").exists()
    written = read_jsonl(out_dir / "ingredient_health_summaries.jsonl")
    assert written[0]["canonical_beverage_id"] == "ingredient-1"

    resumed = run_ingredient_health_summary_generation(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=FailingProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
        limit=1,
        resume=True,
    )

    assert resumed.summary.processed_rows == 1
    assert resumed.summary.failed_rows == 0


def test_run_ingredient_health_summary_generation_records_failures(tmp_path) -> None:
    inputs = tmp_path / "ingredient_health_summary_inputs.jsonl"
    out_dir = tmp_path / "ingredient_health_summaries"
    write_jsonl(inputs, [ingredient_summary_input_row()])
    provider = FailingProvider()

    result = run_ingredient_health_summary_generation(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=provider,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
        llm_attempts=2,
    )

    assert provider.calls == 2
    assert result.summary.processed_rows == 0
    assert result.summary.failed_rows == 1
    failures = read_jsonl(out_dir / "ingredient_health_summaries_failures.jsonl")
    assert failures[0]["error_message"] == "provider_failed"


def test_run_ingredient_health_summary_generation_rejects_too_many_workers(tmp_path) -> None:
    inputs = tmp_path / "ingredient_health_summary_inputs.jsonl"
    write_jsonl(inputs, [ingredient_summary_input_row()])

    with pytest.raises(ValueError, match="ingredient_summary_workers_must_be_at_most_30"):
        run_ingredient_health_summary_generation(
            inputs_path=inputs,
            out_dir=tmp_path / "out",
            provider=StaticProvider(llm_payload()),
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            workers=31,
        )
