from __future__ import annotations

import csv
from pathlib import Path

import orjson

from sipz_agent.core.composition_nutriment_summary import (
    nutriment_summary_prompt,
    read_evidence_match_records,
    read_reference_table,
    run_nutriment_health_summary_generation,
)


class FakeNutrimentSummaryProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        payload = orjson.loads(prompt.split("Input:\n", 1)[1])
        nutriment = payload["nutriment"]
        effects = payload["matched_health_effect_rows"]
        return adapter.validate_python(
            {
                "canonical_bioactive_name": nutriment["canonical_bioactive_name"],
                "amount_context": nutriment["amount_context"],
                "food_level_relevance": (
                    "This amount is meaningful at food level for the supplied evidence context."
                ),
                "strong_evidence_effects": [
                    {
                        "effect_slug": effects[0]["effect_slug"],
                        "effect_label": effects[0]["effect_label"],
                        "summary": "Strong evidence effect kept at food level.",
                        "evidence_level": effects[0]["evidence_level"],
                        "score": effects[0]["score"],
                    }
                ],
                "medium_evidence_effects": [],
                "low_evidence_effects": [],
                "supplement_level_relevance": [
                    {
                        "effect_slug": effects[-1]["effect_slug"],
                        "effect_label": effects[-1]["effect_label"],
                        "summary": (
                            "This effect appears to require a higher supplemental exposure "
                            "than the ingredient amount."
                        ),
                        "evidence_level": effects[-1]["evidence_level"],
                        "score": effects[-1]["score"],
                    }
                ],
                "caveats": ["Composition-based summary, not ingredient-specific trial evidence."],
            }
        )


class NoisyNutrimentSummaryProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        payload = orjson.loads(prompt.split("Input:\n", 1)[1])
        nutriment = payload["nutriment"]
        effects = payload["matched_health_effect_rows"]
        strong_effects = [
            {
                "effect_slug": effect["effect_slug"],
                "effect_label": effect["effect_label"],
                "summary": "Food-level effect summary.",
                "evidence_level": effect["evidence_level"],
                "score": effect["score"],
            }
            for effect in effects
        ]
        strong_effects.append(
            {
                "effect_slug": "invented_combined_slug",
                "effect_label": "Invented effect",
                "summary": "This should be discarded.",
                "evidence_level": "strong",
                "score": 0.9,
            }
        )
        return adapter.validate_python(
            {
                "canonical_bioactive_name": nutriment["canonical_bioactive_name"],
                "amount_context": nutriment["amount_context"],
                "food_level_relevance": "Relevant at food level.",
                "strong_evidence_effects": strong_effects,
                "medium_evidence_effects": [],
                "low_evidence_effects": [],
                "supplement_level_relevance": [
                    {
                        "effect_slug": "invented_supplement_slug",
                        "effect_label": "Invented supplement effect",
                        "summary": "This should also be discarded.",
                        "evidence_level": "limited",
                        "score": 0.45,
                    }
                ],
                "caveats": ["c1", "c2", "c3", "c4", "c5", "c6"],
            }
        )


class FlakyNutrimentSummaryProvider(FakeNutrimentSummaryProvider):
    def complete_json(self, prompt, adapter):
        if not self.prompts:
            self.prompts.append(prompt)
            raise RuntimeError("transient_failure")
        return super().complete_json(prompt, adapter)


class DuplicateSlugNutrimentSummaryProvider:
    def complete_json(self, prompt, adapter):
        payload = orjson.loads(prompt.split("Input:\n", 1)[1])
        nutriment = payload["nutriment"]
        effect = payload["matched_health_effect_rows"][0]
        duplicate = {
            "effect_slug": effect["effect_slug"],
            "effect_label": effect["effect_label"],
            "summary": "Duplicate effect summary.",
            "evidence_level": effect["evidence_level"],
            "score": effect["score"],
        }
        return adapter.validate_python(
            {
                "canonical_bioactive_name": nutriment["canonical_bioactive_name"],
                "amount_context": nutriment["amount_context"],
                "food_level_relevance": "Relevant at food level.",
                "strong_evidence_effects": [duplicate],
                "medium_evidence_effects": [duplicate],
                "low_evidence_effects": [duplicate],
                "supplement_level_relevance": [duplicate],
                "caveats": [],
            }
        )


class EmptyNutrimentSummaryProvider:
    def complete_json(self, prompt, adapter):
        payload = orjson.loads(prompt.split("Input:\n", 1)[1])
        nutriment = payload["nutriment"]
        return adapter.validate_python(
            {
                "canonical_bioactive_name": nutriment["canonical_bioactive_name"],
                "amount_context": nutriment["amount_context"],
                "food_level_relevance": "Only general nutrition relevance remains.",
                "strong_evidence_effects": [],
                "medium_evidence_effects": [],
                "low_evidence_effects": [],
                "supplement_level_relevance": [],
                "caveats": ["No matched effect applies at this food dose."],
            }
        )


class FailingNutrimentSummaryProvider:
    def complete_json(self, prompt, adapter):
        _ = prompt
        _ = adapter
        raise RuntimeError("llm_failed")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows)
    )


def write_reference_table(path: Path) -> None:
    rows = [
        {
            "canonical_bioactive_name": "Vitamin C",
            "reference_amount": "90",
            "reference_unit": "mg/day",
            "reference_type": "beneficial_daily_value",
            "caution_limit": "",
            "caution_limit_unit": "",
            "threshold_notes": "Adult daily value context.",
            "source": "test",
            "review_status": "reviewed",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_match_payload(**overrides) -> dict:
    payload = {
        "canonical_beverage_id": "ingredient-1",
        "ingredient_name": "orange juice",
        "serving_basis": "100g",
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
        "significance_confidence": 0.91,
        "significance_reasoning": "30 mg vitamin C is a meaningful fraction of daily intake.",
        "classification_status": "classified",
        "matched": True,
        "match_count": 2,
        "evidence_matches": [
            {
                "effect_slug": "normal_immune_function",
                "effect_label": "Normal immune function",
                "description": "Vitamin C supports normal immune function.",
                "score": 0.85,
                "evidence_level": "strong",
                "tags": ["immune"],
                "sources": [{"pmid": "1"}],
            },
            {
                "effect_slug": "high_dose_cold_duration",
                "effect_label": "Cold duration at supplemental doses",
                "description": "Supplemental vitamin C at gram-level doses may affect cold duration.",
                "score": 0.45,
                "evidence_level": "limited",
                "tags": ["supplement"],
                "sources": [{"pmid": "2"}],
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_prompt_contains_reference_significance_and_evidence(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.csv"
    evidence_path = tmp_path / "evidence_matches.jsonl"
    write_reference_table(reference_path)
    write_jsonl(evidence_path, [evidence_match_payload()])
    reference_rows = read_reference_table(reference_path)
    record = read_evidence_match_records(evidence_path)[0]

    prompt = nutriment_summary_prompt(record=record, reference_rows=reference_rows)

    assert "Separate food-level relevance from supplement-level relevance" in prompt
    assert "Every output effect_slug must be copied exactly from allowed_effect_slugs" in prompt
    assert "Return at most 3 effects in each food-level bucket" in prompt
    assert "Do not use supplement_level_relevance for inverse, comparator" in prompt
    assert "blood pressure" in prompt
    assert "orange juice" in prompt
    assert "30 mg/100g" in prompt
    assert "30 mg vitamin C is a meaningful fraction" in prompt
    assert "Adult daily value context" in prompt
    assert "normal_immune_function" in prompt
    assert "high_dose_cold_duration" in prompt


def test_run_writes_summaries_and_skips_unmatched_rows(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(
        evidence_path,
        [
            evidence_match_payload(),
            evidence_match_payload(
                canonical_beverage_id="ingredient-2",
                ingredient_name="unmatched drink",
                matched=False,
                match_count=0,
                evidence_matches=[],
            ),
        ],
    )
    write_reference_table(reference_path)
    provider = FakeNutrimentSummaryProvider()

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=provider,
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
    )

    assert len(provider.prompts) == 1
    assert result.summary.total_evidence_match_rows == 2
    assert result.summary.eligible_matched_rows == 1
    assert result.summary.selected_matched_rows == 1
    assert result.summary.not_processed_due_to_limit == 0
    assert result.summary.processed_rows == 1
    assert result.summary.failed_rows == 0
    assert result.summary.skipped_unmatched_rows == 1
    assert result.summary.skipped_empty_summaries == 0
    assert result.summary.llm_attempts == 3
    assert result.summary.strong_effects == 1
    assert result.summary.supplement_level_effects == 1
    assert (out_dir / "nutriment_summaries.jsonl").exists()
    assert (out_dir / "nutriment_summaries_failures.jsonl").exists()
    assert (out_dir / "nutriment_summaries_summary.json").exists()
    assert (out_dir / "nutriment_summaries_batch_log.jsonl").exists()

    written = [
        orjson.loads(line)
        for line in (out_dir / "nutriment_summaries.jsonl").read_bytes().splitlines()
    ]
    assert written[0]["ingredient_name"] == "orange juice"
    assert written[0]["canonical_bioactive_name"] == "Vitamin C"
    assert written[0]["model_provider"] == "deepseek"
    assert written[0]["model_name"] == "deepseek-v4-pro"


def test_limit_accounting_does_not_count_unprocessed_rows_as_unmatched(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(
        evidence_path,
        [
            evidence_match_payload(canonical_beverage_id="ingredient-1"),
            evidence_match_payload(canonical_beverage_id="ingredient-2"),
            evidence_match_payload(
                canonical_beverage_id="ingredient-3",
                matched=False,
                match_count=0,
                evidence_matches=[],
            ),
        ],
    )
    write_reference_table(reference_path)

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=FakeNutrimentSummaryProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
        limit=1,
    )

    assert result.summary.total_evidence_match_rows == 3
    assert result.summary.eligible_matched_rows == 2
    assert result.summary.selected_matched_rows == 1
    assert result.summary.not_processed_due_to_limit == 1
    assert result.summary.skipped_unmatched_rows == 1


def test_output_sanitizes_invented_slugs_and_caps_lists(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    payload = evidence_match_payload()
    payload["evidence_matches"] = [
        {
            "effect_slug": f"effect_{index}",
            "effect_label": f"Effect {index}",
            "description": "Effect description.",
            "score": 0.8,
            "evidence_level": "strong",
            "tags": [],
            "sources": [],
        }
        for index in range(6)
    ]
    payload["match_count"] = 6
    write_jsonl(evidence_path, [payload])
    write_reference_table(reference_path)

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=NoisyNutrimentSummaryProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
    )

    summary = result.results[0]
    assert len(summary.strong_evidence_effects) == 3
    assert [effect.effect_slug for effect in summary.strong_evidence_effects] == [
        "effect_0",
        "effect_1",
        "effect_2",
    ]
    assert summary.supplement_level_relevance == []
    assert len(summary.caveats) == 5


def test_output_dedupes_effect_slugs_across_buckets(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(evidence_path, [evidence_match_payload()])
    write_reference_table(reference_path)

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=DuplicateSlugNutrimentSummaryProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
    )

    summary = result.results[0]
    assert len(summary.strong_evidence_effects) == 1
    assert summary.medium_evidence_effects == []
    assert summary.low_evidence_effects == []
    assert summary.supplement_level_relevance == []
    assert result.summary.processed_rows == 1
    assert result.summary.skipped_empty_summaries == 0


def test_empty_summaries_are_skipped(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(evidence_path, [evidence_match_payload()])
    write_reference_table(reference_path)

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=EmptyNutrimentSummaryProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
    )

    assert result.results == []
    assert result.summary.processed_rows == 0
    assert result.summary.skipped_empty_summaries == 1
    written = (out_dir / "nutriment_summaries.jsonl").read_text(encoding="utf-8")
    assert written == ""


def test_transient_failures_are_retried(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(evidence_path, [evidence_match_payload()])
    write_reference_table(reference_path)
    provider = FlakyNutrimentSummaryProvider()

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=provider,
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
        llm_attempts=2,
    )

    assert len(provider.prompts) == 2
    assert result.summary.processed_rows == 1
    assert result.summary.failed_rows == 0
    assert result.summary.llm_attempts == 2


def test_failures_are_row_level_and_workers_are_capped(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence_matches.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "summaries"
    write_jsonl(evidence_path, [evidence_match_payload()])
    write_reference_table(reference_path)

    result = run_nutriment_health_summary_generation(
        evidence_matches_path=evidence_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=FailingNutrimentSummaryProvider(),
        model_provider="deepseek",
        model_name="deepseek-v4-pro",
        workers=1,
    )

    assert result.summary.processed_rows == 0
    assert result.summary.failed_rows == 1
    assert result.summary.llm_attempts == 3
    assert result.failures[0].error_type == "RuntimeError"
    assert result.failures[0].error_message == "llm_failed"

    try:
        run_nutriment_health_summary_generation(
            evidence_matches_path=evidence_path,
            reference_path=reference_path,
            out_dir=out_dir,
            provider=FailingNutrimentSummaryProvider(),
            model_provider="deepseek",
            model_name="deepseek-v4-pro",
            workers=31,
        )
    except ValueError as exc:
        assert str(exc) == "nutriment_summary_workers_must_be_at_most_30"
    else:
        raise AssertionError("Expected worker count above 10 to fail.")
