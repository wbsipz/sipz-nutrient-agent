from __future__ import annotations

import csv
from pathlib import Path

import orjson

from sipz_agent.core.composition_significance import (
    classify_ingredient_significance,
    read_ingredient_profiles,
    read_reference_table,
    run_nutriment_significance_classification,
    significance_prompt,
)


class FakeSignificanceProvider:
    def __init__(self, missing_keys: set[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.missing_keys = missing_keys or set()

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        payload = orjson.loads(prompt.split("Input:\n", 1)[1])
        classifications = []
        for nutriment in payload["nutriments"]:
            if nutriment["source_key"] in self.missing_keys:
                continue
            classifications.append(
                {
                    "source_key": nutriment["source_key"],
                    "canonical_bioactive_name": nutriment["canonical_bioactive_name"],
                    "significance": (
                        "significant"
                        if nutriment["canonical_bioactive_name"] in {"Vitamin C", "Sugars"}
                        else "minor"
                    ),
                    "confidence": 0.82,
                    "reasoning": "Classified from normalized amount and reference context.",
                    "amount_context": nutriment["amount_context"],
                }
            )
        return adapter.validate_python({"classifications": classifications})


def write_profiles(path: Path, rows: list[dict]) -> None:
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
        },
        {
            "canonical_bioactive_name": "Sodium",
            "reference_amount": "",
            "reference_unit": "",
            "reference_type": "limit_or_caution",
            "caution_limit": "2300",
            "caution_limit_unit": "mg/day",
            "threshold_notes": "Use as cautionary sodium context.",
            "source": "test",
            "review_status": "reviewed",
        },
        {
            "canonical_bioactive_name": "Sugars",
            "reference_amount": "",
            "reference_unit": "",
            "reference_type": "limit_or_caution",
            "caution_limit": "50",
            "caution_limit_unit": "g/day added sugars context",
            "threshold_notes": "Use as cautionary sugar context.",
            "source": "test",
            "review_status": "reviewed",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_payload(**overrides) -> dict:
    payload = {
        "canonical_beverage_id": "ingredient-1",
        "canonical_beverage_name": "test lemonade",
        "canonical_category": "test lemonade",
        "canonical_slug": "test-lemonade",
        "beverage_family": "other_beverage",
        "serving_basis": "100g",
        "nutriments": [
            {
                "source_key": "vitamin-c_100g",
                "canonical_bioactive_name": "Vitamin C",
                "raw_amount": 0.03,
                "raw_unit": "g/100g",
                "display_amount": 30,
                "display_unit": "mg/100g",
                "amount_quality_status": "ok",
                "amount_quality_flags": [],
            },
            {
                "source_key": "sodium_100g",
                "canonical_bioactive_name": "Sodium",
                "raw_amount": 12924.1665,
                "raw_unit": "g/100g",
                "display_amount": 12924166.5,
                "display_unit": "mg/100g",
                "amount_quality_status": "review_source_data",
                "amount_quality_flags": ["raw_g_per_100g_gt_100_impossible"],
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_quality_flagged_nutriment_is_skipped_before_llm(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    write_profiles(profiles_path, [profile_payload()])
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)
    provider = FakeSignificanceProvider()

    result = classify_ingredient_significance(
        profile=profile,
        reference_rows=reference_rows,
        provider=provider,
    )

    assert len(provider.prompts) == 1
    assert "12924" not in provider.prompts[0]
    by_key = {item.source_key: item for item in result.classifications}
    assert by_key["vitamin-c_100g"].significance == "significant"
    assert by_key["vitamin-c_100g"].used_llm is True
    assert by_key["sodium_100g"].used_llm is False
    assert by_key["sodium_100g"].significance == "unknown_threshold"
    assert by_key["sodium_100g"].classification_status == "skipped_amount_quality_flag"
    assert "raw_g_per_100g_gt_100_impossible" in by_key["sodium_100g"].amount_quality_flags


def test_significance_prompt_contains_reference_and_caution_instructions(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    payload = profile_payload(
        nutriments=[
            {
                "source_key": "sugars_100g",
                "canonical_bioactive_name": "Sugars",
                "raw_amount": 25,
                "raw_unit": "g/100g",
                "display_amount": 25,
                "display_unit": "g/100g",
                "amount_quality_status": "ok",
                "amount_quality_flags": [],
            }
        ]
    )
    write_profiles(profiles_path, [payload])
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)

    prompt = significance_prompt(
        profile=profile,
        nutriments=profile.nutriments,
        reference_rows=reference_rows,
    )

    assert "serving_basis" in prompt
    assert "100g" in prompt
    assert "cautionary nutrients" in prompt
    assert "50" in prompt
    assert "g/day added sugars context" in prompt
    assert "Use the reference row as context, not as a deterministic rule" in prompt
    assert "Soft calibration, not hard rules" in prompt
    assert "Do not use unknown_threshold merely because total sugars are not confirmed added sugars" in prompt
    assert "This row is total sugars, not confirmed added sugars" in prompt
    assert "values just under 10% of the 50 g/day added-sugar context" in prompt


def test_caffeine_prompt_calibrates_common_beverage_concentrations(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    payload = profile_payload(
        nutriments=[
            {
                "source_key": "caffeine_100g",
                "canonical_bioactive_name": "Caffeine",
                "raw_amount": 0.032,
                "raw_unit": "g/100g",
                "display_amount": 32,
                "display_unit": "mg/100g",
                "amount_quality_status": "ok",
                "amount_quality_flags": [],
                "reference_type": "limit_or_caution",
                "caution_limit": "400",
                "caution_limit_unit": "mg/day",
            }
        ]
    )
    write_profiles(profiles_path, [payload])
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)

    prompt = significance_prompt(
        profile=profile,
        nutriments=profile.nutriments,
        reference_rows=reference_rows,
    )

    assert "Classify caffeine consistently" in prompt
    assert "Around 25-35 mg/100g is usually minor" in prompt


def test_vitamin_a_prompt_distinguishes_ug_from_iu(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    payload = profile_payload(
        nutriments=[
            {
                "source_key": "vitamin-a_100g",
                "canonical_bioactive_name": "Vitamin A and Carotenoids",
                "raw_amount": 0.0000846,
                "raw_unit": "g/100g",
                "display_amount": 84.6,
                "display_unit": "ug/100g",
                "amount_quality_status": "ok",
                "amount_quality_flags": [],
                "reference_type": "beneficial_daily_value",
            },
            {
                "source_key": "vitamin-a-iu_100g",
                "canonical_bioactive_name": "Vitamin A and Carotenoids",
                "raw_amount": 242,
                "raw_unit": "IU/100g",
                "display_amount": 242,
                "display_unit": "IU/100g",
                "amount_quality_status": "ok",
                "amount_quality_flags": [],
                "reference_type": "beneficial_daily_value",
            },
        ]
    )
    write_profiles(profiles_path, [payload])
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)

    prompt = significance_prompt(
        profile=profile,
        nutriments=profile.nutriments,
        reference_rows=reference_rows,
    )

    assert "Treat ug or mcg vitamin A values as RAE-like significance context" in prompt
    assert "Vitamin A IU values are not directly comparable to mcg RAE" in prompt


def test_missing_llm_classification_becomes_row_level_unknown(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    write_profiles(profiles_path, [profile_payload()])
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)
    provider = FakeSignificanceProvider(missing_keys={"vitamin-c_100g"})

    result = classify_ingredient_significance(
        profile=profile,
        reference_rows=reference_rows,
        provider=provider,
    )

    by_key = {item.source_key: item for item in result.classifications}
    assert by_key["vitamin-c_100g"].classification_status == "failed_llm_classification"
    assert by_key["vitamin-c_100g"].significance == "unknown_threshold"
    assert by_key["vitamin-c_100g"].confidence == 0.0
    assert by_key["vitamin-c_100g"].used_llm is True
    assert by_key["sodium_100g"].classification_status == "skipped_amount_quality_flag"


def test_missing_reference_context_is_flagged_without_llm(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    write_profiles(
        profiles_path,
        [
            profile_payload(
                nutriments=[
                    {
                        "source_key": "unknown_100g",
                        "canonical_bioactive_name": "Unknown Nutrient",
                        "raw_amount": 1,
                        "raw_unit": "g/100g",
                        "display_amount": 1,
                        "display_unit": "g/100g",
                        "amount_quality_status": "ok",
                        "amount_quality_flags": [],
                    }
                ]
            )
        ],
    )
    write_reference_table(reference_path)
    profile = read_ingredient_profiles(profiles_path)[0]
    reference_rows = read_reference_table(reference_path)
    provider = FakeSignificanceProvider()

    result = classify_ingredient_significance(
        profile=profile,
        reference_rows=reference_rows,
        provider=provider,
    )

    assert provider.prompts == []
    assert result.classifications[0].source_key == "unknown_100g"
    assert result.classifications[0].classification_status == "missing_reference_context"
    assert result.classifications[0].significance == "unknown_threshold"


def test_run_writes_significance_artifacts_and_caps_workers(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.jsonl"
    reference_path = tmp_path / "reference.csv"
    out_dir = tmp_path / "run"
    write_profiles(
        profiles_path,
        [
            profile_payload(),
            profile_payload(
                canonical_beverage_id="ingredient-2",
                canonical_beverage_name="second lemonade",
            ),
        ],
    )
    write_reference_table(reference_path)

    result = run_nutriment_significance_classification(
        profiles_path=profiles_path,
        reference_path=reference_path,
        out_dir=out_dir,
        provider=FakeSignificanceProvider(),
        workers=2,
    )

    assert result.summary.processed_ingredients == 2
    assert result.summary.classified_nutriments == 2
    assert result.summary.skipped_invalid_nutriments == 2
    assert (out_dir / "nutriment_significance.jsonl").exists()
    assert (out_dir / "nutriment_significance_flat.csv").exists()
    assert (out_dir / "nutriment_significance_summary.json").exists()
    assert (out_dir / "nutriment_significance_failures.jsonl").exists()
    assert (out_dir / "significance_batch_log.jsonl").exists()

    try:
        run_nutriment_significance_classification(
            profiles_path=profiles_path,
            reference_path=reference_path,
            out_dir=out_dir,
            provider=FakeSignificanceProvider(),
            workers=11,
        )
    except ValueError as exc:
        assert str(exc) == "significance_workers_must_be_at_most_10"
    else:
        raise AssertionError("Expected worker count above 10 to fail.")
