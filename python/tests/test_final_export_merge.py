from __future__ import annotations

import orjson
import pytest
from pydantic import TypeAdapter

from sipz_agent.core.final_export_merge import (
    FINAL_HEALTH_SUMMARY_MERGE_FAILURES_JSONL,
    FINAL_HEALTH_SUMMARY_MERGE_SUMMARY_JSON,
    FINAL_HEALTH_SUMMARY_MERGES_JSONL,
    final_merge_prompt,
    run_final_health_summary_merge,
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


def normalized_row(
    canonical_beverage_id: str = "ingredient-1",
    canonical_beverage_name: str = "Test Ingredient",
) -> dict:
    return {
        "canonical_beverage_id": canonical_beverage_id,
        "canonical_beverage_name": canonical_beverage_name,
        "summary_source_type": "composition_based",
        "summary_confidence_status": "ready",
        "strong_evidence": [
            {
                "effect_slug": "hydration",
                "effect_label": "Hydration",
                "summary": "Supports hydration.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            }
        ],
        "medium_evidence": [
            {
                "effect_slug": "satiety",
                "effect_label": "Satiety",
                "summary": "May support satiety.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            }
        ],
        "low_evidence": [],
        "negative_or_cautionary_effects": [
            {
                "effect_slug": "sugar_risk",
                "effect_label": "Sugar risk",
                "summary": "Can increase sugar-related risk.",
                "evidence_level": "strong",
                "score": 0.85,
                "supporting_nutrients": ["Sugars"],
            }
        ],
        "supplement_level_only_effects": [
            {
                "effect_slug": "supplement_effect",
                "effect_label": "Supplement effect",
                "summary": "Requires supplemental dose.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            }
        ],
        "caveats": ["Existing caveat."],
        "source_summary_text": "Existing summary.",
        "legacy": {
            "audited_claims": [
                {
                    "source_file": "health_reports_ingredients_v1_rows.csv",
                    "row_number": 2,
                    "ingredient": canonical_beverage_name,
                    "effect_type": "positive",
                    "claim": "Legacy accepted hydration claim.",
                    "verdict": "accept",
                    "sources": ["https://example.com/accepted"],
                },
                {
                    "source_file": "health_reports_ingredients_v1_rows.csv",
                    "row_number": 2,
                    "ingredient": canonical_beverage_name,
                    "effect_type": "positive",
                    "claim": "Legacy rejected alertness claim.",
                    "verdict": "reject",
                    "sources": ["https://example.com/rejected"],
                },
            ]
        },
        "source_paths": {},
        "normalization_warnings": [],
    }


def no_effect_row() -> dict:
    row = normalized_row("no-effects", "No Effects")
    row["summary_confidence_status"] = "partial"
    row["strong_evidence"] = []
    row["medium_evidence"] = []
    row["low_evidence"] = []
    row["negative_or_cautionary_effects"] = []
    row["supplement_level_only_effects"] = []
    return row


def llm_payload() -> dict:
    return {
        "final_positive_summary": "Provides hydration benefits and may support satiety.",
        "final_negative_summary": "Main caution is sugar-related risk when relevant.",
        "strong_evidence": [
            {
                "effect_slug": "hydration",
                "effect_label": "Hydration",
                "summary": "Supports hydration.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "hallucinated",
                "effect_label": "Hallucinated",
                "summary": "Should be removed.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            },
        ],
        "medium_evidence": [
            {
                "effect_slug": "hydration",
                "effect_label": "Hydration duplicate",
                "summary": "Duplicate should be removed.",
                "evidence_level": "medium",
                "score": 0.7,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "satiety",
                "effect_label": "Satiety",
                "summary": "May support satiety.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            },
        ],
        "low_evidence": [],
        "negative_or_cautionary_effects": [
            {
                "effect_slug": "sugar_risk",
                "effect_label": "Sugar risk",
                "summary": "Can increase sugar-related risk.",
                "evidence_level": "strong",
                "score": 0.85,
                "supporting_nutrients": ["Sugars"],
            },
            {
                "effect_slug": "supplement_effect",
                "effect_label": "Wrong bucket",
                "summary": "Supplement effect should not be allowed in caution.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            },
        ],
        "supplement_level_only_effects": [
            {
                "effect_slug": "supplement_effect",
                "effect_label": "Supplement effect",
                "summary": "Requires supplemental dose.",
                "evidence_level": "medium",
                "score": 0.6,
                "supporting_nutrients": [],
            },
            {
                "effect_slug": "hydration",
                "effect_label": "Wrong supplement",
                "summary": "Food effect should not be allowed in supplement bucket.",
                "evidence_level": "strong",
                "score": 0.9,
                "supporting_nutrients": [],
            },
        ],
        "caveats": ["Caveat one."],
        "legacy_claims_used": [
            "Legacy accepted hydration claim.",
            "Legacy rejected alertness claim.",
        ],
        "legacy_claims_excluded": [
            "Legacy rejected alertness claim.",
            "Legacy accepted hydration claim.",
        ],
        "merge_warnings": ["test warning"],
    }


def test_final_merge_prompt_splits_legacy_claim_context() -> None:
    prompt = final_merge_prompt(normalized_row())

    assert "legacy_claims_to_consider" in prompt
    assert "Legacy accepted hydration claim." in prompt
    assert "legacy_claims_to_exclude" in prompt
    assert "Legacy rejected alertness claim." in prompt


def test_run_final_health_summary_merge_writes_sanitized_results_and_resumes(tmp_path) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(
        inputs,
        [
            normalized_row("ingredient-1", "First"),
            normalized_row("ingredient-2", "Second"),
        ],
    )

    first = run_final_health_summary_merge(
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
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)
    assert written[0]["canonical_beverage_id"] == "ingredient-1"
    assert [item["effect_slug"] for item in written[0]["strong_evidence"]] == ["hydration"]
    assert [item["effect_slug"] for item in written[0]["medium_evidence"]] == ["satiety"]
    assert [item["effect_slug"] for item in written[0]["negative_or_cautionary_effects"]] == [
        "sugar_risk"
    ]
    assert [item["effect_slug"] for item in written[0]["supplement_level_only_effects"]] == [
        "supplement_effect"
    ]
    assert written[0]["legacy_claims_used"] == ["Legacy accepted hydration claim."]
    assert written[0]["legacy_claims_excluded"] == ["Legacy rejected alertness claim."]
    assert written[0]["merge_warnings"] == [
        "test warning",
        "legacy_claims_used_sanitized_to_exact_audit_matches",
    ]

    resumed = run_final_health_summary_merge(
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
    assert (out_dir / FINAL_HEALTH_SUMMARY_MERGE_SUMMARY_JSON).exists()


def test_run_final_health_summary_merge_rejects_more_than_30_workers(tmp_path) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])

    with pytest.raises(ValueError, match="final_merge_workers_must_be_at_most_30"):
        run_final_health_summary_merge(
            inputs_path=inputs,
            out_dir=out_dir,
            provider=StaticProvider(llm_payload()),
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            workers=31,
        )


def test_run_final_health_summary_merge_sanitizes_legacy_objects_blank_slugs_and_global_duplicates(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    row = normalized_row()
    row["supplement_level_only_effects"].append(
        {
            "effect_slug": "hydration",
            "effect_label": "Hydration at supplement level",
            "summary": "Should not duplicate food-level hydration.",
            "evidence_level": "strong",
            "score": 0.9,
            "supporting_nutrients": [],
        }
    )
    write_jsonl(inputs, [row])
    payload = llm_payload()
    payload["low_evidence"] = [
        {
            "effect_slug": "",
            "effect_label": "Legacy prose effect",
            "summary": "Legacy prose should not become an unkeyed structured effect.",
            "evidence_level": "low",
            "score": 0.3,
            "supporting_nutrients": [],
        }
    ]
    payload["legacy_claims_excluded"] = [
        {
            "claim": "Legacy rejected alertness claim.",
            "source_file": "health_reports_ingredients_v1_rows.csv",
        }
    ]

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert written["low_evidence"] == []
    assert [item["effect_slug"] for item in written["supplement_level_only_effects"]] == [
        "supplement_effect"
    ]
    assert written["legacy_claims_excluded"] == ["Legacy rejected alertness claim."]


def test_run_final_health_summary_merge_normalizes_0_to_10_scores_and_caps_legacy(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    row = normalized_row()
    for index in range(6):
        row["legacy"]["audited_claims"].append(
            {
                "source_file": "health_reports_ingredients_v2_rows.csv",
                "row_number": 2,
                "ingredient": "Test Ingredient",
                "effect_type": "positive",
                "claim": f"Additional accepted legacy claim {index}.",
                "verdict": "accept",
                "sources": [],
            }
        )
    write_jsonl(inputs, [row])
    payload = llm_payload()
    payload["strong_evidence"][0]["score"] = 9
    payload["legacy_claims_used"] = [
        "Legacy accepted hydration claim.",
        *[f"Additional accepted legacy claim {index}." for index in range(6)],
    ]

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert written["strong_evidence"][0]["score"] == 0.9
    assert len(written["legacy_claims_used"]) == 5


def test_run_final_health_summary_merge_allows_no_effect_legacy_summary_as_ready(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [no_effect_row()])
    payload = llm_payload()
    payload["strong_evidence"] = []
    payload["medium_evidence"] = []
    payload["low_evidence"] = []
    payload["negative_or_cautionary_effects"] = []
    payload["supplement_level_only_effects"] = []
    payload["legacy_claims_used"] = ["Legacy accepted hydration claim."]

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert written["summary_confidence_status"] == "ready"
    assert "legacy_only_or_prose_only_summary_needs_review" not in written["merge_warnings"]


def test_run_final_health_summary_merge_allows_negative_summary_without_caution_bucket(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])
    payload = llm_payload()
    payload["final_negative_summary"] = "May increase dental caries risk with frequent use."
    payload["negative_or_cautionary_effects"] = []

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert "negative_summary_has_no_caution_bucket_effect" not in written["merge_warnings"]
    assert (
        "legacy_caution_prose_without_traceable_caution_bucket_needs_review"
        not in written["merge_warnings"]
    )
    assert written["summary_confidence_status"] == "ready"


def test_run_final_health_summary_merge_does_not_warn_for_no_adverse_effects(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])
    payload = llm_payload()
    payload["final_negative_summary"] = "No adverse effects were identified in the reviewed evidence."
    payload["negative_or_cautionary_effects"] = []

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert "negative_summary_has_no_caution_bucket_effect" not in written["merge_warnings"]


def test_run_final_health_summary_merge_removes_stale_dropped_effect_warnings(
    tmp_path,
) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])
    payload = llm_payload()
    payload["merge_warnings"] = [
        "Supplement effects from extract moved to supplement_level_only_effects.",
        "Useful non-routing warning.",
    ]
    payload["supplement_level_only_effects"] = []

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=StaticProvider(payload),
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
    )

    assert result.summary.failed_rows == 0
    written = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL)[0]
    assert "Supplement effects from extract moved to supplement_level_only_effects." not in written[
        "merge_warnings"
    ]
    assert "Useful non-routing warning." in written["merge_warnings"]


def test_run_final_health_summary_merge_rejects_rejected_legacy_claim_usage(tmp_path) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])
    payload = llm_payload()
    payload["final_positive_summary"] = "Legacy rejected alertness claim."
    provider = StaticProvider(payload)

    result = run_final_health_summary_merge(
        inputs_path=inputs,
        out_dir=out_dir,
        provider=provider,
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        workers=1,
        llm_attempts=2,
    )

    assert len(provider.prompts) == 2
    assert result.summary.processed_rows == 0
    assert result.summary.failed_rows == 1
    failures = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGE_FAILURES_JSONL)
    assert failures[0]["error_message"] == "llm_response_used_rejected_legacy_claim"


def test_run_final_health_summary_merge_records_provider_failures(tmp_path) -> None:
    inputs = tmp_path / "normalized.jsonl"
    out_dir = tmp_path / "merge"
    write_jsonl(inputs, [normalized_row()])
    provider = FailingProvider()

    result = run_final_health_summary_merge(
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
    failures = read_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGE_FAILURES_JSONL)
    assert failures[0]["error_message"] == "provider_failed"
