from pathlib import Path

from sipz_agent.core.added_ingredient_mapping import build_added_ingredient_mapping_rows


def ingredient_row(row_id: str, name: str) -> dict[str, str]:
    return {
        "canonical_beverage_id": row_id,
        "canonical_beverage_name": name,
    }


def equivalence_row(
    row_id: str,
    name: str,
    *,
    decision: str = "research_direct",
    research_target_name: str = "",
) -> dict[str, str]:
    return {
        "group_id": name.replace(" ", "_"),
        "canonical_search_name": name,
        "group_role": "research_target" if not research_target_name else "reuse_member",
        "research_target_group_id": research_target_name.replace(" ", "_"),
        "research_target_name": research_target_name,
        "canonical_beverage_id": row_id,
        "canonical_beverage_name": name,
        "decision": decision,
        "relationship": "same_ingredient",
        "adaptation": "Reuse evidence." if research_target_name else "Use directly.",
        "confidence": "0.8",
        "reason": "Test fixture.",
    }


def target_row(target_id: str, name: str, covered: str) -> dict[str, str]:
    return {
        "run_target_id": target_id,
        "run_target_name": name,
        "representative_canonical_beverage_id": f"{target_id}-rep",
        "representative_canonical_beverage_name": name,
        "covered_target_names": covered,
        "collapse_reason_code": "test_reason",
        "collapse_reason": "Test target.",
    }


def test_added_mapping_prefers_original_target_before_added_target() -> None:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=[ingredient_row("coffee-1", "cafe americano")],
        equivalence_rows=[
            equivalence_row("coffee-1", "cafe americano", research_target_name="coffee")
        ],
        skip_rows=[],
        original_target_rows=[
            target_row("coffee", "coffee", '["coffee", "cafe americano"]')
        ],
        added_target_rows=[
            target_row("cafe_americano", "cafe americano", '["cafe americano"]')
        ],
        runs_root=Path("/tmp/does-not-exist"),
    )

    assert summary["mapping_status_counts"] == {"mapped_to_original_corpus_target": 1}
    assert rows[0]["mapped_target_source"] == "original_corpus"
    assert rows[0]["final_run_target_name"] == "coffee"


def test_added_mapping_uses_added_target_when_original_has_no_match() -> None:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=[ingredient_row("xylitol-1", "xylitol")],
        equivalence_rows=[equivalence_row("xylitol-1", "xylitol")],
        skip_rows=[],
        original_target_rows=[],
        added_target_rows=[target_row("xylitol", "xylitol", '["xylitol"]')],
        runs_root=Path("/tmp/does-not-exist"),
    )

    assert summary["mapping_status_counts"] == {"mapped_to_added_batch_target": 1}
    assert rows[0]["mapped_target_source"] == "added_82_batch"
    assert rows[0]["final_run_target_name"] == "xylitol"


def test_added_mapping_uses_already_covered_aliases_as_original_matches() -> None:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=[ingredient_row("date-1", "fresh medjool dates")],
        equivalence_rows=[
            equivalence_row("date-1", "medjool dates", research_target_name="medjool dates")
        ],
        skip_rows=[],
        original_target_rows=[
            target_row("date", "date", '["date"]'),
        ],
        original_alias_target_rows=[
            target_row("date", "date", '["date", "medjool dates"]'),
        ],
        added_target_rows=[],
        runs_root=Path("/tmp/does-not-exist"),
    )

    assert summary["mapping_status_counts"] == {"mapped_to_original_corpus_target": 1}
    assert rows[0]["mapped_target_source"] == "original_corpus"
    assert rows[0]["final_run_target_name"] == "date"


def test_added_mapping_does_not_auto_include_manual_review_rows_with_target_match() -> None:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=[ingredient_row("stevia-1", "vanilla stevia concentrate")],
        equivalence_rows=[],
        skip_rows=[
            {
                "canonical_beverage_id": "stevia-1",
                "canonical_beverage_name": "vanilla stevia concentrate",
                "decision": "manual_review",
                "canonical_search_name": "stevia",
                "relationship": "extract_or_supplement_form",
                "confidence": "0.6",
                "reason": "Flavored concentrate needs manual review.",
            }
        ],
        original_target_rows=[
            target_row("stevia", "stevia", '["stevia"]'),
        ],
        added_target_rows=[],
        runs_root=Path("/tmp/does-not-exist"),
    )

    assert summary["mapping_status_counts"] == {"manual_review_unmapped": 1}
    assert rows[0]["mapped_target_source"] == "manual_review"
    assert rows[0]["final_run_target_name"] == ""
    assert rows[0]["include_in_export"] == "false"
    assert rows[0]["needs_manual_review"] == "true"


def test_added_mapping_classifies_skip_manual_and_unmatched() -> None:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=[
            ingredient_row("skip-1", "flavored syrup"),
            ingredient_row("manual-1", "complex blend"),
            ingredient_row("unknown-1", "unknown ingredient"),
        ],
        equivalence_rows=[
            equivalence_row("unknown-1", "unknown ingredient"),
        ],
        skip_rows=[
            {
                "canonical_beverage_id": "skip-1",
                "canonical_beverage_name": "flavored syrup",
                "decision": "skip_low_value",
                "relationship": "sweetened_or_formulated_product",
                "confidence": "0.9",
                "reason": "Formulated product.",
            },
            {
                "canonical_beverage_id": "manual-1",
                "canonical_beverage_name": "complex blend",
                "decision": "manual_review",
                "relationship": "same_ingredient",
                "confidence": "0.5",
                "reason": "Needs review.",
            },
        ],
        original_target_rows=[],
        added_target_rows=[],
        runs_root=Path("/tmp/does-not-exist"),
    )

    by_id = {row["canonical_beverage_id"]: row for row in rows}
    assert summary["mapping_status_counts"] == {
        "skipped_low_value": 1,
        "manual_review_unmapped": 1,
        "unmatched": 1,
    }
    assert by_id["skip-1"]["mapped_target_source"] == "skipped_low_value"
    assert by_id["manual-1"]["needs_manual_review"] == "true"
    assert by_id["unknown-1"]["mapping_status"] == "unmatched"
