from sipz_agent.core.ingredient_target_mapping import build_ingredient_target_map_rows


def ingredient_row(row_id: str, name: str) -> dict[str, str]:
    return {
        "canonical_beverage_id": row_id,
        "canonical_beverage_name": name,
    }


def test_build_ingredient_target_map_rows_flattens_all_mapping_statuses() -> None:
    ingredient_rows = [
        ingredient_row("coffee-1", "americano coffee"),
        ingredient_row("apple-1", "apple"),
        ingredient_row("caramel-1", "caramel"),
        ingredient_row("tea-1", "green tea with citrus"),
        ingredient_row("skip-1", "peach tea syrup"),
        ingredient_row("manual-1", "green tea extract"),
        ingredient_row("unknown-1", "unknown ingredient"),
    ]
    equivalence_rows = [
        {
            "group_id": "americano_coffee",
            "canonical_search_name": "americano coffee",
            "group_role": "research_target",
            "research_target_group_id": "",
            "research_target_name": "",
            "canonical_beverage_id": "coffee-1",
            "canonical_beverage_name": "americano coffee",
            "decision": "research_direct",
            "relationship": "same_ingredient",
            "adaptation": "Use coffee evidence.",
            "confidence": "0.7",
            "reason": "Clear coffee variant.",
        },
        {
            "group_id": "apple",
            "canonical_search_name": "apple",
            "group_role": "research_target",
            "research_target_group_id": "",
            "research_target_name": "",
            "canonical_beverage_id": "apple-1",
            "canonical_beverage_name": "apple",
            "decision": "research_direct",
            "relationship": "same_ingredient",
            "adaptation": "Use apple evidence directly.",
            "confidence": "0.8",
            "reason": "Direct target.",
        },
        {
            "group_id": "caramel",
            "canonical_search_name": "caramel",
            "group_role": "research_target",
            "research_target_group_id": "",
            "research_target_name": "",
            "canonical_beverage_id": "caramel-1",
            "canonical_beverage_name": "caramel",
            "decision": "research_direct",
            "relationship": "same_ingredient",
            "adaptation": "Use caramel evidence.",
            "confidence": "0.6",
            "reason": "Direct target before final exclusion.",
        },
        {
            "group_id": "green_tea_with_citrus",
            "canonical_search_name": "green tea with citrus",
            "group_role": "research_target",
            "research_target_group_id": "",
            "research_target_name": "",
            "canonical_beverage_id": "tea-1",
            "canonical_beverage_name": "green tea with citrus",
            "decision": "research_direct",
            "relationship": "same_ingredient",
            "adaptation": "Use tea evidence.",
            "confidence": "0.6",
            "reason": "Direct target before PubMed-ready exclusion.",
        },
    ]
    final_target_rows = [
        {
            "run_target_id": "coffee",
            "run_target_name": "coffee",
            "representative_canonical_beverage_id": "coffee-rep",
            "representative_canonical_beverage_name": "coffee",
            "covered_target_names": '["coffee", "americano coffee", "cafe con leche"]',
            "collapse_reason_code": "coffee_variant",
            "collapse_reason": "Collapsed coffee variants.",
        },
        {
            "run_target_id": "apple",
            "run_target_name": "apple",
            "representative_canonical_beverage_id": "apple-1",
            "representative_canonical_beverage_name": "apple",
            "covered_target_names": '["apple"]',
            "collapse_reason_code": "unchanged",
            "collapse_reason": "Kept distinct.",
        },
    ]
    final_excluded_rows = [
        {
            "run_target_id": "caramel",
            "run_target_name": "caramel",
            "representative_canonical_beverage_id": "caramel-1",
            "representative_canonical_beverage_name": "caramel",
            "covered_target_names": '["caramel"]',
            "collapse_reason_code": "unchanged",
            "collapse_reason": "Kept before final cleanup.",
            "final_exclusion_code": "product_or_flavor_construct",
            "final_exclusion_reason": "Flavor construct.",
        }
    ]
    pubmed_excluded_rows = [
        {
            "canonical_search_name": "green tea with citrus",
            "exclusion_code": "flavored_tea_product",
            "exclusion_reason": "Flavored tea product should reuse broader tea evidence.",
        }
    ]
    skip_rows = [
        {
            "canonical_beverage_id": "skip-1",
            "canonical_beverage_name": "peach tea syrup",
            "decision": "skip_low_value",
            "relationship": "sweetened_or_formulated_product",
            "confidence": "0.9",
            "reason": "Formulated product.",
        },
        {
            "canonical_beverage_id": "manual-1",
            "canonical_beverage_name": "green tea extract",
            "decision": "manual_review",
            "relationship": "extract_or_supplement_form",
            "confidence": "0.6",
            "reason": "Needs human review.",
        },
    ]

    rows = build_ingredient_target_map_rows(
        ingredient_rows=ingredient_rows,
        equivalence_rows=equivalence_rows,
        final_target_rows=final_target_rows,
        final_excluded_rows=final_excluded_rows,
        pubmed_excluded_rows=pubmed_excluded_rows,
        skip_rows=skip_rows,
    )

    by_id = {row["canonical_beverage_id"]: row for row in rows}
    assert len(rows) == len(ingredient_rows)

    assert by_id["coffee-1"]["final_mapping_status"] == "mapped_to_run_target"
    assert by_id["coffee-1"]["final_run_target_name"] == "coffee"
    assert by_id["coffee-1"]["collapse_reason_code"] == "coffee_variant"

    assert by_id["apple-1"]["final_mapping_status"] == "mapped_to_run_target"
    assert by_id["apple-1"]["final_run_target_name"] == "apple"

    assert by_id["caramel-1"]["final_mapping_status"] == "excluded_final"
    assert by_id["caramel-1"]["final_exclusion_code"] == "product_or_flavor_construct"

    assert by_id["tea-1"]["final_mapping_status"] == "excluded_pubmed_ready"
    assert by_id["tea-1"]["pubmed_exclusion_code"] == "flavored_tea_product"

    assert by_id["skip-1"]["final_mapping_status"] == "skipped_low_value"
    assert by_id["skip-1"]["decision"] == "skip_low_value"

    assert by_id["manual-1"]["final_mapping_status"] == "manual_review_unmapped"
    assert by_id["manual-1"]["decision"] == "manual_review"

    assert by_id["unknown-1"]["final_mapping_status"] == "unmapped"


def test_build_ingredient_target_map_rows_uses_reuse_research_target_name() -> None:
    rows = build_ingredient_target_map_rows(
        ingredient_rows=[ingredient_row("reuse-1", "coffee milk")],
        equivalence_rows=[
            {
                "group_id": "coffee",
                "canonical_search_name": "coffee",
                "group_role": "reuse_member",
                "research_target_group_id": "coffee",
                "research_target_name": "coffee",
                "canonical_beverage_id": "reuse-1",
                "canonical_beverage_name": "coffee milk",
                "decision": "reuse_group_evidence",
                "relationship": "same_ingredient",
                "adaptation": "Reuse coffee evidence with caveats.",
                "confidence": "0.7",
                "reason": "Derived ingredient form.",
            }
        ],
        final_target_rows=[
            {
                "run_target_id": "coffee",
                "run_target_name": "coffee",
                "representative_canonical_beverage_id": "coffee-rep",
                "representative_canonical_beverage_name": "coffee",
                "covered_target_names": '["coffee"]',
                "collapse_reason_code": "coffee_variant",
                "collapse_reason": "Collapsed coffee variants.",
            }
        ],
        final_excluded_rows=[],
        skip_rows=[],
    )

    assert rows[0]["final_mapping_status"] == "mapped_to_run_target"
    assert rows[0]["final_run_target_name"] == "coffee"
    assert rows[0]["decision"] == "reuse_group_evidence"
    assert rows[0]["adaptation"] == "Reuse coffee evidence with caveats."
