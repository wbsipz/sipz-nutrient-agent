from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from sipz_agent.cli_ingredients import app
from sipz_agent.core.ingredient_preparation import (
    INGREDIENT_COLUMNS,
    canonical_research_target_rows,
    normalize_decision,
    prepare_ingredients,
)
from sipz_agent.core.models import HeuristicProvider
from sipz_agent.schemas.ingredients import IngredientPreparationDecision


def write_lookup(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INGREDIENT_COLUMNS)
        writer.writeheader()
        for index, name in enumerate(names, start=1):
            writer.writerow(
                {
                    "canonical_beverage_id": f"ingredient-{index}",
                    "canonical_beverage_name": name,
                    "health_effect_positive": "",
                    "health_effect_negative": "",
                    "health_effect_positive_embedding": "",
                    "health_effect_negative_embedding": "",
                    "health_effect_positive_tags": "",
                    "health_effect_negative_tags": "",
                    "payload_nutrients_count": "0",
                    "matched_nutrients_count": "0",
                    "skipped_keys_count": "0",
                    "missing_summary_count": "0",
                    "missing_amount_count": "0",
                    "source": "",
                    "embedding_model": "",
                    "embedded_at": "",
                    "created_at": "",
                    "updated_at": "",
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_ingredients_classifies_and_groups_heuristically(tmp_path: Path) -> None:
    lookup = tmp_path / "ingredients.csv"
    out_dir = tmp_path / "ingredient_preparation"
    write_lookup(
        lookup,
        [
            "blueberry",
            "blueberry puree",
            "fruit punch juice",
            "green tea extract",
            "Oats",
            "pistachio milk",
            "organic carrot juice",
            "orange-carrot juice",
            "strawberry kiwi juice",
        ],
    )

    result = prepare_ingredients(
        lookup_path=lookup,
        out_dir=out_dir,
        provider=HeuristicProvider(),
    )

    by_name = {decision.canonical_beverage_name: decision for decision in result.decisions}
    assert by_name["blueberry"].decision == "research_direct"
    assert by_name["blueberry"].canonical_search_name == "blueberry"
    assert by_name["blueberry puree"].decision == "reuse_group_evidence"
    assert by_name["blueberry puree"].canonical_search_name == "blueberry"
    assert by_name["blueberry puree"].group_id == "blueberry"
    assert by_name["fruit punch juice"].decision == "skip_low_value"
    assert by_name["green tea extract"].decision == "manual_review"
    assert by_name["green tea extract"].canonical_search_name == "green tea"
    assert by_name["Oats"].decision == "research_direct"
    assert by_name["Oats"].canonical_search_name == "oats"
    assert by_name["pistachio milk"].decision == "reuse_group_evidence"
    assert by_name["pistachio milk"].canonical_search_name == "pistachio"
    assert by_name["pistachio milk"].group_id == "pistachio"
    assert by_name["organic carrot juice"].decision == "reuse_group_evidence"
    assert by_name["organic carrot juice"].canonical_search_name == "carrot"
    assert by_name["organic carrot juice"].group_id == "carrot"
    assert by_name["orange-carrot juice"].decision == "skip_low_value"
    assert by_name["strawberry kiwi juice"].decision == "skip_low_value"

    candidates = read_csv(out_dir / "ingredient_research_candidates.csv")
    skips = read_csv(out_dir / "ingredient_skip_list.csv")
    groups = read_csv(out_dir / "ingredient_equivalence_groups.csv")
    decisions = json.loads((out_dir / "ingredient_preparation_decisions.json").read_text())
    assert [row["canonical_beverage_name"] for row in candidates] == ["blueberry", "Oats"]
    assert all(row["group_role"] == "research_target" for row in candidates)
    assert {row["canonical_beverage_name"] for row in skips} == {
        "fruit punch juice",
        "green tea extract",
        "orange-carrot juice",
        "strawberry kiwi juice",
    }
    assert {row["canonical_beverage_name"] for row in groups} == {
        "blueberry",
        "blueberry puree",
        "Oats",
        "pistachio milk",
        "organic carrot juice",
    }
    pistachio_group = [
        row for row in groups if row["canonical_beverage_name"] == "pistachio milk"
    ][0]
    assert pistachio_group["group_role"] == "reuse_member"
    assert pistachio_group["research_target_group_id"] == "pistachio"
    assert pistachio_group["research_target_name"] == "pistachio"
    assert decisions["summary"] == {
        "total_rows": 9,
        "processed_rows": 9,
        "research_direct": 2,
        "reuse_group_evidence": 3,
        "skip_low_value": 3,
        "manual_review": 1,
    }


def test_sipz_ingredients_prepare_cli_writes_artifacts(tmp_path: Path) -> None:
    lookup = tmp_path / "ingredients.csv"
    out_dir = tmp_path / "out"
    write_lookup(lookup, ["blueberry", "blueberry puree", "fruit punch juice"])

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--lookup",
            str(lookup),
            "--out",
            str(out_dir),
            "--provider",
            "heuristic",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    decisions = json.loads((out_dir / "ingredient_preparation_decisions.json").read_text())
    assert decisions["summary"]["processed_rows"] == 2
    assert (out_dir / "ingredient_research_candidates.csv").exists()
    assert (out_dir / "ingredient_canonical_research_targets.csv").exists()
    assert (out_dir / "ingredient_skip_list.csv").exists()
    assert (out_dir / "ingredient_equivalence_groups.csv").exists()


def test_prepare_ingredients_applies_hard_skip_and_alcohol_group_rules(
    tmp_path: Path,
) -> None:
    lookup = tmp_path / "ingredients.csv"
    out_dir = tmp_path / "ingredient_preparation"
    write_lookup(
        lookup,
        [
            "code red mountain dew",
            "berry energy drink",
            "key lime sparkling water",
            "organic chocolate milk",
            "chocolate milk powder",
            "cookie dough milkshake",
            "rum spirit",
            "grauburgunder wine",
            "kolsch beer",
            "root beer",
        ],
    )

    result = prepare_ingredients(
        lookup_path=lookup,
        out_dir=out_dir,
        provider=HeuristicProvider(),
    )

    by_name = {decision.canonical_beverage_name: decision for decision in result.decisions}
    assert by_name["code red mountain dew"].decision == "skip_low_value"
    assert by_name["berry energy drink"].decision == "skip_low_value"
    assert by_name["key lime sparkling water"].decision == "skip_low_value"
    assert by_name["organic chocolate milk"].decision == "skip_low_value"
    assert by_name["chocolate milk powder"].decision == "skip_low_value"
    assert by_name["cookie dough milkshake"].decision == "skip_low_value"
    assert by_name["root beer"].decision == "skip_low_value"
    assert by_name["rum spirit"].decision == "reuse_group_evidence"
    assert by_name["rum spirit"].canonical_search_name == "spirits"
    assert by_name["grauburgunder wine"].decision == "reuse_group_evidence"
    assert by_name["grauburgunder wine"].canonical_search_name == "wine"
    assert by_name["kolsch beer"].decision == "reuse_group_evidence"
    assert by_name["kolsch beer"].canonical_search_name == "beer"

    skips = read_csv(out_dir / "ingredient_skip_list.csv")
    groups = read_csv(out_dir / "ingredient_equivalence_groups.csv")
    assert {row["canonical_beverage_name"] for row in skips} == {
        "code red mountain dew",
        "berry energy drink",
        "key lime sparkling water",
        "organic chocolate milk",
        "chocolate milk powder",
        "cookie dough milkshake",
        "root beer",
    }
    alcohol_targets = {
        row["canonical_beverage_name"]: row["research_target_name"] for row in groups
    }
    assert alcohol_targets["rum spirit"] == "spirits"
    assert alcohol_targets["grauburgunder wine"] == "wine"
    assert alcohol_targets["kolsch beer"] == "beer"


def test_normalize_decision_repairs_llm_group_ids_and_juice_rules() -> None:
    pistachio = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-1",
            canonical_beverage_name="pistachio milk",
            decision="reuse_group_evidence",
            canonical_search_name="pistachio",
            group_id="pistachio_milk",
            relationship="minimally_processed_form",
            adaptation="Reuse pistachio evidence.",
            reason="LLM supplied a row-specific group id.",
            confidence=0.8,
        )
    )
    assert pistachio.decision == "reuse_group_evidence"
    assert pistachio.group_id == "pistachio"

    tart_cherry = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-2",
            canonical_beverage_name="tart cherry juice",
            decision="skip_low_value",
            canonical_search_name="",
            group_id="",
            relationship="juice_or_beverage_form",
            adaptation="",
            reason="LLM treated this as a juice blend.",
            confidence=0.7,
        )
    )
    assert tart_cherry.decision == "reuse_group_evidence"
    assert tart_cherry.canonical_search_name == "tart cherry"
    assert tart_cherry.group_id == "tart_cherry"

    orange_with_pulp = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-3",
            canonical_beverage_name="orange juice from concentrate with pulp",
            decision="skip_low_value",
            canonical_search_name="",
            group_id="",
            relationship="juice_or_beverage_form",
            adaptation="",
            reason="LLM treated pulp as another ingredient.",
            confidence=0.7,
        )
    )
    assert orange_with_pulp.decision == "reuse_group_evidence"
    assert orange_with_pulp.canonical_search_name == "orange"
    assert orange_with_pulp.group_id == "orange"

    fresh_orange = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-4",
            canonical_beverage_name="fresh orange juice",
            decision="reuse_group_evidence",
            canonical_search_name="orange juice",
            group_id="orange_juice",
            relationship="juice_or_beverage_form",
            adaptation="Reuse orange juice evidence.",
            reason="LLM targeted the juice form.",
            confidence=0.8,
        )
    )
    assert fresh_orange.decision == "reuse_group_evidence"
    assert fresh_orange.canonical_search_name == "orange"
    assert fresh_orange.group_id == "orange"


def test_normalize_decision_demotes_formulated_direct_research_targets() -> None:
    matcha_latte = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-1",
            canonical_beverage_name="matcha latte",
            decision="research_direct",
            canonical_search_name="matcha latte",
            group_id="matcha_latte",
            relationship="same_ingredient",
            adaptation="Use directly.",
            reason="LLM called this a distinct research target.",
            confidence=0.9,
        )
    )

    assert matcha_latte.decision == "manual_review"
    assert matcha_latte.relationship == "sweetened_or_formulated_product"
    assert matcha_latte.group_id == "matcha_latte"

    coffee_mix = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-2",
            canonical_beverage_name="3 in 1 coffee mix",
            decision="research_direct",
            canonical_search_name="3 in 1 coffee mix",
            group_id="3_in_1_coffee_mix",
            relationship="same_ingredient",
            adaptation="Use directly.",
            reason="LLM called this a clear target.",
            confidence=0.9,
        )
    )
    assert coffee_mix.decision == "skip_low_value"
    assert coffee_mix.canonical_search_name == ""
    assert coffee_mix.group_id == ""

    chicory_mix = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-3",
            canonical_beverage_name="chicory coffee mix",
            decision="research_direct",
            canonical_search_name="chicory coffee mix",
            group_id="chicory_coffee_mix",
            relationship="same_ingredient",
            adaptation="Use directly.",
            reason="LLM called this a clear target.",
            confidence=0.9,
        )
    )
    assert chicory_mix.decision == "manual_review"
    assert chicory_mix.relationship == "unclear_relationship"

    generic_juice = normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id="ingredient-4",
            canonical_beverage_name="organic 100% juice",
            decision="research_direct",
            canonical_search_name="100",
            group_id="100",
            relationship="juice_or_beverage_form",
            adaptation="Use directly.",
            reason="LLM extracted a numeric target.",
            confidence=0.9,
        )
    )
    assert generic_juice.decision == "skip_low_value"
    assert generic_juice.group_id == ""


def test_canonical_research_targets_are_deduped_and_include_reuse_targets() -> None:
    decisions = [
        normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id="ingredient-1",
                canonical_beverage_name="blueberry",
                decision="research_direct",
                canonical_search_name="blueberry",
                group_id="blueberry",
                relationship="same_ingredient",
                adaptation="Use directly.",
                reason="Direct target.",
                confidence=0.8,
            )
        ),
        normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id="ingredient-2",
                canonical_beverage_name="blueberry puree",
                decision="reuse_group_evidence",
                canonical_search_name="blueberry",
                group_id="blueberry_puree",
                relationship="minimally_processed_form",
                adaptation="Reuse blueberry.",
                reason="Reuse target.",
                confidence=0.9,
            )
        ),
        normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id="ingredient-3",
                canonical_beverage_name="pistachio milk",
                decision="reuse_group_evidence",
                canonical_search_name="pistachio",
                group_id="pistachio_milk",
                relationship="minimally_processed_form",
                adaptation="Reuse pistachio.",
                reason="Reuse target not present as a direct row.",
                confidence=0.7,
            )
        ),
        normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id="ingredient-4",
                canonical_beverage_name="organic 100% juice",
                decision="research_direct",
                canonical_search_name="100",
                group_id="100",
                relationship="juice_or_beverage_form",
                adaptation="Use directly.",
                reason="Bad numeric target.",
                confidence=0.7,
            )
        ),
        normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id="ingredient-5",
                canonical_beverage_name="hot cocoa mix",
                decision="research_direct",
                canonical_search_name="hot cocoa mix",
                group_id="hot_cocoa_mix",
                relationship="same_ingredient",
                adaptation="Use directly.",
                reason="Bad mix target.",
                confidence=0.7,
            )
        ),
    ]

    rows = {
        row["group_id"]: row for row in canonical_research_target_rows(decisions)
    }
    assert set(rows) == {"blueberry", "pistachio"}
    assert rows["blueberry"]["direct_row_count"] == "1"
    assert rows["blueberry"]["reuse_row_count"] == "1"
    assert rows["pistachio"]["direct_row_count"] == "0"
    assert rows["pistachio"]["reuse_row_count"] == "1"
