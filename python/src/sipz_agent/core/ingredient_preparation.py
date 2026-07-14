from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import orjson
from pydantic import TypeAdapter, ValidationError

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.models import HeuristicProvider, LlmProvider
from sipz_agent.schemas.ingredients import (
    IngredientLookupRow,
    IngredientPreparationDecision,
    IngredientPreparationResponse,
    IngredientPreparationSummary,
)


INGREDIENT_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "health_effect_positive",
    "health_effect_negative",
    "health_effect_positive_embedding",
    "health_effect_negative_embedding",
    "health_effect_positive_tags",
    "health_effect_negative_tags",
    "payload_nutrients_count",
    "matched_nutrients_count",
    "skipped_keys_count",
    "missing_summary_count",
    "missing_amount_count",
    "source",
    "embedding_model",
    "embedded_at",
    "created_at",
    "updated_at",
]

RESEARCH_CANDIDATE_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "canonical_search_name",
    "group_id",
    "group_role",
    "relationship",
    "confidence",
    "reason",
]

CANONICAL_TARGET_COLUMNS = [
    "group_id",
    "canonical_search_name",
    "direct_row_count",
    "reuse_row_count",
    "representative_canonical_beverage_id",
    "representative_canonical_beverage_name",
    "relationship_examples",
    "confidence",
    "reason",
]

SKIP_LIST_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "confidence",
    "reason",
]

GROUP_COLUMNS = [
    "group_id",
    "canonical_search_name",
    "group_role",
    "research_target_group_id",
    "research_target_name",
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "adaptation",
    "confidence",
    "reason",
]

MANUAL_REVIEW_KEYWORDS = {
    "blend",
    "mix",
    "concentrate",
    "cocktail",
    "drink",
    "beverage",
    "flavored",
    "flavour",
    "energy",
    "smoothie",
}

SKIP_KEYWORDS = {
    "cookie",
    "cookies",
    "coca cola",
    "cola",
    "cordial",
    "artificial",
    "beverage mix",
    "drink mix",
    "flavor",
    "flavoured",
    "flavored",
    "flavour",
    "fruit punch",
    "hot chocolate",
    "lemonade",
    "mixer",
    "punch",
    "soda",
    "soft drink",
    "sports drink",
    "sweetened",
    "syrup",
    "tropicana",
}

FORMULATED_REVIEW_KEYWORDS = {
    "beverage",
    "bloody mary mix",
    "drink",
    "energy drink",
    "energy tea",
    "fruit milk",
    "juice blend",
    "sparkling water",
}

HARD_SKIP_TERMS = {
    "beverage mix",
    "candy",
    "chocolate milk",
    "chocolate oat drink",
    "chocolate sandwich cookies",
    "cocktail mix",
    "cola syrup",
    "cookie",
    "cookies",
    "dessert",
    "drink mix",
    "energy drink",
    "energy tea",
    "flavored water",
    "flavored milk",
    "flavoured water",
    "flavoured milk",
    "gatorade",
    "hot chocolate",
    "milkshake",
    "mixer",
    "monster energy",
    "powerade",
    "red bull",
    "rockstar",
    "seltzer",
    "soda",
    "sparkling water",
    "sports drink",
    "syrup",
}

MIX_MANUAL_REVIEW_TERMS = {
    "chicory coffee mix",
    "coffee chicory mix",
    "instant chicory coffee mix",
    "mixed plant milk",
}

FORMULATED_RESEARCH_DEMOTION_TERMS = {
    "alcoholic beverage",
    "cafe latte",
    "café latte",
    "cappuccino",
    "latte",
    "matcha latte",
    "mocha",
    "smoothie",
}

GENERIC_JUICE_TARGETS = {
    "fruit",
    "fruit juice",
    "juice",
    "organic",
    "vegetable",
    "vegetable juice",
}

SODA_TERMS = {
    "coca cola",
    "coke",
    "cola",
    "dr pepper",
    "fanta",
    "mountain dew",
    "pepsi",
    "root beer",
    "sprite",
}

BEER_TERMS = {
    "beer",
    "bock",
    "doppelbock",
    "kellerbier",
    "kolsch",
    "lager",
    "radler",
}

WINE_TERMS = {
    "beaujolais",
    "barolo",
    "dolcetto",
    "grauburgunder",
    "merlot",
    "montepulciano",
    "moscato",
    "port wine",
    "sangria",
    "spritzer",
    "wine",
}

SPIRIT_TERMS = {
    "aperol",
    "brandy",
    "cognac",
    "gin",
    "liqueur",
    "rum",
    "schnapps",
    "spirit",
    "tequila",
    "vodka",
    "whiskey",
    "whisky",
}

BLEND_CONNECTORS = {
    "-",
    " and ",
    " & ",
}

JUICE_DESCRIPTORS = {
    "black",
    "fresh",
    "green",
    "organic",
    "red",
    "sour",
    "tart",
    "white",
}

JUICE_FORM_TAIL_WORDS = {
    "concentrate",
    "from",
    "pulp",
    "with",
}

COMPOUND_JUICE_BASES = {
    "black cherry",
    "black currant",
    "black grape",
    "blood orange",
    "dark cherry",
    "green apple",
    "red cherry",
    "red grape",
    "sour cherry",
    "tart cherry",
    "white grape",
}

WHOLE_FOOD_OVERRIDES = {
    "oats": "oats",
}

DERIVED_INGREDIENT_FORMS = {
    "milk": "same_ingredient",
    "butter": "same_ingredient",
    "oil": "same_ingredient",
}

MILK_DESCRIPTORS = {
    "1",
    "2",
    "cow",
    "dairy",
    "fat",
    "free",
    "light",
    "low",
    "non",
    "nonfat",
    "pasteurized",
    "reduced",
    "semi",
    "skim",
    "skimmed",
    "whole",
}

EXTRACT_KEYWORDS = {
    "extract",
    "capsule",
    "supplement",
    "isolate",
}

POWDER_KEYWORDS = {
    "powder",
    "dried",
    "freeze dried",
    "freeze-dried",
}

PUREE_KEYWORDS = {
    "puree",
    "pulp",
    "paste",
}

JUICE_KEYWORDS = {
    "juice",
    "nectar",
}

LEADING_DESCRIPTORS = {
    "fresh",
    "raw",
    "whole",
    "dried",
    "frozen",
    "organic",
    "unsweetened",
    "sweetened",
}

FORM_WORDS = (
    *EXTRACT_KEYWORDS,
    *POWDER_KEYWORDS,
    *PUREE_KEYWORDS,
    *JUICE_KEYWORDS,
    "concentrate",
    "drink",
    "beverage",
    "lemonade",
    "nectar",
    "sparkling water",
    "syrup",
    "flavor",
    "flavour",
)

PREPARATION_RESPONSE_ADAPTER = TypeAdapter(IngredientPreparationResponse)


@dataclass(frozen=True)
class IngredientPreparationResult:
    decisions: list[IngredientPreparationDecision]
    summary: IngredientPreparationSummary


def read_lookup_rows(path: Path, limit: int | None = None) -> list[IngredientLookupRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, raw in enumerate(reader):
            if limit is not None and index >= limit:
                break
            rows.append(IngredientLookupRow.model_validate(raw))
    return rows


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def chunked(items: list[IngredientLookupRow], size: int) -> Iterable[list[IngredientLookupRow]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def slug_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold())
    return re.sub(r"_+", "_", normalized).strip("_")


def contains_any(name: str, keywords: set[str]) -> bool:
    lowered = name.casefold()
    return any(keyword in lowered for keyword in keywords)


def contains_word_or_phrase(name: str, keywords: set[str]) -> bool:
    lowered = name.casefold()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", lowered)
        for keyword in keywords
    )


def group_role(decision: IngredientPreparationDecision) -> str:
    if decision.decision == "research_direct":
        return "research_target"
    if decision.decision == "reuse_group_evidence":
        return "reuse_member"
    return ""


def research_target_group_id(decision: IngredientPreparationDecision) -> str:
    return decision.group_id if decision.decision == "reuse_group_evidence" else ""


def research_target_name(decision: IngredientPreparationDecision) -> str:
    return decision.canonical_search_name if decision.decision == "reuse_group_evidence" else ""


def canonical_group_id(search_name: str) -> str:
    return slug_text(search_name)


def canonical_search_name(name: str) -> str:
    lowered = name.casefold()
    for word in FORM_WORDS:
        lowered = re.sub(rf"\b{re.escape(word)}\b", " ", lowered)
    tokens = [
        token
        for token in re.sub(r"[^a-z0-9]+", " ", lowered).split()
        if token not in LEADING_DESCRIPTORS
    ]
    if not tokens:
        return " ".join(name.strip().split())
    return " ".join(tokens)


def split_derived_ingredient_form(name: str) -> tuple[str, str] | None:
    tokens = re.sub(r"[^a-z0-9]+", " ", name.casefold()).split()
    if len(tokens) < 2:
        return None
    form = tokens[-1]
    if form not in DERIVED_INGREDIENT_FORMS:
        return None
    base = " ".join(tokens[:-1])
    if not base:
        return None
    if form == "milk" and all(token in MILK_DESCRIPTORS for token in tokens[:-1]):
        return "milk", DERIVED_INGREDIENT_FORMS[form]
    return base, DERIVED_INGREDIENT_FORMS[form]


def juice_base_name(name: str) -> str | None:
    tokens = re.sub(r"[^a-z0-9]+", " ", name.casefold()).split()
    form_indices = [
        index for index, token in enumerate(tokens) if token in {"juice", "nectar"}
    ]
    if not form_indices:
        return None
    form_index = form_indices[0]
    raw_base = " ".join(tokens[:form_index])
    if not raw_base or raw_base in GENERIC_JUICE_TARGETS:
        return None
    if any(token.isdigit() for token in tokens[:form_index]):
        return None
    if all(token.isdigit() for token in tokens[:form_index]):
        return None
    if raw_base in COMPOUND_JUICE_BASES:
        return raw_base
    base_tokens = [
        token for token in tokens[:form_index] if token not in JUICE_DESCRIPTORS
    ]
    if not base_tokens:
        return None
    base = " ".join(base_tokens)
    if base in GENERIC_JUICE_TARGETS:
        return None
    if all(token.isdigit() for token in base_tokens):
        return None
    return base if len(base_tokens) == 1 else None


def is_juice_blend(name: str) -> bool:
    lowered = f" {name.casefold()} "
    if "juice" not in lowered and "nectar" not in lowered:
        return False
    if "blend" in lowered:
        return True
    if any(connector in lowered for connector in BLEND_CONNECTORS):
        return True
    if juice_base_name(name) is not None:
        return False
    tokens = re.sub(r"[^a-z0-9]+", " ", name.casefold()).split()
    form_indices = [
        index for index, token in enumerate(tokens) if token in {"juice", "nectar"}
    ]
    if not form_indices:
        return False
    ingredient_tokens = [
        token for token in tokens[: form_indices[0]] if token not in JUICE_DESCRIPTORS
    ]
    return len(ingredient_tokens) > 1


def is_generic_percent_juice(name: str) -> bool:
    tokens = re.sub(r"[^a-z0-9]+", " ", name.casefold()).split()
    form_indices = [
        index for index, token in enumerate(tokens) if token in {"juice", "nectar"}
    ]
    if not form_indices:
        return False
    return any(token.isdigit() for token in tokens[: form_indices[0]])


def hard_skip_reason(name: str) -> str | None:
    lowered = name.casefold()
    if contains_word_or_phrase(lowered, MIX_MANUAL_REVIEW_TERMS):
        return None
    if contains_word_or_phrase(lowered, {"mix"}):
        return "Formulated mix; skip direct ingredient literature research."
    if is_generic_percent_juice(name):
        return "Generic percent juice; skip because no specific ingredient target is present."
    if is_juice_blend(name):
        return (
            "Juice blend; skip direct literature research because the result would "
            "not map cleanly to a single ingredient."
        )
    if contains_word_or_phrase(lowered, SODA_TERMS):
        return "Soda or cola product; skip direct ingredient literature research."
    if contains_word_or_phrase(lowered, HARD_SKIP_TERMS):
        return "Formulated product; skip direct ingredient literature research."
    return None


def direct_research_demotion(name: str) -> tuple[str, str, str] | None:
    lowered = name.casefold()
    if contains_word_or_phrase(lowered, MIX_MANUAL_REVIEW_TERMS):
        return (
            "manual_review",
            "unclear_relationship",
            "Ambiguous ingredient mix; review before selecting a canonical research target.",
        )
    if contains_word_or_phrase(lowered, FORMULATED_RESEARCH_DEMOTION_TERMS):
        return (
            "manual_review",
            "sweetened_or_formulated_product",
            "Formulated beverage; do not treat as an independent literature target.",
        )
    return None


def alcohol_group(name: str) -> tuple[str, str] | None:
    lowered = name.casefold()
    if contains_word_or_phrase(lowered, BEER_TERMS):
        return "beer", "Beer variant; reuse broad beer/alcohol evidence instead of researching this variety."
    if contains_word_or_phrase(lowered, WINE_TERMS):
        return "wine", "Wine variant; reuse broad wine/alcohol evidence instead of researching this variety."
    if contains_word_or_phrase(lowered, SPIRIT_TERMS):
        return "spirits", "Spirit variant; reuse broad spirits/alcohol evidence instead of researching this variety."
    return None


def relationship_for_name(name: str) -> str:
    lowered = name.casefold()
    if contains_any(lowered, EXTRACT_KEYWORDS):
        return "extract_or_supplement_form"
    if contains_any(lowered, POWDER_KEYWORDS):
        return "powder_or_concentrate_form"
    if contains_any(lowered, PUREE_KEYWORDS):
        return "minimally_processed_form"
    if contains_any(lowered, JUICE_KEYWORDS):
        return "juice_or_beverage_form"
    if contains_any(lowered, {"syrup", "sweetened", "flavor", "flavour"}):
        return "sweetened_or_formulated_product"
    return "same_ingredient"


def normalize_decision(decision: IngredientPreparationDecision) -> IngredientPreparationDecision:
    name = decision.canonical_beverage_name
    lowered = name.casefold().strip()
    skip_reason = hard_skip_reason(name)
    if skip_reason is not None:
        return decision.model_copy(
            update={
                "decision": "skip_low_value",
                "canonical_search_name": "",
                "group_id": "",
                "relationship": (
                    "juice_or_beverage_form"
                    if "juice" in lowered or "nectar" in lowered
                    else "sweetened_or_formulated_product"
                ),
                "adaptation": "",
                "reason": skip_reason,
                "confidence": max(decision.confidence, 0.9),
            }
        )
    alcohol = alcohol_group(name)
    if alcohol is not None:
        search_name, reason = alcohol
        return decision.model_copy(
            update={
                "decision": "reuse_group_evidence",
                "canonical_search_name": search_name,
                "group_id": slug_text(search_name),
                "relationship": "same_ingredient",
                "adaptation": (
                    f"Reuse {search_name} evidence with caveats for the specific product form."
                ),
                "reason": reason,
                "confidence": max(min(decision.confidence, 0.9), 0.8),
            }
        )
    if lowered in WHOLE_FOOD_OVERRIDES:
        search_name = WHOLE_FOOD_OVERRIDES[lowered]
        return decision.model_copy(
            update={
                "decision": "research_direct",
                "canonical_search_name": search_name,
                "group_id": slug_text(search_name),
                "relationship": "same_ingredient",
                "adaptation": "Use validated claims directly for this ingredient.",
                "reason": "Raw whole ingredient; do not infer beverage form from table naming.",
                "confidence": max(decision.confidence, 0.9),
            }
        )
    relationship = relationship_for_name(name)
    if decision.decision == "research_direct":
        demotion = direct_research_demotion(name)
        if demotion is not None:
            demoted_decision, relationship, reason = demotion
            return canonicalize_decision(
                decision.model_copy(
                    update={
                        "decision": demoted_decision,
                        "canonical_search_name": "",
                        "group_id": "",
                        "relationship": relationship,
                        "adaptation": "",
                        "reason": reason,
                        "confidence": min(max(decision.confidence, 0.75), 0.9),
                    }
                )
            )
    juice_base = juice_base_name(name)
    if juice_base is not None and decision.decision in {
        "research_direct",
        "reuse_group_evidence",
        "skip_low_value",
        "manual_review",
    }:
        return canonicalize_decision(
            decision.model_copy(
                update={
                    "decision": "reuse_group_evidence",
                    "canonical_search_name": juice_base,
                    "group_id": canonical_group_id(juice_base),
                    "relationship": "juice_or_beverage_form",
                    "adaptation": (
                        f"Reuse {juice_base} evidence with a caveat that this row is {name}."
                    ),
                    "reason": (
                        f"Single-ingredient juice or nectar form; research '{juice_base}' "
                        "instead of treating this row as an independent literature target."
                    ),
                    "confidence": min(max(decision.confidence, 0.8), 0.9),
                }
            )
        )
    if (
        decision.decision == "research_direct"
        and decision.canonical_search_name
        and not valid_canonical_research_target(decision.canonical_search_name)
    ):
        return canonicalize_decision(
            decision.model_copy(
                update={
                    "decision": "manual_review",
                    "relationship": "unclear_relationship",
                    "adaptation": "",
                    "reason": (
                        "Invalid canonical research target shape; review before selecting "
                        "a literature search target."
                    ),
                    "confidence": min(max(decision.confidence, 0.6), 0.8),
                }
            )
        )
    if decision.decision == "research_direct" and relationship == "juice_or_beverage_form":
        search_name = decision.canonical_search_name or canonical_search_name(name)
        if slug_text(search_name) != slug_text(name):
            return canonicalize_decision(decision.model_copy(
                update={
                    "decision": "reuse_group_evidence",
                    "canonical_search_name": search_name,
                    "group_id": canonical_group_id(search_name),
                    "relationship": relationship,
                    "adaptation": (
                        f"Reuse {search_name} evidence with a caveat that this row is {name}."
                    ),
                    "reason": (
                        f"Single-ingredient juice or beverage form; research '{search_name}' "
                        "instead of treating this row as an independent literature target."
                    ),
                    "confidence": min(max(decision.confidence, 0.8), 0.9),
                }
            ))
    derived = split_derived_ingredient_form(name)
    if derived and decision.decision == "research_direct":
        search_name, relationship = derived
        return canonicalize_decision(decision.model_copy(
            update={
                "decision": "reuse_group_evidence",
                "canonical_search_name": search_name,
                "group_id": canonical_group_id(search_name),
                "relationship": relationship,
                "adaptation": (
                    f"Reuse {search_name} evidence with a caveat that this row is {name}."
                ),
                "reason": (
                    f"Derived ingredient form; research the base ingredient '{search_name}' "
                    "instead of the derivative product."
                ),
                "confidence": min(decision.confidence, 0.85),
            }
        ))
    return canonicalize_decision(decision)


def canonicalize_decision(
    decision: IngredientPreparationDecision,
) -> IngredientPreparationDecision:
    if decision.decision in {"research_direct", "reuse_group_evidence"}:
        search_name = decision.canonical_search_name or canonical_search_name(
            decision.canonical_beverage_name
        )
        return decision.model_copy(
            update={
                "canonical_search_name": search_name,
                "group_id": canonical_group_id(search_name),
            }
        )
    if decision.decision == "manual_review":
        search_name = decision.canonical_search_name or canonical_search_name(
            decision.canonical_beverage_name
        )
        return decision.model_copy(
            update={
                "canonical_search_name": search_name,
                "group_id": canonical_group_id(search_name),
            }
        )
    return decision.model_copy(update={"canonical_search_name": "", "group_id": ""})


def heuristic_decision(row: IngredientLookupRow) -> IngredientPreparationDecision:
    name = row.canonical_beverage_name
    lowered = name.casefold()
    relationship = relationship_for_name(name)
    search_name = canonical_search_name(name)
    group_id = slug_text(search_name)

    if contains_any(lowered, SKIP_KEYWORDS):
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="skip_low_value",
                canonical_search_name="",
                group_id="",
                relationship=relationship,  # type: ignore[arg-type]
                adaptation="",
                reason=(
                    "Likely low-value literature target because the row appears to be a "
                    "sweetened, flavored, generic, or formulated product."
                ),
                confidence=0.72,
            )
        )

    if contains_any(lowered, FORMULATED_REVIEW_KEYWORDS):
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="manual_review",
                canonical_search_name=search_name,
                group_id=group_id,
                relationship=relationship,  # type: ignore[arg-type]
                adaptation="Review manually because the row appears to be a beverage or mixed product.",
                reason="Ingredient name suggests a mixed, formulated, or beverage-specific product.",
                confidence=0.64,
            )
        )

    if relationship == "same_ingredient":
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="research_direct",
                canonical_search_name=search_name,
                group_id=group_id,
                relationship="same_ingredient",
                adaptation="Use validated claims directly for this ingredient.",
                reason="Clear ingredient identity suitable for direct literature lookup.",
                confidence=0.7,
            )
        )

    if relationship in {"minimally_processed_form", "juice_or_beverage_form"}:
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="reuse_group_evidence",
                canonical_search_name=search_name,
                group_id=group_id,
                relationship=relationship,  # type: ignore[arg-type]
                adaptation=(
                    "Reuse the canonical ingredient evidence only with wording that preserves "
                    "the ingredient form and avoids implying direct study of this exact row."
                ),
                reason="Ingredient form is close enough to a canonical ingredient to consider reuse.",
                confidence=0.64,
            )
        )

    if relationship in {"powder_or_concentrate_form", "extract_or_supplement_form"}:
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="manual_review",
                canonical_search_name=search_name,
                group_id=group_id,
                relationship=relationship,  # type: ignore[arg-type]
                adaptation="Review manually because form and dose may differ from whole-food evidence.",
                reason="Concentrated or extract-like form may not share whole-food health effects.",
                confidence=0.67,
            )
        )

    if contains_any(lowered, MANUAL_REVIEW_KEYWORDS):
        return normalize_decision(
            IngredientPreparationDecision(
                canonical_beverage_id=row.canonical_beverage_id,
                canonical_beverage_name=name,
                decision="manual_review",
                canonical_search_name=search_name,
                group_id=group_id,
                relationship="unclear_relationship",
                adaptation="Review manually before grouping or researching.",
                reason="Ingredient name suggests a mixed or formulated product.",
                confidence=0.62,
            )
        )

    return normalize_decision(
        IngredientPreparationDecision(
            canonical_beverage_id=row.canonical_beverage_id,
            canonical_beverage_name=name,
            decision="manual_review",
            canonical_search_name=search_name,
            group_id=group_id,
            relationship="unclear_relationship",
            adaptation="Review manually before grouping or researching.",
            reason="The ingredient could not be confidently classified by deterministic rules.",
            confidence=0.5,
        )
    )


def preparation_prompt(rows: list[IngredientLookupRow]) -> str:
    payload = [
        {
            "canonical_beverage_id": row.canonical_beverage_id,
            "canonical_beverage_name": row.canonical_beverage_name,
            "existing_positive": row.health_effect_positive[:500],
            "existing_negative": row.health_effect_negative[:500],
        }
        for row in rows
    ]
    return f"""Classify ingredient rows before literature research for the Sipz ingredient health report pipeline.

Return JSON only with a top-level "decisions" array. Return exactly one decision for each input row.

Decision values:
- research_direct: clear ingredient with likely human nutrition/health literature.
- reuse_group_evidence: variant that can likely reuse another canonical ingredient's evidence.
- skip_low_value: poor literature target, such as fruit punch, syrup, flavoring, soda, sweetened mix, or highly formulated product.
- manual_review: ambiguous, concentrated, extract-like, mixed, or too risky to classify automatically.

Relationship values:
- same_ingredient
- minimally_processed_form
- juice_or_beverage_form
- powder_or_concentrate_form
- extract_or_supplement_form
- sweetened_or_formulated_product
- unclear_relationship

Rules:
- Do not run literature for every variant independently.
- Treat each row name as an ingredient name, not automatically as a finished beverage.
- If the name is a plain whole food such as oats, blueberry, orange, pineapple, or cardamom, classify it as research_direct unless it is clearly a mix/formulated product.
- Prefer one canonical search target for close variants.
- Blueberry puree can reuse blueberry evidence with caveats.
- Pistachio milk should usually reuse pistachio evidence with form caveats, not be researched as pistachio milk.
- Juice blends such as orange-carrot juice, strawberry kiwi juice, multifruit juice blend, apple berry juice, and orange peach apricot nectar should be skip_low_value. Do not research blends directly.
- Single-ingredient juices such as carrot juice or white grape juice should usually reuse the base ingredient evidence, not be treated as independent research targets.
- Soda/cola, energy drinks, sports drinks, syrups, drink mixes, cocktail mixes, flavored waters, seltzers, flavored milks, chocolate milk, cookies, desserts, candy, and milkshakes should be skip_low_value.
- Alcohol variants should not each be researched independently. Beer varieties should reuse beer evidence, wine varieties should reuse wine evidence, and spirits/liqueurs should reuse spirits evidence.
- Fruit punch juice should usually be skip_low_value or manual_review.
- Extracts and supplements should not automatically reuse whole-food evidence.
- canonical_search_name should be the target to research, like "blueberry" or "green tea".
- group_id should be a lowercase snake_case identifier for the canonical search target.
- Keep reasons concise.
- Confidence must be between 0 and 1.

Input rows:
{orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


def llm_decisions(
    rows: list[IngredientLookupRow],
    provider: LlmProvider,
) -> list[IngredientPreparationDecision]:
    response = provider.complete_json(preparation_prompt(rows), PREPARATION_RESPONSE_ADAPTER)
    use_positional_mapping = any(
        not decision.canonical_beverage_id for decision in response.decisions
    )
    decisions = []
    if use_positional_mapping:
        for index, row in enumerate(rows):
            decision = (
                response.decisions[index]
                if index < len(response.decisions)
                else heuristic_decision(row)
            )
            search_name = decision.canonical_search_name or canonical_search_name(
                row.canonical_beverage_name
            )
            group_id = slug_text(decision.group_id or search_name)
            relationship = (
                relationship_for_name(row.canonical_beverage_name)
                if decision.relationship == "unclear_relationship"
                else decision.relationship
            )
            decisions.append(
                normalize_decision(decision.model_copy(
                    update={
                        "canonical_beverage_id": row.canonical_beverage_id,
                        "canonical_beverage_name": row.canonical_beverage_name,
                        "canonical_search_name": search_name,
                        "group_id": group_id,
                        "relationship": relationship,
                    }
                ))
            )
        return decisions

    by_id = {decision.canonical_beverage_id: decision for decision in response.decisions}
    for row in rows:
        decision = by_id.get(row.canonical_beverage_id)
        if decision is None:
            decisions.append(heuristic_decision(row))
            continue
        search_name = decision.canonical_search_name or canonical_search_name(
            row.canonical_beverage_name
        )
        group_id = slug_text(decision.group_id or search_name)
        relationship = (
            relationship_for_name(row.canonical_beverage_name)
            if decision.relationship == "unclear_relationship"
            else decision.relationship
        )
        if decision.canonical_beverage_name != row.canonical_beverage_name:
            decision = decision.model_copy(
                update={"canonical_beverage_name": row.canonical_beverage_name}
            )
        decisions.append(
            normalize_decision(decision.model_copy(
                update={
                    "canonical_search_name": search_name,
                    "group_id": group_id,
                    "relationship": relationship,
                }
            ))
        )
    return decisions


def classify_rows(
    rows: list[IngredientLookupRow],
    provider: LlmProvider,
    *,
    chunk_size: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> list[IngredientPreparationDecision]:
    if isinstance(provider, HeuristicProvider):
        return [heuristic_decision(row) for row in rows]
    decisions: list[IngredientPreparationDecision] = []
    chunks = list(chunked(rows, chunk_size))
    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress(index, len(chunks))
        try:
            decisions.extend(llm_decisions(chunk, provider))
        except (RuntimeError, ValidationError, json.JSONDecodeError):
            decisions.extend(heuristic_decision(row) for row in chunk)
    return decisions


def summary_for_decisions(
    total_rows: int,
    decisions: list[IngredientPreparationDecision],
) -> IngredientPreparationSummary:
    counts = {decision: 0 for decision in IngredientPreparationSummary.model_fields if decision not in {"total_rows", "processed_rows"}}
    for decision in decisions:
        counts[decision.decision] += 1
    return IngredientPreparationSummary(
        total_rows=total_rows,
        processed_rows=len(decisions),
        research_direct=counts["research_direct"],
        reuse_group_evidence=counts["reuse_group_evidence"],
        skip_low_value=counts["skip_low_value"],
        manual_review=counts["manual_review"],
    )


def decision_json(decision: IngredientPreparationDecision) -> dict[str, object]:
    return decision.model_dump(mode="json")


def valid_canonical_research_target(search_name: str) -> bool:
    group_id = canonical_group_id(search_name)
    lowered = search_name.casefold().strip()
    if not group_id:
        return False
    if group_id.replace("_", "").isdigit():
        return False
    if lowered in GENERIC_JUICE_TARGETS or lowered in {"beverage", "drink", "nectar"}:
        return False
    if contains_word_or_phrase(lowered, {"mix"}):
        return False
    return True


def canonical_research_target_rows(
    decisions: list[IngredientPreparationDecision],
) -> list[dict[str, str]]:
    grouped: dict[str, list[IngredientPreparationDecision]] = {}
    for decision in decisions:
        if decision.decision not in {"research_direct", "reuse_group_evidence"}:
            continue
        if not decision.group_id or not decision.canonical_search_name:
            continue
        if not valid_canonical_research_target(decision.canonical_search_name):
            continue
        grouped.setdefault(decision.group_id, []).append(decision)

    rows = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        direct_members = [
            member for member in members if member.decision == "research_direct"
        ]
        reuse_members = [
            member for member in members if member.decision == "reuse_group_evidence"
        ]
        representative = direct_members[0] if direct_members else members[0]
        relationship_examples = sorted({member.relationship for member in members})
        rows.append(
            {
                "group_id": group_id,
                "canonical_search_name": representative.canonical_search_name,
                "direct_row_count": str(len(direct_members)),
                "reuse_row_count": str(len(reuse_members)),
                "representative_canonical_beverage_id": (
                    representative.canonical_beverage_id
                ),
                "representative_canonical_beverage_name": (
                    representative.canonical_beverage_name
                ),
                "relationship_examples": json.dumps(relationship_examples),
                "confidence": str(max(member.confidence for member in members)),
                "reason": representative.reason,
            }
        )
    return rows


def write_preparation_artifacts(
    *,
    out_dir: Path,
    decisions: list[IngredientPreparationDecision],
    summary: IngredientPreparationSummary,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "ingredient_preparation_decisions.json",
        {
            "summary": summary.model_dump(mode="json"),
            "decisions": [decision_json(decision) for decision in decisions],
        },
    )
    write_json(out_dir / "ingredient_preparation_summary.json", summary.model_dump(mode="json"))
    write_csv(
        out_dir / "ingredient_research_candidates.csv",
        RESEARCH_CANDIDATE_COLUMNS,
        [
            {
                "canonical_beverage_id": decision.canonical_beverage_id,
                "canonical_beverage_name": decision.canonical_beverage_name,
                "canonical_search_name": decision.canonical_search_name,
                "group_id": decision.group_id,
                "group_role": group_role(decision),
                "relationship": decision.relationship,
                "confidence": str(decision.confidence),
                "reason": decision.reason,
            }
            for decision in decisions
            if decision.decision == "research_direct"
        ],
    )
    write_csv(
        out_dir / "ingredient_canonical_research_targets.csv",
        CANONICAL_TARGET_COLUMNS,
        canonical_research_target_rows(decisions),
    )
    write_csv(
        out_dir / "ingredient_skip_list.csv",
        SKIP_LIST_COLUMNS,
        [
            {
                "canonical_beverage_id": decision.canonical_beverage_id,
                "canonical_beverage_name": decision.canonical_beverage_name,
                "decision": decision.decision,
                "relationship": decision.relationship,
                "confidence": str(decision.confidence),
                "reason": decision.reason,
            }
            for decision in decisions
            if decision.decision in {"skip_low_value", "manual_review"}
        ],
    )
    write_csv(
        out_dir / "ingredient_equivalence_groups.csv",
        GROUP_COLUMNS,
        [
            {
                "group_id": decision.group_id,
                "canonical_search_name": decision.canonical_search_name,
                "group_role": group_role(decision),
                "research_target_group_id": research_target_group_id(decision),
                "research_target_name": research_target_name(decision),
                "canonical_beverage_id": decision.canonical_beverage_id,
                "canonical_beverage_name": decision.canonical_beverage_name,
                "decision": decision.decision,
                "relationship": decision.relationship,
                "adaptation": decision.adaptation,
                "confidence": str(decision.confidence),
                "reason": decision.reason,
            }
            for decision in decisions
            if decision.decision in {"research_direct", "reuse_group_evidence"}
        ],
    )


def prepare_ingredients(
    *,
    lookup_path: Path,
    out_dir: Path,
    provider: LlmProvider,
    limit: int | None = None,
    chunk_size: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> IngredientPreparationResult:
    rows = read_lookup_rows(lookup_path, limit=limit)
    decisions = classify_rows(rows, provider, chunk_size=chunk_size, progress=progress)
    summary = summary_for_decisions(total_rows=len(rows), decisions=decisions)
    write_preparation_artifacts(out_dir=out_dir, decisions=decisions, summary=summary)
    return IngredientPreparationResult(decisions=decisions, summary=summary)
