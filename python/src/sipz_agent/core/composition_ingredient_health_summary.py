from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable

import orjson
from pydantic import TypeAdapter

from sipz_agent.core.artifacts import model_dump_jsonable
from sipz_agent.core.models import LlmProvider
from sipz_agent.schemas.composition import (
    IngredientDominantNutrient,
    IngredientHealthSummaryFailure,
    IngredientHealthSummaryRecord,
    IngredientHealthSummaryRunSummary,
    IngredientIgnoredTraceNutrient,
    IngredientLevelEffectSummary,
    LlmIngredientHealthSummary,
)


LLM_INGREDIENT_SUMMARY_ADAPTER = TypeAdapter(LlmIngredientHealthSummary)
INGREDIENT_SUMMARY_ADAPTER = TypeAdapter(IngredientHealthSummaryRecord)

INGREDIENT_HEALTH_SUMMARIES_JSONL = "ingredient_health_summaries.jsonl"
INGREDIENT_HEALTH_SUMMARIES_FAILURES_JSONL = "ingredient_health_summaries_failures.jsonl"
INGREDIENT_HEALTH_SUMMARIES_SUMMARY_JSON = "ingredient_health_summaries_summary.json"
INGREDIENT_HEALTH_SUMMARIES_BATCH_LOG_JSONL = "ingredient_health_summaries_batch_log.jsonl"

MAX_EFFECTS_PER_BUCKET = 5
MAX_CAUTIONARY_EFFECTS = 6
MAX_DOMINANT_NUTRIENTS = 8
MAX_SUPPLEMENT_EFFECTS = 6
MAX_IGNORED_TRACE_NUTRIENTS = 10
INITIAL_EFFECT_BUCKET_LIMIT = 20
CAUTIONARY_NUTRIENTS = {
    "alcohol",
    "caffeine",
    "carbohydrates",
    "cholesterol",
    "erythritol",
    "fat",
    "fructose",
    "glucose",
    "green tea extract / egcg",
    "lactose",
    "maltodextrins",
    "maltose",
    "ph",
    "polyols",
    "salt",
    "saturated fat",
    "sodium",
    "sucrose",
    "sugars",
    "sulfate",
}
CAUTIONARY_EFFECT_PATTERN = re.compile(
    "|".join(
        [
            r"adverse",
            r"blood[-_ ]?pressure",
            r"body[-_ ]?weight",
            r"cancer",
            r"cardiometabolic",
            r"caries",
            r"cardiovascular[-_ ]?risk",
            r"diabetes",
            r"dental",
            r"glycemic",
            r"harm",
            r"hypertension",
            r"inflammation",
            r"kidney",
            r"ldl",
            r"mortality",
            r"obesity",
            r"risk",
            r"toxic",
            r"weight[-_ ]?gain",
        ]
    ),
    re.IGNORECASE,
)
RARE_SUGAR_EFFECT_PATTERN = re.compile(
    r"allulose|tagatose|rare[-_ ]?sugar|postprandial[-_ ]?glucose|hba1c|glycemic[-_ ]?response",
    re.IGNORECASE,
)


IngredientSummaryProgress = Callable[[int, int, dict[str, Any]], None]


@dataclass(frozen=True)
class IngredientHealthSummaryRunResult:
    out_dir: Path
    results: list[IngredientHealthSummaryRecord]
    failures: list[IngredientHealthSummaryFailure]
    summary: IngredientHealthSummaryRunSummary


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[object]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    atomic_write(path, payload)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def summary_key(row: dict[str, Any] | IngredientHealthSummaryRecord) -> str:
    if isinstance(row, IngredientHealthSummaryRecord):
        return row.canonical_beverage_id
    return str(row["canonical_beverage_id"])


def effect_sort_key(effect: dict[str, Any]) -> tuple[float, str]:
    score = effect.get("score")
    return (float(score) if isinstance(score, (int, float)) else -1.0, str(effect.get("effect_slug", "")))


def compact_effect(effect: dict[str, Any]) -> dict[str, Any]:
    return {
        "effect_slug": effect.get("effect_slug", ""),
        "effect_label": effect.get("effect_label", ""),
        "summary": effect.get("summary", ""),
        "evidence_level": effect.get("evidence_level", ""),
        "score": effect.get("score"),
    }


def compact_food_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": summary.get("source_key", ""),
        "canonical_bioactive_name": summary.get("canonical_bioactive_name", ""),
        "amount_context": summary.get("amount_context", ""),
        "dose_band": summary.get("dose_band", ""),
        "dose_band_basis": summary.get("dose_band_basis", ""),
        "food_level_relevance": summary.get("food_level_relevance", ""),
        "strong_evidence_effects": [
            compact_effect(effect) for effect in summary.get("strong_evidence_effects", [])
        ],
        "medium_evidence_effects": [
            compact_effect(effect) for effect in summary.get("medium_evidence_effects", [])
        ],
        "low_evidence_effects": [
            compact_effect(effect) for effect in summary.get("low_evidence_effects", [])
        ],
        "caveats": summary.get("caveats", [])[:3],
        "significance_reasoning": summary.get("significance_reasoning", ""),
    }


def compact_supplement_summary(summary: dict[str, Any]) -> dict[str, Any]:
    effects = [
        compact_effect(effect)
        for effect in summary.get("effects", [])
        if not is_excluded_supplement_effect(summary=summary, effect=effect)
    ]
    return {
        "source_key": summary.get("source_key", ""),
        "canonical_bioactive_name": summary.get("canonical_bioactive_name", ""),
        "amount_context": summary.get("amount_context", ""),
        "dose_band": summary.get("dose_band", ""),
        "dose_band_basis": summary.get("dose_band_basis", ""),
        "effects": effects,
        "caveats": summary.get("caveats", [])[:3],
    }


def compact_nutriment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": item.get("source_key", ""),
        "canonical_bioactive_name": item.get("canonical_bioactive_name", ""),
        "amount_context": item.get("amount_context", ""),
        "significance": item.get("significance", ""),
        "dose_band": item.get("dose_band", ""),
        "dose_band_basis": item.get("dose_band_basis", ""),
        "reasoning": item.get("reasoning", ""),
    }


def ingredient_prompt_payload(row: dict[str, Any]) -> dict[str, Any]:
    supplement_summaries = [
        compact_supplement_summary(summary)
        for summary in row.get("supplement_level_only_effects", [])
    ]
    return {
        "ingredient": {
            "canonical_beverage_id": row["canonical_beverage_id"],
            "canonical_beverage_name": row["canonical_beverage_name"],
            "ingredient_name": row["ingredient_name"],
            "serving_basis": row.get("serving_basis") or "100g",
            "summary_type": row.get("summary_type", ""),
        },
        "food_level_summaries": [
            compact_food_summary(summary) for summary in row.get("food_level_summaries", [])
        ],
        "supplement_level_only_effects": [
            summary for summary in supplement_summaries if summary["effects"]
        ],
        "background_nutrients": [
            compact_nutriment(item) for item in row.get("minor_or_trace_background_nutrients", [])
        ][:20],
        "ignored_trace_nutrients": [
            compact_nutriment(item) for item in row.get("ignored_trace_nutrients", [])
        ][:20],
        "skipped_no_effect_summary_nutrients": [
            compact_nutriment(item) for item in row.get("skipped_no_effect_summary_nutrients", [])
        ],
        "input_warnings": row.get("input_warnings", []),
    }


def allowed_food_effect_slugs(row: dict[str, Any]) -> set[str]:
    slugs: set[str] = set()
    for summary in row.get("food_level_summaries", []):
        for bucket in ["strong_evidence_effects", "medium_evidence_effects", "low_evidence_effects"]:
            for effect in summary.get(bucket, []):
                if effect.get("effect_slug"):
                    slugs.add(effect["effect_slug"])
    return slugs


def allowed_supplement_effect_slugs(row: dict[str, Any]) -> set[str]:
    slugs: set[str] = set()
    for summary in row.get("supplement_level_only_effects", []):
        for effect in summary.get("effects", []):
            if is_excluded_supplement_effect(summary=summary, effect=effect):
                continue
            if effect.get("effect_slug"):
                slugs.add(effect["effect_slug"])
    return slugs


def allowed_food_nutrients(row: dict[str, Any]) -> set[str]:
    return {
        summary["canonical_bioactive_name"]
        for summary in row.get("food_level_summaries", [])
        if summary.get("canonical_bioactive_name")
    }


def food_nutrients_by_effect_slug(row: dict[str, Any]) -> dict[str, list[str]]:
    nutrients_by_slug: dict[str, list[str]] = {}
    for summary in row.get("food_level_summaries", []):
        nutrient = summary.get("canonical_bioactive_name")
        if not nutrient:
            continue
        for bucket in ["strong_evidence_effects", "medium_evidence_effects", "low_evidence_effects"]:
            for effect in summary.get(bucket, []):
                slug = effect.get("effect_slug")
                if not slug:
                    continue
                nutrients_by_slug.setdefault(slug, [])
                if nutrient not in nutrients_by_slug[slug]:
                    nutrients_by_slug[slug].append(nutrient)
    return nutrients_by_slug


def allowed_supplement_nutrients(row: dict[str, Any]) -> set[str]:
    nutrients: set[str] = set()
    for summary in row.get("supplement_level_only_effects", []):
        nutrient = summary.get("canonical_bioactive_name")
        if not nutrient:
            continue
        if any(
            not is_excluded_supplement_effect(summary=summary, effect=effect)
            for effect in summary.get("effects", [])
        ):
            nutrients.add(nutrient)
    return nutrients


def supplement_nutrients_by_effect_slug(row: dict[str, Any]) -> dict[str, list[str]]:
    nutrients_by_slug: dict[str, list[str]] = {}
    for summary in row.get("supplement_level_only_effects", []):
        nutrient = summary.get("canonical_bioactive_name")
        if not nutrient:
            continue
        for effect in summary.get("effects", []):
            if is_excluded_supplement_effect(summary=summary, effect=effect):
                continue
            slug = effect.get("effect_slug")
            if not slug:
                continue
            nutrients_by_slug.setdefault(slug, [])
            if nutrient not in nutrients_by_slug[slug]:
                nutrients_by_slug[slug].append(nutrient)
    return nutrients_by_slug


def is_excluded_supplement_effect(*, summary: dict[str, Any], effect: dict[str, Any]) -> bool:
    if str(summary.get("canonical_bioactive_name", "")).casefold() != "sugars":
        return False
    text = " ".join(
        str(value or "")
        for value in [
            effect.get("effect_slug"),
            effect.get("effect_label"),
            effect.get("summary"),
            effect.get("description"),
        ]
    )
    return bool(RARE_SUGAR_EFFECT_PATTERN.search(text))


def allowed_ignored_trace_nutrients(row: dict[str, Any]) -> set[str]:
    return {
        item["canonical_bioactive_name"]
        for item in row.get("ignored_trace_nutrients", [])
        if item.get("canonical_bioactive_name")
    }


def ingredient_health_summary_prompt(row: dict[str, Any]) -> str:
    payload = ingredient_prompt_payload(row)
    food_slugs = sorted(allowed_food_effect_slugs(row))
    supplement_slugs = sorted(allowed_supplement_effect_slugs(row))
    food_nutrients = sorted(allowed_food_nutrients(row))
    ignored_nutrients = sorted(allowed_ignored_trace_nutrients(row))
    return (
        "Create a composition-based, literature-derived health summary for one ingredient.\n\n"
        "Use only the supplied Step 8a input. Do not invent nutrients, effects, mechanisms, "
        "doses, or clinical evidence. This is not direct clinical evidence for the ingredient; "
        "it is a synthesis from nutrient/bioactive composition and existing nutrient evidence.\n\n"
        "Rules:\n"
        "- Use food_level_summaries for strong_evidence, medium_evidence, low_evidence, "
        "negative_or_cautionary_effects, dominant_nutrients, and overall_summary.\n"
        "- Use supplement_level_only_effects only in supplement_level_only_effects.\n"
        "- ignored_trace_nutrients means below meaningful threshold / not claim-driving; do not "
        "use those nutrients to support health effects.\n"
        "- background_nutrients are context only and should not drive conclusions.\n"
        "- Include cautionary effects when supported by food-level summaries, especially sugar, "
        "sodium, salt, saturated fat, caffeine, alcohol, cholesterol, pH/acidity, and polyols.\n"
        "- Put harmful, risk-increasing, or cautionary effects only in "
        "negative_or_cautionary_effects, even when evidence is strong. Do not place dental caries, "
        "weight gain, obesity, diabetes risk, LDL, blood pressure, cardiometabolic risk, kidney "
        "risk, cancer risk, or mortality risk in strong_evidence/medium_evidence/low_evidence.\n"
        "- strong_evidence, medium_evidence, and low_evidence are for beneficial or neutral "
        "food-level effects only.\n"
        "- If the ingredient has little claim-driving evidence, say so plainly and keep buckets "
        "small or empty.\n"
        "- Every effect_slug in strong/medium/low/negative must be copied from allowed_food_effect_slugs.\n"
        "- Every effect_slug in supplement_level_only_effects must be copied from "
        "allowed_supplement_effect_slugs.\n"
        "- dominant_nutrients must be copied from allowed_food_nutrients.\n"
        "- ignored_trace_nutrients should be selected only from allowed_ignored_trace_nutrients.\n\n"
        f"allowed_food_effect_slugs: {food_slugs}\n"
        f"allowed_supplement_effect_slugs: {supplement_slugs}\n"
        f"allowed_food_nutrients: {food_nutrients}\n"
        f"allowed_ignored_trace_nutrients: {ignored_nutrients}\n\n"
        "Keep the response compact: at most 3 strong, 3 medium, 3 low, 6 negative/cautionary, "
        "5 dominant nutrients, 4 supplement-only effects, and 5 ignored trace nutrients. "
        "Use one concise sentence for every summary/reason/caveat.\n\n"
        "Return JSON with exactly these top-level keys:\n"
        "{\n"
        '  "strong_evidence": [],\n'
        '  "medium_evidence": [],\n'
        '  "low_evidence": [],\n'
        '  "negative_or_cautionary_effects": [],\n'
        '  "dominant_nutrients": [],\n'
        '  "supplement_level_only_effects": [],\n'
        '  "ignored_trace_nutrients": [],\n'
        '  "overall_summary": "Brief complete ingredient-level summary.",\n'
        '  "caveats": []\n'
        "}\n\n"
        "Effect objects require: effect_slug, effect_label, summary, evidence_level, score, "
        "supporting_nutrients.\n"
        "Dominant nutrient objects require: canonical_bioactive_name, amount_context, dose_band, reason.\n"
        "Ignored trace objects require: canonical_bioactive_name, amount_context, reason.\n\n"
        "Input:\n"
        + orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
    )


def dedupe_effects(items: list[IngredientLevelEffectSummary], *, limit: int) -> list[IngredientLevelEffectSummary]:
    seen: set[str] = set()
    kept: list[IngredientLevelEffectSummary] = []
    for item in items:
        key = item.effect_slug or f"{item.effect_label}:{item.summary}"
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def sanitize_effect_bucket(
    items: list[IngredientLevelEffectSummary],
    *,
    allowed_slugs: set[str],
    allowed_nutrients: set[str],
    limit: int,
    nutrients_by_slug: dict[str, list[str]] | None = None,
) -> list[IngredientLevelEffectSummary]:
    sanitized: list[IngredientLevelEffectSummary] = []
    for item in items:
        if item.effect_slug and item.effect_slug not in allowed_slugs:
            continue
        source_supporting = nutrients_by_slug.get(item.effect_slug, []) if nutrients_by_slug else []
        raw_supporting = item.supporting_nutrients or source_supporting
        supporting = [name for name in raw_supporting if name in allowed_nutrients]
        if item.supporting_nutrients and not supporting:
            continue
        sanitized.append(
            IngredientLevelEffectSummary(
                effect_slug=item.effect_slug,
                effect_label=item.effect_label,
                summary=item.summary,
                evidence_level=item.evidence_level,
                score=item.score,
                supporting_nutrients=supporting,
            )
        )
    return dedupe_effects(sanitized, limit=limit)


def has_cautionary_support(item: IngredientLevelEffectSummary) -> bool:
    return any(name.casefold() in CAUTIONARY_NUTRIENTS for name in item.supporting_nutrients)


def is_cautionary_effect(item: IngredientLevelEffectSummary) -> bool:
    text = " ".join([item.effect_slug, item.effect_label, item.summary])
    if not CAUTIONARY_EFFECT_PATTERN.search(text):
        return False
    return has_cautionary_support(item) or not item.supporting_nutrients


def split_cautionary_effects(
    items: list[IngredientLevelEffectSummary],
) -> tuple[list[IngredientLevelEffectSummary], list[IngredientLevelEffectSummary]]:
    regular: list[IngredientLevelEffectSummary] = []
    cautionary: list[IngredientLevelEffectSummary] = []
    for item in items:
        if is_cautionary_effect(item):
            cautionary.append(item)
        else:
            regular.append(item)
    return regular, cautionary


def fill_supplement_supporting_nutrients(
    items: list[IngredientLevelEffectSummary],
    *,
    input_row: dict[str, Any],
) -> list[IngredientLevelEffectSummary]:
    nutrients_by_slug = supplement_nutrients_by_effect_slug(input_row)
    filled: list[IngredientLevelEffectSummary] = []
    for item in items:
        supporting = item.supporting_nutrients or nutrients_by_slug.get(item.effect_slug, [])
        filled.append(
            IngredientLevelEffectSummary(
                effect_slug=item.effect_slug,
                effect_label=item.effect_label,
                summary=item.summary,
                evidence_level=item.evidence_level,
                score=item.score,
                supporting_nutrients=supporting,
            )
        )
    return filled


def remove_seen_effects(
    items: list[IngredientLevelEffectSummary],
    seen_slugs: set[str],
) -> list[IngredientLevelEffectSummary]:
    kept: list[IngredientLevelEffectSummary] = []
    for item in items:
        key = item.effect_slug or f"{item.effect_label}:{item.summary}"
        if key in seen_slugs:
            continue
        seen_slugs.add(key)
        kept.append(item)
    return kept


def sanitize_dominant_nutrients(
    items: list[IngredientDominantNutrient],
    *,
    input_row: dict[str, Any],
) -> list[IngredientDominantNutrient]:
    source_by_name = {
        summary["canonical_bioactive_name"]: summary
        for summary in input_row.get("food_level_summaries", [])
        if summary.get("canonical_bioactive_name")
    }
    kept: list[IngredientDominantNutrient] = []
    seen: set[str] = set()
    for item in items:
        if item.canonical_bioactive_name in seen or item.canonical_bioactive_name not in source_by_name:
            continue
        source = source_by_name[item.canonical_bioactive_name]
        kept.append(
            IngredientDominantNutrient(
                canonical_bioactive_name=item.canonical_bioactive_name,
                amount_context=item.amount_context or source.get("amount_context", ""),
                dose_band=item.dose_band or source.get("dose_band", ""),
                reason=item.reason,
            )
        )
        seen.add(item.canonical_bioactive_name)
        if len(kept) >= MAX_DOMINANT_NUTRIENTS:
            break
    return kept


def sanitize_ignored_trace_nutrients(
    items: list[IngredientIgnoredTraceNutrient],
    *,
    input_row: dict[str, Any],
) -> list[IngredientIgnoredTraceNutrient]:
    source_by_name = {
        item["canonical_bioactive_name"]: item
        for item in input_row.get("ignored_trace_nutrients", [])
        if item.get("canonical_bioactive_name")
    }
    kept: list[IngredientIgnoredTraceNutrient] = []
    seen: set[str] = set()
    for item in items:
        if item.canonical_bioactive_name in seen or item.canonical_bioactive_name not in source_by_name:
            continue
        source = source_by_name[item.canonical_bioactive_name]
        kept.append(
            IngredientIgnoredTraceNutrient(
                canonical_bioactive_name=item.canonical_bioactive_name,
                amount_context=item.amount_context or source.get("amount_context", ""),
                reason=item.reason,
            )
        )
        seen.add(item.canonical_bioactive_name)
        if len(kept) >= MAX_IGNORED_TRACE_NUTRIENTS:
            break
    return kept


def sanitize_llm_summary(
    *,
    response: LlmIngredientHealthSummary,
    input_row: dict[str, Any],
) -> LlmIngredientHealthSummary:
    food_slugs = allowed_food_effect_slugs(input_row)
    supplement_slugs = allowed_supplement_effect_slugs(input_row)
    food_nutrients = allowed_food_nutrients(input_row)
    food_support_by_slug = food_nutrients_by_effect_slug(input_row)
    supplement_nutrients = allowed_supplement_nutrients(input_row)
    cautionary_food_nutrients = {
        name for name in food_nutrients if name.casefold() in CAUTIONARY_NUTRIENTS
    }
    raw_strong_evidence = sanitize_effect_bucket(
        response.strong_evidence,
        allowed_slugs=food_slugs,
        allowed_nutrients=food_nutrients,
        limit=INITIAL_EFFECT_BUCKET_LIMIT,
        nutrients_by_slug=food_support_by_slug,
    )
    raw_medium_evidence = sanitize_effect_bucket(
        response.medium_evidence,
        allowed_slugs=food_slugs,
        allowed_nutrients=food_nutrients,
        limit=INITIAL_EFFECT_BUCKET_LIMIT,
        nutrients_by_slug=food_support_by_slug,
    )
    raw_low_evidence = sanitize_effect_bucket(
        response.low_evidence,
        allowed_slugs=food_slugs,
        allowed_nutrients=food_nutrients,
        limit=INITIAL_EFFECT_BUCKET_LIMIT,
        nutrients_by_slug=food_support_by_slug,
    )
    raw_negative_or_cautionary_effects = sanitize_effect_bucket(
        response.negative_or_cautionary_effects,
        allowed_slugs=food_slugs,
        allowed_nutrients=food_nutrients | cautionary_food_nutrients,
        limit=INITIAL_EFFECT_BUCKET_LIMIT,
        nutrients_by_slug=food_support_by_slug,
    )

    strong_regular, strong_cautionary = split_cautionary_effects(raw_strong_evidence)
    medium_regular, medium_cautionary = split_cautionary_effects(raw_medium_evidence)
    low_regular, low_cautionary = split_cautionary_effects(raw_low_evidence)

    seen_food_effects: set[str] = set()
    strong_evidence = dedupe_effects(
        remove_seen_effects(strong_regular, seen_food_effects),
        limit=MAX_EFFECTS_PER_BUCKET,
    )
    medium_evidence = remove_seen_effects(
        dedupe_effects(medium_regular, limit=MAX_EFFECTS_PER_BUCKET),
        seen_food_effects,
    )
    low_evidence = remove_seen_effects(
        dedupe_effects(low_regular, limit=MAX_EFFECTS_PER_BUCKET),
        seen_food_effects,
    )
    negative_or_cautionary_effects = dedupe_effects(
        [
            *strong_cautionary,
            *medium_cautionary,
            *low_cautionary,
            *raw_negative_or_cautionary_effects,
        ],
        limit=MAX_CAUTIONARY_EFFECTS,
    )
    negative_or_cautionary_effects = remove_seen_effects(
        negative_or_cautionary_effects,
        seen_food_effects,
    )
    return LlmIngredientHealthSummary(
        strong_evidence=strong_evidence,
        medium_evidence=medium_evidence,
        low_evidence=low_evidence,
        negative_or_cautionary_effects=negative_or_cautionary_effects,
        dominant_nutrients=sanitize_dominant_nutrients(
            response.dominant_nutrients,
            input_row=input_row,
        ),
        supplement_level_only_effects=sanitize_effect_bucket(
            fill_supplement_supporting_nutrients(
                response.supplement_level_only_effects,
                input_row=input_row,
            ),
            allowed_slugs=supplement_slugs,
            allowed_nutrients=supplement_nutrients,
            limit=MAX_SUPPLEMENT_EFFECTS,
        ),
        ignored_trace_nutrients=sanitize_ignored_trace_nutrients(
            response.ignored_trace_nutrients,
            input_row=input_row,
        ),
        overall_summary=response.overall_summary,
        caveats=response.caveats[:5],
    )


def summarize_ingredient_record(
    *,
    input_row: dict[str, Any],
    provider: LlmProvider,
    model_provider: str,
    model_name: str,
    llm_attempts: int = 3,
) -> IngredientHealthSummaryRecord:
    prompt = ingredient_health_summary_prompt(input_row)
    last_error: Exception | None = None
    for _ in range(llm_attempts):
        try:
            response = provider.complete_json(prompt, LLM_INGREDIENT_SUMMARY_ADAPTER)
            response = sanitize_llm_summary(response=response, input_row=input_row)
            break
        except Exception as exc:
            last_error = exc
    else:
        if last_error is None:
            raise RuntimeError("ingredient_summary_llm_failed")
        raise last_error

    return IngredientHealthSummaryRecord(
        canonical_beverage_id=input_row["canonical_beverage_id"],
        canonical_beverage_name=input_row["canonical_beverage_name"],
        ingredient_name=input_row["ingredient_name"],
        serving_basis=input_row.get("serving_basis") or "100g",
        strong_evidence=response.strong_evidence,
        medium_evidence=response.medium_evidence,
        low_evidence=response.low_evidence,
        negative_or_cautionary_effects=response.negative_or_cautionary_effects,
        dominant_nutrients=response.dominant_nutrients,
        supplement_level_only_effects=response.supplement_level_only_effects,
        ignored_trace_nutrients=response.ignored_trace_nutrients,
        overall_summary=response.overall_summary,
        caveats=response.caveats,
        input_warnings=input_row.get("input_warnings", []),
        model_provider=model_provider,
        model_name=model_name,
    )


def summarize_results(
    *,
    total_input_rows: int,
    selected_input_rows: int,
    not_processed_due_to_limit: int,
    llm_attempts: int,
    results: list[IngredientHealthSummaryRecord],
    failures: list[IngredientHealthSummaryFailure],
) -> IngredientHealthSummaryRunSummary:
    return IngredientHealthSummaryRunSummary(
        total_input_rows=total_input_rows,
        selected_input_rows=selected_input_rows,
        not_processed_due_to_limit=not_processed_due_to_limit,
        processed_rows=len(results),
        failed_rows=len(failures),
        llm_attempts=llm_attempts,
        strong_effects=sum(len(result.strong_evidence) for result in results),
        medium_effects=sum(len(result.medium_evidence) for result in results),
        low_effects=sum(len(result.low_evidence) for result in results),
        negative_or_cautionary_effects=sum(
            len(result.negative_or_cautionary_effects) for result in results
        ),
        dominant_nutrients=sum(len(result.dominant_nutrients) for result in results),
        supplement_level_only_effects=sum(
            len(result.supplement_level_only_effects) for result in results
        ),
        ignored_trace_nutrients=sum(len(result.ignored_trace_nutrients) for result in results),
    )


def run_ingredient_health_summary_generation(
    *,
    inputs_path: Path,
    out_dir: Path,
    provider: LlmProvider,
    model_provider: str,
    model_name: str,
    workers: int = 1,
    limit: int | None = None,
    resume: bool = True,
    force: bool = False,
    llm_attempts: int = 3,
    progress: IngredientSummaryProgress | None = None,
) -> IngredientHealthSummaryRunResult:
    if workers < 1:
        raise ValueError("ingredient_summary_workers_must_be_positive")
    if workers > 30:
        raise ValueError("ingredient_summary_workers_must_be_at_most_30")
    if limit is not None and limit < 1:
        raise ValueError("ingredient_summary_limit_must_be_positive")
    if llm_attempts < 1:
        raise ValueError("ingredient_summary_llm_attempts_must_be_positive")

    all_rows = read_jsonl(inputs_path)
    selected_rows = list(all_rows)
    if limit is not None:
        selected_rows = selected_rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_results: list[IngredientHealthSummaryRecord] = []
    skipped_keys: set[str] = set()
    if resume and not force:
        result_path = out_dir / INGREDIENT_HEALTH_SUMMARIES_JSONL
        if result_path.exists():
            existing_results = [
                INGREDIENT_SUMMARY_ADAPTER.validate_python(row) for row in read_jsonl(result_path)
            ]
            skipped_keys = {summary_key(result) for result in existing_results}

    pending = [row for row in selected_rows if force or summary_key(row) not in skipped_keys]
    append_jsonl(
        out_dir / INGREDIENT_HEALTH_SUMMARIES_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "ingredient_summaries_started",
            "total_input_rows": len(all_rows),
            "selected_input_rows": len(selected_rows),
            "not_processed_due_to_limit": len(all_rows) - len(selected_rows),
            "pending": len(pending),
            "resumed_existing": len(existing_results),
            "workers": workers,
            "llm_attempts": llm_attempts,
        },
    )

    indexed_results: list[tuple[int, IngredientHealthSummaryRecord]] = []
    indexed_failures: list[tuple[int, IngredientHealthSummaryFailure]] = []

    def summarize_one(index: int, row: dict[str, Any]) -> tuple[
        int,
        IngredientHealthSummaryRecord | None,
        IngredientHealthSummaryFailure | None,
    ]:
        try:
            result = summarize_ingredient_record(
                input_row=row,
                provider=provider,
                model_provider=model_provider,
                model_name=model_name,
                llm_attempts=llm_attempts,
            )
            return index, result, None
        except Exception as exc:
            return (
                index,
                None,
                IngredientHealthSummaryFailure(
                    canonical_beverage_id=row.get("canonical_beverage_id", "unknown"),
                    canonical_beverage_name=row.get("canonical_beverage_name", ""),
                    ingredient_name=row.get("ingredient_name", ""),
                    error_type=type(exc).__name__,
                    error_message=str(exc) or type(exc).__name__,
                ),
            )

    if workers == 1:
        for index, row in enumerate(pending, start=1):
            if progress:
                progress(index, len(pending), row)
            _, result, failure = summarize_one(index, row)
            if result is not None:
                indexed_results.append((index, result))
            if failure is not None:
                indexed_failures.append((index, failure))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, row in enumerate(pending, start=1):
                if progress:
                    progress(index, len(pending), row)
                future = executor.submit(summarize_one, index, row)
                futures[future] = index
            for future in as_completed(futures):
                index, result, failure = future.result()
                if result is not None:
                    indexed_results.append((index, result))
                if failure is not None:
                    indexed_failures.append((index, failure))

    results = [
        *existing_results,
        *(result for _, result in sorted(indexed_results, key=lambda item: item[0])),
    ]
    failures = [failure for _, failure in sorted(indexed_failures, key=lambda item: item[0])]
    summary = summarize_results(
        total_input_rows=len(all_rows),
        selected_input_rows=len(selected_rows),
        not_processed_due_to_limit=len(all_rows) - len(selected_rows),
        llm_attempts=llm_attempts,
        results=results,
        failures=failures,
    )

    write_jsonl(out_dir / INGREDIENT_HEALTH_SUMMARIES_JSONL, results)
    write_jsonl(out_dir / INGREDIENT_HEALTH_SUMMARIES_FAILURES_JSONL, failures)
    atomic_write(
        out_dir / INGREDIENT_HEALTH_SUMMARIES_SUMMARY_JSON,
        orjson.dumps(
            summary.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        ),
    )
    append_jsonl(
        out_dir / INGREDIENT_HEALTH_SUMMARIES_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "ingredient_summaries_completed",
            "processed": len(indexed_results),
            "failures": len(failures),
            "selected_input_rows": summary.selected_input_rows,
            "not_processed_due_to_limit": summary.not_processed_due_to_limit,
            "strong_effects": summary.strong_effects,
            "medium_effects": summary.medium_effects,
            "low_effects": summary.low_effects,
            "negative_or_cautionary_effects": summary.negative_or_cautionary_effects,
            "supplement_level_only_effects": summary.supplement_level_only_effects,
        },
    )

    return IngredientHealthSummaryRunResult(
        out_dir=out_dir,
        results=results,
        failures=failures,
        summary=summary,
    )
