from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import orjson
from pydantic import TypeAdapter

from sipz_agent.core.artifacts import model_dump_jsonable
from sipz_agent.core.models import LlmProvider
from sipz_agent.schemas.composition import (
    IngredientNutrimentProfile,
    IngredientNutrimentSignificanceResult,
    LlmNutrimentSignificanceResponse,
    NormalizedNutrimentProfile,
    NutrimentSignificanceClassification,
    NutrimentSignificanceFailure,
    NutrimentSignificanceRunSummary,
    ReferenceIntakeRow,
)


LLM_SIGNIFICANCE_RESPONSE_ADAPTER = TypeAdapter(LlmNutrimentSignificanceResponse)
INGREDIENT_PROFILE_ADAPTER = TypeAdapter(IngredientNutrimentProfile)
REFERENCE_ROW_ADAPTER = TypeAdapter(ReferenceIntakeRow)
SIGNIFICANCE_RESULT_ADAPTER = TypeAdapter(IngredientNutrimentSignificanceResult)

SIGNIFICANCE_JSONL = "nutriment_significance.jsonl"
SIGNIFICANCE_FLAT_CSV = "nutriment_significance_flat.csv"
SIGNIFICANCE_SUMMARY_JSON = "nutriment_significance_summary.json"
SIGNIFICANCE_FAILURES_JSONL = "nutriment_significance_failures.jsonl"
SIGNIFICANCE_BATCH_LOG_JSONL = "significance_batch_log.jsonl"


SignificanceProgress = Callable[[int, int, IngredientNutrimentProfile], None]


@dataclass(frozen=True)
class NutrimentSignificanceRunResult:
    out_dir: Path
    results: list[IngredientNutrimentSignificanceResult]
    failures: list[NutrimentSignificanceFailure]
    summary: NutrimentSignificanceRunSummary


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def read_ingredient_profiles(path: Path, *, limit: int | None = None) -> list[IngredientNutrimentProfile]:
    profiles = [INGREDIENT_PROFILE_ADAPTER.validate_python(row) for row in read_jsonl(path)]
    if limit is not None:
        return profiles[:limit]
    return profiles


def read_reference_table(path: Path) -> dict[str, ReferenceIntakeRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"canonical_bioactive_name"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("reference_intake_columns_missing:" + ",".join(missing))
        rows = [REFERENCE_ROW_ADAPTER.validate_python(row) for row in reader]
    return {row.canonical_bioactive_name.casefold(): row for row in rows}


def amount_context(nutriment: NormalizedNutrimentProfile) -> str:
    unit = nutriment.display_unit or nutriment.raw_unit or "per 100g"
    return f"{nutriment.display_amount:g} {unit}"


def reference_for_nutriment(
    nutriment: NormalizedNutrimentProfile,
    reference_rows: dict[str, ReferenceIntakeRow],
) -> ReferenceIntakeRow | None:
    return reference_rows.get(nutriment.canonical_bioactive_name.casefold())


def reference_value(
    nutriment: NormalizedNutrimentProfile,
    reference: ReferenceIntakeRow | None,
    field: str,
) -> str:
    if reference is not None:
        return str(getattr(reference, field))
    value = getattr(nutriment, field, "")
    return "" if value is None else str(value)


def skipped_quality_classification(
    nutriment: NormalizedNutrimentProfile,
    reference: ReferenceIntakeRow | None,
) -> NutrimentSignificanceClassification:
    flags = list(nutriment.amount_quality_flags)
    if nutriment.amount_quality_status and nutriment.amount_quality_status != "ok":
        flags.append(f"amount_quality_status:{nutriment.amount_quality_status}")
    flag_text = ", ".join(flags) if flags else "source amount quality flag"
    return NutrimentSignificanceClassification(
        source_key=nutriment.source_key,
        canonical_bioactive_name=nutriment.canonical_bioactive_name,
        raw_amount=nutriment.raw_amount,
        raw_unit=nutriment.raw_unit,
        display_amount=nutriment.display_amount,
        display_unit=nutriment.display_unit,
        amount_context=amount_context(nutriment),
        reference_amount=reference_value(nutriment, reference, "reference_amount"),
        reference_unit=reference_value(nutriment, reference, "reference_unit"),
        reference_type=reference_value(nutriment, reference, "reference_type"),
        caution_limit=reference_value(nutriment, reference, "caution_limit"),
        caution_limit_unit=reference_value(nutriment, reference, "caution_limit_unit"),
        significance="unknown_threshold",
        confidence=1.0,
        reasoning=(
            "Skipped before LLM classification because the source amount is implausible "
            f"or needs source-data review: {flag_text}."
        ),
        classification_status="skipped_amount_quality_flag",
        used_llm=False,
        amount_quality_flags=flags,
    )


def missing_reference_classification(
    nutriment: NormalizedNutrimentProfile,
) -> NutrimentSignificanceClassification:
    return NutrimentSignificanceClassification(
        source_key=nutriment.source_key,
        canonical_bioactive_name=nutriment.canonical_bioactive_name,
        raw_amount=nutriment.raw_amount,
        raw_unit=nutriment.raw_unit,
        display_amount=nutriment.display_amount,
        display_unit=nutriment.display_unit,
        amount_context=amount_context(nutriment),
        reference_amount="",
        reference_unit="",
        reference_type="",
        caution_limit="",
        caution_limit_unit="",
        significance="unknown_threshold",
        confidence=1.0,
        reasoning=(
            "Skipped before LLM classification because no reference-intake context was "
            "available from the profile or reference table."
        ),
        classification_status="missing_reference_context",
        used_llm=False,
        amount_quality_flags=[],
    )


def failed_llm_classification(
    nutriment: NormalizedNutrimentProfile,
    reference: ReferenceIntakeRow | None,
    *,
    reason: str,
) -> NutrimentSignificanceClassification:
    return NutrimentSignificanceClassification(
        source_key=nutriment.source_key,
        canonical_bioactive_name=nutriment.canonical_bioactive_name,
        raw_amount=nutriment.raw_amount,
        raw_unit=nutriment.raw_unit,
        display_amount=nutriment.display_amount,
        display_unit=nutriment.display_unit,
        amount_context=amount_context(nutriment),
        reference_amount=reference_value(nutriment, reference, "reference_amount"),
        reference_unit=reference_value(nutriment, reference, "reference_unit"),
        reference_type=reference_value(nutriment, reference, "reference_type"),
        caution_limit=reference_value(nutriment, reference, "caution_limit"),
        caution_limit_unit=reference_value(nutriment, reference, "caution_limit_unit"),
        significance="unknown_threshold",
        confidence=0.0,
        reasoning=reason,
        classification_status="failed_llm_classification",
        used_llm=True,
        amount_quality_flags=[],
    )


def nutriment_interpretation_guidance(nutriment: NormalizedNutrimentProfile) -> list[str]:
    guidance: list[str] = []
    canonical_name = nutriment.canonical_bioactive_name.casefold()
    source_key = nutriment.source_key.casefold()
    display_unit = nutriment.display_unit.casefold()

    if canonical_name == "sugars":
        if source_key == "sugars_100g":
            guidance.append(
                "This row is total sugars, not confirmed added sugars. Do not return "
                "unknown_threshold only because the caution reference is for added sugars; "
                "classify by sugar concentration and mention added-sugar uncertainty in "
                "reasoning or confidence."
            )
        elif source_key == "added-sugars_100g":
            guidance.append(
                "This row is added sugars, so the added-sugar caution reference is directly "
                "applicable."
            )
    if canonical_name == "caffeine":
        guidance.append(
            "Classify caffeine consistently by concentration and plausible serving "
            "accumulation. Around 25-35 mg/100g is usually minor for a 100g serving, "
            "not significant, unless the ingredient is normally consumed in a much larger "
            "serving or the amount is closer to a clearly high caffeine dose."
        )
    if canonical_name == "vitamin a and carotenoids":
        if "iu/" in display_unit:
            guidance.append(
                "Vitamin A IU values are not directly comparable to mcg RAE without conversion; "
                "use unknown_threshold or lower confidence unless context makes significance clear."
            )
        elif "ug/" in display_unit or "mcg/" in display_unit:
            guidance.append(
                "Treat ug or mcg vitamin A values as RAE-like significance context from the "
                "normalized profile. Do not return unknown_threshold solely because carotenoid "
                "conversion can vary."
            )
    return guidance


def nutriment_prompt_payload(
    nutriment: NormalizedNutrimentProfile,
    reference: ReferenceIntakeRow | None,
) -> dict[str, object]:
    return {
        "source_key": nutriment.source_key,
        "canonical_bioactive_name": nutriment.canonical_bioactive_name,
        "raw_amount": nutriment.raw_amount,
        "raw_unit": nutriment.raw_unit,
        "display_amount": nutriment.display_amount,
        "display_unit": nutriment.display_unit,
        "amount_context": amount_context(nutriment),
        "reference_row": {
            "reference_amount": reference_value(nutriment, reference, "reference_amount"),
            "reference_unit": reference_value(nutriment, reference, "reference_unit"),
            "reference_type": reference_value(nutriment, reference, "reference_type"),
            "caution_limit": reference_value(nutriment, reference, "caution_limit"),
            "caution_limit_unit": reference_value(nutriment, reference, "caution_limit_unit"),
            "threshold_notes": reference.threshold_notes if reference is not None else "",
            "source": reference.source if reference is not None else "",
            "review_status": reference.review_status if reference is not None else "",
        },
        "interpretation_guidance": nutriment_interpretation_guidance(nutriment),
    }


def significance_prompt(
    *,
    profile: IngredientNutrimentProfile,
    nutriments: list[NormalizedNutrimentProfile],
    reference_rows: dict[str, ReferenceIntakeRow],
) -> str:
    payload = {
        "ingredient": {
            "canonical_beverage_id": profile.canonical_beverage_id,
            "name": profile.canonical_beverage_name,
            "serving_basis": profile.serving_basis or "100g",
        },
        "nutriments": [
            nutriment_prompt_payload(
                nutriment,
                reference_for_nutriment(nutriment, reference_rows),
            )
            for nutriment in nutriments
        ],
    }
    return (
        "Classify whether each listed nutriment amount is nutritionally meaningful for "
        "human health at a 100g serving.\n\n"
        "Use health-normalized dosage, not raw gram size. For example, 1 g protein is "
        "usually minor, 30 mg vitamin C can be meaningful, and 1 g vitamin C would be "
        "extremely high. Treat sodium, salt, saturated fat, sugars, caffeine, and alcohol "
        "as cautionary nutrients: they can be significant because the amount may be "
        "health-relevant, not because they are beneficial targets. Use the reference row "
        "as context, not as a deterministic rule.\n\n"
        "Soft calibration, not hard rules: for beneficial daily-value or adequate-intake "
        "nutrients, under about 2% of reference intake is usually trace, about 2-10% is "
        "usually minor, and about 10% or more is usually significant for micronutrients "
        "unless context argues otherwise. For caution nutrients, about 10% or more of a "
        "caution limit in 100g is usually significant; lower amounts can be minor or trace "
        "depending on concentration and serving context. Avoid marking values below 10% "
        "of a caution limit as significant solely because the ingredient is a beverage or "
        "could be consumed in a larger serving; use minor unless the absolute concentration "
        "is clearly high for that nutriment. For added sugars specifically, values just "
        "under 10% of the 50 g/day added-sugar context, such as 4-5 g/100g, are usually "
        "minor, while values at or above about 10% are usually significant. Macronutrients "
        "should be judged by health relevance and energy/concentration context; a low "
        "protein amount should not dominate an ingredient summary.\n\n"
        "Use unknown_threshold only when the amount cannot be reasonably interpreted from "
        "the available unit/reference context. Do not use unknown_threshold merely because "
        "total sugars are not confirmed added sugars; classify the concentration as "
        "cautionary and mention the uncertainty. Do not infer specific health effects in "
        "this step.\n\n"
        "Allowed significance values: significant, minor, trace, unknown_threshold.\n"
        "Return JSON with exactly this top-level shape:\n"
        "{\n"
        '  "classifications": [\n'
        "    {\n"
        '      "source_key": "vitamin-c_100g",\n'
        '      "canonical_bioactive_name": "Vitamin C",\n'
        '      "significance": "significant",\n'
        '      "confidence": 0.88,\n'
        '      "reasoning": "30 mg vitamin C per 100g is a meaningful fraction of adult daily intake.",\n'
        '      "amount_context": "30 mg/100g"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Input:\n"
        + orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
    )


def llm_classification_to_result(
    *,
    classification_source_key: str,
    classification_name: str,
    significance: str,
    confidence: float,
    reasoning: str,
    amount_context_text: str,
    nutriment: NormalizedNutrimentProfile,
    reference: ReferenceIntakeRow | None,
) -> NutrimentSignificanceClassification:
    return NutrimentSignificanceClassification(
        source_key=classification_source_key,
        canonical_bioactive_name=classification_name,
        raw_amount=nutriment.raw_amount,
        raw_unit=nutriment.raw_unit,
        display_amount=nutriment.display_amount,
        display_unit=nutriment.display_unit,
        amount_context=amount_context_text,
        reference_amount=reference_value(nutriment, reference, "reference_amount"),
        reference_unit=reference_value(nutriment, reference, "reference_unit"),
        reference_type=reference_value(nutriment, reference, "reference_type"),
        caution_limit=reference_value(nutriment, reference, "caution_limit"),
        caution_limit_unit=reference_value(nutriment, reference, "caution_limit_unit"),
        significance=significance,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
        classification_status="classified",
        used_llm=True,
        amount_quality_flags=[],
    )


def classify_ingredient_significance(
    *,
    profile: IngredientNutrimentProfile,
    reference_rows: dict[str, ReferenceIntakeRow],
    provider: LlmProvider,
) -> IngredientNutrimentSignificanceResult:
    preclassified: dict[str, NutrimentSignificanceClassification] = {}
    eligible: list[NormalizedNutrimentProfile] = []
    for nutriment in profile.nutriments:
        reference = reference_for_nutriment(nutriment, reference_rows)
        if nutriment.amount_quality_flags or nutriment.amount_quality_status != "ok":
            preclassified[nutriment.source_key] = skipped_quality_classification(nutriment, reference)
        elif reference is None and not nutriment.reference_type:
            preclassified[nutriment.source_key] = missing_reference_classification(nutriment)
        else:
            eligible.append(nutriment)

    classified: dict[str, NutrimentSignificanceClassification] = {}
    if eligible:
        response = provider.complete_json(
            significance_prompt(
                profile=profile,
                nutriments=eligible,
                reference_rows=reference_rows,
            ),
            LLM_SIGNIFICANCE_RESPONSE_ADAPTER,
        )
        by_key = {nutriment.source_key: nutriment for nutriment in eligible}
        returned_keys = {classification.source_key for classification in response.classifications}
        missing = sorted(set(by_key) - returned_keys)
        extra = sorted(returned_keys - set(by_key))
        for source_key in missing:
            nutriment = by_key[source_key]
            classified[source_key] = failed_llm_classification(
                nutriment,
                reference_for_nutriment(nutriment, reference_rows),
                reason=(
                    "The LLM response omitted this nutriment, so it was preserved as "
                    "unknown_threshold instead of failing the full ingredient."
                ),
            )
        for classification in response.classifications:
            if classification.source_key in extra:
                continue
            nutriment = by_key[classification.source_key]
            reference = reference_for_nutriment(nutriment, reference_rows)
            classified[classification.source_key] = llm_classification_to_result(
                classification_source_key=classification.source_key,
                classification_name=classification.canonical_bioactive_name,
                significance=classification.significance,
                confidence=classification.confidence,
                reasoning=classification.reasoning,
                amount_context_text=classification.amount_context,
                nutriment=nutriment,
                reference=reference,
            )

    ordered_classifications = [
        classified.get(nutriment.source_key) or preclassified[nutriment.source_key]
        for nutriment in profile.nutriments
        if nutriment.source_key in classified or nutriment.source_key in preclassified
    ]

    return IngredientNutrimentSignificanceResult(
        canonical_beverage_id=profile.canonical_beverage_id,
        ingredient_name=profile.canonical_beverage_name,
        serving_basis=profile.serving_basis or "100g",
        classifications=ordered_classifications,
    )


def write_jsonl(path: Path, rows: list[object]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    atomic_write(path, payload)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


FLAT_FIELDS = [
    "canonical_beverage_id",
    "ingredient_name",
    "source_key",
    "canonical_bioactive_name",
    "raw_amount",
    "raw_unit",
    "display_amount",
    "display_unit",
    "amount_context",
    "reference_amount",
    "reference_unit",
    "reference_type",
    "caution_limit",
    "caution_limit_unit",
    "significance",
    "confidence",
    "classification_status",
    "used_llm",
    "amount_quality_flags",
    "reasoning",
]


def write_flat_csv(path: Path, results: list[IngredientNutrimentSignificanceResult]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_FIELDS)
        writer.writeheader()
        for result in results:
            for classification in result.classifications:
                row = classification.model_dump(mode="json")
                writer.writerow(
                    {
                        "canonical_beverage_id": result.canonical_beverage_id,
                        "ingredient_name": result.ingredient_name,
                        "source_key": row["source_key"],
                        "canonical_bioactive_name": row["canonical_bioactive_name"],
                        "raw_amount": row["raw_amount"],
                        "raw_unit": row["raw_unit"],
                        "display_amount": row["display_amount"],
                        "display_unit": row["display_unit"],
                        "amount_context": row["amount_context"],
                        "reference_amount": row["reference_amount"],
                        "reference_unit": row["reference_unit"],
                        "reference_type": row["reference_type"],
                        "caution_limit": row["caution_limit"],
                        "caution_limit_unit": row["caution_limit_unit"],
                        "significance": row["significance"],
                        "confidence": row["confidence"],
                        "classification_status": row["classification_status"],
                        "used_llm": row["used_llm"],
                        "amount_quality_flags": ";".join(row["amount_quality_flags"]),
                        "reasoning": row["reasoning"],
                    }
                )
    temporary.replace(path)


def summarize_results(
    *,
    total_ingredients: int,
    results: list[IngredientNutrimentSignificanceResult],
    failures: list[NutrimentSignificanceFailure],
) -> NutrimentSignificanceRunSummary:
    classifications = [
        classification for result in results for classification in result.classifications
    ]
    return NutrimentSignificanceRunSummary(
        total_ingredients=total_ingredients,
        processed_ingredients=len(results),
        failed_ingredients=len(failures),
        classified_nutriments=sum(1 for item in classifications if item.used_llm),
        skipped_invalid_nutriments=sum(
            1
            for item in classifications
            if item.classification_status == "skipped_amount_quality_flag"
        ),
        significant=sum(1 for item in classifications if item.significance == "significant"),
        minor=sum(1 for item in classifications if item.significance == "minor"),
        trace=sum(1 for item in classifications if item.significance == "trace"),
        unknown_threshold=sum(
            1 for item in classifications if item.significance == "unknown_threshold"
        ),
    )


def run_nutriment_significance_classification(
    *,
    profiles_path: Path,
    reference_path: Path,
    out_dir: Path,
    provider: LlmProvider,
    workers: int = 1,
    limit: int | None = None,
    resume: bool = True,
    force: bool = False,
    progress: SignificanceProgress | None = None,
) -> NutrimentSignificanceRunResult:
    if workers < 1:
        raise ValueError("significance_workers_must_be_positive")
    if workers > 10:
        raise ValueError("significance_workers_must_be_at_most_10")
    if limit is not None and limit < 1:
        raise ValueError("significance_limit_must_be_positive")

    profiles = read_ingredient_profiles(profiles_path, limit=limit)
    reference_rows = read_reference_table(reference_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_results: list[IngredientNutrimentSignificanceResult] = []
    skipped_ids: set[str] = set()
    if resume and not force:
        result_path = out_dir / SIGNIFICANCE_JSONL
        if result_path.exists():
            existing_results = [
                SIGNIFICANCE_RESULT_ADAPTER.validate_python(row) for row in read_jsonl(result_path)
            ]
            skipped_ids = {result.canonical_beverage_id for result in existing_results}

    pending = [profile for profile in profiles if force or profile.canonical_beverage_id not in skipped_ids]
    append_jsonl(
        out_dir / SIGNIFICANCE_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "nutriment_significance_started",
            "profiles": len(profiles),
            "pending": len(pending),
            "resumed_existing": len(existing_results),
            "workers": workers,
        },
    )

    indexed_results: list[tuple[int, IngredientNutrimentSignificanceResult]] = []
    indexed_failures: list[tuple[int, NutrimentSignificanceFailure]] = []

    def classify_one(index: int, profile: IngredientNutrimentProfile) -> tuple[
        int,
        IngredientNutrimentSignificanceResult | None,
        NutrimentSignificanceFailure | None,
    ]:
        try:
            result = classify_ingredient_significance(
                profile=profile,
                reference_rows=reference_rows,
                provider=provider,
            )
            return index, result, None
        except Exception as exc:
            return (
                index,
                None,
                NutrimentSignificanceFailure(
                    canonical_beverage_id=profile.canonical_beverage_id,
                    ingredient_name=profile.canonical_beverage_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or type(exc).__name__,
                ),
            )

    if workers == 1:
        for index, profile in enumerate(pending, start=1):
            if progress:
                progress(index, len(pending), profile)
            _, result, failure = classify_one(index, profile)
            if result is not None:
                indexed_results.append((index, result))
            if failure is not None:
                indexed_failures.append((index, failure))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, profile in enumerate(pending, start=1):
                if progress:
                    progress(index, len(pending), profile)
                future = executor.submit(classify_one, index, profile)
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
        total_ingredients=len(profiles),
        results=results,
        failures=failures,
    )

    write_jsonl(out_dir / SIGNIFICANCE_JSONL, results)
    write_flat_csv(out_dir / SIGNIFICANCE_FLAT_CSV, results)
    write_jsonl(out_dir / SIGNIFICANCE_FAILURES_JSONL, failures)
    atomic_write(
        out_dir / SIGNIFICANCE_SUMMARY_JSON,
        orjson.dumps(
            summary.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        ),
    )
    append_jsonl(
        out_dir / SIGNIFICANCE_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "nutriment_significance_completed",
            "processed": len(indexed_results),
            "failures": len(failures),
            "significant": summary.significant,
            "minor": summary.minor,
            "trace": summary.trace,
            "unknown_threshold": summary.unknown_threshold,
        },
    )

    return NutrimentSignificanceRunResult(
        out_dir=out_dir,
        results=results,
        failures=failures,
        summary=summary,
    )
