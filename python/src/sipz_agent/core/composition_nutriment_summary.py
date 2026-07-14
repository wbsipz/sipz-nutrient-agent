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
    LlmNutrimentHealthSummary,
    NutrimentEvidenceEffect,
    NutrimentEvidenceMatchRecord,
    NutrimentHealthSummaryFailure,
    NutrimentHealthSummaryRecord,
    NutrimentHealthSummaryRunSummary,
    ReferenceIntakeRow,
)


EVIDENCE_MATCH_ADAPTER = TypeAdapter(NutrimentEvidenceMatchRecord)
LLM_NUTRIMENT_SUMMARY_ADAPTER = TypeAdapter(LlmNutrimentHealthSummary)
NUTRIMENT_SUMMARY_ADAPTER = TypeAdapter(NutrimentHealthSummaryRecord)
REFERENCE_ROW_ADAPTER = TypeAdapter(ReferenceIntakeRow)

NUTRIMENT_SUMMARIES_JSONL = "nutriment_summaries.jsonl"
NUTRIMENT_SUMMARIES_FAILURES_JSONL = "nutriment_summaries_failures.jsonl"
NUTRIMENT_SUMMARIES_SUMMARY_JSON = "nutriment_summaries_summary.json"
NUTRIMENT_SUMMARIES_BATCH_LOG_JSONL = "nutriment_summaries_batch_log.jsonl"
MAX_FOOD_LEVEL_EFFECTS_PER_BUCKET = 3
MAX_SUPPLEMENT_LEVEL_EFFECTS = 5
SUMMARY_EFFECT_BUCKETS = [
    "strong_evidence_effects",
    "medium_evidence_effects",
    "low_evidence_effects",
    "supplement_level_relevance",
]

NutrimentSummaryProgress = Callable[[int, int, NutrimentEvidenceMatchRecord], None]


@dataclass(frozen=True)
class NutrimentHealthSummaryRunResult:
    out_dir: Path
    results: list[NutrimentHealthSummaryRecord]
    failures: list[NutrimentHealthSummaryFailure]
    summary: NutrimentHealthSummaryRunSummary


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


def write_jsonl(path: Path, rows: list[object]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    atomic_write(path, payload)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def read_evidence_match_records(path: Path) -> list[NutrimentEvidenceMatchRecord]:
    return [EVIDENCE_MATCH_ADAPTER.validate_python(row) for row in read_jsonl(path)]


def read_reference_table(path: Path) -> dict[str, ReferenceIntakeRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"canonical_bioactive_name"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("reference_intake_columns_missing:" + ",".join(missing))
        rows = [REFERENCE_ROW_ADAPTER.validate_python(row) for row in reader]
    return {row.canonical_bioactive_name.casefold(): row for row in rows}


def summary_key(record: NutrimentEvidenceMatchRecord | NutrimentHealthSummaryRecord) -> str:
    return f"{record.canonical_beverage_id}\t{record.source_key}"


def reference_for_record(
    record: NutrimentEvidenceMatchRecord,
    reference_rows: dict[str, ReferenceIntakeRow],
) -> ReferenceIntakeRow | None:
    return reference_rows.get(record.canonical_bioactive_name.casefold())


def evidence_sort_key(effect: NutrimentEvidenceEffect) -> tuple[int, float]:
    rank = {
        "strong": 4,
        "high": 4,
        "moderate": 3,
        "medium": 3,
        "limited": 2,
        "low": 2,
        "insufficient": 1,
    }.get(effect.evidence_level.casefold(), 0)
    return rank, effect.score if effect.score is not None else -1.0


def evidence_payload(effect: NutrimentEvidenceEffect) -> dict[str, object]:
    return {
        "effect_slug": effect.effect_slug,
        "effect_label": effect.effect_label,
        "description": effect.description,
        "score": effect.score,
        "evidence_level": effect.evidence_level,
        "tags": effect.tags,
        "sources": effect.sources,
    }


def reference_payload(
    record: NutrimentEvidenceMatchRecord,
    reference: ReferenceIntakeRow | None,
) -> dict[str, object]:
    return {
        "reference_amount": reference.reference_amount if reference else record.reference_amount,
        "reference_unit": reference.reference_unit if reference else record.reference_unit,
        "reference_type": reference.reference_type if reference else record.reference_type,
        "caution_limit": reference.caution_limit if reference else record.caution_limit,
        "caution_limit_unit": reference.caution_limit_unit if reference else record.caution_limit_unit,
        "threshold_notes": reference.threshold_notes if reference else "",
        "source": reference.source if reference else "",
        "review_status": reference.review_status if reference else "",
    }


def valid_effect_slugs(record: NutrimentEvidenceMatchRecord) -> set[str]:
    return {effect.effect_slug for effect in record.evidence_matches if effect.effect_slug}


def filter_effect_summaries(items: list, allowed_slugs: set[str], *, limit: int) -> list:
    filtered = [item for item in items if item.effect_slug in allowed_slugs]
    return filtered[:limit]


def dedupe_effect_summary_buckets(
    *,
    strong_evidence_effects: list,
    medium_evidence_effects: list,
    low_evidence_effects: list,
    supplement_level_relevance: list,
) -> dict[str, list]:
    seen_slugs: set[str] = set()
    deduped: dict[str, list] = {}
    for bucket_name, items in [
        ("strong_evidence_effects", strong_evidence_effects),
        ("medium_evidence_effects", medium_evidence_effects),
        ("low_evidence_effects", low_evidence_effects),
        ("supplement_level_relevance", supplement_level_relevance),
    ]:
        kept = []
        for item in items:
            if item.effect_slug in seen_slugs:
                continue
            seen_slugs.add(item.effect_slug)
            kept.append(item)
        deduped[bucket_name] = kept
    return deduped


def sanitize_llm_summary(
    *,
    response: LlmNutrimentHealthSummary,
    record: NutrimentEvidenceMatchRecord,
) -> LlmNutrimentHealthSummary:
    allowed_slugs = valid_effect_slugs(record)
    filtered_buckets = dedupe_effect_summary_buckets(
        strong_evidence_effects=filter_effect_summaries(
            response.strong_evidence_effects,
            allowed_slugs,
            limit=MAX_FOOD_LEVEL_EFFECTS_PER_BUCKET,
        ),
        medium_evidence_effects=filter_effect_summaries(
            response.medium_evidence_effects,
            allowed_slugs,
            limit=MAX_FOOD_LEVEL_EFFECTS_PER_BUCKET,
        ),
        low_evidence_effects=filter_effect_summaries(
            response.low_evidence_effects,
            allowed_slugs,
            limit=MAX_FOOD_LEVEL_EFFECTS_PER_BUCKET,
        ),
        supplement_level_relevance=filter_effect_summaries(
            response.supplement_level_relevance,
            allowed_slugs,
            limit=MAX_SUPPLEMENT_LEVEL_EFFECTS,
        ),
    )
    return LlmNutrimentHealthSummary(
        canonical_bioactive_name=response.canonical_bioactive_name,
        amount_context=response.amount_context,
        food_level_relevance=response.food_level_relevance,
        strong_evidence_effects=filtered_buckets["strong_evidence_effects"],
        medium_evidence_effects=filtered_buckets["medium_evidence_effects"],
        low_evidence_effects=filtered_buckets["low_evidence_effects"],
        supplement_level_relevance=filtered_buckets["supplement_level_relevance"],
        caveats=response.caveats[:5],
    )


def nutriment_summary_prompt(
    *,
    record: NutrimentEvidenceMatchRecord,
    reference_rows: dict[str, ReferenceIntakeRow],
) -> str:
    sorted_effects = sorted(record.evidence_matches, key=evidence_sort_key, reverse=True)
    payload = {
        "ingredient": {
            "canonical_beverage_id": record.canonical_beverage_id,
            "name": record.ingredient_name,
            "serving_basis": record.serving_basis or "100g",
        },
        "nutriment": {
            "source_key": record.source_key,
            "canonical_bioactive_name": record.canonical_bioactive_name,
            "raw_amount": record.raw_amount,
            "raw_unit": record.raw_unit,
            "display_amount": record.display_amount,
            "display_unit": record.display_unit,
            "amount_context": record.amount_context,
            "reference_row": reference_payload(
                record,
                reference_for_record(record, reference_rows),
            ),
            "significance_classifier": {
                "significance": record.significance,
                "confidence": record.significance_confidence,
                "reasoning": record.significance_reasoning,
                "classification_status": record.classification_status,
            },
        },
        "matched_health_effect_rows": [evidence_payload(effect) for effect in sorted_effects],
        "allowed_effect_slugs": [effect.effect_slug for effect in sorted_effects],
    }
    return (
        "Create a nutriment-level health summary for one ingredient at a 100g serving.\n\n"
        "Use only the matched health-effect rows supplied in the input. Do not invent "
        "health effects. Do not imply the ingredient itself has clinical trial evidence "
        "unless an evidence row is specifically about this ingredient. Every output "
        "effect_slug must be copied exactly from allowed_effect_slugs. Do not create, "
        "combine, rename, or paraphrase effect slugs. If no supplied slug fits a summary, "
        "omit that summary.\n\n"
        "Separate food-level relevance from supplement-level relevance. Food-level "
        "relevance means the ingredient's listed amount per 100g plausibly supports the "
        "effect or caution. Supplement-level relevance means the effect is known for the "
        "nutriment, but the evidence description appears to involve supplemental, "
        "isolated, therapeutic, or much higher doses than this 100g ingredient amount. "
        "Down-rank or move effects to supplement_level_relevance only when the evidence "
        "requires a meaningfully higher dose or isolated supplement form. Do not use "
        "supplement_level_relevance for inverse, comparator, absence, or restriction "
        "effects that do not apply to consuming this ingredient amount; omit those or "
        "mention them briefly as caveats.\n\n"
        "Order effects from strongest to weakest evidence using evidence_level and score, "
        "then adjust for dose relevance. Map strong/high evidence to "
        "strong_evidence_effects, moderate/medium to medium_evidence_effects, and "
        "limited/low/uncertain to low_evidence_effects unless dose relevance requires "
        "supplement_level_relevance instead. Return at most 3 effects in each food-level "
        "bucket and at most 5 supplement-level effects. Prefer fewer, higher-signal "
        "effects over exhaustive coverage. Consolidate duplicate rows only by selecting "
        "one original input effect_slug; never create a new combined slug.\n\n"
        "For cautionary nutriments such as sugars, sodium, salt, saturated fat, caffeine, "
        "alcohol, and cholesterol, summarize supported adverse or cautionary relevance "
        "plainly. If the significance classifier says a cautionary nutriment amount is "
        "meaningful at 100g, evidence about habitual intake, blood pressure, "
        "cardiometabolic risk, dental caries, LDL cholesterol, or intoxication can be "
        "food-level relevance when it matches the nutriment and amount direction. Keep "
        "all strings concise.\n\n"
        "Return JSON with exactly this top-level shape:\n"
        "{\n"
        '  "canonical_bioactive_name": "Vitamin C",\n'
        '  "amount_context": "30 mg/100g",\n'
        '  "food_level_relevance": "At this concentration, vitamin C may meaningfully contribute to normal immune and antioxidant-related nutrition.",\n'
        '  "strong_evidence_effects": [\n'
        "    {\n"
        '      "effect_slug": "immune_support",\n'
        '      "effect_label": "Immune support",\n'
        '      "summary": "Brief effect summary at this ingredient amount.",\n'
        '      "evidence_level": "strong",\n'
        '      "score": 0.8\n'
        "    }\n"
        "  ],\n"
        '  "medium_evidence_effects": [],\n'
        '  "low_evidence_effects": [],\n'
        '  "supplement_level_relevance": [\n'
        "    {\n"
        '      "effect_slug": "example_effect",\n'
        '      "effect_label": "Example effect",\n'
        '      "summary": "The nutriment has evidence for this effect at higher supplemental doses, but this ingredient amount alone is not enough to support it.",\n'
        '      "evidence_level": "limited",\n'
        '      "score": 0.45\n'
        "    }\n"
        "  ],\n"
        '  "caveats": ["Brief caveat."]\n'
        "}\n\n"
        "Input:\n"
        + orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
    )


def summarize_nutriment_record(
    *,
    record: NutrimentEvidenceMatchRecord,
    reference_rows: dict[str, ReferenceIntakeRow],
    provider: LlmProvider,
    model_provider: str,
    model_name: str,
    llm_attempts: int = 3,
) -> NutrimentHealthSummaryRecord:
    prompt = nutriment_summary_prompt(record=record, reference_rows=reference_rows)
    last_error: Exception | None = None
    for _ in range(llm_attempts):
        try:
            response = provider.complete_json(prompt, LLM_NUTRIMENT_SUMMARY_ADAPTER)
            response = sanitize_llm_summary(response=response, record=record)
            break
        except Exception as exc:
            last_error = exc
    else:
        if last_error is None:
            raise RuntimeError("nutriment_summary_llm_failed")
        raise last_error

    return NutrimentHealthSummaryRecord(
        canonical_beverage_id=record.canonical_beverage_id,
        ingredient_name=record.ingredient_name,
        serving_basis=record.serving_basis,
        source_key=record.source_key,
        canonical_bioactive_name=record.canonical_bioactive_name,
        raw_amount=record.raw_amount,
        raw_unit=record.raw_unit,
        display_amount=record.display_amount,
        display_unit=record.display_unit,
        amount_context=record.amount_context,
        reference_amount=record.reference_amount,
        reference_unit=record.reference_unit,
        reference_type=record.reference_type,
        caution_limit=record.caution_limit,
        caution_limit_unit=record.caution_limit_unit,
        significance=record.significance,
        significance_confidence=record.significance_confidence,
        significance_reasoning=record.significance_reasoning,
        evidence_match_count=record.match_count,
        food_level_relevance=response.food_level_relevance,
        strong_evidence_effects=response.strong_evidence_effects,
        medium_evidence_effects=response.medium_evidence_effects,
        low_evidence_effects=response.low_evidence_effects,
        supplement_level_relevance=response.supplement_level_relevance,
        caveats=response.caveats,
        model_provider=model_provider,
        model_name=model_name,
    )


def summarize_results(
    *,
    total_evidence_match_rows: int,
    eligible_matched_rows: int,
    selected_matched_rows: int,
    not_processed_due_to_limit: int,
    skipped_unmatched_rows: int,
    skipped_empty_summaries: int,
    llm_attempts: int,
    results: list[NutrimentHealthSummaryRecord],
    failures: list[NutrimentHealthSummaryFailure],
) -> NutrimentHealthSummaryRunSummary:
    return NutrimentHealthSummaryRunSummary(
        total_evidence_match_rows=total_evidence_match_rows,
        eligible_matched_rows=eligible_matched_rows,
        selected_matched_rows=selected_matched_rows,
        not_processed_due_to_limit=not_processed_due_to_limit,
        processed_rows=len(results),
        failed_rows=len(failures),
        skipped_unmatched_rows=skipped_unmatched_rows,
        skipped_empty_summaries=skipped_empty_summaries,
        llm_attempts=llm_attempts,
        strong_effects=sum(len(result.strong_evidence_effects) for result in results),
        medium_effects=sum(len(result.medium_evidence_effects) for result in results),
        low_effects=sum(len(result.low_evidence_effects) for result in results),
        supplement_level_effects=sum(len(result.supplement_level_relevance) for result in results),
    )


def summary_effect_count(result: NutrimentHealthSummaryRecord) -> int:
    return (
        len(result.strong_evidence_effects)
        + len(result.medium_evidence_effects)
        + len(result.low_evidence_effects)
        + len(result.supplement_level_relevance)
    )


def run_nutriment_health_summary_generation(
    *,
    evidence_matches_path: Path,
    reference_path: Path,
    out_dir: Path,
    provider: LlmProvider,
    model_provider: str,
    model_name: str,
    workers: int = 1,
    limit: int | None = None,
    resume: bool = True,
    force: bool = False,
    llm_attempts: int = 3,
    progress: NutrimentSummaryProgress | None = None,
) -> NutrimentHealthSummaryRunResult:
    if workers < 1:
        raise ValueError("nutriment_summary_workers_must_be_positive")
    if workers > 30:
        raise ValueError("nutriment_summary_workers_must_be_at_most_30")
    if limit is not None and limit < 1:
        raise ValueError("nutriment_summary_limit_must_be_positive")
    if llm_attempts < 1:
        raise ValueError("nutriment_summary_llm_attempts_must_be_positive")

    all_records = read_evidence_match_records(evidence_matches_path)
    all_matched_records = [
        record for record in all_records if record.matched and record.evidence_matches
    ]
    matched_records = list(all_matched_records)
    if limit is not None:
        matched_records = matched_records[:limit]
    reference_rows = read_reference_table(reference_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_results: list[NutrimentHealthSummaryRecord] = []
    skipped_keys: set[str] = set()
    if resume and not force:
        result_path = out_dir / NUTRIMENT_SUMMARIES_JSONL
        if result_path.exists():
            existing_results = [
                NUTRIMENT_SUMMARY_ADAPTER.validate_python(row) for row in read_jsonl(result_path)
            ]
            skipped_keys = {summary_key(result) for result in existing_results}

    pending = [
        record for record in matched_records if force or summary_key(record) not in skipped_keys
    ]
    append_jsonl(
        out_dir / NUTRIMENT_SUMMARIES_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "nutriment_summaries_started",
            "evidence_match_rows": len(all_records),
            "eligible_matched_rows": len(all_matched_records),
            "selected_matched_rows": len(matched_records),
            "not_processed_due_to_limit": len(all_matched_records) - len(matched_records),
            "pending": len(pending),
            "resumed_existing": len(existing_results),
            "workers": workers,
            "llm_attempts": llm_attempts,
        },
    )

    indexed_results: list[tuple[int, NutrimentHealthSummaryRecord]] = []
    indexed_failures: list[tuple[int, NutrimentHealthSummaryFailure]] = []

    def summarize_one(index: int, record: NutrimentEvidenceMatchRecord) -> tuple[
        int,
        NutrimentHealthSummaryRecord | None,
        NutrimentHealthSummaryFailure | None,
    ]:
        try:
            result = summarize_nutriment_record(
                record=record,
                reference_rows=reference_rows,
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
                NutrimentHealthSummaryFailure(
                    canonical_beverage_id=record.canonical_beverage_id,
                    ingredient_name=record.ingredient_name,
                    source_key=record.source_key,
                    canonical_bioactive_name=record.canonical_bioactive_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or type(exc).__name__,
                ),
            )

    if workers == 1:
        for index, record in enumerate(pending, start=1):
            if progress:
                progress(index, len(pending), record)
            _, result, failure = summarize_one(index, record)
            if result is not None:
                indexed_results.append((index, result))
            if failure is not None:
                indexed_failures.append((index, failure))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, record in enumerate(pending, start=1):
                if progress:
                    progress(index, len(pending), record)
                future = executor.submit(summarize_one, index, record)
                futures[future] = index
            for future in as_completed(futures):
                index, result, failure = future.result()
                if result is not None:
                    indexed_results.append((index, result))
                if failure is not None:
                    indexed_failures.append((index, failure))

    raw_results = [
        *existing_results,
        *(result for _, result in sorted(indexed_results, key=lambda item: item[0])),
    ]
    results = [result for result in raw_results if summary_effect_count(result) > 0]
    skipped_empty_summaries = len(raw_results) - len(results)
    failures = [failure for _, failure in sorted(indexed_failures, key=lambda item: item[0])]
    summary = summarize_results(
        total_evidence_match_rows=len(all_records),
        eligible_matched_rows=len(all_matched_records),
        selected_matched_rows=len(matched_records),
        not_processed_due_to_limit=len(all_matched_records) - len(matched_records),
        skipped_unmatched_rows=len(all_records) - len(all_matched_records),
        skipped_empty_summaries=skipped_empty_summaries,
        llm_attempts=llm_attempts,
        results=results,
        failures=failures,
    )

    write_jsonl(out_dir / NUTRIMENT_SUMMARIES_JSONL, results)
    write_jsonl(out_dir / NUTRIMENT_SUMMARIES_FAILURES_JSONL, failures)
    atomic_write(
        out_dir / NUTRIMENT_SUMMARIES_SUMMARY_JSON,
        orjson.dumps(
            summary.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        ),
    )
    append_jsonl(
        out_dir / NUTRIMENT_SUMMARIES_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "nutriment_summaries_completed",
            "processed": len(indexed_results),
            "failures": len(failures),
            "selected_matched_rows": summary.selected_matched_rows,
            "not_processed_due_to_limit": summary.not_processed_due_to_limit,
            "skipped_empty_summaries": summary.skipped_empty_summaries,
            "strong_effects": summary.strong_effects,
            "medium_effects": summary.medium_effects,
            "low_effects": summary.low_effects,
            "supplement_level_effects": summary.supplement_level_effects,
        },
    )

    return NutrimentHealthSummaryRunResult(
        out_dir=out_dir,
        results=results,
        failures=failures,
        summary=summary,
    )
