from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import orjson
import re

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from sipz_agent.core.artifacts import model_dump_jsonable
from sipz_agent.core.final_export_inputs import atomic_write, read_jsonl, write_jsonl
from sipz_agent.core.models import LlmProvider


FINAL_HEALTH_SUMMARY_MERGES_JSONL = "final_health_summary_merges.jsonl"
FINAL_HEALTH_SUMMARY_MERGE_FAILURES_JSONL = "final_health_summary_merge_failures.jsonl"
FINAL_HEALTH_SUMMARY_MERGE_SUMMARY_JSON = "final_health_summary_merge_summary.json"
FINAL_HEALTH_SUMMARY_MERGE_BATCH_LOG_JSONL = "final_health_summary_merge_batch_log.jsonl"

MAX_REGULAR_EFFECTS = 5
MAX_CAUTIONARY_EFFECTS = 8
MAX_SUPPLEMENT_EFFECTS = 6
MAX_CAVEATS = 6
MAX_LEGACY_CLAIMS_USED = 5
MAX_LEGACY_CLAIMS_EXCLUDED = 8

CAUTIONARY_SUMMARY_PATTERN = re.compile(
    r"\b("
    r"risk|caution|adverse|harm|increase|high sugar|added sugar|sodium|salt|"
    r"saturated fat|weight gain|dental caries|caries|insulin|blood pressure|"
    r"hypertension|kidney|hyperkalemia|not reliable|not sufficient|concern|"
    r"inadequate|low in|limited satiety"
    r")\b",
    re.IGNORECASE,
)
NON_CAUTIONARY_NEGATION_PATTERN = re.compile(
    r"\b("
    r"no adverse effects|no negative effects|no significant negative effects|"
    r"no adverse|no harms?|not identified|were identified|none identified"
    r")\b",
    re.IGNORECASE,
)
DROPPED_EFFECT_WARNING_PATTERN = re.compile(
    r"\b(moved|included|added|placed|routed)\b.*\b("
    r"supplement|supplement_level_only_effects|negative_or_cautionary|caution|bucket|effect"
    r")\b",
    re.IGNORECASE,
)

FinalMergeProgress = Callable[[int, int, dict[str, Any]], None]


class FinalMergeEffect(BaseModel):
    effect_slug: str = ""
    effect_label: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=900)
    evidence_level: str = ""
    score: float | None = Field(default=None, ge=0, le=1)
    supporting_nutrients: list[str] = Field(default_factory=list)
    source_claim: dict[str, Any] | None = None

    @field_validator("effect_slug", "evidence_level", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value

    @field_validator("score", mode="before")
    @classmethod
    def normalize_score(cls, value: object) -> object:
        if value is None or value == "":
            return None
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        if 1 < score <= 10:
            return score / 10
        return score


class LlmFinalHealthSummaryMerge(BaseModel):
    final_positive_summary: str = Field(min_length=1, max_length=1800)
    final_negative_summary: str = Field(min_length=1, max_length=1800)
    strong_evidence: list[FinalMergeEffect] = Field(default_factory=list)
    medium_evidence: list[FinalMergeEffect] = Field(default_factory=list)
    low_evidence: list[FinalMergeEffect] = Field(default_factory=list)
    negative_or_cautionary_effects: list[FinalMergeEffect] = Field(default_factory=list)
    supplement_level_only_effects: list[FinalMergeEffect] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    legacy_claims_used: list[str] = Field(default_factory=list)
    legacy_claims_excluded: list[str] = Field(default_factory=list)
    merge_warnings: list[str] = Field(default_factory=list)

    @field_validator("legacy_claims_used", "legacy_claims_excluded", mode="before")
    @classmethod
    def normalize_legacy_claim_list(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                claim = str(item.get("claim", "")).strip()
                if claim:
                    normalized.append(claim)
        return normalized


class FinalHealthSummaryMergeRecord(LlmFinalHealthSummaryMerge):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = Field(min_length=1)
    summary_source_type: str = Field(min_length=1)
    summary_confidence_status: str = "ready"
    used_llm: bool = True
    model_provider: str = ""
    model_name: str = ""


class FinalHealthSummaryMergeFailure(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = ""
    summary_source_type: str = ""
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class FinalHealthSummaryMergeRunSummary(BaseModel):
    total_input_rows: int
    selected_input_rows: int
    not_processed_due_to_limit: int
    processed_rows: int
    failed_rows: int
    llm_attempts: int
    strong_effects: int
    medium_effects: int
    low_effects: int
    negative_or_cautionary_effects: int
    supplement_level_only_effects: int
    legacy_claims_used: int
    legacy_claims_excluded: int
    rows_with_warnings: int


@dataclass(frozen=True)
class FinalHealthSummaryMergeRunResult:
    out_dir: Path
    results: list[FinalHealthSummaryMergeRecord]
    failures: list[FinalHealthSummaryMergeFailure]
    summary: FinalHealthSummaryMergeRunSummary


LLM_FINAL_MERGE_ADAPTER = TypeAdapter(LlmFinalHealthSummaryMerge)
FINAL_MERGE_RECORD_ADAPTER = TypeAdapter(FinalHealthSummaryMergeRecord)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def merge_key(row: dict[str, Any] | FinalHealthSummaryMergeRecord) -> str:
    if isinstance(row, FinalHealthSummaryMergeRecord):
        return row.canonical_beverage_id
    return str(row["canonical_beverage_id"])


def compact_effect(effect: dict[str, Any]) -> dict[str, Any]:
    return {
        "effect_slug": effect.get("effect_slug", ""),
        "effect_label": effect.get("effect_label", ""),
        "summary": effect.get("summary", ""),
        "evidence_level": effect.get("evidence_level", ""),
        "score": effect.get("score"),
        "supporting_nutrients": effect.get("supporting_nutrients", []),
        "source_claim": effect.get("source_claim"),
    }


def accepted_legacy_claims(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        claim
        for claim in row.get("legacy", {}).get("audited_claims", [])
        if str(claim.get("verdict", "")).casefold() == "accept"
    ]


def rejected_legacy_claims(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        claim
        for claim in row.get("legacy", {}).get("audited_claims", [])
        if str(claim.get("verdict", "")).casefold() == "reject"
    ]


def prompt_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_beverage_id": row.get("canonical_beverage_id"),
        "canonical_beverage_name": row.get("canonical_beverage_name"),
        "summary_source_type": row.get("summary_source_type"),
        "summary_confidence_status": row.get("summary_confidence_status"),
        "normalized_new_summary": {
            "strong_evidence": [compact_effect(item) for item in row.get("strong_evidence", [])],
            "medium_evidence": [compact_effect(item) for item in row.get("medium_evidence", [])],
            "low_evidence": [compact_effect(item) for item in row.get("low_evidence", [])],
            "negative_or_cautionary_effects": [
                compact_effect(item) for item in row.get("negative_or_cautionary_effects", [])
            ],
            "supplement_level_only_effects": [
                compact_effect(item) for item in row.get("supplement_level_only_effects", [])
            ],
            "caveats": row.get("caveats", []),
            "source_summary_text": row.get("source_summary_text", ""),
            "normalization_warnings": row.get("normalization_warnings", []),
        },
        "legacy_claims_to_consider": accepted_legacy_claims(row),
        "legacy_claims_to_exclude": rejected_legacy_claims(row),
    }


def food_effect_slugs(row: dict[str, Any]) -> set[str]:
    slugs: set[str] = set()
    for bucket in [
        "strong_evidence",
        "medium_evidence",
        "low_evidence",
        "negative_or_cautionary_effects",
    ]:
        slugs.update(str(item.get("effect_slug", "")) for item in row.get(bucket, []))
    return {slug for slug in slugs if slug}


def supplement_effect_slugs(row: dict[str, Any]) -> set[str]:
    return {
        str(item.get("effect_slug", ""))
        for item in row.get("supplement_level_only_effects", [])
        if item.get("effect_slug")
    }


def final_merge_prompt(row: dict[str, Any]) -> str:
    food_slugs = sorted(food_effect_slugs(row))
    supplement_slugs = sorted(supplement_effect_slugs(row))
    return (
        "Create the final ingredient health summary by merging normalized new evidence with "
        "audited legacy context.\n\n"
        "Use the normalized new summary as the primary source of truth. Accepted legacy claims "
        "may improve clarity or fill a gap, but rejected legacy claims must not be used. Do not "
        "invent health effects, doses, mechanisms, nutrients, or ingredient-specific clinical "
        "evidence. Direct-literature claims are normal ingredient-level evidence, not "
        "supplement-level evidence. Composition supplement effects must remain only in "
        "supplement_level_only_effects.\n\n"
        "Routing rules:\n"
        "- Put beneficial or neutral food-level effects in strong_evidence, medium_evidence, "
        "or low_evidence.\n"
        "- Put harmful, risk-increasing, or cautionary effects only in "
        "negative_or_cautionary_effects.\n"
        "- Keep supplement-level-only effects separate.\n"
        "- You may move a food-level effect between regular and caution buckets if the text "
        "clearly warrants it.\n"
        "- Keep final_positive_summary and final_negative_summary concise and user-readable.\n"
        "- If little evidence exists, say so plainly rather than overstating effects.\n\n"
        "Legacy rules:\n"
        "- If you use a legacy claim, copy the exact claim string into legacy_claims_used.\n"
        f"- Use at most {MAX_LEGACY_CLAIMS_USED} legacy claims total; prefer none when normalized "
        "new evidence is sufficient.\n"
        "- If a legacy claim is not an exact fit, leave it out.\n\n"
        "Negative/caution rules:\n"
        "- If final_negative_summary describes a concrete risk and a supplied food-level effect "
        "supports that risk, include that effect in negative_or_cautionary_effects.\n"
        "- If there is no supplied effect for the negative summary, keep the summary cautious "
        "and add a caveat instead of inventing a bucket effect.\n\n"
        "Score rules:\n"
        "- score must be a decimal from 0.0 to 1.0, never a 0-10 score.\n\n"
        f"allowed_food_effect_slugs: {food_slugs}\n"
        f"allowed_supplement_effect_slugs: {supplement_slugs}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        "{\n"
        '  "final_positive_summary": "...",\n'
        '  "final_negative_summary": "...",\n'
        '  "strong_evidence": [],\n'
        '  "medium_evidence": [],\n'
        '  "low_evidence": [],\n'
        '  "negative_or_cautionary_effects": [],\n'
        '  "supplement_level_only_effects": [],\n'
        '  "caveats": [],\n'
        '  "legacy_claims_used": [],\n'
        '  "legacy_claims_excluded": [],\n'
        '  "merge_warnings": []\n'
        "}\n\n"
        "Effect objects require: effect_slug, effect_label, summary, evidence_level, score, "
        "supporting_nutrients.\n"
        "Keep at most 5 strong, 5 medium, 5 low, 8 negative/cautionary, 6 supplement-only "
        "effects, and 6 caveats.\n\n"
        "Input:\n"
        + orjson.dumps(prompt_payload(row), option=orjson.OPT_INDENT_2).decode("utf-8")
    )


def effect_key(item: FinalMergeEffect) -> str:
    return item.effect_slug or f"{item.effect_label}:{item.summary}"


def sanitize_effects(
    items: list[FinalMergeEffect],
    *,
    allowed_slugs: set[str],
    seen: set[str],
    limit: int,
) -> list[FinalMergeEffect]:
    kept: list[FinalMergeEffect] = []
    for item in items:
        if not item.effect_slug:
            continue
        key = effect_key(item)
        if item.effect_slug not in allowed_slugs:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def contains_rejected_claim_text(response: LlmFinalHealthSummaryMerge, rejected_claims: list[dict[str, Any]]) -> bool:
    rejected_texts = [
        str(claim.get("claim", "")).strip().casefold()
        for claim in rejected_claims
        if len(str(claim.get("claim", "")).strip()) >= 20
    ]
    if not rejected_texts:
        return False
    response_text = " ".join(
        [
            response.final_positive_summary,
            response.final_negative_summary,
            *[effect.summary for effect in response.strong_evidence],
            *[effect.summary for effect in response.medium_evidence],
            *[effect.summary for effect in response.low_evidence],
            *[effect.summary for effect in response.negative_or_cautionary_effects],
            *[effect.summary for effect in response.supplement_level_only_effects],
        ]
    ).casefold()
    return any(text in response_text for text in rejected_texts)


def sanitize_legacy_claims(response_claims: list[str], allowed_claims: list[dict[str, Any]]) -> list[str]:
    allowed = {str(claim.get("claim", "")).strip() for claim in allowed_claims}
    return [claim for claim in response_claims if claim in allowed]


def claimed_legacy_usage_was_dropped(
    response_claims: list[str],
    sanitized_claims: list[str],
) -> bool:
    return bool(response_claims) and len(sanitized_claims) < len(response_claims)


def has_input_effects(input_row: dict[str, Any]) -> bool:
    return any(
        input_row.get(bucket)
        for bucket in [
            "strong_evidence",
            "medium_evidence",
            "low_evidence",
            "negative_or_cautionary_effects",
            "supplement_level_only_effects",
        ]
    )


def has_output_effects(response: LlmFinalHealthSummaryMerge) -> bool:
    return any(
        [
            response.strong_evidence,
            response.medium_evidence,
            response.low_evidence,
            response.negative_or_cautionary_effects,
            response.supplement_level_only_effects,
        ]
    )


def has_cautionary_negative_summary(response: LlmFinalHealthSummaryMerge) -> bool:
    negative_summary = response.final_negative_summary or ""
    if NON_CAUTIONARY_NEGATION_PATTERN.search(negative_summary):
        return False
    return bool(CAUTIONARY_SUMMARY_PATTERN.search(negative_summary))


def warning_refs_dropped_effect(warning: str) -> bool:
    return bool(DROPPED_EFFECT_WARNING_PATTERN.search(warning))


def sanitize_final_merge_response(
    *,
    response: LlmFinalHealthSummaryMerge,
    input_row: dict[str, Any],
) -> LlmFinalHealthSummaryMerge:
    rejected = rejected_legacy_claims(input_row)
    if contains_rejected_claim_text(response, rejected):
        raise ValueError("llm_response_used_rejected_legacy_claim")

    food_slugs = food_effect_slugs(input_row)
    supplement_slugs = supplement_effect_slugs(input_row)
    seen: set[str] = set()
    strong = sanitize_effects(
        response.strong_evidence,
        allowed_slugs=food_slugs,
        seen=seen,
        limit=MAX_REGULAR_EFFECTS,
    )
    medium = sanitize_effects(
        response.medium_evidence,
        allowed_slugs=food_slugs,
        seen=seen,
        limit=MAX_REGULAR_EFFECTS,
    )
    low = sanitize_effects(
        response.low_evidence,
        allowed_slugs=food_slugs,
        seen=seen,
        limit=MAX_REGULAR_EFFECTS,
    )
    caution = sanitize_effects(
        response.negative_or_cautionary_effects,
        allowed_slugs=food_slugs,
        seen=seen,
        limit=MAX_CAUTIONARY_EFFECTS,
    )
    supplement = sanitize_effects(
        response.supplement_level_only_effects,
        allowed_slugs=supplement_slugs,
        seen=seen,
        limit=MAX_SUPPLEMENT_EFFECTS,
    )
    accepted = accepted_legacy_claims(input_row)
    excluded = rejected_legacy_claims(input_row)
    legacy_used = sanitize_legacy_claims(
        response.legacy_claims_used,
        accepted,
    )[:MAX_LEGACY_CLAIMS_USED]
    legacy_excluded = sanitize_legacy_claims(
        response.legacy_claims_excluded,
        excluded,
    )[:MAX_LEGACY_CLAIMS_EXCLUDED]
    warnings = [
        warning
        for warning in response.merge_warnings
        if not warning_refs_dropped_effect(warning)
    ][:MAX_CAVEATS]
    if claimed_legacy_usage_was_dropped(response.legacy_claims_used, legacy_used):
        warnings.append("legacy_claims_used_sanitized_to_exact_audit_matches")
    warnings = warnings[:MAX_CAVEATS]
    return LlmFinalHealthSummaryMerge(
        final_positive_summary=response.final_positive_summary,
        final_negative_summary=response.final_negative_summary,
        strong_evidence=strong,
        medium_evidence=medium,
        low_evidence=low,
        negative_or_cautionary_effects=caution,
        supplement_level_only_effects=supplement,
        caveats=response.caveats[:MAX_CAVEATS],
        legacy_claims_used=legacy_used,
        legacy_claims_excluded=legacy_excluded,
        merge_warnings=warnings,
    )


def final_confidence_status(
    *,
    input_row: dict[str, Any],
    response: LlmFinalHealthSummaryMerge,
) -> str:
    _ = input_row
    _ = response
    return "ready"


def merge_one_record(
    *,
    input_row: dict[str, Any],
    provider: LlmProvider,
    model_provider: str,
    model_name: str,
    llm_attempts: int,
) -> FinalHealthSummaryMergeRecord:
    prompt = final_merge_prompt(input_row)
    last_error: Exception | None = None
    for _ in range(llm_attempts):
        try:
            response = provider.complete_json(prompt, LLM_FINAL_MERGE_ADAPTER)
            response = sanitize_final_merge_response(response=response, input_row=input_row)
            break
        except Exception as exc:
            last_error = exc
    else:
        if last_error is None:
            raise RuntimeError("final_merge_llm_failed")
        raise last_error

    return FinalHealthSummaryMergeRecord(
        canonical_beverage_id=input_row["canonical_beverage_id"],
        canonical_beverage_name=input_row["canonical_beverage_name"],
        summary_source_type=input_row["summary_source_type"],
        summary_confidence_status=final_confidence_status(
            input_row=input_row,
            response=response,
        ),
        final_positive_summary=response.final_positive_summary,
        final_negative_summary=response.final_negative_summary,
        strong_evidence=response.strong_evidence,
        medium_evidence=response.medium_evidence,
        low_evidence=response.low_evidence,
        negative_or_cautionary_effects=response.negative_or_cautionary_effects,
        supplement_level_only_effects=response.supplement_level_only_effects,
        caveats=response.caveats,
        legacy_claims_used=response.legacy_claims_used,
        legacy_claims_excluded=response.legacy_claims_excluded,
        merge_warnings=response.merge_warnings,
        model_provider=model_provider,
        model_name=model_name,
    )


def summarize_merge_results(
    *,
    total_input_rows: int,
    selected_input_rows: int,
    not_processed_due_to_limit: int,
    llm_attempts: int,
    results: list[FinalHealthSummaryMergeRecord],
    failures: list[FinalHealthSummaryMergeFailure],
) -> FinalHealthSummaryMergeRunSummary:
    return FinalHealthSummaryMergeRunSummary(
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
        supplement_level_only_effects=sum(
            len(result.supplement_level_only_effects) for result in results
        ),
        legacy_claims_used=sum(len(result.legacy_claims_used) for result in results),
        legacy_claims_excluded=sum(len(result.legacy_claims_excluded) for result in results),
        rows_with_warnings=sum(bool(result.merge_warnings) for result in results),
    )


def run_final_health_summary_merge(
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
    progress: FinalMergeProgress | None = None,
) -> FinalHealthSummaryMergeRunResult:
    if workers < 1:
        raise ValueError("final_merge_workers_must_be_positive")
    if workers > 30:
        raise ValueError("final_merge_workers_must_be_at_most_30")
    if limit is not None and limit < 1:
        raise ValueError("final_merge_limit_must_be_positive")
    if llm_attempts < 1:
        raise ValueError("final_merge_llm_attempts_must_be_positive")

    all_rows = read_jsonl(inputs_path)
    selected_rows = list(all_rows)
    if limit is not None:
        selected_rows = selected_rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_results: list[FinalHealthSummaryMergeRecord] = []
    skipped_keys: set[str] = set()
    result_path = out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL
    if resume and not force and result_path.exists():
        existing_results = [
            FINAL_MERGE_RECORD_ADAPTER.validate_python(row) for row in read_jsonl(result_path)
        ]
        skipped_keys = {merge_key(result) for result in existing_results}

    pending = [row for row in selected_rows if force or merge_key(row) not in skipped_keys]
    append_jsonl(
        out_dir / FINAL_HEALTH_SUMMARY_MERGE_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "final_merge_started",
            "total_input_rows": len(all_rows),
            "selected_input_rows": len(selected_rows),
            "not_processed_due_to_limit": len(all_rows) - len(selected_rows),
            "pending": len(pending),
            "resumed_existing": len(existing_results),
            "workers": workers,
            "llm_attempts": llm_attempts,
        },
    )

    indexed_results: list[tuple[int, FinalHealthSummaryMergeRecord]] = []
    indexed_failures: list[tuple[int, FinalHealthSummaryMergeFailure]] = []

    def merge_one(
        index: int,
        row: dict[str, Any],
    ) -> tuple[int, FinalHealthSummaryMergeRecord | None, FinalHealthSummaryMergeFailure | None]:
        try:
            result = merge_one_record(
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
                FinalHealthSummaryMergeFailure(
                    canonical_beverage_id=row.get("canonical_beverage_id", "unknown"),
                    canonical_beverage_name=row.get("canonical_beverage_name", ""),
                    summary_source_type=row.get("summary_source_type", ""),
                    error_type=type(exc).__name__,
                    error_message=str(exc) or type(exc).__name__,
                ),
            )

    if workers == 1:
        for index, row in enumerate(pending, start=1):
            if progress:
                progress(index, len(pending), row)
            _, result, failure = merge_one(index, row)
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
                future = executor.submit(merge_one, index, row)
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
    summary = summarize_merge_results(
        total_input_rows=len(all_rows),
        selected_input_rows=len(selected_rows),
        not_processed_due_to_limit=len(all_rows) - len(selected_rows),
        llm_attempts=llm_attempts,
        results=results,
        failures=failures,
    )

    write_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGES_JSONL, results)
    write_jsonl(out_dir / FINAL_HEALTH_SUMMARY_MERGE_FAILURES_JSONL, failures)
    atomic_write(
        out_dir / FINAL_HEALTH_SUMMARY_MERGE_SUMMARY_JSON,
        orjson.dumps(
            summary.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        ),
    )
    append_jsonl(
        out_dir / FINAL_HEALTH_SUMMARY_MERGE_BATCH_LOG_JSONL,
        {
            "ts": utc_now(),
            "event": "final_merge_completed",
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

    return FinalHealthSummaryMergeRunResult(
        out_dir=out_dir,
        results=results,
        failures=failures,
        summary=summary,
    )
