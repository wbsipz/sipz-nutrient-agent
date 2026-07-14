from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from pydantic import TypeAdapter

from sipz_agent.core.artifacts import model_dump_jsonable
from sipz_agent.schemas.composition import (
    IngredientNutrimentProfile,
    IngredientNutrimentSignificanceResult,
    NutrimentHealthSummaryRecord,
    NutrimentSignificanceFailure,
)


INGREDIENT_PROFILE_ADAPTER = TypeAdapter(IngredientNutrimentProfile)
SIGNIFICANCE_RESULT_ADAPTER = TypeAdapter(IngredientNutrimentSignificanceResult)
SIGNIFICANCE_FAILURE_ADAPTER = TypeAdapter(NutrimentSignificanceFailure)
NUTRIMENT_SUMMARY_ADAPTER = TypeAdapter(NutrimentHealthSummaryRecord)

INGREDIENT_SUMMARY_INPUTS_JSONL = "ingredient_health_summary_inputs.jsonl"
INGREDIENT_SUMMARY_INPUTS_SUMMARY_JSON = "ingredient_health_summary_inputs_summary.json"

DOSE_TABLE_REQUIRED_COLUMNS = {
    "canonical_bioactive_name",
    "meaningful_threshold_amount",
    "meaningful_threshold_unit",
    "meaningful_threshold_basis",
    "meaningful_threshold_source",
    "meaningful_review_status",
    "supplement_threshold_amount",
    "supplement_threshold_unit",
    "supplement_threshold_basis",
    "supplement_threshold_source",
    "supplement_review_status",
    "trace_definition",
    "meaningful_definition",
    "supplement_definition",
    "notes",
}

SUPPLEMENT_ROUTE_ALLOWED_STATUSES = {
    "supplement_dose_reviewed",
    "supplement_dose_reviewed_caution",
}

MASS_FACTORS_TO_MG = {
    "g": 1000.0,
    "mg": 1.0,
    "ug": 0.001,
    "mcg": 0.001,
}


@dataclass(frozen=True)
class IngredientSummaryInputRunResult:
    out_dir: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    atomic_write(path, payload)


def read_profiles(path: Path) -> list[IngredientNutrimentProfile]:
    return [INGREDIENT_PROFILE_ADAPTER.validate_python(row) for row in read_jsonl(path)]


def read_significance_results(path: Path) -> dict[str, IngredientNutrimentSignificanceResult]:
    results = [SIGNIFICANCE_RESULT_ADAPTER.validate_python(row) for row in read_jsonl(path)]
    return {result.canonical_beverage_id: result for result in results}


def read_significance_failures(path: Path | None) -> dict[str, NutrimentSignificanceFailure]:
    if path is None or not path.exists():
        return {}
    failures = [SIGNIFICANCE_FAILURE_ADAPTER.validate_python(row) for row in read_jsonl(path)]
    return {failure.canonical_beverage_id: failure for failure in failures}


def read_nutriment_summaries(path: Path) -> dict[tuple[str, str], NutrimentHealthSummaryRecord]:
    summaries = [NUTRIMENT_SUMMARY_ADAPTER.validate_python(row) for row in read_jsonl(path)]
    return {(summary.canonical_beverage_id, summary.source_key): summary for summary in summaries}


def read_dose_band_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(DOSE_TABLE_REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError("dose_band_table_columns_missing:" + ",".join(missing))
        rows = list(reader)
    return {row["canonical_bioactive_name"].casefold(): row for row in rows}


def primary_unit(unit: str) -> str:
    return unit.strip().split()[0].replace("/100g", "").replace("/serving", "")


def comparable_amount(amount: float, unit: str) -> tuple[float, str] | None:
    base_unit = primary_unit(unit).casefold()
    if base_unit in MASS_FACTORS_TO_MG:
        return amount * MASS_FACTORS_TO_MG[base_unit], "mg"
    if base_unit == "pH".casefold():
        return amount, "pH"
    if base_unit == "kcal":
        return amount, "kcal"
    return None


def parse_threshold_amount(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compare_amount_to_threshold(
    *,
    display_amount: float | None,
    display_unit: str,
    threshold_amount: str,
    threshold_unit: str,
) -> tuple[int | None, str | None]:
    if display_amount is None or not display_unit or not threshold_amount or not threshold_unit:
        return None, "missing_amount_or_threshold"
    parsed_threshold = parse_threshold_amount(threshold_amount)
    if parsed_threshold is None:
        return None, "non_numeric_threshold"
    amount_value = comparable_amount(display_amount, display_unit)
    threshold_value = comparable_amount(parsed_threshold, threshold_unit)
    if amount_value is None or threshold_value is None:
        return None, "unsupported_unit"
    if amount_value[1] != threshold_value[1]:
        return None, f"incompatible_units:{display_unit}:{threshold_unit}"
    if amount_value[0] < threshold_value[0]:
        return -1, None
    if amount_value[0] > threshold_value[0]:
        return 1, None
    return 0, None


def dose_band_basis(
    *,
    amount_context: str,
    dose_band: str,
    dose_row: dict[str, str] | None,
    reason: str,
) -> str:
    if dose_row is None:
        return reason
    meaningful = (
        f"{dose_row['meaningful_threshold_amount']} {dose_row['meaningful_threshold_unit']}".strip()
    )
    supplement = (
        f"{dose_row['supplement_threshold_amount']} {dose_row['supplement_threshold_unit']}".strip()
    )
    if dose_band == "trace":
        return f"{amount_context} is below the meaningful threshold ({meaningful})."
    if dose_band == "meaningful" and supplement:
        return f"{amount_context} is at or above the meaningful threshold ({meaningful}) and below the supplement threshold ({supplement})."
    if dose_band == "meaningful":
        return f"{amount_context} is at or above the meaningful threshold ({meaningful}); no supplement threshold is routable."
    if dose_band == "supplement":
        return f"{amount_context} is at or above the supplement threshold ({supplement})."
    if dose_band == "not_applicable":
        return reason
    return reason


def classify_dose_band(
    *,
    display_amount: float | None,
    display_unit: str,
    amount_context: str,
    dose_row: dict[str, str] | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if dose_row is None:
        return {
            "dose_band": "unknown",
            "dose_band_basis": "No dose-band table row found for nutriment.",
            "allow_food_level_effects": False,
            "allow_supplement_level_effects": False,
            "warnings": ["missing_dose_band_row"],
        }

    name = dose_row["canonical_bioactive_name"]
    if name == "Ph":
        amount = display_amount
        threshold = parse_threshold_amount(dose_row["meaningful_threshold_amount"])
        if amount is None or threshold is None:
            return {
                "dose_band": "unknown",
                "dose_band_basis": "pH amount or threshold is missing.",
                "allow_food_level_effects": False,
                "allow_supplement_level_effects": False,
                "warnings": ["missing_ph_amount_or_threshold"],
            }
        dose_band = "not_applicable" if amount > threshold else "meaningful"
        return {
            "dose_band": dose_band,
            "dose_band_basis": (
                f"{amount_context} uses inverse pH routing; pH <= {threshold:g} is acidity context."
            ),
            "allow_food_level_effects": dose_band == "meaningful",
            "allow_supplement_level_effects": False,
            "warnings": [],
        }

    meaningful_cmp, meaningful_warning = compare_amount_to_threshold(
        display_amount=display_amount,
        display_unit=display_unit,
        threshold_amount=dose_row["meaningful_threshold_amount"],
        threshold_unit=dose_row["meaningful_threshold_unit"],
    )
    if meaningful_warning:
        warnings.append("meaningful_threshold_compare_failed:" + meaningful_warning)
        return {
            "dose_band": "unknown",
            "dose_band_basis": dose_band_basis(
                amount_context=amount_context,
                dose_band="unknown",
                dose_row=dose_row,
                reason="Could not compare amount to meaningful threshold.",
            ),
            "allow_food_level_effects": False,
            "allow_supplement_level_effects": False,
            "warnings": warnings,
        }

    if meaningful_cmp is not None and meaningful_cmp < 0:
        return {
            "dose_band": "trace",
            "dose_band_basis": dose_band_basis(
                amount_context=amount_context,
                dose_band="trace",
                dose_row=dose_row,
                reason="Amount is below meaningful threshold.",
            ),
            "allow_food_level_effects": False,
            "allow_supplement_level_effects": False,
            "warnings": warnings,
        }

    supplement_status = dose_row["supplement_review_status"]
    supplement_threshold = dose_row["supplement_threshold_amount"]
    if supplement_status not in SUPPLEMENT_ROUTE_ALLOWED_STATUSES or not supplement_threshold:
        return {
            "dose_band": "meaningful",
            "dose_band_basis": dose_band_basis(
                amount_context=amount_context,
                dose_band="meaningful",
                dose_row=dose_row,
                reason="Amount is meaningful; supplement routing is disabled for this nutriment.",
            ),
            "allow_food_level_effects": True,
            "allow_supplement_level_effects": False,
            "warnings": warnings,
        }

    supplement_cmp, supplement_warning = compare_amount_to_threshold(
        display_amount=display_amount,
        display_unit=display_unit,
        threshold_amount=dose_row["supplement_threshold_amount"],
        threshold_unit=dose_row["supplement_threshold_unit"],
    )
    if supplement_warning:
        warnings.append("supplement_threshold_compare_failed:" + supplement_warning)
        return {
            "dose_band": "meaningful",
            "dose_band_basis": dose_band_basis(
                amount_context=amount_context,
                dose_band="meaningful",
                dose_row=dose_row,
                reason="Amount is meaningful; supplement threshold could not be compared.",
            ),
            "allow_food_level_effects": True,
            "allow_supplement_level_effects": False,
            "warnings": warnings,
        }
    if supplement_cmp is not None and supplement_cmp >= 0:
        return {
            "dose_band": "supplement",
            "dose_band_basis": dose_band_basis(
                amount_context=amount_context,
                dose_band="supplement",
                dose_row=dose_row,
                reason="Amount is at or above supplement threshold.",
            ),
            "allow_food_level_effects": True,
            "allow_supplement_level_effects": True,
            "warnings": warnings,
        }
    return {
        "dose_band": "meaningful",
        "dose_band_basis": dose_band_basis(
            amount_context=amount_context,
            dose_band="meaningful",
            dose_row=dose_row,
            reason="Amount is meaningful but below supplement threshold.",
        ),
        "allow_food_level_effects": True,
        "allow_supplement_level_effects": False,
        "warnings": warnings,
    }


def summary_payload(summary: NutrimentHealthSummaryRecord, *, include_supplement: bool) -> dict[str, Any]:
    payload = {
        "source_key": summary.source_key,
        "canonical_bioactive_name": summary.canonical_bioactive_name,
        "amount_context": summary.amount_context,
        "food_level_relevance": summary.food_level_relevance,
        "strong_evidence_effects": model_dump_jsonable(summary.strong_evidence_effects),
        "medium_evidence_effects": model_dump_jsonable(summary.medium_evidence_effects),
        "low_evidence_effects": model_dump_jsonable(summary.low_evidence_effects),
        "caveats": summary.caveats,
        "significance": summary.significance,
        "significance_confidence": summary.significance_confidence,
        "significance_reasoning": summary.significance_reasoning,
    }
    if include_supplement:
        payload["supplement_level_relevance"] = model_dump_jsonable(
            summary.supplement_level_relevance
        )
    return payload


def dose_table_payload(dose_row: dict[str, str] | None) -> dict[str, str]:
    if dose_row is None:
        return {}
    keys = [
        "meaningful_threshold_amount",
        "meaningful_threshold_unit",
        "meaningful_threshold_basis",
        "meaningful_review_status",
        "supplement_threshold_amount",
        "supplement_threshold_unit",
        "supplement_threshold_basis",
        "supplement_review_status",
    ]
    return {key: dose_row.get(key, "") for key in keys}


def classification_payload(classification) -> dict[str, Any]:
    return model_dump_jsonable(classification)


def nutriment_profile_payload(profile: IngredientNutrimentProfile) -> list[dict[str, Any]]:
    return [model_dump_jsonable(nutriment) for nutriment in profile.nutriments]


def build_ingredient_summary_input_row(
    *,
    profile: IngredientNutrimentProfile,
    significance: IngredientNutrimentSignificanceResult | None,
    significance_failure: NutrimentSignificanceFailure | None,
    summaries_by_key: dict[tuple[str, str], NutrimentHealthSummaryRecord],
    dose_rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    input_warnings: list[str] = []
    row = {
        "canonical_beverage_id": profile.canonical_beverage_id,
        "canonical_beverage_name": profile.canonical_beverage_name,
        "ingredient_name": significance.ingredient_name if significance else profile.canonical_beverage_name,
        "serving_basis": profile.serving_basis or (significance.serving_basis if significance else "100g"),
        "summary_type": "composition_based_literature_derived_input",
        "full_normalized_nutriment_profile": nutriment_profile_payload(profile),
        "dose_banded_nutriments": [],
        "food_level_summaries": [],
        "supplement_level_only_effects": [],
        "minor_or_trace_background_nutrients": [],
        "ignored_trace_nutrients": [],
        "unmatched_significant_nutrients": [],
        "skipped_no_effect_summary_nutrients": [],
        "input_warnings": input_warnings,
    }
    if significance_failure is not None:
        input_warnings.append(
            "significance_classification_failed:"
            + significance_failure.error_type
            + ":"
            + significance_failure.error_message
        )
    if significance is None:
        input_warnings.append("missing_significance_result")
        return row

    for classification in significance.classifications:
        dose_row = dose_rows.get(classification.canonical_bioactive_name.casefold())
        routed = classify_dose_band(
            display_amount=classification.display_amount,
            display_unit=classification.display_unit,
            amount_context=classification.amount_context,
            dose_row=dose_row,
        )
        if routed["warnings"]:
            input_warnings.extend(
                f"{classification.source_key}:{warning}" for warning in routed["warnings"]
            )
        dose_banded = {
            **classification_payload(classification),
            "dose_band": routed["dose_band"],
            "dose_band_basis": routed["dose_band_basis"],
            "allow_food_level_effects": routed["allow_food_level_effects"],
            "allow_supplement_level_effects": routed["allow_supplement_level_effects"],
            "dose_band_table_row": dose_table_payload(dose_row),
        }
        row["dose_banded_nutriments"].append(dose_banded)

        if classification.significance in {"minor", "unknown_threshold"}:
            row["minor_or_trace_background_nutrients"].append(dose_banded)
            continue
        if classification.significance == "trace" or routed["dose_band"] == "trace":
            row["ignored_trace_nutrients"].append(dose_banded)
            continue

        summary = summaries_by_key.get((profile.canonical_beverage_id, classification.source_key))
        if summary is None:
            if classification.significance == "significant":
                row["skipped_no_effect_summary_nutrients"].append(dose_banded)
            continue
        if not routed["allow_food_level_effects"]:
            row["ignored_trace_nutrients"].append(dose_banded)
            continue

        row["food_level_summaries"].append(
            {
                "dose_band": routed["dose_band"],
                "dose_band_basis": routed["dose_band_basis"],
                **summary_payload(summary, include_supplement=False),
            }
        )
        if routed["allow_supplement_level_effects"] and summary.supplement_level_relevance:
            row["supplement_level_only_effects"].append(
                {
                    "source_key": summary.source_key,
                    "canonical_bioactive_name": summary.canonical_bioactive_name,
                    "amount_context": summary.amount_context,
                    "dose_band": routed["dose_band"],
                    "dose_band_basis": routed["dose_band_basis"],
                    "effects": model_dump_jsonable(summary.supplement_level_relevance),
                    "caveats": summary.caveats,
                }
            )
    return row


def summarize_input_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ingredient_rows": len(rows),
        "ingredients_with_input_warnings": sum(1 for row in rows if row["input_warnings"]),
        "dose_banded_nutriments": sum(len(row["dose_banded_nutriments"]) for row in rows),
        "food_level_summary_nutrients": sum(len(row["food_level_summaries"]) for row in rows),
        "supplement_level_summary_nutrients": sum(
            len(row["supplement_level_only_effects"]) for row in rows
        ),
        "ignored_trace_nutrients": sum(len(row["ignored_trace_nutrients"]) for row in rows),
        "minor_or_trace_background_nutrients": sum(
            len(row["minor_or_trace_background_nutrients"]) for row in rows
        ),
        "skipped_no_effect_summary_nutrients": sum(
            len(row["skipped_no_effect_summary_nutrients"]) for row in rows
        ),
    }


def run_ingredient_summary_input_assembly(
    *,
    profiles_path: Path,
    significance_path: Path,
    nutriment_summaries_path: Path,
    dose_band_table_path: Path,
    out_dir: Path,
    significance_failures_path: Path | None = None,
    limit: int | None = None,
) -> IngredientSummaryInputRunResult:
    profiles = read_profiles(profiles_path)
    if limit is not None:
        profiles = profiles[:limit]
    significance_results = read_significance_results(significance_path)
    significance_failures = read_significance_failures(significance_failures_path)
    summaries_by_key = read_nutriment_summaries(nutriment_summaries_path)
    dose_rows = read_dose_band_table(dose_band_table_path)

    rows = [
        build_ingredient_summary_input_row(
            profile=profile,
            significance=significance_results.get(profile.canonical_beverage_id),
            significance_failure=significance_failures.get(profile.canonical_beverage_id),
            summaries_by_key=summaries_by_key,
            dose_rows=dose_rows,
        )
        for profile in profiles
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / INGREDIENT_SUMMARY_INPUTS_JSONL, rows)
    summary = summarize_input_rows(rows)
    atomic_write(
        out_dir / INGREDIENT_SUMMARY_INPUTS_SUMMARY_JSON,
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE),
    )
    return IngredientSummaryInputRunResult(out_dir=out_dir, rows=rows, summary=summary)
