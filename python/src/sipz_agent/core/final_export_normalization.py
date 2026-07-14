from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.final_export_inputs import read_jsonl, slugify, write_jsonl


NORMALIZED_NEW_SUMMARIES_JSONL = "normalized_new_summaries.jsonl"
NORMALIZED_NEW_SUMMARIES_SUMMARY_JSON = "normalized_new_summaries_summary.json"
NORMALIZED_NEW_SUMMARIES_FAILURES_JSONL = "normalized_new_summaries_failures.jsonl"


SUPPORT_LEVEL_RULES: dict[str, tuple[str, float, str]] = {
    "human_systematic_review": ("strong_evidence", 0.84, "strong"),
    "human_rct": ("strong_evidence", 0.75, "strong"),
    "human_observational": ("medium_evidence", 0.60, "medium"),
    "human_mechanistic": ("medium_evidence", 0.50, "medium"),
    "review_author_interpretation": ("low_evidence", 0.45, "low"),
}
DEFAULT_SUPPORT_RULE = ("low_evidence", 0.40, "low")

EFFECT_BUCKETS = [
    "strong_evidence",
    "medium_evidence",
    "low_evidence",
    "negative_or_cautionary_effects",
    "supplement_level_only_effects",
]


@dataclass(frozen=True)
class FinalExportNormalizationResult:
    out_dir: Path
    rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    summary: dict[str, Any]


def support_level_rule(support_level: str) -> tuple[str, float, str]:
    return SUPPORT_LEVEL_RULES.get(support_level, DEFAULT_SUPPORT_RULE)


def quote_count(claim: dict[str, Any]) -> int:
    quotes = claim.get("supporting_quotes")
    return len(quotes) if isinstance(quotes, list) else 0


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def direct_effect_slug(claim: dict[str, Any], index: int) -> str:
    effect_row_id = str(claim.get("effect_row_id") or "").strip()
    if effect_row_id:
        return "direct_literature_" + slugify(effect_row_id)
    citation = str(claim.get("citation_id") or "").strip()
    proposed = str(claim.get("proposed_ingredient_claim_id") or "").strip()
    fallback = slugify("_".join(part for part in [citation, proposed] if part))
    return "direct_literature_" + (fallback or f"claim_{index}")


def normalize_direct_claim(claim: dict[str, Any], index: int) -> tuple[str, dict[str, Any]]:
    support_level = str(claim.get("support_level") or "").strip()
    bucket, score, evidence_level = support_level_rule(support_level)
    statement = str(claim.get("validated_statement") or "").strip()
    if not statement:
        statement = str(claim.get("claim_scope") or "").strip()
    if not statement:
        statement = "Accepted direct-literature claim without validated statement."

    return bucket, {
        "effect_slug": direct_effect_slug(claim, index),
        "effect_label": "Direct literature claim",
        "summary": statement,
        "evidence_level": evidence_level,
        "score": score,
        "supporting_nutrients": [],
        "source_claim": {
            "effect_row_id": str(claim.get("effect_row_id") or ""),
            "citation_id": str(claim.get("citation_id") or ""),
            "support_level": support_level,
            "verdict": str(claim.get("verdict") or ""),
            "claim_scope": str(claim.get("claim_scope") or ""),
            "validator_reasoning": str(claim.get("validator_reasoning") or ""),
            "limitations": string_list(claim.get("limitations")),
            "supporting_quotes_count": quote_count(claim),
            "web_audit_verdict": str(claim.get("web_audit_verdict") or ""),
        },
    }


def base_normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical_id = str(row.get("canonical_beverage_id") or "").strip()
    canonical_name = str(row.get("canonical_beverage_name") or "").strip()
    if not canonical_id:
        raise ValueError("missing_canonical_beverage_id")
    if not canonical_name:
        raise ValueError("missing_canonical_beverage_name")
    source_type = str(row.get("summary_source_type") or "").strip()
    if source_type not in {"direct_literature", "composition_based"}:
        raise ValueError(f"invalid_summary_source_type:{source_type}")
    return {
        "canonical_beverage_id": canonical_id,
        "canonical_beverage_name": canonical_name,
        "summary_source_type": source_type,
        "summary_confidence_status": row.get("summary_confidence_status") or "ready",
        "strong_evidence": [],
        "medium_evidence": [],
        "low_evidence": [],
        "negative_or_cautionary_effects": [],
        "supplement_level_only_effects": [],
        "caveats": [],
        "source_summary_text": "",
        "legacy": row.get("legacy") or {},
        "source_paths": row.get("source_paths") or {},
        "normalization_warnings": [],
    }


def normalize_composition_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = base_normalized_row(row)
    composition = row.get("composition_summary")
    if not isinstance(composition, dict):
        raise ValueError("missing_composition_summary")
    for bucket in EFFECT_BUCKETS:
        value = composition.get(bucket)
        if not isinstance(value, list):
            raise ValueError(f"composition_bucket_not_list:{bucket}")
        normalized[bucket] = value
    caveats = composition.get("caveats")
    if not isinstance(caveats, list):
        raise ValueError("composition_caveats_not_list")
    normalized["caveats"] = caveats
    normalized["source_summary_text"] = str(composition.get("overall_summary") or "")
    if not any(normalized[bucket] for bucket in EFFECT_BUCKETS):
        normalized["normalization_warnings"].append("composition_summary_has_no_effects")
    return normalized


def normalize_direct_literature_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = base_normalized_row(row)
    direct = row.get("direct_literature")
    if not isinstance(direct, dict):
        raise ValueError("missing_direct_literature")
    claims = direct.get("claims")
    if not isinstance(claims, list):
        raise ValueError("direct_claims_not_list")
    accepted_claims = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("accepted") is not False
    ]
    if not accepted_claims:
        raise ValueError("no_accepted_direct_claims")

    for index, claim in enumerate(accepted_claims, start=1):
        bucket, effect = normalize_direct_claim(claim, index)
        normalized[bucket].append(effect)
    normalized["direct_literature_context"] = {
        "final_run_target_id": direct.get("final_run_target_id", ""),
        "final_run_target_name": direct.get("final_run_target_name", ""),
        "run_path": direct.get("run_path", ""),
        "claim_file": direct.get("claim_file", ""),
        "claim_count": len(accepted_claims),
    }
    if len(accepted_claims) != len(claims):
        normalized["normalization_warnings"].append("some_direct_claims_were_not_accepted")
    return normalized


def normalize_final_export_row(row: dict[str, Any]) -> dict[str, Any]:
    source_type = str(row.get("summary_source_type") or "").strip()
    if source_type == "composition_based":
        return normalize_composition_row(row)
    if source_type == "direct_literature":
        return normalize_direct_literature_row(row)
    return base_normalized_row(row)


def failure_row(row: dict[str, Any], index: int, exc: Exception) -> dict[str, Any]:
    return {
        "line_number": index,
        "canonical_beverage_id": str(row.get("canonical_beverage_id") or ""),
        "canonical_beverage_name": str(row.get("canonical_beverage_name") or ""),
        "summary_source_type": str(row.get("summary_source_type") or ""),
        "error_message": str(exc),
    }


def normalize_final_export_summaries(
    *,
    inputs_path: Path,
    out_dir: Path,
) -> FinalExportNormalizationResult:
    if not inputs_path.exists():
        raise ValueError(f"missing_inputs:{inputs_path}")

    input_rows = read_jsonl(inputs_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows, start=1):
        try:
            rows.append(normalize_final_export_row(row))
        except Exception as exc:
            failures.append(failure_row(row, index, exc))

    summary = {
        "input_rows": len(input_rows),
        "normalized_rows": len(rows),
        "failed_rows": len(failures),
        "direct_literature_rows": sum(
            1 for row in rows if row["summary_source_type"] == "direct_literature"
        ),
        "composition_based_rows": sum(
            1 for row in rows if row["summary_source_type"] == "composition_based"
        ),
        "strong_effects": sum(len(row["strong_evidence"]) for row in rows),
        "medium_effects": sum(len(row["medium_evidence"]) for row in rows),
        "low_effects": sum(len(row["low_evidence"]) for row in rows),
        "negative_or_cautionary_effects": sum(
            len(row["negative_or_cautionary_effects"]) for row in rows
        ),
        "supplement_level_only_effects": sum(
            len(row["supplement_level_only_effects"]) for row in rows
        ),
        "rows_with_normalization_warnings": sum(
            bool(row["normalization_warnings"]) for row in rows
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_JSONL, rows)
    write_jsonl(out_dir / NORMALIZED_NEW_SUMMARIES_FAILURES_JSONL, failures)
    write_json(out_dir / NORMALIZED_NEW_SUMMARIES_SUMMARY_JSON, summary)
    return FinalExportNormalizationResult(
        out_dir=out_dir,
        rows=rows,
        failures=failures,
        summary=summary,
    )
