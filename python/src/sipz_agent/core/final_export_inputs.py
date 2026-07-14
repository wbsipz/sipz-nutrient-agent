from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import orjson

from sipz_agent.core.artifacts import model_dump_jsonable, write_json


FINAL_EXPORT_MERGE_INPUTS_JSONL = "final_export_merge_inputs.jsonl"
SKIPPED_OR_UNCOVERED_INGREDIENTS_CSV = "skipped_or_uncovered_ingredients.csv"
FINAL_EXPORT_MERGE_INPUT_SUMMARY_JSON = "final_export_merge_input_summary.json"

DIRECT_CLAIM_FILE_PRECEDENCE = [
    "web_audited_ingredient_claims.json",
    "audited_ingredient_claims.json",
    "validated_ingredient_claims.json",
]


@dataclass(frozen=True)
class FinalExportInputAssemblyResult:
    out_dir: Path
    rows: list[dict[str, Any]]
    skipped_rows: list[dict[str, str]]
    summary: dict[str, Any]


@dataclass
class LegacyAuditAttachmentStats:
    attached_claims: int = 0
    mismatched_claims_skipped: int = 0


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        orjson.dumps(model_dump_jsonable(row), option=orjson.OPT_APPEND_NEWLINE) for row in rows
    )
    atomic_write(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def read_csv_rows_with_row_number(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: value or "" for key, value in row.items() if key is not None}
            normalized["_csv_row_number"] = str(row_number)
            rows.append(normalized)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_cell(value: str) -> Any:
    if not value.strip():
        return []
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return []


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug


def normalized_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def load_legacy_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows_with_row_number(path):
        canonical_id = row.get("canonical_beverage_id", "").strip()
        if not canonical_id:
            continue
        rows_by_id[canonical_id] = {
            "row_number": int(row["_csv_row_number"]),
            "canonical_beverage_name": row.get("canonical_beverage_name", ""),
            "positive": row.get("health_effect_positive", ""),
            "negative": row.get("health_effect_negative", ""),
            "positive_tags": json_cell(row.get("health_effect_positive_tags", "")),
            "negative_tags": json_cell(row.get("health_effect_negative_tags", "")),
        }
    return rows_by_id


def load_audited_legacy_claims(path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    claims_by_source_row: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_csv_rows_with_row_number(path):
        source_file = Path(row.get("source_file", "")).name
        row_number_text = row.get("row_number", "").strip()
        if not source_file or not row_number_text:
            continue
        try:
            source_row_number = int(row_number_text)
        except ValueError:
            continue
        claim = {
            "source_file": source_file,
            "row_number": source_row_number,
            "ingredient": row.get("ingredient", ""),
            "effect_type": row.get("effect_type", ""),
            "claim": row.get("claim", ""),
            "verdict": row.get("verdict", ""),
            "sources": [
                source.strip()
                for source in row.get("sources", "").split(";")
                if source.strip()
            ],
        }
        claims_by_source_row.setdefault((source_file, source_row_number), []).append(claim)
    return claims_by_source_row


def legacy_payload(
    *,
    canonical_id: str,
    legacy_v1: dict[str, dict[str, Any]],
    legacy_v2: dict[str, dict[str, Any]],
    audited_claims: dict[tuple[str, int], list[dict[str, str]]],
    legacy_v1_path: Path,
    legacy_v2_path: Path,
    stats: LegacyAuditAttachmentStats | None = None,
) -> dict[str, Any]:
    v1 = legacy_v1.get(canonical_id)
    v2 = legacy_v2.get(canonical_id)
    audit_rows: list[dict[str, str]] = []
    for legacy_path, legacy_row in [(legacy_v1_path, v1), (legacy_v2_path, v2)]:
        if legacy_row is None:
            continue
        legacy_name = normalized_name(str(legacy_row.get("canonical_beverage_name", "")))
        for claim in audited_claims.get((legacy_path.name, legacy_row["row_number"]), []):
            claim_name = normalized_name(str(claim.get("ingredient", "")))
            if claim_name != legacy_name:
                if stats is not None:
                    stats.mismatched_claims_skipped += 1
                continue
            audit_rows.append(claim)
            if stats is not None:
                stats.attached_claims += 1
    return {
        "v1": v1
        or {
            "row_number": None,
            "canonical_beverage_name": "",
            "positive": "",
            "negative": "",
            "positive_tags": [],
            "negative_tags": [],
        },
        "v2": v2
        or {
            "row_number": None,
            "canonical_beverage_name": "",
            "positive": "",
            "negative": "",
            "positive_tags": [],
            "negative_tags": [],
        },
        "audited_claims": audit_rows,
    }


def load_composition_summaries(path: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        canonical_id = str(row.get("canonical_beverage_id", "")).strip()
        if canonical_id:
            summaries[canonical_id] = row
    return summaries


def compact_composition_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "serving_basis": row.get("serving_basis", "100g"),
        "strong_evidence": row.get("strong_evidence", []),
        "medium_evidence": row.get("medium_evidence", []),
        "low_evidence": row.get("low_evidence", []),
        "negative_or_cautionary_effects": row.get("negative_or_cautionary_effects", []),
        "supplement_level_only_effects": row.get("supplement_level_only_effects", []),
        "caveats": row.get("caveats", []),
        "overall_summary": row.get("overall_summary", ""),
        "summary_status": row.get("summary_status", ""),
        "model_provider": row.get("model_provider", ""),
        "model_name": row.get("model_name", ""),
    }


def run_suffix(path: Path) -> str:
    name = path.name
    if "_" not in name:
        return name
    return name.split("_", 1)[1]


def discover_run_dirs(ingredient_runs_dir: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    if not ingredient_runs_dir.exists():
        return runs
    for packet_path in ingredient_runs_dir.glob("*/ingredient_packet.json"):
        runs.setdefault(run_suffix(packet_path.parent), packet_path.parent)
    return runs


def resolve_direct_run(row: dict[str, str], ingredient_runs_dir: Path, run_dirs: dict[str, Path]) -> Path | None:
    explicit = row.get("run_path", "").strip()
    if explicit:
        explicit_path = Path(explicit)
        candidates = [explicit_path]
        if not explicit_path.is_absolute():
            candidates.append((Path.cwd() / explicit_path).resolve())
        for candidate in candidates:
            if (candidate / "ingredient_packet.json").exists():
                return candidate

    for value in [row.get("final_run_target_id", ""), row.get("final_run_target_name", "")]:
        target_slug = slugify(value)
        if target_slug in run_dirs:
            return run_dirs[target_slug]

    # The caller may pass a temp ingredient_runs directory while rows contain relative paths.
    for value in [row.get("final_run_target_id", ""), row.get("final_run_target_name", "")]:
        target_slug = slugify(value)
        candidate = ingredient_runs_dir / f"unused-prefix_{target_slug}"
        if (candidate / "ingredient_packet.json").exists():
            return candidate
    return None


def resolve_claim_file(run_dir: Path, mapped_claim_file: str) -> Path | None:
    mapped = mapped_claim_file.strip()
    if mapped and (run_dir / mapped).exists():
        return run_dir / mapped
    for filename in DIRECT_CLAIM_FILE_PRECEDENCE:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_json_file(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def build_direct_literature_index(
    included_rows: list[dict[str, str]],
    ingredient_runs_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    run_dirs = discover_run_dirs(ingredient_runs_dir)
    direct_by_id: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, str] = {}

    for row in included_rows:
        canonical_id = row.get("canonical_beverage_id", "").strip()
        if not canonical_id:
            continue
        run_dir = resolve_direct_run(row, ingredient_runs_dir, run_dirs)
        if run_dir is None:
            unresolved[canonical_id] = "missing_direct_literature_run"
            continue
        claim_file = resolve_claim_file(run_dir, row.get("claim_file", ""))
        if claim_file is None:
            unresolved[canonical_id] = "missing_direct_literature_claims"
            continue
        claims = load_json_file(claim_file)
        if not isinstance(claims, list) or not claims:
            unresolved[canonical_id] = "missing_direct_literature_claims"
            continue
        direct_by_id[canonical_id] = {
            "final_run_target_id": row.get("final_run_target_id", ""),
            "final_run_target_name": row.get("final_run_target_name", ""),
            "run_path": str(run_dir),
            "claim_file": claim_file.name,
            "claims": claims,
        }
    return direct_by_id, unresolved


def source_paths_payload(
    *,
    legacy_v1_path: Path,
    legacy_v2_path: Path,
    claim_audit_path: Path,
    combined_map_path: Path,
    included_map_path: Path,
    composition_summaries_path: Path,
) -> dict[str, str]:
    return {
        "legacy_v1": str(legacy_v1_path),
        "legacy_v2": str(legacy_v2_path),
        "claim_audit": str(claim_audit_path),
        "combined_map": str(combined_map_path),
        "included_map": str(included_map_path),
        "composition_summaries": str(composition_summaries_path),
    }


def confidence_status(
    *,
    source_type: str,
    direct: dict[str, Any] | None,
    composition: dict[str, Any] | None,
) -> str:
    if source_type == "direct_literature":
        return "ready" if direct and direct.get("claims") else "partial"
    if composition is None:
        return "partial"
    effect_count = sum(
        len(composition.get(key, []))
        for key in [
            "strong_evidence",
            "medium_evidence",
            "low_evidence",
            "negative_or_cautionary_effects",
            "supplement_level_only_effects",
        ]
    )
    return "ready" if effect_count else "partial"


def assemble_final_export_inputs(
    *,
    legacy_v1_path: Path,
    legacy_v2_path: Path,
    claim_audit_path: Path,
    combined_map_path: Path,
    included_map_path: Path,
    composition_summaries_path: Path,
    ingredient_runs_dir: Path,
    out_dir: Path,
) -> FinalExportInputAssemblyResult:
    for path in [
        legacy_v1_path,
        legacy_v2_path,
        claim_audit_path,
        combined_map_path,
        included_map_path,
        composition_summaries_path,
    ]:
        if not path.exists():
            raise ValueError(f"missing_input:{path}")
    if not ingredient_runs_dir.exists():
        raise ValueError(f"missing_ingredient_runs_dir:{ingredient_runs_dir}")

    combined_rows = read_csv_rows_with_row_number(combined_map_path)
    included_rows = read_csv_rows_with_row_number(included_map_path)
    legacy_v1 = load_legacy_rows(legacy_v1_path)
    legacy_v2 = load_legacy_rows(legacy_v2_path)
    audited_claims = load_audited_legacy_claims(claim_audit_path)
    composition_by_id = load_composition_summaries(composition_summaries_path)
    direct_by_id, unresolved_direct = build_direct_literature_index(
        included_rows,
        ingredient_runs_dir,
    )
    included_ids = {
        row.get("canonical_beverage_id", "").strip()
        for row in included_rows
        if row.get("canonical_beverage_id", "").strip()
    }
    paths_payload = source_paths_payload(
        legacy_v1_path=legacy_v1_path,
        legacy_v2_path=legacy_v2_path,
        claim_audit_path=claim_audit_path,
        combined_map_path=combined_map_path,
        included_map_path=included_map_path,
        composition_summaries_path=composition_summaries_path,
    )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    both_sources = 0
    legacy_v1_attached = 0
    legacy_v2_attached = 0
    legacy_stats = LegacyAuditAttachmentStats()

    for map_row in combined_rows:
        canonical_id = map_row.get("canonical_beverage_id", "").strip()
        canonical_name = map_row.get("canonical_beverage_name", "").strip()
        if not canonical_id:
            continue
        direct = direct_by_id.get(canonical_id)
        composition_raw = composition_by_id.get(canonical_id)
        composition = compact_composition_summary(composition_raw) if composition_raw else None

        if direct is None and composition is None:
            reason = unresolved_direct.get(canonical_id, "no_new_summary_source")
            if canonical_id not in included_ids:
                reason = "no_new_summary_source"
            skipped.append(
                {
                    "canonical_beverage_id": canonical_id,
                    "canonical_beverage_name": canonical_name,
                    "reason": reason,
                    "details": "",
                }
            )
            continue

        source_type = "direct_literature" if direct is not None else "composition_based"
        if direct is not None and composition is not None:
            both_sources += 1
        legacy = legacy_payload(
            canonical_id=canonical_id,
            legacy_v1=legacy_v1,
            legacy_v2=legacy_v2,
            audited_claims=audited_claims,
            legacy_v1_path=legacy_v1_path,
            legacy_v2_path=legacy_v2_path,
            stats=legacy_stats,
        )
        if legacy["v1"]["row_number"] is not None:
            legacy_v1_attached += 1
        if legacy["v2"]["row_number"] is not None:
            legacy_v2_attached += 1

        rows.append(
            {
                "canonical_beverage_id": canonical_id,
                "canonical_beverage_name": canonical_name,
                "summary_source_type": source_type,
                "summary_confidence_status": confidence_status(
                    source_type=source_type,
                    direct=direct,
                    composition=composition,
                ),
                "direct_literature": direct,
                "composition_summary": composition,
                "secondary_composition_summary": composition
                if direct is not None and composition is not None
                else None,
                "legacy": legacy,
                "source_paths": paths_payload,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / FINAL_EXPORT_MERGE_INPUTS_JSONL, rows)
    write_csv(
        out_dir / SKIPPED_OR_UNCOVERED_INGREDIENTS_CSV,
        skipped,
        ["canonical_beverage_id", "canonical_beverage_name", "reason", "details"],
    )
    summary = {
        "combined_map_rows": len(combined_rows),
        "written_rows": len(rows),
        "direct_literature_rows": sum(
            1 for row in rows if row["summary_source_type"] == "direct_literature"
        ),
        "composition_based_rows": sum(
            1 for row in rows if row["summary_source_type"] == "composition_based"
        ),
        "both_sources_rows": both_sources,
        "skipped_rows": len(skipped),
        "legacy_v1_attached_rows": legacy_v1_attached,
        "legacy_v2_attached_rows": legacy_v2_attached,
        "legacy_audit_claims_attached": legacy_stats.attached_claims,
        "legacy_audit_claims_skipped_name_mismatch": legacy_stats.mismatched_claims_skipped,
        "resolver_warnings": len(unresolved_direct),
        "unresolved_direct_literature_rows": len(unresolved_direct),
    }
    write_json(out_dir / FINAL_EXPORT_MERGE_INPUT_SUMMARY_JSON, summary)

    return FinalExportInputAssemblyResult(
        out_dir=out_dir,
        rows=rows,
        skipped_rows=skipped,
        summary=summary,
    )
