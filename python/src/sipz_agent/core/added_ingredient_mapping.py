from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

import orjson

from sipz_agent.core.ingredient_target_mapping import (
    load_preparation_decisions,
    merge_preparation_row,
    normalize_key,
    parse_json_list,
    read_csv_rows,
)


ADDED_INGREDIENT_MAP_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "canonical_search_name",
    "group_role",
    "adaptation",
    "preparation_reason",
    "preparation_confidence",
    "mapping_status",
    "mapped_target_source",
    "final_run_target_id",
    "final_run_target_name",
    "final_representative_canonical_beverage_id",
    "final_representative_canonical_beverage_name",
    "collapse_reason_code",
    "collapse_reason",
    "run_path",
    "claim_file",
    "web_audited_claim_count",
    "include_in_export",
    "needs_manual_review",
    "exclusion_reason",
]


def write_csv_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def slugify_run_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold())
    return re.sub(r"-+", "-", slug).strip("-")


def mapping_lookup_names(preparation: dict[str, str], ingredient: dict[str, str]) -> list[str]:
    names = [
        preparation.get("research_target_name", ""),
        preparation.get("canonical_search_name", ""),
        preparation.get("group_id", "").replace("_", " "),
        preparation.get("research_target_group_id", "").replace("_", " "),
        ingredient.get("canonical_beverage_name", ""),
    ]
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = normalize_key(name)
        if key and key not in seen:
            output.append(name)
            seen.add(key)
    return output


def target_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        names = [row.get("run_target_name", "")]
        names.extend(parse_json_list(row.get("covered_target_names", "[]")))
        for name in names:
            key = normalize_key(name)
            if key and key not in index:
                index[key] = row
    return index


def find_target(
    *,
    preparation: dict[str, str],
    ingredient: dict[str, str],
    index: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    for name in mapping_lookup_names(preparation, ingredient):
        target = index.get(normalize_key(name))
        if target is not None:
            return target
    return None


def latest_run_for_target(target_name: str, runs_root: Path) -> Path | None:
    slug = slugify_run_name(target_name)
    matches = sorted(runs_root.glob(f"*_{slug}"), key=lambda path: path.name)
    return matches[-1] if matches else None


def claim_artifact_for_run(run_path: Path | None) -> tuple[str, str]:
    if run_path is None:
        return "", "0"
    for filename in ("web_audited_ingredient_claims.json", "audited_ingredient_claims.json"):
        path = run_path / filename
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return filename, "0"
        if isinstance(raw, list):
            return filename, str(len(raw))
        return filename, "0"
    return "", "0"


def target_fields(target: dict[str, str]) -> dict[str, str]:
    return {
        "final_run_target_id": target.get("run_target_id", ""),
        "final_run_target_name": target.get("run_target_name", ""),
        "final_representative_canonical_beverage_id": target.get(
            "representative_canonical_beverage_id", ""
        ),
        "final_representative_canonical_beverage_name": target.get(
            "representative_canonical_beverage_name", ""
        ),
        "collapse_reason_code": target.get("collapse_reason_code", ""),
        "collapse_reason": target.get("collapse_reason", ""),
    }


def build_added_ingredient_mapping_rows(
    *,
    ingredient_rows: list[dict[str, str]],
    equivalence_rows: list[dict[str, str]],
    skip_rows: list[dict[str, str]],
    original_target_rows: list[dict[str, str]],
    added_target_rows: list[dict[str, str]],
    original_alias_target_rows: list[dict[str, str]] | None = None,
    added_excluded_target_rows: list[dict[str, str]] | None = None,
    preparation_decisions: dict[str, dict[str, str]] | None = None,
    runs_root: Path = Path("ingredient_runs"),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    decisions_by_id = preparation_decisions or {}
    equivalence_by_id = {row["canonical_beverage_id"]: row for row in equivalence_rows}
    skip_by_id = {row["canonical_beverage_id"]: row for row in skip_rows}
    original_index = target_index([*original_target_rows, *(original_alias_target_rows or [])])
    added_index = target_index(added_target_rows)
    added_excluded_index = target_index(added_excluded_target_rows or [])

    output: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for ingredient in ingredient_rows:
        canonical_id = ingredient.get("canonical_beverage_id", "")
        if canonical_id in seen_ids:
            raise ValueError(f"duplicate_canonical_beverage_id:{canonical_id}")
        seen_ids.add(canonical_id)

        preparation = merge_preparation_row(
            canonical_id=canonical_id,
            decisions_by_id=decisions_by_id,
            equivalence_by_id=equivalence_by_id,
            skip_by_id=skip_by_id,
        )
        decision = preparation.get("decision", "")

        if decision == "skip_low_value":
            mapping_status = "skipped_low_value"
            mapped_source = "skipped_low_value"
            exclusion_reason = preparation.get("reason", "")
            needs_manual_review = "false"
            target = {}
        elif decision == "manual_review":
            mapping_status = "manual_review_unmapped"
            mapped_source = "manual_review"
            exclusion_reason = preparation.get("reason", "")
            needs_manual_review = "true"
            target = {}
        else:
            target = find_target(
                preparation=preparation,
                ingredient=ingredient,
                index=original_index,
            )
            if target is not None:
                mapping_status = "mapped_to_original_corpus_target"
                mapped_source = "original_corpus"
                exclusion_reason = ""
                needs_manual_review = "false"
            else:
                target = find_target(
                    preparation=preparation,
                    ingredient=ingredient,
                    index=added_index,
                )
                if target is not None:
                    mapping_status = "mapped_to_added_batch_target"
                    mapped_source = "added_82_batch"
                    exclusion_reason = ""
                    needs_manual_review = "false"
                else:
                    target = find_target(
                        preparation=preparation,
                        ingredient=ingredient,
                        index=added_excluded_index,
                    )
                    if target is not None:
                        mapping_status = "excluded_after_crunch"
                        mapped_source = "excluded_after_crunch"
                        exclusion_reason = target.get("collapse_reason", "")
                        needs_manual_review = "false"
                    else:
                        mapping_status = "unmatched"
                        mapped_source = "unmatched"
                        exclusion_reason = "No matching original or added research target was found."
                        needs_manual_review = "true"
                        target = {}

        if mapped_source in {"original_corpus", "added_82_batch"}:
            needs_manual_review = "false"

        target_name = target.get("run_target_name", "")
        run_path = latest_run_for_target(target_name, runs_root) if target_name else None
        claim_file, claim_count = claim_artifact_for_run(run_path)
        include_in_export = (
            "true"
            if mapped_source in {"original_corpus", "added_82_batch"} and int(claim_count or "0") > 0
            else "false"
        )
        row = {
            "canonical_beverage_id": canonical_id,
            "canonical_beverage_name": ingredient.get("canonical_beverage_name", ""),
            "decision": decision,
            "relationship": preparation.get("relationship", ""),
            "canonical_search_name": preparation.get("canonical_search_name", ""),
            "group_role": preparation.get("group_role", ""),
            "adaptation": preparation.get("adaptation", ""),
            "preparation_reason": preparation.get("reason", ""),
            "preparation_confidence": preparation.get("confidence", ""),
            "mapping_status": mapping_status,
            "mapped_target_source": mapped_source,
            **target_fields(target),
            "run_path": str(run_path) if run_path else "",
            "claim_file": claim_file,
            "web_audited_claim_count": claim_count,
            "include_in_export": include_in_export,
            "needs_manual_review": needs_manual_review,
            "exclusion_reason": exclusion_reason,
        }
        output.append(row)
        status_counts[mapping_status] = status_counts.get(mapping_status, 0) + 1
        source_counts[mapped_source] = source_counts.get(mapped_source, 0) + 1

    summary = {
        "total_rows": len(output),
        "mapping_status_counts": status_counts,
        "mapped_target_source_counts": source_counts,
        "include_in_export_count": sum(1 for row in output if row["include_in_export"] == "true"),
        "not_included_count": sum(1 for row in output if row["include_in_export"] != "true"),
    }
    return output, summary


def build_added_ingredient_mapping(
    *,
    lookup_path: Path,
    equivalence_path: Path,
    skip_list_path: Path,
    original_targets_path: Path,
    added_targets_path: Path,
    out_path: Path,
    not_included_out_path: Path,
    summary_path: Path,
    original_alias_targets_path: Path | None = None,
    added_excluded_targets_path: Path | None = None,
    preparation_decisions_path: Path | None = None,
    runs_root: Path = Path("ingredient_runs"),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows, summary = build_added_ingredient_mapping_rows(
        ingredient_rows=read_csv_rows(lookup_path),
        equivalence_rows=read_csv_rows(equivalence_path),
        skip_rows=read_csv_rows(skip_list_path),
        original_target_rows=read_csv_rows(original_targets_path),
        original_alias_target_rows=(
            read_csv_rows(original_alias_targets_path)
            if original_alias_targets_path is not None and original_alias_targets_path.exists()
            else []
        ),
        added_target_rows=read_csv_rows(added_targets_path),
        added_excluded_target_rows=(
            read_csv_rows(added_excluded_targets_path)
            if added_excluded_targets_path is not None and added_excluded_targets_path.exists()
            else []
        ),
        preparation_decisions=load_preparation_decisions(preparation_decisions_path),
        runs_root=runs_root,
    )
    not_included_rows = [row for row in rows if row["include_in_export"] != "true"]
    write_csv_rows(out_path, rows, ADDED_INGREDIENT_MAP_COLUMNS)
    write_csv_rows(not_included_out_path, not_included_rows, ADDED_INGREDIENT_MAP_COLUMNS)
    summary_path.write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    return rows, summary
