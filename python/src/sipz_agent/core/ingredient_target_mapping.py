from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any


INGREDIENT_TARGET_MAP_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "canonical_search_name",
    "group_role",
    "adaptation",
    "preparation_reason",
    "preparation_confidence",
    "final_mapping_status",
    "final_run_target_id",
    "final_run_target_name",
    "final_representative_canonical_beverage_id",
    "final_representative_canonical_beverage_name",
    "collapse_reason_code",
    "collapse_reason",
    "pubmed_exclusion_code",
    "pubmed_exclusion_reason",
    "final_exclusion_code",
    "final_exclusion_reason",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INGREDIENT_TARGET_MAP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in INGREDIENT_TARGET_MAP_COLUMNS})


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def parse_json_list(value: str) -> list[str]:
    if not value.strip():
        return []
    raw = json.loads(value)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def load_preparation_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    decisions = raw.get("decisions", raw) if isinstance(raw, dict) else raw
    if not isinstance(decisions, list):
        return {}
    by_id: dict[str, dict[str, str]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        canonical_id = str(decision.get("canonical_beverage_id") or "")
        if not canonical_id:
            continue
        by_id[canonical_id] = {key: "" if value is None else str(value) for key, value in decision.items()}
    return by_id


def coverage_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        names = parse_json_list(row.get("covered_target_names", "[]"))
        for name in names:
            key = normalize_key(name)
            if key and key not in index:
                index[key] = row
    return index


def merge_preparation_row(
    *,
    canonical_id: str,
    decisions_by_id: dict[str, dict[str, str]],
    equivalence_by_id: dict[str, dict[str, str]],
    skip_by_id: dict[str, dict[str, str]],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (
        decisions_by_id.get(canonical_id, {}),
        equivalence_by_id.get(canonical_id, {}),
        skip_by_id.get(canonical_id, {}),
    ):
        for key, value in source.items():
            if value != "" or key not in merged:
                merged[key] = value
    return merged


def mapping_lookup_names(preparation: dict[str, str]) -> list[str]:
    names = [
        preparation.get("research_target_name", ""),
        preparation.get("canonical_search_name", ""),
        preparation.get("group_id", "").replace("_", " "),
        preparation.get("research_target_group_id", "").replace("_", " "),
    ]
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = normalize_key(name)
        if key and key not in seen:
            output.append(name)
            seen.add(key)
    return output


def find_target(
    preparation: dict[str, str],
    index: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    for name in mapping_lookup_names(preparation):
        target = index.get(normalize_key(name))
        if target is not None:
            return target
    return None


def build_ingredient_target_map_rows(
    *,
    ingredient_rows: list[dict[str, str]],
    equivalence_rows: list[dict[str, str]],
    final_target_rows: list[dict[str, str]],
    final_excluded_rows: list[dict[str, str]],
    skip_rows: list[dict[str, str]],
    pubmed_excluded_rows: list[dict[str, str]] | None = None,
    preparation_decisions: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    decisions_by_id = preparation_decisions or {}
    equivalence_by_id = {row["canonical_beverage_id"]: row for row in equivalence_rows}
    skip_by_id = {row["canonical_beverage_id"]: row for row in skip_rows}
    final_by_covered_name = coverage_index(final_target_rows)
    excluded_by_covered_name = coverage_index(final_excluded_rows)
    pubmed_excluded_by_name = {
        normalize_key(row.get("canonical_search_name", "")): row
        for row in pubmed_excluded_rows or []
        if row.get("canonical_search_name", "")
    }

    output: list[dict[str, str]] = []
    seen_ids: set[str] = set()
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
        final_target = find_target(preparation, final_by_covered_name)
        excluded_target = find_target(preparation, excluded_by_covered_name)
        pubmed_excluded_target = None
        for name in mapping_lookup_names(preparation):
            pubmed_excluded_target = pubmed_excluded_by_name.get(normalize_key(name))
            if pubmed_excluded_target is not None:
                break
        decision = preparation.get("decision", "")
        if final_target is not None:
            status = "mapped_to_run_target"
            target = final_target
        elif excluded_target is not None:
            status = "excluded_final"
            target = excluded_target
        elif pubmed_excluded_target is not None:
            status = "excluded_pubmed_ready"
            target = pubmed_excluded_target
        elif decision == "skip_low_value":
            status = "skipped_low_value"
            target = {}
        elif decision == "manual_review":
            status = "manual_review_unmapped"
            target = {}
        else:
            status = "unmapped"
            target = {}

        output.append(
            {
                "canonical_beverage_id": canonical_id,
                "canonical_beverage_name": ingredient.get("canonical_beverage_name", ""),
                "decision": decision,
                "relationship": preparation.get("relationship", ""),
                "canonical_search_name": preparation.get("canonical_search_name", ""),
                "group_role": preparation.get("group_role", ""),
                "adaptation": preparation.get("adaptation", ""),
                "preparation_reason": preparation.get("reason", ""),
                "preparation_confidence": preparation.get("confidence", ""),
                "final_mapping_status": status,
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
                "pubmed_exclusion_code": target.get("exclusion_code", ""),
                "pubmed_exclusion_reason": target.get("exclusion_reason", ""),
                "final_exclusion_code": target.get("final_exclusion_code", ""),
                "final_exclusion_reason": target.get("final_exclusion_reason", ""),
            }
        )
    return output


def build_ingredient_target_map(
    *,
    lookup_path: Path,
    equivalence_path: Path,
    final_targets_path: Path,
    final_excluded_path: Path,
    skip_list_path: Path,
    pubmed_excluded_path: Path | None = None,
    out_path: Path,
    preparation_decisions_path: Path | None = None,
) -> list[dict[str, str]]:
    rows = build_ingredient_target_map_rows(
        ingredient_rows=read_csv_rows(lookup_path),
        equivalence_rows=read_csv_rows(equivalence_path),
        final_target_rows=read_csv_rows(final_targets_path),
        pubmed_excluded_rows=(
            read_csv_rows(pubmed_excluded_path)
            if pubmed_excluded_path is not None and pubmed_excluded_path.exists()
            else []
        ),
        final_excluded_rows=read_csv_rows(final_excluded_path) if final_excluded_path.exists() else [],
        skip_rows=read_csv_rows(skip_list_path),
        preparation_decisions=load_preparation_decisions(preparation_decisions_path),
    )
    write_csv_rows(out_path, rows)
    return rows
