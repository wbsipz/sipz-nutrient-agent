from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, NEW_EFFECT_COLUMNS, stable_uuid


DEFAULT_INPUT = Path("combined_internal_export")
DEFAULT_OUTPUT = Path("combined_internal_export/canonicalized")

ALIASES = {
    "adhd_symptom_improvement": (
        "adhd_symptom_reduction",
        "Same ADHD symptom concept with reduction/improvement wording.",
    ),
    "anxiolytic": (
        "anxiety_reduction",
        "Consumer-facing duplicate of anxiety reduction.",
    ),
    "cholesterol_lowering": (
        "cholesterol_reduction",
        "Same cholesterol-lowering concept.",
    ),
    "lipid_lowering": (
        "improved_lipid_profile",
        "Same broad lipid profile improvement concept.",
    ),
    "glycemic_control": (
        "improved_glycemic_control",
        "Same glycemic regulation improvement concept.",
    ),
    "postprandial_glycemia_reduction": (
        "reduced_postprandial_glucose",
        "Same post-meal glucose reduction concept.",
    ),
    "sleep_quality_improvement": (
        "improved_sleep_quality",
        "Same sleep quality improvement concept.",
    ),
    "lowered_systolic_blood_pressure": (
        "systolic_blood_pressure_reduction",
        "Same systolic blood pressure reduction concept.",
    ),
    "reduced_diabetes_risk": (
        "reduced_type_2_diabetes_risk",
        "Current evidence rows describe type 2 diabetes risk.",
    ),
}

CANONICAL_LABELS = {
    "menopausal_symptom_relief": "Menopausal symptom relief",
    "reduced_delayed_onset_muscle_soreness": "Reduced delayed-onset muscle soreness",
    "reduced_postprandial_glucose": "Reduced postprandial glucose",
    "systolic_blood_pressure_reduction": "Systolic blood pressure reduction",
}

EVIDENCE_LEVEL_RANK = {
    "limited": 1,
    "moderate": 2,
    "strong": 3,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_json_list(value: str) -> list:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def json_cell(value: list) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_slug(slug: str) -> str:
    seen = set()
    current = slug
    while current in ALIASES:
        if current in seen:
            raise ValueError(f"cyclic_alias:{slug}")
        seen.add(current)
        current = ALIASES[current][0]
    return current


def merge_tags(rows: list[dict[str, str]]) -> list[str]:
    tags: list[str] = []
    for row in rows:
        for tag in parse_json_list(row.get("tags", "")):
            if isinstance(tag, str):
                normalized = tag.strip().lower().replace("-", "_").replace(" ", "_")
                if normalized and normalized.replace("_", "").isalnum():
                    tags.append(normalized)
    return list(dict.fromkeys(tags))[:12]


def earliest(rows: list[dict[str, str]], column: str) -> str:
    values = sorted(row.get(column, "") for row in rows if row.get(column, "").strip())
    return values[0] if values else datetime.now(UTC).isoformat()


def latest(rows: list[dict[str, str]], column: str) -> str:
    values = sorted(row.get(column, "") for row in rows if row.get(column, "").strip())
    return values[-1] if values else datetime.now(UTC).isoformat()


def best_evidence_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            EVIDENCE_LEVEL_RANK.get(row.get("evidence_level", ""), 0),
            float(row.get("score", "0") or 0),
            len(row.get("description", "")),
        ),
    )


def merge_sources(rows: list[dict[str, str]]) -> list[dict]:
    sources: list[dict] = []
    seen = set()
    for row in rows:
        for source in parse_json_list(row.get("sources", "")):
            if not isinstance(source, dict):
                continue
            key = (
                source.get("doi") or "",
                source.get("pmid") or "",
                source.get("url") or "",
                source.get("title") or "",
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
    return sources


def merge_review_notes(rows: list[dict[str, str]]) -> str:
    notes = [
        row.get("review_notes", "").strip()
        for row in rows
        if row.get("review_notes", "").strip()
    ]
    return " | ".join(dict.fromkeys(notes))


def canonicalize_health_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(canonical_slug(row["effect_slug"]), []).append(row)

    output: list[dict[str, str]] = []
    by_slug: dict[str, dict[str, str]] = {}
    for slug, group in sorted(grouped.items()):
        base = next((row for row in group if row["effect_slug"] == slug), group[-1])
        merged = {column: base.get(column, "") for column in NEW_EFFECT_COLUMNS}
        merged["id"] = stable_uuid("health-effect", slug)
        merged["effect_slug"] = slug
        merged["effect_label"] = CANONICAL_LABELS.get(slug, merged["effect_label"])
        merged["tags"] = json_cell(merge_tags(group))
        merged["created_at"] = earliest(group, "created_at")
        merged["updated_at"] = latest(group, "updated_at")
        output.append(merged)
        by_slug[slug] = merged
    return output, by_slug


def canonicalize_evidence_rows(
    rows: list[dict[str, str]],
    health_by_slug: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    moved_rows: list[dict[str, str]] = []

    for row in rows:
        original_slug = row["effect_slug"]
        target_slug = canonical_slug(original_slug)
        updated = {column: row.get(column, "") for column in EVIDENCE_COLUMNS}
        updated["effect_slug"] = target_slug
        if target_slug in health_by_slug:
            updated["effect_label"] = health_by_slug[target_slug]["effect_label"]
        updated["id"] = stable_uuid(
            "evidence",
            updated["bioactive_type"],
            updated["bioactive_id"],
            target_slug,
        )
        grouped.setdefault(
            (updated["bioactive_type"], updated["bioactive_id"], target_slug),
            [],
        ).append(updated)
        if target_slug != original_slug:
            moved_rows.append(
                {
                    "old_effect_slug": original_slug,
                    "canonical_effect_slug": target_slug,
                    "bioactive_name": row.get("bioactive_name", ""),
                    "reason": ALIASES[original_slug][1],
                }
            )

    output: list[dict[str, str]] = []
    duplicate_report: list[dict[str, str]] = []
    for key, group in sorted(grouped.items()):
        base = best_evidence_row(group)
        merged = {column: base.get(column, "") for column in EVIDENCE_COLUMNS}
        merged["id"] = stable_uuid("evidence", key[0], key[1], key[2])
        merged["effect_slug"] = key[2]
        if key[2] in health_by_slug:
            merged["effect_label"] = health_by_slug[key[2]]["effect_label"]
        merged["tags"] = json_cell(merge_tags(group))
        merged["sources"] = json_cell(merge_sources(group))
        merged["review_notes"] = merge_review_notes(group)
        merged["created_at"] = earliest(group, "created_at")
        merged["updated_at"] = latest(group, "updated_at")
        output.append(merged)
        if len(group) > 1:
            duplicate_report.append(
                {
                    "bioactive_type": key[0],
                    "bioactive_id": key[1],
                    "bioactive_name": base.get("bioactive_name", ""),
                    "canonical_effect_slug": key[2],
                    "merged_row_count": str(len(group)),
                    "kept_evidence_level": merged["evidence_level"],
                    "kept_score": merged["score"],
                }
            )
    return output, moved_rows + duplicate_report


def alias_report_rows(health_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_slug = {row["effect_slug"]: row for row in health_rows}
    rows = []
    for old_slug, (target_slug, reason) in sorted(ALIASES.items()):
        old_row = by_slug.get(old_slug, {})
        target_row = by_slug.get(target_slug, {})
        rows.append(
            {
                "old_effect_slug": old_slug,
                "canonical_effect_slug": target_slug,
                "old_effect_label": old_row.get("effect_label", ""),
                "canonical_effect_label": CANONICAL_LABELS.get(
                    target_slug,
                    target_row.get("effect_label", ""),
                ),
                "reason": reason,
            }
        )
    return rows


def validate_outputs(
    health_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    existing_effect_slugs: set[str] | None = None,
) -> dict[str, int]:
    existing_effect_slugs = existing_effect_slugs or set()
    health_slugs = [row["effect_slug"] for row in health_rows]
    evidence_ids = [row["id"] for row in evidence_rows]
    evidence_keys = [
        (row["bioactive_type"], row["bioactive_id"], row["effect_slug"])
        for row in evidence_rows
    ]
    evidence_slugs_not_in_new_health_effects = {
        row["effect_slug"] for row in evidence_rows if row["effect_slug"] not in health_slugs
    }
    evidence_slugs_not_in_new_or_lookup = {
        slug
        for slug in evidence_slugs_not_in_new_health_effects
        if slug not in existing_effect_slugs
    }
    remaining_alias_slugs = {
        row["effect_slug"]
        for row in [*health_rows, *evidence_rows]
        if row["effect_slug"] in ALIASES
    }
    return {
        "health_rows": len(health_rows),
        "evidence_rows": len(evidence_rows),
        "duplicate_health_slugs": len(health_slugs) - len(set(health_slugs)),
        "duplicate_evidence_ids": len(evidence_ids) - len(set(evidence_ids)),
        "duplicate_evidence_entity_effects": len(evidence_keys) - len(set(evidence_keys)),
        "evidence_slugs_not_in_new_health_effects": len(
            evidence_slugs_not_in_new_health_effects
        ),
        "evidence_slugs_not_in_new_or_lookup": len(evidence_slugs_not_in_new_or_lookup),
        "remaining_alias_slugs": len(remaining_alias_slugs),
    }


def lookup_effect_slugs(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    rows = read_csv(path)
    return {row.get("effect_slug", "").strip() for row in rows if row.get("effect_slug", "").strip()}


def apply_aliases(input_dir: Path, output_dir: Path, lookup_path: Path | None = None) -> dict[str, int]:
    health_rows = read_csv(input_dir / "health_effects_new.combined.csv")
    evidence_rows = read_csv(input_dir / "bioactive_health_evidence_rows.generated.combined.csv")
    existing_effect_slugs = lookup_effect_slugs(lookup_path)

    canonical_health_rows, health_by_slug = canonicalize_health_rows(health_rows)
    canonical_evidence_rows, moved_rows = canonicalize_evidence_rows(
        evidence_rows,
        health_by_slug,
    )
    validation = validate_outputs(
        canonical_health_rows,
        canonical_evidence_rows,
        existing_effect_slugs,
    )

    write_csv(
        output_dir / "health_effects_new.combined.csv",
        NEW_EFFECT_COLUMNS,
        canonical_health_rows,
    )
    write_csv(
        output_dir / "bioactive_health_evidence_rows.generated.combined.csv",
        EVIDENCE_COLUMNS,
        canonical_evidence_rows,
    )
    write_csv(
        output_dir / "health_effect_slug_aliases_applied.csv",
        [
            "old_effect_slug",
            "canonical_effect_slug",
            "old_effect_label",
            "canonical_effect_label",
            "reason",
        ],
        alias_report_rows(health_rows),
    )
    write_csv(
        output_dir / "evidence_alias_merges.csv",
        [
            "old_effect_slug",
            "canonical_effect_slug",
            "bioactive_name",
            "reason",
            "bioactive_type",
            "bioactive_id",
            "merged_row_count",
            "kept_evidence_level",
            "kept_score",
        ],
        moved_rows,
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "lookup_path": str(lookup_path.resolve()) if lookup_path and lookup_path.exists() else "",
        "input_health_rows": len(health_rows),
        "input_evidence_rows": len(evidence_rows),
        "lookup_effect_slugs": len(existing_effect_slugs),
        "aliases_applied": len(ALIASES),
        **validation,
    }
    (output_dir / "canonicalization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "IMPORT_INSTRUCTIONS.md").write_text(
        """# Canonicalized Combined Export Import Order

Import these files into Supabase in this order:

1. `health_effects_new.combined.csv` -> `beverage.health_effects`
2. `bioactive_health_evidence_rows.generated.combined.csv` -> `beverage.bioactive_health_evidence`

Review these reports before import:

- `health_effect_slug_aliases_applied.csv`
- `evidence_alias_merges.csv`
- `canonicalization_summary.json`

`evidence_slugs_not_in_new_health_effects` can be greater than zero when those slugs
already exist in Supabase. `evidence_slugs_not_in_new_or_lookup` should be zero before
importing evidence rows.
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply reviewed health-effect slug aliases to combined export CSVs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path("bioactive_health_evidence_rows.csv"),
        help="Existing Supabase lookup CSV used to validate already-present effect slugs.",
    )
    args = parser.parse_args()

    summary = apply_aliases(args.input, args.out, args.lookup)
    print(f"Wrote canonicalized exports under: {args.out}")
    print(f"Health effects: {summary['health_rows']}")
    print(f"Evidence rows: {summary['evidence_rows']}")
    print(f"Aliases applied: {summary['aliases_applied']}")
    print(f"Duplicate health slugs: {summary['duplicate_health_slugs']}")
    print(f"Duplicate evidence IDs: {summary['duplicate_evidence_ids']}")
    print(
        "Evidence slugs not in new health effects: "
        f"{summary['evidence_slugs_not_in_new_health_effects']}"
    )
    print(
        "Evidence slugs not in new health effects or lookup: "
        f"{summary['evidence_slugs_not_in_new_or_lookup']}"
    )
    print(f"Remaining alias slugs: {summary['remaining_alias_slugs']}")


if __name__ == "__main__":
    main()
