from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, NEW_EFFECT_COLUMNS, stable_uuid


DEFAULT_INPUT = Path("combined_internal_export/canonicalized")
DEFAULT_OUTPUT = Path("combined_internal_export/import_ready")
DEFAULT_EXISTING = Path("health_effects_rows(1).csv")


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


def unique_slug(base_slug: str, unavailable: set[str]) -> str:
    candidate = f"{base_slug}_generated"
    if candidate not in unavailable:
        unavailable.add(candidate)
        return candidate
    counter = 2
    while True:
        candidate = f"{base_slug}_generated_{counter}"
        if candidate not in unavailable:
            unavailable.add(candidate)
            return candidate
        counter += 1


def build_slug_renames(
    *,
    health_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
) -> dict[str, str]:
    existing_slugs = {
        row.get("effect_slug", "").strip()
        for row in existing_rows
        if row.get("effect_slug", "").strip()
    }
    unavailable = {
        row.get("effect_slug", "").strip()
        for row in [*existing_rows, *health_rows]
        if row.get("effect_slug", "").strip()
    }
    return {
        row["effect_slug"]: unique_slug(row["effect_slug"], unavailable)
        for row in health_rows
        if row.get("effect_slug", "") in existing_slugs
    }


def apply_health_renames(
    health_rows: list[dict[str, str]],
    slug_renames: dict[str, str],
) -> list[dict[str, str]]:
    output = []
    for row in health_rows:
        updated = {column: row.get(column, "") for column in NEW_EFFECT_COLUMNS}
        new_slug = slug_renames.get(updated["effect_slug"])
        if new_slug:
            updated["effect_slug"] = new_slug
            updated["id"] = stable_uuid("health-effect", new_slug)
        output.append(updated)
    return output


def apply_evidence_renames(
    evidence_rows: list[dict[str, str]],
    slug_renames: dict[str, str],
    health_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    output = []
    for row in evidence_rows:
        updated = {column: row.get(column, "") for column in EVIDENCE_COLUMNS}
        new_slug = slug_renames.get(updated["effect_slug"])
        if new_slug:
            updated["effect_slug"] = new_slug
            updated["id"] = stable_uuid(
                "evidence",
                updated["bioactive_type"],
                updated["bioactive_id"],
                new_slug,
            )
        health_row = health_by_slug.get(updated["effect_slug"])
        if health_row is not None:
            updated["effect_label"] = health_row["effect_label"]
        output.append(updated)
    return output


def validate_outputs(
    *,
    existing_rows: list[dict[str, str]],
    health_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
) -> dict[str, int]:
    existing_slugs = {
        row.get("effect_slug", "").strip()
        for row in existing_rows
        if row.get("effect_slug", "").strip()
    }
    health_slugs = [row["effect_slug"] for row in health_rows]
    evidence_ids = [row["id"] for row in evidence_rows]
    evidence_keys = [
        (row["bioactive_type"], row["bioactive_id"], row["effect_slug"])
        for row in evidence_rows
    ]
    import_existing_conflicts = set(health_slugs) & existing_slugs
    evidence_slugs_not_in_import_or_existing = {
        row["effect_slug"]
        for row in evidence_rows
        if row["effect_slug"] not in health_slugs and row["effect_slug"] not in existing_slugs
    }
    return {
        "health_rows": len(health_rows),
        "evidence_rows": len(evidence_rows),
        "existing_health_rows": len(existing_rows),
        "existing_health_slugs": len(existing_slugs),
        "import_health_slugs_conflicting_with_existing": len(import_existing_conflicts),
        "duplicate_import_health_slugs": len(health_slugs) - len(set(health_slugs)),
        "duplicate_evidence_ids": len(evidence_ids) - len(set(evidence_ids)),
        "duplicate_evidence_entity_effects": len(evidence_keys) - len(set(evidence_keys)),
        "evidence_slugs_not_in_import_or_existing": len(
            evidence_slugs_not_in_import_or_existing
        ),
    }


def prepare_import(
    *,
    input_dir: Path,
    output_dir: Path,
    existing_health_effects_path: Path,
) -> dict[str, int]:
    health_rows = read_csv(input_dir / "health_effects_new.combined.csv")
    evidence_rows = read_csv(input_dir / "bioactive_health_evidence_rows.generated.combined.csv")
    existing_rows = read_csv(existing_health_effects_path)
    slug_renames = build_slug_renames(
        health_rows=health_rows,
        existing_rows=existing_rows,
    )

    renamed_health_rows = apply_health_renames(health_rows, slug_renames)
    health_by_slug = {row["effect_slug"]: row for row in renamed_health_rows}
    renamed_evidence_rows = apply_evidence_renames(
        evidence_rows,
        slug_renames,
        health_by_slug,
    )
    validation = validate_outputs(
        existing_rows=existing_rows,
        health_rows=renamed_health_rows,
        evidence_rows=renamed_evidence_rows,
    )

    write_csv(
        output_dir / "health_effects_new.combined.csv",
        NEW_EFFECT_COLUMNS,
        renamed_health_rows,
    )
    write_csv(
        output_dir / "bioactive_health_evidence_rows.generated.combined.csv",
        EVIDENCE_COLUMNS,
        renamed_evidence_rows,
    )
    write_csv(
        output_dir / "health_effect_existing_slug_renames.csv",
        ["old_effect_slug", "new_effect_slug", "reason"],
        [
            {
                "old_effect_slug": old_slug,
                "new_effect_slug": new_slug,
                "reason": "The old slug already exists in the exported health_effects table.",
            }
            for old_slug, new_slug in sorted(slug_renames.items())
        ],
    )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "existing_health_effects_path": str(existing_health_effects_path.resolve()),
        "renamed_health_effect_slugs": len(slug_renames),
        **validation,
    }
    (output_dir / "import_ready_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "IMPORT_INSTRUCTIONS.md").write_text(
        """# Import-Ready Combined Export

These files have had new health-effect slug collisions renamed against the current
health_effects export.

Import into Supabase in this order:

1. `health_effects_new.combined.csv` -> `beverage.health_effects`
2. `bioactive_health_evidence_rows.generated.combined.csv` -> `beverage.bioactive_health_evidence`

Review before import:

- `health_effect_existing_slug_renames.csv`
- `import_ready_summary.json`
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename new health-effect slugs that already exist in Supabase."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--existing-health-effects", type=Path, default=DEFAULT_EXISTING)
    args = parser.parse_args()

    summary = prepare_import(
        input_dir=args.input,
        output_dir=args.out,
        existing_health_effects_path=args.existing_health_effects,
    )
    print(f"Wrote import-ready exports under: {args.out}")
    print(f"Health effects: {summary['health_rows']}")
    print(f"Evidence rows: {summary['evidence_rows']}")
    print(f"Renamed health-effect slugs: {summary['renamed_health_effect_slugs']}")
    print(
        "Import health slugs conflicting with existing: "
        f"{summary['import_health_slugs_conflicting_with_existing']}"
    )
    print(f"Duplicate import health slugs: {summary['duplicate_import_health_slugs']}")
    print(f"Duplicate evidence IDs: {summary['duplicate_evidence_ids']}")
    print(
        "Evidence slugs not in import or existing: "
        f"{summary['evidence_slugs_not_in_import_or_existing']}"
    )


if __name__ == "__main__":
    main()
