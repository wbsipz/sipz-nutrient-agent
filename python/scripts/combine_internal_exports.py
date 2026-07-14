from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Iterable

import orjson
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, NEW_EFFECT_COLUMNS
from sipz_agent.core.models import LlmProvider, create_llm_provider

DEFAULT_OUTPUT = Path("combined_internal_export")


class ResolvedHealthEffect(BaseModel):
    effect_label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        tags = []
        for tag in value:
            normalized = tag.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized and normalized.replace("_", "").isalnum():
                tags.append(normalized)
        return list(dict.fromkeys(tags))[:12]


RESOLVED_HEALTH_EFFECT_ADAPTER = TypeAdapter(ResolvedHealthEffect)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def required_columns_present(path: Path, required: list[str]) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    return set(required).issubset(fieldnames)


def run_dir_for_export_file(path: Path) -> Path:
    return path.parent.parent


def sort_key(path: Path) -> str:
    return run_dir_for_export_file(path).name


def non_empty_value(row: dict[str, str], columns: list[str]) -> bool:
    return any(row.get(column, "").strip() for column in columns)


def same_row(left: dict[str, str], right: dict[str, str], columns: list[str]) -> bool:
    return all(left.get(column, "") == right.get(column, "") for column in columns)


def merge_unique_rows(
    *,
    files: list[Path],
    columns: list[str],
    key_column: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    by_key: dict[str, tuple[str, dict[str, str]]] = {}
    conflicts: list[dict[str, str]] = []
    duplicate_count = 0

    for path in sorted(files, key=sort_key):
        if not required_columns_present(path, columns):
            raise ValueError(f"missing_required_columns:{path}")
        run_name = run_dir_for_export_file(path).name
        for row in read_csv_rows(path):
            if not non_empty_value(row, columns):
                continue
            key = row.get(key_column, "").strip()
            if not key:
                continue
            normalized_row = {column: row.get(column, "") for column in columns}
            existing = by_key.get(key)
            if existing is not None:
                duplicate_count += 1
                existing_run, existing_row = existing
                if not same_row(existing_row, normalized_row, columns):
                    conflicts.append(
                        {
                            "key": key,
                            "kept_run": existing_run,
                            "replaced_by_run": run_name,
                            "file": str(path),
                        }
                    )
                # Keep the newest run because sorted run IDs are chronological.
            by_key[key] = (run_name, normalized_row)

    merged = [row for _, row in sorted(by_key.values(), key=lambda item: item[0])]
    return merged, conflicts, duplicate_count


def parse_tags(value: str) -> list[str]:
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]


def json_tags(tags: list[str]) -> str:
    return json.dumps(tags, ensure_ascii=False, separators=(",", ":"))


def health_effect_definition_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("effect_label", ""),
        row.get("description", ""),
        row.get("tags", ""),
    )


def collect_health_effect_definitions(
    files: list[Path],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for path in sorted(files, key=sort_key):
        if not required_columns_present(path, NEW_EFFECT_COLUMNS):
            raise ValueError(f"missing_required_columns:{path}")
        run_name = run_dir_for_export_file(path).name
        for row in read_csv_rows(path):
            if not non_empty_value(row, NEW_EFFECT_COLUMNS):
                continue
            slug = row.get("effect_slug", "").strip()
            if not slug:
                continue
            normalized = {column: row.get(column, "") for column in NEW_EFFECT_COLUMNS}
            normalized["source_run"] = run_name
            grouped.setdefault(slug, []).append(normalized)
    return grouped


def resolution_prompt(effect_slug: str, definitions: list[dict[str, str]]) -> str:
    payload = [
        {
            "source_run": definition["source_run"],
            "effect_label": definition["effect_label"],
            "description": definition["description"],
            "tags": parse_tags(definition.get("tags", "")),
        }
        for definition in definitions
    ]
    return f"""You are standardizing a health-effect taxonomy row for Sipz.

The same effect_slug was generated by multiple research runs with slightly different wording.
Create one canonical definition for this existing slug. Do not change the slug.

Rules:
- Keep the label broad enough for all definitions.
- Do not mention a specific nutrient, bioactive, study, population, or dose.
- The description should define the health effect itself, not make a claim.
- Use concise sentence-case wording.
- Tags must be snake_case and useful for lookup.
- Return JSON only with: effect_label, description, tags.

effect_slug: {effect_slug}
candidate_definitions:
{orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


def resolve_health_effect_conflict(
    *,
    effect_slug: str,
    definitions: list[dict[str, str]],
    provider: LlmProvider,
) -> ResolvedHealthEffect:
    return provider.complete_json(
        resolution_prompt(effect_slug, definitions),
        RESOLVED_HEALTH_EFFECT_ADAPTER,
    )


def representative_times(definitions: list[dict[str, str]]) -> tuple[str, str]:
    created = sorted(
        item.get("created_at", "")
        for item in definitions
        if item.get("created_at", "").strip()
    )
    updated = sorted(
        item.get("updated_at", "")
        for item in definitions
        if item.get("updated_at", "").strip()
    )
    now = datetime.now(UTC).isoformat()
    return (created[0] if created else now, updated[-1] if updated else now)


def combine_tags(definitions: list[dict[str, str]]) -> list[str]:
    tags = []
    for definition in definitions:
        tags.extend(parse_tags(definition.get("tags", "")))
    normalized = []
    for tag in tags:
        clean = tag.strip().lower().replace("-", "_").replace(" ", "_")
        if clean and clean.replace("_", "").isalnum():
            normalized.append(clean)
    return list(dict.fromkeys(normalized))[:12]


def resolve_health_effect_rows(
    *,
    health_files: list[Path],
    provider: LlmProvider | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped = collect_health_effect_definitions(health_files)
    rows: list[dict[str, str]] = []
    resolutions: list[dict[str, str]] = []

    for effect_slug, definitions in sorted(grouped.items()):
        unique_definitions = {
            health_effect_definition_key(definition) for definition in definitions
        }
        # IDs are stable by slug in the internal exporter, so any definition can supply it.
        base = definitions[-1]
        created_at, updated_at = representative_times(definitions)
        if len(unique_definitions) <= 1:
            rows.append({column: base.get(column, "") for column in NEW_EFFECT_COLUMNS})
            continue
        if provider is None:
            label = base["effect_label"]
            description = base["description"]
            tags = combine_tags(definitions)
            method = "tag_union_latest_wording"
        else:
            resolved = resolve_health_effect_conflict(
                effect_slug=effect_slug,
                definitions=definitions,
                provider=provider,
            )
            label = resolved.effect_label
            description = resolved.description
            tags = resolved.tags or combine_tags(definitions)
            method = "llm_resolved"
        rows.append(
            {
                "id": base["id"],
                "effect_slug": effect_slug,
                "effect_label": label,
                "description": description,
                "tags": json_tags(tags),
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        resolutions.append(
            {
                "effect_slug": effect_slug,
                "definition_count": str(len(definitions)),
                "resolution_method": method,
                "effect_label": label,
                "description": description,
                "tags": json_tags(tags),
            }
        )
    return rows, resolutions


def combine_exports(
    runs_dir: Path,
    out_dir: Path,
    *,
    resolve_health_conflicts: bool = False,
    provider: LlmProvider | None = None,
) -> dict[str, object]:
    health_files = list(runs_dir.glob("*/internal_export/health_effects_new.csv"))
    evidence_files = list(
        runs_dir.glob("*/internal_export/bioactive_health_evidence_rows.generated.csv")
    )

    health_rows, health_conflicts, health_duplicates = merge_unique_rows(
        files=health_files,
        columns=NEW_EFFECT_COLUMNS,
        key_column="effect_slug",
    )
    evidence_rows, evidence_conflicts, evidence_duplicates = merge_unique_rows(
        files=evidence_files,
        columns=EVIDENCE_COLUMNS,
        key_column="id",
    )

    if resolve_health_conflicts:
        health_rows, health_resolutions = resolve_health_effect_rows(
            health_files=health_files,
            provider=provider,
        )
    else:
        health_resolutions = []

    write_csv(out_dir / "health_effects_new.combined.csv", NEW_EFFECT_COLUMNS, health_rows)
    write_csv(
        out_dir / "bioactive_health_evidence_rows.generated.combined.csv",
        EVIDENCE_COLUMNS,
        evidence_rows,
    )
    write_csv(
        out_dir / "health_effects_conflicts.csv",
        ["key", "kept_run", "replaced_by_run", "file"],
        health_conflicts,
    )
    write_csv(
        out_dir / "bioactive_health_evidence_conflicts.csv",
        ["key", "kept_run", "replaced_by_run", "file"],
        evidence_conflicts,
    )
    write_csv(
        out_dir / "health_effects_resolutions.csv",
        [
            "effect_slug",
            "definition_count",
            "resolution_method",
            "effect_label",
            "description",
            "tags",
        ],
        health_resolutions,
    )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "runs_dir": str(runs_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "health_effect_files": len(health_files),
        "evidence_files": len(evidence_files),
        "health_effect_rows": len(health_rows),
        "evidence_rows": len(evidence_rows),
        "health_effect_duplicate_rows": health_duplicates,
        "evidence_duplicate_rows": evidence_duplicates,
        "health_effect_conflicts": len(health_conflicts),
        "evidence_conflicts": len(evidence_conflicts),
        "health_effect_conflicts_resolved": len(health_resolutions),
        "health_effect_resolution_mode": (
            "llm" if resolve_health_conflicts and provider else (
                "deterministic" if resolve_health_conflicts else "disabled"
            )
        ),
    }
    (out_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "IMPORT_INSTRUCTIONS.md").write_text(
        """# Combined Internal Export Import Order

Import these files into Supabase in this order:

1. `health_effects_new.combined.csv` -> `beverage.health_effects`
2. `bioactive_health_evidence_rows.generated.combined.csv` -> `beverage.bioactive_health_evidence`

Review conflict and resolution reports before import:

- `health_effects_conflicts.csv`
- `health_effects_resolutions.csv`
- `bioactive_health_evidence_conflicts.csv`
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine all per-run internal export CSVs into Supabase import CSVs."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("research_runs"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resolve-health-conflicts",
        action="store_true",
        help=(
            "Resolve conflicting health-effect definitions into one canonical row per slug. "
            "Uses deterministic tag union unless --provider is set."
        ),
    )
    parser.add_argument("--provider", default=None, help="LLM provider for conflict resolution.")
    parser.add_argument("--model", default=None, help="Provider model name.")
    args = parser.parse_args()

    provider = None
    if args.resolve_health_conflicts and args.provider:
        model_config = resolve_model_config(provider=args.provider, model=args.model)
        provider = create_llm_provider(model_config)

    summary = combine_exports(
        runs_dir=args.runs_dir,
        out_dir=args.out,
        resolve_health_conflicts=args.resolve_health_conflicts,
        provider=provider,
    )
    print(f"Wrote combined exports under: {args.out}")
    print(f"Health effects: {summary['health_effect_rows']}")
    print(f"Evidence rows: {summary['evidence_rows']}")
    print(f"Health-effect conflicts: {summary['health_effect_conflicts']}")
    print(f"Health-effect conflicts resolved: {summary['health_effect_conflicts_resolved']}")
    print(f"Evidence conflicts: {summary['evidence_conflicts']}")


if __name__ == "__main__":
    main()
