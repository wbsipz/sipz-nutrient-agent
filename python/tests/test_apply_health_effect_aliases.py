from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, NEW_EFFECT_COLUMNS, stable_uuid

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "apply_health_effect_aliases.py"
SPEC = importlib.util.spec_from_file_location("apply_health_effect_aliases", SCRIPT_PATH)
assert SPEC is not None
alias_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(alias_module)
apply_aliases = alias_module.apply_aliases


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def health_row(slug: str, label: str, tags: list[str]) -> dict[str, str]:
    return {
        "id": stable_uuid("health-effect", slug),
        "effect_slug": slug,
        "effect_label": label,
        "description": f"{label} description.",
        "tags": json.dumps(tags),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }


def evidence_row(
    *,
    slug: str,
    level: str,
    score: str,
    source_title: str,
) -> dict[str, str]:
    bioactive_id = "bioactive-1"
    return {
        "id": stable_uuid("evidence", "polyphenol", bioactive_id, slug),
        "bioactive_type": "polyphenol",
        "bioactive_id": bioactive_id,
        "bioactive_name": "Ashwagandha",
        "effect_slug": slug,
        "effect_label": slug.replace("_", " ").title(),
        "description": f"{slug} evidence.",
        "score": score,
        "evidence_level": level,
        "tags": json.dumps([slug]),
        "sources": json.dumps([{"title": source_title, "url": f"https://example.test/{slug}"}]),
        "review_status": "generated",
        "review_notes": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }


def test_apply_aliases_collapses_health_and_evidence_rows(tmp_path: Path) -> None:
    input_dir = tmp_path / "combined"
    output_dir = tmp_path / "canonicalized"
    write_csv(
        input_dir / "health_effects_new.combined.csv",
        NEW_EFFECT_COLUMNS,
        [
            health_row("improved_sleep_quality", "Improved sleep quality", ["sleep_quality"]),
            health_row(
                "sleep_quality_improvement",
                "Sleep quality improvement",
                ["sleep_improvement"],
            ),
        ],
    )
    write_csv(
        input_dir / "bioactive_health_evidence_rows.generated.combined.csv",
        EVIDENCE_COLUMNS,
        [
            evidence_row(
                slug="improved_sleep_quality",
                level="limited",
                score="0.45",
                source_title="First paper",
            ),
            evidence_row(
                slug="sleep_quality_improvement",
                level="strong",
                score="0.84",
                source_title="Second paper",
            ),
        ],
    )

    summary = apply_aliases(input_dir, output_dir)

    health_rows = read_csv(output_dir / "health_effects_new.combined.csv")
    evidence_rows = read_csv(output_dir / "bioactive_health_evidence_rows.generated.combined.csv")
    assert summary["health_rows"] == 1
    assert summary["evidence_rows"] == 1
    assert summary["duplicate_health_slugs"] == 0
    assert summary["duplicate_evidence_ids"] == 0
    assert summary["remaining_alias_slugs"] == 0
    assert health_rows[0]["effect_slug"] == "improved_sleep_quality"
    assert json.loads(health_rows[0]["tags"]) == ["sleep_quality", "sleep_improvement"]
    assert evidence_rows[0]["effect_slug"] == "improved_sleep_quality"
    assert evidence_rows[0]["effect_label"] == "Improved sleep quality"
    assert evidence_rows[0]["evidence_level"] == "strong"
    assert len(json.loads(evidence_rows[0]["sources"])) == 2
