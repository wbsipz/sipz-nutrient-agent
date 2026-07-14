from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, NEW_EFFECT_COLUMNS, stable_uuid

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "avoid_existing_health_effect_conflicts.py"
)
SPEC = importlib.util.spec_from_file_location("avoid_existing_health_effect_conflicts", SCRIPT_PATH)
assert SPEC is not None
conflict_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(conflict_module)
prepare_import = conflict_module.prepare_import


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_import_renames_health_slugs_that_already_exist(tmp_path: Path) -> None:
    input_dir = tmp_path / "canonicalized"
    output_dir = tmp_path / "import_ready"
    existing_path = tmp_path / "health_effects_rows.csv"
    write_csv(
        existing_path,
        NEW_EFFECT_COLUMNS,
        [
            {
                "id": "existing-health-effect",
                "effect_slug": "weight_loss",
                "effect_label": "Weight loss",
                "description": "",
                "tags": "",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
            }
        ],
    )
    write_csv(
        input_dir / "health_effects_new.combined.csv",
        NEW_EFFECT_COLUMNS,
        [
            {
                "id": stable_uuid("health-effect", "weight_loss"),
                "effect_slug": "weight_loss",
                "effect_label": "Weight loss",
                "description": "A decrease in body weight.",
                "tags": json.dumps(["weight_loss"]),
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )
    write_csv(
        input_dir / "bioactive_health_evidence_rows.generated.combined.csv",
        EVIDENCE_COLUMNS,
        [
            {
                "id": stable_uuid("evidence", "polyphenol", "bioactive-1", "weight_loss"),
                "bioactive_type": "polyphenol",
                "bioactive_id": "bioactive-1",
                "bioactive_name": "Test bioactive",
                "effect_slug": "weight_loss",
                "effect_label": "Weight loss",
                "description": "Evidence description.",
                "score": "0.45",
                "evidence_level": "limited",
                "tags": json.dumps(["weight_loss"]),
                "sources": json.dumps([{"title": "Paper"}]),
                "review_status": "generated",
                "review_notes": "",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )

    summary = prepare_import(
        input_dir=input_dir,
        output_dir=output_dir,
        existing_health_effects_path=existing_path,
    )

    health_rows = read_csv(output_dir / "health_effects_new.combined.csv")
    evidence_rows = read_csv(output_dir / "bioactive_health_evidence_rows.generated.combined.csv")
    renames = read_csv(output_dir / "health_effect_existing_slug_renames.csv")
    assert summary["renamed_health_effect_slugs"] == 1
    assert summary["import_health_slugs_conflicting_with_existing"] == 0
    assert summary["evidence_slugs_not_in_import_or_existing"] == 0
    assert health_rows[0]["effect_slug"] == "weight_loss_generated"
    assert evidence_rows[0]["effect_slug"] == "weight_loss_generated"
    assert evidence_rows[0]["id"] == stable_uuid(
        "evidence",
        "polyphenol",
        "bioactive-1",
        "weight_loss_generated",
    )
    assert renames[0]["old_effect_slug"] == "weight_loss"
    assert renames[0]["new_effect_slug"] == "weight_loss_generated"
