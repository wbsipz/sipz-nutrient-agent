from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from sipz_agent.core.internal_export import NEW_EFFECT_COLUMNS

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "combine_internal_exports.py"
SPEC = importlib.util.spec_from_file_location("combine_internal_exports", SCRIPT_PATH)
assert SPEC is not None
combine_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(combine_module)
combine_exports = combine_module.combine_exports


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def health_row(
    *,
    label: str,
    description: str,
    tags: list[str],
    updated_at: str,
) -> dict[str, str]:
    return {
        "id": "health-effect-1",
        "effect_slug": "blood_pressure_support",
        "effect_label": label,
        "description": description,
        "tags": json.dumps(tags),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": updated_at,
    }


def test_combine_exports_can_resolve_health_effect_conflicts_deterministically(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "research_runs"
    first = runs_dir / "2026-01-01T00-00-00+00-00_quercetin" / "internal_export"
    second = runs_dir / "2026-01-02T00-00-00+00-00_anthocyanins" / "internal_export"
    write_csv(
        first / "health_effects_new.csv",
        NEW_EFFECT_COLUMNS,
        [
            health_row(
                label="Blood pressure support",
                description="Effects related to maintenance of blood pressure.",
                tags=["blood_pressure", "vascular_health"],
                updated_at="2026-01-01T00:00:00+00:00",
            )
        ],
    )
    write_csv(
        second / "health_effects_new.csv",
        NEW_EFFECT_COLUMNS,
        [
            health_row(
                label="Blood pressure and vascular function",
                description="Effects on blood pressure and vascular function.",
                tags=["cardiometabolic", "blood_pressure"],
                updated_at="2026-01-02T00:00:00+00:00",
            )
        ],
    )

    summary = combine_exports(
        runs_dir=runs_dir,
        out_dir=tmp_path / "combined",
        resolve_health_conflicts=True,
    )

    rows = read_csv(tmp_path / "combined" / "health_effects_new.combined.csv")
    resolutions = read_csv(tmp_path / "combined" / "health_effects_resolutions.csv")
    assert summary["health_effect_conflicts"] == 1
    assert summary["health_effect_conflicts_resolved"] == 1
    assert summary["health_effect_resolution_mode"] == "deterministic"
    assert len(rows) == 1
    assert rows[0]["effect_slug"] == "blood_pressure_support"
    assert rows[0]["effect_label"] == "Blood pressure and vascular function"
    assert json.loads(rows[0]["tags"]) == [
        "blood_pressure",
        "vascular_health",
        "cardiometabolic",
    ]
    assert resolutions == [
        {
            "effect_slug": "blood_pressure_support",
            "definition_count": "2",
            "resolution_method": "tag_union_latest_wording",
            "effect_label": "Blood pressure and vascular function",
            "description": "Effects on blood pressure and vascular function.",
            "tags": '["blood_pressure","vascular_health","cardiometabolic"]',
        }
    ]
