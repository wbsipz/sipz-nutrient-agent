from __future__ import annotations

import csv
from pathlib import Path

import orjson
from pydantic import TypeAdapter

from sipz_agent.schemas.claims import ValidatedClaim
from sipz_agent.schemas.effects import EFFECT_CSV_COLUMNS, EffectRow


class AuditResult:
    def __init__(self, ok: bool, issues: list[str]) -> None:
        self.ok = ok
        self.issues = issues

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AuditResult):
            return False
        return self.ok == other.ok and self.issues == other.issues


def read_effect_rows(path: Path) -> list[EffectRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EFFECT_CSV_COLUMNS:
            raise ValueError("effects_csv_columns_invalid")
        rows = []
        for row in reader:
            rows.append(
                EffectRow.model_validate(
                    {
                        **row,
                        "score": float(row["score"]),
                        "match_confidence": float(row["match_confidence"]),
                        "tags": orjson.loads(row["tags"]),
                        "sources": orjson.loads(row["sources"]),
                    }
                )
            )
        return rows


def audit_run(run_dir: str | Path) -> AuditResult:
    run_path = Path(run_dir)
    issues: list[str] = []
    effect_rows = read_effect_rows(run_path / "effects.csv")
    validated_claims = TypeAdapter(list[ValidatedClaim]).validate_python(
        orjson.loads((run_path / "validated_claims.json").read_bytes())
    )

    effect_ids = {row.id for row in effect_rows}
    for claim in validated_claims:
        if claim.effect_row_id not in effect_ids:
            issues.append(
                f"validated claim {claim.proposed_claim_id} references missing effect row "
                f"{claim.effect_row_id}"
            )
        if not any(quote.match_status != "not_found" for quote in claim.supporting_quotes):
            issues.append(f"validated claim {claim.proposed_claim_id} has no grounded quote")

    return AuditResult(ok=len(issues) == 0, issues=issues)
