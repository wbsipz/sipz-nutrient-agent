from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVIDENCE_FIELDS = [
    "effect_slug",
    "effect_label",
    "description",
    "score",
    "evidence_level",
    "tags",
    "sources",
]


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_json_field(value: str) -> Any:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_score(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def read_evidence_by_bioactive(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"bioactive_name", *EVIDENCE_FIELDS}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("evidence_columns_missing:" + ",".join(missing))

        by_bioactive: dict[str, list[dict[str, Any]]] = {}
        for row in reader:
            bioactive_name = (row.get("bioactive_name") or "").strip()
            if not bioactive_name:
                continue
            evidence = {
                "effect_slug": row.get("effect_slug", ""),
                "effect_label": row.get("effect_label", ""),
                "description": row.get("description", ""),
                "score": parse_score(row.get("score", "")),
                "evidence_level": row.get("evidence_level", ""),
                "tags": parse_json_field(row.get("tags", "")),
                "sources": parse_json_field(row.get("sources", "")),
            }
            by_bioactive.setdefault(normalize_name(bioactive_name), []).append(evidence)
    return by_bioactive


def read_significance_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def significant_nutriment_records(significance_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ingredient in significance_results:
        for classification in ingredient.get("classifications", []):
            if classification.get("significance") != "significant":
                continue
            records.append(
                {
                    "canonical_beverage_id": ingredient.get("canonical_beverage_id", ""),
                    "ingredient_name": ingredient.get("ingredient_name", ""),
                    "serving_basis": ingredient.get("serving_basis", "100g"),
                    "source_key": classification.get("source_key", ""),
                    "canonical_bioactive_name": classification.get("canonical_bioactive_name", ""),
                    "raw_amount": classification.get("raw_amount"),
                    "raw_unit": classification.get("raw_unit", ""),
                    "display_amount": classification.get("display_amount"),
                    "display_unit": classification.get("display_unit", ""),
                    "amount_context": classification.get("amount_context", ""),
                    "reference_amount": classification.get("reference_amount", ""),
                    "reference_unit": classification.get("reference_unit", ""),
                    "reference_type": classification.get("reference_type", ""),
                    "caution_limit": classification.get("caution_limit", ""),
                    "caution_limit_unit": classification.get("caution_limit_unit", ""),
                    "significance": classification.get("significance", ""),
                    "significance_confidence": classification.get("confidence"),
                    "significance_reasoning": classification.get("reasoning", ""),
                    "classification_status": classification.get("classification_status", ""),
                }
            )
    return records


def build_evidence_match_records(
    *,
    significance_results: list[dict[str, Any]],
    evidence_by_bioactive: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for nutriment in significant_nutriment_records(significance_results):
        canonical_name = nutriment["canonical_bioactive_name"]
        matches = evidence_by_bioactive.get(normalize_name(canonical_name), [])
        records.append(
            {
                **nutriment,
                "matched": bool(matches),
                "match_count": len(matches),
                "evidence_matches": matches,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lookup evidence rows for significant nutriments from Step 5."
    )
    parser.add_argument(
        "--significance",
        type=Path,
        required=True,
        help="Path to nutriment_significance.jsonl.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Path to bioactive_health_evidence_rows CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path for nutriment evidence matches.",
    )
    args = parser.parse_args()

    if not args.significance.exists():
        raise FileNotFoundError(args.significance)
    if not args.evidence.exists():
        raise FileNotFoundError(args.evidence)

    significance_results = read_significance_results(args.significance)
    evidence_by_bioactive = read_evidence_by_bioactive(args.evidence)
    records = build_evidence_match_records(
        significance_results=significance_results,
        evidence_by_bioactive=evidence_by_bioactive,
    )
    write_jsonl(args.out, records)

    matched = sum(1 for record in records if record["matched"])
    unmatched = len(records) - matched
    unique_nutriments = {record["canonical_bioactive_name"] for record in records}
    unique_unmatched = {
        record["canonical_bioactive_name"] for record in records if not record["matched"]
    }
    print(f"significant_nutriment_rows={len(records)}")
    print(f"matched_rows={matched}")
    print(f"unmatched_rows={unmatched}")
    print(f"unique_significant_nutriments={len(unique_nutriments)}")
    print(f"unique_unmatched_nutriments={len(unique_unmatched)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
