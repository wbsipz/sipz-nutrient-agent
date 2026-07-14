from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
from typing import Any

import orjson


REGULAR_BUCKETS = ["strong_evidence", "medium_evidence", "low_evidence"]
NEGATIVE_BUCKET = "negative_or_cautionary_effects"
COPY_FILES = [
    "ingredient_health_summaries_summary.json",
    "ingredient_health_summaries_failures.jsonl",
    "ingredient_health_summaries_batch_log.jsonl",
    "supporting_nutrients_cleanup_report.json",
    "duplicate_effect_slug_cleanup_report.json",
]
BENEFICIAL_CONTEXT_TERMS = [
    "dietary fiber",
    "fermentable fiber",
    "fiber and whole grains",
    "fiber can",
    "fiber in whole",
    "fiber intake",
    "higher fiber",
    "inulin",
    "non-digestible fiber",
    "nondigestible fiber",
    "prebiotic fiber",
    "soluble fiber",
    "whole grain",
    "whole-grain",
    "whole wheat",
]
BENEFICIAL_EFFECT_TERMS = [
    "bifidogenic",
    "blood glucose regulation",
    "blunt postprandial glucose",
    "bowel regularity",
    "butyrate",
    "cardiovascular health",
    "colon health",
    "colorectal cancer risk",
    "digestive health",
    "glycemic control",
    "gut microbiota",
    "insulin sensitivity",
    "ldl cholesterol",
    "lower colorectal cancer",
    "lower ldl",
    "regularity",
    "satiety",
    "short-chain fatty acid",
]
NEGATIVE_CONTEXT_TERMS = [
    "added sugar",
    "added sugars",
    "adverse",
    "blood glucose spikes",
    "discomfort",
    "bloating",
    "gas",
    "high intake",
    "high glycemic load",
    "high sugar",
    "higher risk",
    "impair",
    "increase risk",
    "increased risk",
    "lack of fiber",
    "large dose",
    "negative",
    "poor glycemic",
    "poorer",
    "rapidly absorbed",
    "refined",
    "sugar content",
    "sugars",
    "uncertain",
    "worsen",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows))


def target_bucket(effect: dict[str, Any]) -> str:
    evidence = str(effect.get("evidence_level", "")).casefold()
    if "strong" in evidence or effect.get("score", 0) >= 0.8:
        return "strong_evidence"
    if "moderate" in evidence or "medium" in evidence or effect.get("score", 0) >= 0.55:
        return "medium_evidence"
    return "low_evidence"


def is_beneficial_caution_false_positive(effect: dict[str, Any]) -> bool:
    supporting = {str(name).casefold() for name in effect.get("supporting_nutrients", [])}
    text = " ".join(
        str(effect.get(key) or "")
        for key in ["effect_slug", "effect_label", "summary", "evidence_level"]
    ).casefold()

    if not (supporting & {"fiber", "carbohydrates"}):
        return False
    if not any(term in text for term in BENEFICIAL_CONTEXT_TERMS):
        return False
    if not any(term in text for term in BENEFICIAL_EFFECT_TERMS):
        return False
    if any(term in text for term in NEGATIVE_CONTEXT_TERMS):
        return False
    return True


def reroute_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    moved: list[dict[str, Any]] = []
    regular_slugs = {
        effect.get("effect_slug")
        for bucket in REGULAR_BUCKETS
        for effect in row.get(bucket, [])
        if effect.get("effect_slug")
    }
    remaining_cautionary: list[dict[str, Any]] = []

    for effect in row.get(NEGATIVE_BUCKET, []):
        slug = effect.get("effect_slug", "")
        if slug in regular_slugs:
            remaining_cautionary.append(effect)
            continue
        if not is_beneficial_caution_false_positive(effect):
            remaining_cautionary.append(effect)
            continue

        bucket = target_bucket(effect)
        row.setdefault(bucket, []).append(effect)
        regular_slugs.add(slug)
        moved.append(
            {
                "canonical_beverage_id": row["canonical_beverage_id"],
                "canonical_beverage_name": row["canonical_beverage_name"],
                "effect_slug": slug,
                "supporting_nutrients": effect.get("supporting_nutrients", []),
                "moved_from": NEGATIVE_BUCKET,
                "moved_to": bucket,
                "summary": effect.get("summary", ""),
            }
        )

    row[NEGATIVE_BUCKET] = remaining_cautionary
    return row, moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reroute clearly beneficial fiber/inulin/whole-grain effects that were placed in "
            "negative_or_cautionary_effects into the appropriate regular evidence bucket."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source_dir / "ingredient_health_summaries.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(source_path)
    moved: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned, row_moved = reroute_row(row)
        output_rows.append(cleaned)
        moved.extend(row_moved)

    write_jsonl(args.out_dir / "ingredient_health_summaries.jsonl", output_rows)
    for name in COPY_FILES:
        source_file = args.source_dir / name
        if source_file.exists():
            shutil.copy2(source_file, args.out_dir / name)

    report = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "processed_rows": len(output_rows),
        "moved_effects": len(moved),
        "moved_by_ingredient": dict(Counter(item["canonical_beverage_name"] for item in moved)),
        "moved_by_target_bucket": dict(Counter(item["moved_to"] for item in moved)),
        "moved": moved,
    }
    (args.out_dir / "beneficial_caution_reroute_report.json").write_bytes(
        orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    print(orjson.dumps({key: value for key, value in report.items() if key != "moved"}, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
