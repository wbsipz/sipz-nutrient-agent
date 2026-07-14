from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from sipz_agent.core.ingredient_target_crunch import build_crunched_run_targets, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse added ingredient canonical targets into research run targets."
    )
    parser.add_argument(
        "--preparation-dir",
        type=Path,
        default=Path("ingredient_preparation_added"),
    )
    parser.add_argument(
        "--existing-final-targets",
        type=Path,
        default=Path("ingredient_preparation_full/ingredient_research_run_targets.final.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ingredient_preparation_added/ingredient_research_run_targets.crunched.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.crunched_summary.json"
        ),
    )
    parser.add_argument(
        "--to-run-out",
        type=Path,
        default=Path("ingredient_preparation_added/ingredient_research_run_targets.to_run.csv"),
        help="Write only targets not already covered by the existing final target list.",
    )
    parser.add_argument(
        "--already-covered-out",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.already_covered.csv"
        ),
        help="Write targets mapped to the existing final target list.",
    )
    parser.add_argument(
        "--excluded-out",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.excluded_after_crunch.csv"
        ),
        help="Write targets excluded by deterministic post-crunch cleanup.",
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.to_run_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_crunched_run_targets(
        canonical_targets_path=args.preparation_dir / "ingredient_canonical_research_targets.csv",
        existing_final_targets_path=args.existing_final_targets,
        out_path=args.out,
        summary_path=args.summary,
    )
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.summary}")
    print(f"Source targets: {summary['source_target_count']}")
    print(f"Run targets: {summary['run_target_count']}")
    print(f"Reduction: {summary['reduction_count']} ({summary['reduction_percent']}%)")
    for reason, count in sorted(summary["reason_code_counts"].items()):
        print(f"{reason}: {count}")

    def is_already_covered(row: dict[str, str]) -> bool:
        return any(
            reason_code.startswith("existing_final")
            for reason_code in row["collapse_reason_code"].split("+")
        )

    def is_excluded(row: dict[str, str]) -> bool:
        return any(
            reason_code.startswith("excluded_")
            for reason_code in row["collapse_reason_code"].split("+")
        )

    to_run_rows = [row for row in rows if not is_already_covered(row) and not is_excluded(row)]
    already_covered_rows = [row for row in rows if is_already_covered(row)]
    excluded_rows = [row for row in rows if is_excluded(row)]
    write_csv_rows(args.to_run_out, to_run_rows)
    write_csv_rows(args.already_covered_out, already_covered_rows)
    write_csv_rows(args.excluded_out, excluded_rows)
    split_summary = {
        "crunched_target_count": len(rows),
        "to_run_target_count": len(to_run_rows),
        "already_covered_target_count": len(already_covered_rows),
        "excluded_after_crunch_target_count": len(excluded_rows),
        "to_run_out": str(args.to_run_out),
        "already_covered_out": str(args.already_covered_out),
        "excluded_out": str(args.excluded_out),
    }
    args.split_summary.write_bytes(
        orjson.dumps(split_summary, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    print(f"Wrote: {args.to_run_out}")
    print(f"Wrote: {args.already_covered_out}")
    print(f"Wrote: {args.excluded_out}")
    print(f"Wrote: {args.split_summary}")
    print(f"To-run targets: {len(to_run_rows)}")
    print(f"Already-covered targets: {len(already_covered_rows)}")
    print(f"Excluded after crunch: {len(excluded_rows)}")


if __name__ == "__main__":
    main()
