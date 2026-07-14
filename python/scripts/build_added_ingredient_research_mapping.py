from __future__ import annotations

import argparse
from pathlib import Path

from sipz_agent.core.added_ingredient_mapping import build_added_ingredient_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map every newly added ingredient row to an original-corpus research target, "
            "then to the latest added research batch, otherwise mark it not included."
        )
    )
    parser.add_argument(
        "--preparation-dir",
        type=Path,
        default=Path("ingredient_preparation_added"),
    )
    parser.add_argument(
        "--lookup",
        type=Path,
        default=Path("ingredient_preparation_added/added_ingredients_lookup.csv"),
    )
    parser.add_argument(
        "--original-final-targets",
        type=Path,
        default=Path("ingredient_preparation_full/ingredient_research_run_targets.final.csv"),
    )
    parser.add_argument(
        "--added-targets",
        type=Path,
        default=Path("ingredient_preparation_added/ingredient_research_run_targets.to_run.csv"),
    )
    parser.add_argument(
        "--already-covered-targets",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.already_covered.csv"
        ),
        help=(
            "Added-corpus targets that were collapsed onto original-corpus targets. "
            "These are used as aliases in the original-corpus matching pass."
        ),
    )
    parser.add_argument(
        "--added-excluded-targets",
        type=Path,
        default=Path(
            "ingredient_preparation_added/ingredient_research_run_targets.excluded_after_crunch.csv"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("ingredient_runs"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ingredient_preparation_added/added_ingredient_to_research_target_map.csv"),
    )
    parser.add_argument(
        "--not-included-out",
        type=Path,
        default=Path(
            "ingredient_preparation_added/added_ingredients_not_included_after_research_mapping.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("ingredient_preparation_added/added_ingredient_research_mapping_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_added_ingredient_mapping(
        lookup_path=args.lookup,
        equivalence_path=args.preparation_dir / "ingredient_equivalence_groups.csv",
        skip_list_path=args.preparation_dir / "ingredient_skip_list.csv",
        original_targets_path=args.original_final_targets,
        original_alias_targets_path=args.already_covered_targets,
        added_targets_path=args.added_targets,
        added_excluded_targets_path=args.added_excluded_targets,
        preparation_decisions_path=args.preparation_dir / "ingredient_preparation_decisions.json",
        runs_root=args.runs_root,
        out_path=args.out,
        not_included_out_path=args.not_included_out,
        summary_path=args.summary,
    )
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.not_included_out}")
    print(f"Wrote: {args.summary}")
    print(f"Rows: {len(rows)}")
    for status, count in sorted(summary["mapping_status_counts"].items()):
        print(f"{status}: {count}")
    print(f"Include in export: {summary['include_in_export_count']}")
    print(f"Not included: {summary['not_included_count']}")


if __name__ == "__main__":
    main()
