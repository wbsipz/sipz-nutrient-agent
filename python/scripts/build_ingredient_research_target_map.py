from __future__ import annotations

import argparse
from pathlib import Path

from sipz_agent.core.ingredient_target_mapping import build_ingredient_target_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flattened ingredient row to final research target mapping CSV."
    )
    parser.add_argument("--lookup", type=Path, default=Path("health_reports_ingredients_v1_rows.csv"))
    parser.add_argument(
        "--preparation-dir",
        type=Path,
        default=Path("ingredient_preparation_full"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ingredient_preparation_full/ingredient_to_research_target_map.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prep = args.preparation_dir
    rows = build_ingredient_target_map(
        lookup_path=args.lookup,
        equivalence_path=prep / "ingredient_equivalence_groups.csv",
        final_targets_path=prep / "ingredient_research_run_targets.final.csv",
        final_excluded_path=prep / "ingredient_research_run_targets.final_excluded.csv",
        pubmed_excluded_path=prep / "ingredient_research_run_targets.pubmed_ready_excluded.csv",
        skip_list_path=prep / "ingredient_skip_list.csv",
        preparation_decisions_path=prep / "ingredient_preparation_decisions.json",
        out_path=args.out,
    )
    statuses: dict[str, int] = {}
    for row in rows:
        status = row["final_mapping_status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(rows)}")
    for status, count in sorted(statuses.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
