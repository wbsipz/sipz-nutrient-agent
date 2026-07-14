from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


OUTPUT_COLUMNS = [
    "source_set",
    "canonical_beverage_id",
    "canonical_beverage_name",
    "decision",
    "relationship",
    "canonical_search_name",
    "group_role",
    "adaptation",
    "preparation_reason",
    "preparation_confidence",
    "final_mapping_status",
    "source_mapping_status",
    "mapped_target_source",
    "final_run_target_id",
    "final_run_target_name",
    "final_representative_canonical_beverage_id",
    "final_representative_canonical_beverage_name",
    "collapse_reason_code",
    "collapse_reason",
    "pubmed_exclusion_code",
    "pubmed_exclusion_reason",
    "final_exclusion_code",
    "final_exclusion_reason",
    "run_path",
    "claim_file",
    "web_audited_claim_count",
    "included_or_mapped",
    "include_in_export",
    "needs_manual_review",
    "exclusion_reason",
]


ADDED_STATUS_TO_LEGACY = {
    "mapped_to_original_corpus_target": "mapped_to_run_target",
    "mapped_to_added_batch_target": "mapped_to_run_target",
    "skipped_low_value": "skipped_low_value",
    "manual_review_unmapped": "manual_review_unmapped",
    "excluded_after_crunch": "excluded_final",
    "unmatched": "unmapped",
}


GENERIC_OR_AMBIGUOUS_MANUAL_PATTERNS = [
    r"\bjuice drink\b",
    r"\bjuice beverage\b",
    r"\borganic juice drink\b",
    r"\bvegetable juice beverage\b",
    r"\bsmoothie\b",
    r"\bcappuccino\b",
    r"\blatte\b",
    r"\bfrappuccino\b",
    r"\bmocha\b",
    r"\balcoholic beverage\b",
    r"\benergy drink\b",
    r"\bcarbonated (?:drink|beverage)\b",
    r"\bsupplement\b",
    r"\bmixed plant milk\b",
    r"\boat almond milk\b",
    r"\brice coconut milk\b",
    r"\bcoffee (?:milk|chicory)\b",
    r"\bchicory coffee mix\b",
    r"\binstant chicory coffee mix\b",
    r"\bcashew coffee beverage\b",
]


MANUAL_TARGET_PATTERNS = [
    (r"\b(?:french vanilla |vanilla |salted caramel |caramel |pumpkin spice |coconut |instant |iced |hot |frozen |espresso |caffe |coffee )*(?:cappuccino|latte|mocha)\b", "coffee"),
    (r"\b(?:latte|cappuccino|mocha) coffee\b|\bcoffee milk(?: drink)?\b", "coffee"),
    (r"\bcappuccino capsules?\b", "coffee"),
    (r"\b(?:decaf(?:feinated)?|arabica|espresso|nespresso(?:-compatible)?|coffee)\b.*\bcapsules?\b", "coffee"),
    (r"\b(?:tea)\b.*\bcapsules?\b", "tea"),
    (r"\btea latte\b", "tea"),
    (r"\boat almond milk (?:drink|beverage)?\b", "oat milk"),
    (r"\balmond milk (?:drink|beverage)\b", "almond milk"),
    (r"\bsoy milk (?:drink|beverage)\b", "soy milk"),
    (r"\boat milk (?:drink|beverage|powder)\b", "oat milk"),
    (r"\bcocoa oat drink\b", "oat milk"),
    (r"\brice milk (?:drink|beverage)\b", "rice milk"),
    (r"\brice coconut milk (?:drink|beverage)?\b", "coconut milk"),
    (r"\bcoconut milk (?:drink|beverage|powder)\b", "coconut milk"),
    (r"\bspelt milk drink\b", "spelt"),
    (r"\bpea milk drink\b", "pea"),
    (r"\bfruit (?:and )?milk\b|\bmilk and fruit\b|\bfruit milk\b", "milk"),
    (r"\b(?:vanilla|banana|strawberry|hazelnut|protein|lactose free|lactose-free)?\s*milk drink\b", "milk"),
    (r"\b(?:skimmed|instant) milk powder\b", "milk"),
    (r"\bprobiotic dairy drink\b|\bfermented dairy drink\b|\bdairy drink\b", "milk"),
    (r"\brice based drink\b|\brice-based drink\b", "rice"),
    (r"\bcalcium rice drink\b|\bcocoa rice drink\b|\bvanilla rice drink\b", "rice"),
    (r"\bsoy rice drink\b", "soy"),
    (r"\balmond based drink\b|\balmond-based drink\b", "almond"),
    (r"\bspelt based drink\b|\bspelt-based drink\b", "spelt"),
    (r"\boat milk coffee drink\b", "oat milk"),
    (r"\btea based beverage\b|\btea-based beverage\b", "tea"),
    (r"\boat based drink\b|\boat-based drink\b", "oat"),
    (r"\bsoy based drink\b|\bsoy-based drink\b", "soy"),
    (r"\b(?:plain |vanilla |low fat |low-fat |lactose free |lactose-free |drinkable |drinking |strawberry |banana |mango |orange |lemon |blueberry |berry |mixed fruit |tropical |protein |fruit )*yogurt (?:drink|smoothie)\b", "yogurt"),
    (r"\b(?:plain |vanilla |strawberry )?drinking yogurt\b|\bdrinkable yogurt\b", "yogurt"),
    (r"\bgreek yogurt smoothie\b", "greek yogurt"),
    (r"\bprobiotic drinkable yogurt\b", "probiotic yogurt"),
    (r"\bayran yogurt drink\b|\byogurt drink ayran\b", "ayran"),
    (r"\bkefir (?:dairy|milk|yogurt) drink\b", "kefir"),
    (r"\bpurified drinking water\b|\bdrinking water\b", "water"),
    (r"\bdrinking vinegar\b", "vinegar"),
    (r"\bpowdered sugar\b", "sugar"),
    (r"\bchili[ _-]?powder\b", "chili pepper"),
    (r"\bcocoa[ _-]?powder\b|\binstant cocoa powder\b|\bcocoa powder for beverages\b", "cocoa"),
    (r"\bbrewer s yeast powder\b|\bbrewers yeast powder\b", "yeast"),
    (r"\bmalt beverage\b|\bmalt powder\b|\bmalt drink powder\b|\bmalt extract\b", "base malt"),
    (r"\b(?:vanilla )?whey protein powder\b|\bwhey protein powder\b", "whey protein"),
    (r"\bpartially hydrolyzed whey protein isolate powder\b", "whey protein"),
    (r"\balmond protein powder\b", "almond"),
    (r"\bspirulina tablets?\b", "spirulina"),
    (r"\bmoringa powder\b", "moringa"),
    (r"\b(?:vanilla_?extract|vanilla extract)\b", "vanilla"),
    (r"\bstevia (?:extract|leaf extract|concentrate|liquid|sweetener)\b|\bliquid stevia sweetener\b", "stevia"),
    (r"\bgunpowder green tea\b", "green tea"),
    (r"\bkombucha tea based beverage\b|\bkombucha tea-based beverage\b", "kombucha"),
    (r"\bcane sugar\b", "sugar"),
    (r"\bnutritional yeast\b", "yeast"),
    (r"\btart cherry concentrate\b", "tart cherry"),
    (r"\blemon extract\b", "lemon"),
    (r"\b(?:dried |black |white )?fig (?:jam|preserves)\b", "fig"),
    (r"\bmonk fruit\b", "sugar substitute"),
    (r"\b(?:cayenne pepper|chipotle habanero pepper|crushed red pepper|green jalapeno pepper|green pepper|habanero mustard pepper|habanero pepper|habanero red jalapeno pepper|jalapeno and relleno pepper|jalapeno pepper|jamaican red pepper|red jalapeno pepper|red savina pepper|scotch bonnet pepper|serrano pepper|smoky red pepper|sriracha pepper|sweet and spicy pepper|wasabi pepper) sauce(?: seasoning)?\b", "chili pepper"),
    (r"\bcayenne pepper sauce\b", "cayenne pepper"),
    (r"\bjalapeno pepper sauce\b|\bjalapeño pepper sauce\b", "jalapeno"),
    (r"\bfully hydrogenated menhaden oil\b|\bhighly refined omega 3 fish oil\b", "menhaden fish oil"),
    (r"\bsunflower oil and extra virgin olive oil blend\b", "olive oil"),
    (r"\bbalsamic blend seasoned rice vinegar\b", "vinegar"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def normalize_original_row(row: dict[str, str]) -> dict[str, str]:
    status = row.get("final_mapping_status", "")
    return {
        **row,
        "source_set": "original_corpus",
        "source_mapping_status": status,
        "mapped_target_source": "original_corpus" if status == "mapped_to_run_target" else "",
        "run_path": "",
        "claim_file": "",
        "web_audited_claim_count": "",
        "included_or_mapped": "true" if status == "mapped_to_run_target" else "false",
        "include_in_export": "unknown",
        "needs_manual_review": "true" if status == "manual_review_unmapped" else "false",
        "exclusion_reason": (
            row.get("final_exclusion_reason")
            or row.get("pubmed_exclusion_reason")
            or (row.get("preparation_reason") if status != "mapped_to_run_target" else "")
        ),
    }


def normalize_added_row(row: dict[str, str]) -> dict[str, str]:
    source_status = row.get("mapping_status", "")
    legacy_status = ADDED_STATUS_TO_LEGACY.get(source_status, "unmapped")
    return {
        **row,
        "source_set": "added_2026_06",
        "final_mapping_status": legacy_status,
        "source_mapping_status": source_status,
        "pubmed_exclusion_code": "",
        "pubmed_exclusion_reason": "",
        "final_exclusion_code": "excluded_after_crunch" if legacy_status == "excluded_final" else "",
        "final_exclusion_reason": row.get("exclusion_reason", "")
        if legacy_status == "excluded_final"
        else "",
        "included_or_mapped": "true" if legacy_status == "mapped_to_run_target" else "false",
        "include_in_export": row.get("include_in_export", "false"),
    }


def normalized_text(value: str) -> str:
    text = value.casefold().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def target_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        target_name = row.get("final_run_target_name", "")
        if target_name and row.get("final_mapping_status") == "mapped_to_run_target":
            index.setdefault(normalized_text(target_name), row)
    return index


def proposed_manual_targets(row: dict[str, str]) -> list[str]:
    name = normalized_text(row.get("canonical_beverage_name", ""))
    search = normalized_text(row.get("canonical_search_name", ""))
    text = f"{name} {search}".strip()
    targets: list[str] = []
    for pattern, target in MANUAL_TARGET_PATTERNS:
        if re.search(pattern, text):
            targets.append(target)
    if search:
        targets.append(search)
    stripped_name = re.sub(
        r"\b(?:powder|paste|puree|pur[ée]e|extract|concentrate|liquid|jam|preserves|sauce|seasoning|drink|beverage|capsules?|tablets?|slices?|leather|flavor|flavored|candied|dried|stir in|infused|oil|blend)\b",
        " ",
        name,
    )
    stripped_name = re.sub(r"\s+", " ", stripped_name).strip()
    if stripped_name:
        targets.append(stripped_name)
    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized = normalized_text(target)
        if normalized and normalized not in seen:
            deduped.append(target)
            seen.add(normalized)
    return deduped


def proposed_manual_target(row: dict[str, str]) -> str:
    for target in proposed_manual_targets(row):
        return target
    for pattern in GENERIC_OR_AMBIGUOUS_MANUAL_PATTERNS:
        name = normalized_text(row.get("canonical_beverage_name", ""))
        search = normalized_text(row.get("canonical_search_name", ""))
        if re.search(pattern, f"{name} {search}".strip()):
            return ""
    return ""


def should_resolve_manual_as_unmapped(row: dict[str, str]) -> bool:
    return row.get("final_mapping_status") == "manual_review_unmapped"


def apply_manual_review_resolutions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    index = target_index(rows)
    output: list[dict[str, str]] = []
    for row in rows:
        if row.get("final_mapping_status") != "manual_review_unmapped":
            output.append(row)
            continue
        target_name = ""
        target = None
        for candidate in proposed_manual_targets(row):
            candidate_target = index.get(normalized_text(candidate))
            if candidate_target is not None:
                target_name = candidate
                target = candidate_target
                break
        if target is None:
            if should_resolve_manual_as_unmapped(row):
                resolved = dict(row)
                resolved.update(
                    {
                        "final_mapping_status": "unmapped",
                        "source_mapping_status": "manual_review_resolved_unmapped",
                        "mapped_target_source": "",
                        "included_or_mapped": "false",
                        "include_in_export": "false",
                        "needs_manual_review": "false",
                        "exclusion_reason": (
                            "Manual-review rule left this generic, blended, or ambiguous "
                            "product unmapped."
                        ),
                    }
                )
                output.append(resolved)
                continue
            output.append(row)
            continue

        resolved = dict(row)
        resolved.update(
            {
                "final_mapping_status": "mapped_to_run_target",
                "source_mapping_status": "manual_review_resolved_to_base_target",
                "mapped_target_source": "manual_resolution",
                "final_run_target_id": target.get("final_run_target_id", ""),
                "final_run_target_name": target.get("final_run_target_name", ""),
                "final_representative_canonical_beverage_id": target.get(
                    "final_representative_canonical_beverage_id", ""
                ),
                "final_representative_canonical_beverage_name": target.get(
                    "final_representative_canonical_beverage_name", ""
                ),
                "collapse_reason_code": "manual_base_ingredient_resolution",
                "collapse_reason": (
                    f"Manual-review rule linked this row to clear base ingredient "
                    f"'{target.get('final_run_target_name', '')}'."
                ),
                "run_path": target.get("run_path", ""),
                "claim_file": target.get("claim_file", ""),
                "web_audited_claim_count": target.get("web_audited_claim_count", ""),
                "included_or_mapped": "true",
                "include_in_export": target.get("include_in_export", "unknown"),
                "needs_manual_review": "false",
                "exclusion_reason": "",
            }
        )
        output.append(resolved)
    return output


def count_by(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get(column, "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_combined_mapping(
    *,
    original_path: Path,
    added_path: Path,
    out_dir: Path,
) -> dict[str, object]:
    original_rows = [normalize_original_row(row) for row in read_csv(original_path)]
    added_rows = [normalize_added_row(row) for row in read_csv(added_path)]
    initial_combined_rows = [*original_rows, *added_rows]
    combined_rows = apply_manual_review_resolutions(initial_combined_rows)
    included_rows = [row for row in combined_rows if row["included_or_mapped"] == "true"]
    not_included_rows = [row for row in combined_rows if row["included_or_mapped"] != "true"]

    combined_path = out_dir / "ingredient_to_research_target_map.combined.csv"
    included_path = out_dir / "ingredient_to_research_target_map.included.csv"
    not_included_path = out_dir / "ingredient_to_research_target_map.not_included.csv"
    summary_path = out_dir / "ingredient_to_research_target_map.combined_summary.json"

    write_csv(combined_path, combined_rows)
    write_csv(included_path, included_rows)
    write_csv(not_included_path, not_included_rows)

    summary: dict[str, object] = {
        "original_rows": len(original_rows),
        "added_rows": len(added_rows),
        "total_rows": len(combined_rows),
        "included_or_mapped_rows": len(included_rows),
        "not_included_rows": len(not_included_rows),
        "final_mapping_status_counts": count_by(combined_rows, "final_mapping_status"),
        "source_set_counts": count_by(combined_rows, "source_set"),
        "added_source_mapping_status_counts": count_by(added_rows, "source_mapping_status"),
        "added_include_in_export_counts": count_by(added_rows, "include_in_export"),
        "manual_review_resolved_count": sum(
            1
            for row in combined_rows
            if row.get("source_mapping_status") == "manual_review_resolved_to_base_target"
        ),
        "output_files": {
            "combined": str(combined_path),
            "included": str(included_path),
            "not_included": str(not_included_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine original and added ingredient-to-research-target mapping artifacts."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("ingredient_preparation_full/ingredient_to_research_target_map.csv"),
    )
    parser.add_argument(
        "--added",
        type=Path,
        default=Path("ingredient_preparation_added/added_ingredient_to_research_target_map.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ingredient_preparation_combined"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_combined_mapping(
        original_path=args.original,
        added_path=args.added,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
