from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


FIELDNAMES = [
    "canonical_bioactive_name",
    "meaningful_threshold_amount",
    "meaningful_threshold_unit",
    "meaningful_threshold_basis",
    "meaningful_threshold_source",
    "meaningful_review_status",
    "supplement_threshold_amount",
    "supplement_threshold_unit",
    "supplement_threshold_basis",
    "supplement_threshold_source",
    "supplement_review_status",
    "trace_definition",
    "meaningful_definition",
    "supplement_definition",
    "notes",
]


@dataclass(frozen=True)
class DoseBandOverride:
    meaningful_amount: str
    meaningful_unit: str
    supplement_amount: str
    supplement_unit: str
    threshold_basis: str
    review_status: str
    notes: str = ""


DOSE_BAND_OVERRIDES: dict[str, DoseBandOverride] = {
    "Alcohol": DoseBandOverride("1", "g/serving", "14", "g/serving", "caution threshold; one standard-drink context for higher alcohol effects", "draft_caution_context", "For alcohol, supplement threshold is a high-caution/standard-drink threshold, not a beneficial supplement target."),
    "Beta Carotene": DoseBandOverride("90", "ug RAE/serving vitamin A context", "15000", "ug beta-carotene/serving", "vitamin A context plus high-dose beta-carotene trial territory", "needs_review", "Units mix RAE and beta-carotene mass; review before deterministic use."),
    "Biotin": DoseBandOverride("3", "ug/serving", "300", "ug/serving", "10% DV for meaningful; common high-dose supplement territory for supplement effects", "draft_from_reference_intake"),
    "Caffeine": DoseBandOverride("25", "mg/serving", "200", "mg/serving", "common perceptible caffeine dose; high single-dose caffeine context", "draft_caution_context"),
    "Calcium": DoseBandOverride("130", "mg/serving", "500", "mg/serving", "10% DV for meaningful; common supplemental calcium dose", "draft_from_reference_intake"),
    "Carbohydrates": DoseBandOverride("5", "g/serving", "30", "g/serving", "macronutrient energy contribution; sports-carbohydrate dose context", "draft_macronutrient_context"),
    "Chloride": DoseBandOverride("230", "mg/serving", "1000", "mg/serving", "10% DV for meaningful; high electrolyte/salt context", "draft_from_reference_intake"),
    "Cholesterol": DoseBandOverride("30", "mg/serving", "150", "mg/serving", "10% caution context; high-cholesterol serving context", "draft_caution_context"),
    "Chromium": DoseBandOverride("3.5", "ug/serving", "200", "ug/serving", "10% DV for meaningful; common chromium supplement trial dose", "draft_from_reference_intake"),
    "Cocoa flavanols": DoseBandOverride("200", "mg cocoa flavanols/serving", "500", "mg cocoa flavanols/serving", "human cocoa flavanol intervention dose context", "needs_review", "Current nutriment profiles may use cocoa proxy units; review unit conversion before deterministic use."),
    "Copper": DoseBandOverride("0.09", "mg/serving", "2", "mg/serving", "10% DV for meaningful; high dietary/supplement caution context", "draft_from_reference_intake"),
    "Fat": DoseBandOverride("5", "g/serving", "20", "g/serving", "macronutrient energy and fat-soluble absorption context", "draft_macronutrient_context"),
    "Fiber": DoseBandOverride("2.8", "g/serving", "5", "g/serving", "10% DV for meaningful; common prebiotic/soluble fiber effect dose context", "draft_from_reference_intake"),
    "Folate": DoseBandOverride("40", "ug DFE/serving", "400", "ug DFE/serving", "10% DV for meaningful; common folate/folic-acid supplement dose", "draft_from_reference_intake"),
    "Insoluble Fiber": DoseBandOverride("2.8", "g/serving", "5", "g/serving", "derived from dietary fiber DV; fiber intervention dose context", "draft_derived_context"),
    "Iodine": DoseBandOverride("15", "ug/serving", "150", "ug/serving", "10% DV for meaningful; adult DV-level supplement context", "draft_from_reference_intake"),
    "Iron": DoseBandOverride("1.8", "mg/serving", "18", "mg/serving", "10% DV for meaningful; DV-level iron supplement context", "draft_from_reference_intake"),
    "Lactose": DoseBandOverride("6", "g/serving", "12", "g/serving", "lactose intolerance symptom dose context", "needs_review", "No FDA DV; thresholds are tolerance/symptom-context draft values."),
    "Magnesium": DoseBandOverride("42", "mg/serving", "250", "mg/serving", "10% DV for meaningful; common supplemental magnesium trial dose", "draft_from_reference_intake"),
    "Maltose": DoseBandOverride("5", "g/serving", "25", "g/serving", "sugar concentration context using added-sugar caution analogy", "needs_review", "No distinct daily reference; review if maltose-specific effects are used."),
    "Manganese": DoseBandOverride("0.23", "mg/serving", "5", "mg/serving", "10% DV for meaningful; high manganese supplement/caution context", "draft_from_reference_intake"),
    "Molybdenum": DoseBandOverride("4.5", "ug/serving", "45", "ug/serving", "10% DV for meaningful; adult DV-level supplement context", "draft_from_reference_intake"),
    "Monounsaturated Fat": DoseBandOverride("5", "g/serving", "20", "g/serving", "fat subtype dietary pattern context", "needs_review", "No established DV; effects depend on replacement nutrient and dietary pattern."),
    "Niacin": DoseBandOverride("1.6", "mg NE/serving", "30", "mg NE/serving", "10% DV for meaningful; flushing/high-dose niacin context", "draft_from_reference_intake"),
    "Omega-3 Fatty Acids": DoseBandOverride("0.15", "g ALA/EPA/DHA context/serving", "1", "g EPA/DHA or omega-3/serving", "AI context for meaningful; common omega-3 supplement territory", "needs_review", "Profiles may not distinguish ALA from EPA/DHA; review before deterministic use."),
    "Pantothenic Acid": DoseBandOverride("0.5", "mg/serving", "600", "mg/serving", "10% DV for meaningful; pantethine lipid-study dose context", "draft_from_reference_intake", "Supplement threshold mostly applies to pantethine, not ordinary dietary pantothenic acid."),
    "Phosphorus": DoseBandOverride("125", "mg/serving", "700", "mg/serving", "10% DV for meaningful; high phosphorus serving/supplement context", "draft_from_reference_intake"),
    "Polyunsaturated Fat": DoseBandOverride("5", "g/serving", "20", "g/serving", "fat subtype dietary pattern context", "needs_review", "No established DV; effects depend on omega-3/omega-6 composition and replacement nutrient."),
    "Potassium": DoseBandOverride("470", "mg/serving", "1900", "mg/serving", "10% DV for meaningful; lower end of common potassium intervention dose context", "draft_from_reference_intake"),
    "Proteins": DoseBandOverride("5", "g/serving", "20", "g/serving", "10% DV for meaningful; meal-level protein effect threshold", "draft_macronutrient_context"),
    "Riboflavin": DoseBandOverride("0.13", "mg/serving", "25", "mg/serving", "10% DV for meaningful; high-dose riboflavin supplement context", "draft_from_reference_intake"),
    "Salt": DoseBandOverride("0.58", "g salt/serving", "2", "g salt/serving", "10% salt caution context; high-salt serving context", "draft_caution_context"),
    "Saturated Fat": DoseBandOverride("2", "g/serving", "10", "g/serving", "10% caution context; high saturated-fat serving context", "draft_caution_context"),
    "Selenium": DoseBandOverride("5.5", "ug/serving", "100", "ug/serving", "10% DV for meaningful; common selenium supplement dose", "draft_from_reference_intake"),
    "Sodium": DoseBandOverride("230", "mg/serving", "1000", "mg/serving", "10% sodium limit context; high-sodium serving context", "draft_caution_context"),
    "Soluble Fiber": DoseBandOverride("2.8", "g/serving", "5", "g/serving", "derived from dietary fiber DV; soluble/prebiotic fiber intervention context", "draft_derived_context"),
    "Sucrose": DoseBandOverride("5", "g/serving", "25", "g/serving", "sugar concentration context using added-sugar caution analogy", "needs_review", "No distinct daily reference; review if sucrose-specific effects are used."),
    "Sugars": DoseBandOverride("5", "g/serving", "25", "g/serving", "10% added-sugar caution context; high-sugar serving context", "draft_caution_context"),
    "Taurine": DoseBandOverride("500", "mg/serving", "1500", "mg/serving", "energy-drink exposure context; lower end of taurine trial doses", "needs_review", "No established DV; most therapeutic evidence uses 1.5 g/day or more."),
    "Thiamin": DoseBandOverride("0.12", "mg/serving", "10", "mg/serving", "10% DV for meaningful; high-dose thiamin supplement context", "draft_from_reference_intake"),
    "Vitamin A and Carotenoids": DoseBandOverride("90", "ug RAE/serving", "900", "ug RAE/serving", "10% DV for meaningful; DV-level high dietary/supplement context", "draft_from_reference_intake", "For toxicity, distinguish preformed vitamin A from carotenoids."),
    "Vitamin B12": DoseBandOverride("0.24", "ug/serving", "25", "ug/serving", "10% DV for meaningful; high-dose oral B12 supplement context", "draft_from_reference_intake"),
    "Vitamin B6": DoseBandOverride("0.17", "mg/serving", "10", "mg/serving", "10% DV for meaningful; high-dose B6 supplement context", "draft_from_reference_intake"),
    "Vitamin B9": DoseBandOverride("40", "ug DFE/serving", "400", "ug DFE/serving", "10% DV for meaningful; folate/folic-acid supplement context", "draft_derived_context"),
    "Vitamin C": DoseBandOverride("10", "mg/serving", "200", "mg/serving", "food-level vitamin C contribution and common supplement-effect threshold", "draft_from_reference_intake"),
    "Vitamin D": DoseBandOverride("2", "ug/serving", "25", "ug/serving", "10% DV for meaningful; common vitamin D supplement dose", "draft_from_reference_intake"),
    "Vitamin D3": DoseBandOverride("2", "ug/serving", "25", "ug/serving", "derived from vitamin D DV; common D3 supplement dose", "draft_derived_context"),
    "Vitamin E": DoseBandOverride("1.5", "mg alpha-tocopherol/serving", "100", "mg alpha-tocopherol/serving", "10% DV for meaningful; high-dose vitamin E supplement context", "draft_from_reference_intake"),
    "Vitamin K": DoseBandOverride("12", "ug/serving", "120", "ug/serving", "10% DV for meaningful; DV-level vitamin K supplement context", "draft_from_reference_intake"),
    "Zinc": DoseBandOverride("1.1", "mg/serving", "10", "mg/serving", "10% DV for meaningful; lower end of zinc therapeutic supplement doses", "draft_from_reference_intake"),
}


def parse_low_numeric_amount(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def format_amount(value: float) -> str:
    rounded = round(value, 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:g}"


def serving_unit_from_reference_unit(unit: str) -> str:
    if not unit:
        return ""
    return re.sub(r"/day\b", "/serving", unit)


def primary_unit(unit: str) -> str:
    return unit.split()[0] if unit else ""


def is_non_monotonic_threshold(
    meaningful_amount: str,
    meaningful_unit: str,
    supplement_amount: str,
    supplement_unit: str,
) -> bool:
    if not meaningful_amount or not meaningful_unit or not supplement_amount or not supplement_unit:
        return False
    if primary_unit(meaningful_unit) != primary_unit(supplement_unit):
        return False
    try:
        return float(supplement_amount) <= float(meaningful_amount)
    except ValueError:
        return False


def build_trace_definition(meaningful_amount: str, meaningful_unit: str) -> str:
    if not meaningful_amount or not meaningful_unit:
        return "Not yet defined; reviewer must set meaningful threshold before dose routing."
    return (
        f"< {meaningful_amount} {meaningful_unit}; do not surface direct health-effect claims, "
        "only optional background presence language."
    )


def build_meaningful_definition(
    meaningful_amount: str,
    meaningful_unit: str,
    supplement_amount: str,
    supplement_unit: str,
) -> str:
    if not meaningful_amount or not meaningful_unit:
        return "Not yet defined; reviewer must set meaningful threshold before dose routing."
    if supplement_amount and supplement_unit:
        if is_non_monotonic_threshold(meaningful_amount, meaningful_unit, supplement_amount, supplement_unit):
            return (
                f">= {meaningful_amount} {meaningful_unit}; eligible for normal food-level/naturally "
                "occurring health-effect summaries when evidence context supports the effect. "
                "Supplement threshold is not above the meaningful threshold; review thresholds before "
                "routing supplement-level effects."
            )
        return (
            f">= {meaningful_amount} {meaningful_unit} and below {supplement_amount} "
            f"{supplement_unit}; eligible for normal food-level/naturally occurring "
            "health-effect summaries."
        )
    return (
        f">= {meaningful_amount} {meaningful_unit}; eligible for normal food-level/naturally "
        "occurring health-effect summaries when evidence context supports the effect."
    )


def build_supplement_definition(supplement_amount: str, supplement_unit: str) -> str:
    if not supplement_amount or not supplement_unit:
        return "Not yet reviewed; populate after supplement-dose reference review."
    if supplement_unit == "pH":
        return (
            f"pH <= {supplement_amount}; high-acidity context may be considered when the effect "
            "context matches the recipe acidity. This is not a supplement-dose threshold."
        )
    return (
        f">= {supplement_amount} {supplement_unit}; supplement-level, therapeutic-dose, "
        "high-caution, or high-exposure effects may be considered when the effect context "
        "matches the recipe dose."
    )


def read_reference_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["canonical_bioactive_name"]: row for row in reader}


def read_unique_nutriments(path: Path) -> list[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            names.add(row["canonical_bioactive_name"])
    return sorted(names)


def build_reference_fallback_row(name: str, reference_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    reference = reference_rows.get(name, {})
    reference_amount = reference.get("reference_amount", "")
    reference_unit = reference.get("reference_unit", "")
    caution_limit = reference.get("caution_limit", "")
    caution_limit_unit = reference.get("caution_limit_unit", "")
    reference_type = reference.get("reference_type", "")
    source = reference.get("source", "")
    threshold_notes = reference.get("threshold_notes", "")

    base_amount = parse_low_numeric_amount(reference_amount) if reference_amount else None
    base_unit = reference_unit
    meaningful_basis = "10% of reference intake context"
    if base_amount is None and caution_limit:
        base_amount = parse_low_numeric_amount(caution_limit)
        base_unit = caution_limit_unit
        meaningful_basis = "10% of caution-limit context"

    if base_amount is not None and base_unit:
        meaningful_amount = format_amount(base_amount * 0.1)
        meaningful_unit = serving_unit_from_reference_unit(base_unit)
    else:
        meaningful_amount = ""
        meaningful_unit = ""
        meaningful_basis = "no established intake reference; needs reviewer-defined meaningful threshold"

    if source:
        source = f"{source}; dose-band row added from reference table expansion"
    else:
        source = "Dose-band row added from reference table expansion; no source in reference table"
    meaningful_status = "draft_reference_baseline" if meaningful_amount and meaningful_unit else "needs_meaningful_review"

    return {
        "canonical_bioactive_name": name,
        "meaningful_threshold_amount": meaningful_amount,
        "meaningful_threshold_unit": meaningful_unit,
        "meaningful_threshold_basis": f"{meaningful_basis}; {reference_type}; {threshold_notes}".strip("; "),
        "meaningful_threshold_source": source,
        "meaningful_review_status": meaningful_status,
        "supplement_threshold_amount": "",
        "supplement_threshold_unit": "",
        "supplement_threshold_basis": "Not yet reviewed; populate after supplement-dose reference review.",
        "supplement_threshold_source": "",
        "supplement_review_status": "needs_supplement_dose_review",
        "trace_definition": build_trace_definition(meaningful_amount, meaningful_unit),
        "meaningful_definition": build_meaningful_definition(meaningful_amount, meaningful_unit, "", ""),
        "supplement_definition": build_supplement_definition("", ""),
        "notes": (
            "Added during dose-band table expansion. Supplement threshold intentionally blank until "
            "reference-backed supplement-dose review is completed."
        ),
    }


def build_row(name: str, reference_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    if name not in DOSE_BAND_OVERRIDES:
        return build_reference_fallback_row(name, reference_rows)
    override = DOSE_BAND_OVERRIDES[name]
    reference = reference_rows.get(name, {})
    source = reference.get("source", "")
    if source:
        source = f"{source}; dose-band thresholds are draft Sipz review thresholds"
    else:
        source = "Draft Sipz dose-band review threshold; no reference-intake source row found"

    return {
        "canonical_bioactive_name": name,
        "meaningful_threshold_amount": override.meaningful_amount,
        "meaningful_threshold_unit": override.meaningful_unit,
        "meaningful_threshold_basis": override.threshold_basis,
        "meaningful_threshold_source": source,
        "meaningful_review_status": override.review_status,
        "supplement_threshold_amount": override.supplement_amount,
        "supplement_threshold_unit": override.supplement_unit,
        "supplement_threshold_basis": override.threshold_basis,
        "supplement_threshold_source": source,
        "supplement_review_status": override.review_status,
        "trace_definition": build_trace_definition(override.meaningful_amount, override.meaningful_unit),
        "meaningful_definition": build_meaningful_definition(
            override.meaningful_amount,
            override.meaningful_unit,
            override.supplement_amount,
            override.supplement_unit,
        ),
        "supplement_definition": build_supplement_definition(override.supplement_amount, override.supplement_unit),
        "notes": override.notes,
    }


def build_dose_band_rows(nutriments: list[str], reference_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [build_row(name, reference_rows) for name in nutriments]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_summary(path: Path, *, rows: list[dict[str, str]], nutriments: list[str]) -> None:
    meaningful_review_counts: dict[str, int] = {}
    supplement_review_counts: dict[str, int] = {}
    for row in rows:
        meaningful_status = row["meaningful_review_status"]
        supplement_status = row["supplement_review_status"]
        meaningful_review_counts[meaningful_status] = meaningful_review_counts.get(meaningful_status, 0) + 1
        supplement_review_counts[supplement_status] = supplement_review_counts.get(supplement_status, 0) + 1
    summary = {
        "row_count": len(rows),
        "nutriment_count": len(nutriments),
        "meaningful_review_status_counts": dict(sorted(meaningful_review_counts.items())),
        "supplement_review_status_counts": dict(sorted(supplement_review_counts.items())),
        "nutriments": nutriments,
        "output_csv": str(path.with_name("nutriment_dose_band_table.csv")),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nutriment trace/meaningful/supplement dose-band table.")
    parser.add_argument("--summaries", type=Path, required=True, help="Cleaned nutriment summaries JSONL.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference-intake table CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output nutriment_dose_band_table.csv path.")
    parser.add_argument(
        "--reference-scope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include every nutriment from the reference-intake table, not only summarized nutriments.",
    )
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    args = parser.parse_args()

    reference_rows = read_reference_rows(args.reference)
    nutriments = set(read_unique_nutriments(args.summaries))
    if args.reference_scope:
        nutriments.update(reference_rows)
    nutriment_names = sorted(nutriments)
    rows = build_dose_band_rows(nutriment_names, reference_rows)
    write_csv(args.out, rows)
    summary_path = args.summary or args.out.with_name("nutriment_dose_band_table_summary.json")
    write_summary(summary_path, rows=rows, nutriments=nutriment_names)

    print(f"nutriments={len(nutriment_names)}")
    print(f"rows={len(rows)}")
    print(f"wrote={args.out}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
