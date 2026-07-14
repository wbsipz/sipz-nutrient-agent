from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import orjson


RUN_TARGET_COLUMNS = [
    "run_target_id",
    "run_target_name",
    "representative_canonical_beverage_id",
    "representative_canonical_beverage_name",
    "covered_target_count",
    "covered_direct_row_count",
    "covered_reuse_row_count",
    "covered_target_names",
    "covered_representative_names",
    "collapse_reason_code",
    "collapse_reason",
]


PREFIXES = [
    "baby leaf ",
    "baby ",
    "fresh ",
    "frozen ",
    "freeze dried ",
    "freeze-dried ",
    "dehydrated ",
    "dried ",
    "raw ",
    "roasted ",
    "toasted ",
    "fine ground ",
    "coarse ground ",
    "ground ",
    "whole ",
    "diced ",
    "sliced ",
    "chopped ",
    "minced ",
    "crushed ",
    "grated ",
]

SUFFIXES = [
    " leaves",
    " leaf",
    " strips",
    " strip",
    " hearts",
    " heart",
    " chunks",
    " chunk",
    " pieces",
    " piece",
    " slices",
    " slice",
    " rings",
    " ring",
    " puree",
    " purée",
    " paste",
    " powder",
    " powdered",
    " ground",
    " granules",
    " granule",
]

PLURAL_ALIASES = {
    "blueberries": "blueberry",
    "strawberries": "strawberry",
    "raspberries": "raspberry",
    "blackberries": "blackberry",
    "cranberries": "cranberry",
    "cherries": "cherry",
    "figs": "fig",
    "cashews": "cashew",
    "almonds": "almond",
    "walnuts": "walnut",
    "dates": "date",
    "mushrooms": "mushroom",
    "cucumbers": "cucumber",
}

PHRASE_ALIASES = {
    "agave inulin": "inulin",
    "aleppo pepper": "chili pepper",
    "american grapes": "grape",
    "ancho pepper": "chili pepper",
    "anjou pear": "pear",
    "asian pear": "pear",
    "baby shiitake mushrooms": "shiitake mushroom",
    "banana pepper": "chili pepper",
    "barley malt": "barley",
    "bartlett pear": "pear",
    "bing cherry": "cherry",
    "black tahini": "sesame seed",
    "black walnuts": "walnut",
    "bovine collagen": "collagen",
    "bovine collagen hydrolysate": "collagen",
    "cacao nibs": "cocoa",
    "cacao powder": "cocoa",
    "cashew nuts": "cashew",
    "chicory inulin": "inulin",
    "chipotle chile pepper": "chili pepper",
    "chipotle pepper": "chili pepper",
    "cocoa nibs": "cocoa",
    "coconut meat": "coconut",
    "collagen peptides": "collagen",
    "deglet noor dates": "date",
    "dextrose": "glucose",
    "dried plums": "prune",
    "egg white protein powder": "egg white",
    "fermented cacao powder": "cocoa",
    "figs": "fig",
    "ginger root": "ginger",
    "goji berries": "goji berry",
    "gold kiwifruit": "kiwi",
    "grapes": "grape",
    "granny smith apples": "apple",
    "green bell pepper": "bell pepper",
    "green seedless grapes": "grape",
    "guajillo chile pepper": "chili pepper",
    "hatch green chile pepper": "chili pepper",
    "honeydew melon": "melon",
    "hemp seeds": "hemp seed",
    "hot pepper": "chili pepper",
    "hot pepper leaves": "chili pepper",
    "hulled hemp seed": "hemp seed",
    "hydrolyzed whey protein isolate powder": "whey protein",
    "jalapeno": "chili pepper",
    "jalapeno pepper": "chili pepper",
    "japanese cucumbers": "cucumber",
    "kiwifruit": "kiwi",
    "long hot pepper": "chili pepper",
    "malted barley": "barley",
    "medjool dates": "date",
    "navel oranges": "orange",
    "pickled jalapeno": "chili pepper",
    "pineapple core": "pineapple",
    "pink pitaya": "dragon fruit",
    "pinkglow pineapple": "pineapple",
    "pitaya": "dragon fruit",
    "plain low fat greek yogurt": "greek yogurt",
    "plain whole milk yogurt": "yogurt",
    "pumpkin seeds": "pumpkin seed",
    "rainier cherries": "cherry",
    "red pepper": "bell pepper",
    "red raspberry": "raspberry",
    "rocoto pepper": "chili pepper",
    "sesame": "sesame seed",
    "shiitake mushrooms": "shiitake mushroom",
    "sweet pepper": "bell pepper",
    "tahini": "sesame seed",
    "tart cherries": "tart cherry",
    "whey protein concentrate": "whey protein",
    "watermelon seeds": "watermelon seed",
    "yellow chile pepper": "chili pepper",
}

SUGAR_ALIASES = {
    "brown cane sugar": "brown sugar",
    "cane sugar": "sugar",
    "light brown cane sugar": "brown sugar",
    "panela": "sugar",
    "piloncillo": "sugar",
    "sugar cane": "sugar",
}

TARGET_RENAMES = {
    "algae": "algae oil",
    "baby leaf collard greens": "collard greens",
    "baby shiitake mushrooms": "shiitake mushroom",
    "banana melon": "melon",
    "brown rice vinegar": "vinegar",
    "canola": "canola oil",
    "casaba melon": "melon",
    "cod liver fish": "cod liver oil",
    "coconut mct": "mct oil",
    "corn": "corn oil",
    "dill pickle": "pickle",
    "drumstick leaves": "moringa",
    "durum wheat flour": "wheat flour",
    "durum wheat semolina": "wheat flour",
    "durum wheat whole wheat flour": "whole wheat flour",
    "einkorn wheat berries": "wheat berries",
    "extra virgin olive": "olive oil",
    "grape seed": "grape seed oil",
    "hard red spring wheat berries": "wheat berries",
    "hard red whole wheat flour": "whole wheat flour",
    "herring fish": "herring fish oil",
    "high oleic safflower": "safflower oil",
    "horned melon": "melon",
    "lettuce leaves": "lettuce",
    "little gem lettuce": "lettuce",
    "macadamia nut": "macadamia nut oil",
    "malted wheat flour": "wheat flour",
    "menhaden fish": "menhaden fish oil",
    "mid oleic sunflower": "sunflower oil",
    "mixed nuts": "nuts",
    "nut": "nuts",
    "oat bran flour": "oat bran",
    "panela cheese": "cheese",
    "rice vinegar": "vinegar",
    "romaine lettuce": "lettuce",
    "salmon fish": "salmon fish oil",
    "soybean": "soybean oil",
    "sunflower": "sunflower oil",
}

TARGET_EXCLUDES = {
    "clamato": "Formulated tomato/clam beverage; poor direct literature target.",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_TARGET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUN_TARGET_COLUMNS})


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    text = ascii_text.casefold().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_name(value))
    return re.sub(r"_+", "_", slug).strip("_")


def parse_json_list(value: str) -> list[str]:
    if not value.strip():
        return []
    raw = json.loads(value)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def existing_final_target_index(
    rows: list[dict[str, str]],
) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for row in rows:
        run_name = row["run_target_name"]
        index.setdefault(normalize_name(run_name), (run_name, "existing_final_target"))
        for covered in parse_json_list(row.get("covered_target_names", "[]")):
            index.setdefault(normalize_name(covered), (run_name, "existing_final_covered_target"))
    return index


def candidate_variants(name: str) -> list[str]:
    normalized = normalize_name(name)
    candidates = [normalized]
    if normalized in PLURAL_ALIASES:
        candidates.append(PLURAL_ALIASES[normalized])
    if normalized in PHRASE_ALIASES:
        candidates.append(PHRASE_ALIASES[normalized])
    if normalized in SUGAR_ALIASES:
        candidates.append(SUGAR_ALIASES[normalized])
    if normalized.endswith(" sea salt"):
        candidates.append("sea salt")
    elif normalized.endswith(" salt"):
        candidates.append("salt")

    for prefix in PREFIXES:
        if normalized.startswith(prefix):
            candidates.append(normalized[len(prefix) :])
    for suffix in SUFFIXES:
        if normalized.endswith(suffix):
            candidates.append(normalized[: -len(suffix)])
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            if normalized.startswith(prefix) and normalized.endswith(suffix):
                candidates.append(normalized[len(prefix) : -len(suffix)])

    for candidate in list(candidates):
        if candidate in PLURAL_ALIASES:
            candidates.append(PLURAL_ALIASES[candidate])
        if candidate in PHRASE_ALIASES:
            candidates.append(PHRASE_ALIASES[candidate])
        if candidate in SUGAR_ALIASES:
            candidates.append(SUGAR_ALIASES[candidate])

    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = normalize_name(candidate)
        if clean and clean not in seen:
            output.append(clean)
            seen.add(clean)
    return output


def choose_run_target(
    *,
    name: str,
    added_names: set[str],
    existing_index: dict[str, tuple[str, str]],
) -> tuple[str, str, str]:
    normalized = normalize_name(name)
    if normalized in existing_index:
        run_name, reason_code = existing_index[normalized]
        return run_name, reason_code, f"Mapped to existing final target '{run_name}'."

    if normalized in TARGET_EXCLUDES:
        return normalized, "excluded_low_value", TARGET_EXCLUDES[normalized]

    if normalized in TARGET_RENAMES:
        run_name = TARGET_RENAMES[normalized]
        return run_name, "target_renamed", f"Renamed '{name}' to better research target '{run_name}'."

    for candidate in candidate_variants(name):
        if candidate == normalized:
            continue
        if candidate in existing_index:
            run_name, reason_code = existing_index[candidate]
            return run_name, reason_code, f"Mapped variant '{name}' to existing final target '{run_name}'."
        if candidate in added_names:
            return candidate, "added_variant", f"Collapsed preparation/form variant '{name}' to '{candidate}'."

    return normalized, "unchanged", "Kept as distinct added-ingredient research target."


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def build_crunched_run_target_rows(
    *,
    canonical_target_rows: list[dict[str, str]],
    existing_final_target_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    existing_index = existing_final_target_index(existing_final_target_rows or [])
    added_names = {normalize_name(row["canonical_search_name"]) for row in canonical_target_rows}
    grouped: dict[str, dict[str, Any]] = {}

    for row in canonical_target_rows:
        name = row["canonical_search_name"]
        run_name, reason_code, reason = choose_run_target(
            name=name,
            added_names=added_names,
            existing_index=existing_index,
        )
        key = normalize_name(run_name)
        if key not in grouped:
            grouped[key] = {
                "run_target_id": slugify(run_name),
                "run_target_name": run_name,
                "representative_canonical_beverage_id": row.get(
                    "representative_canonical_beverage_id", ""
                ),
                "representative_canonical_beverage_name": row.get(
                    "representative_canonical_beverage_name", ""
                ),
                "covered_target_names": [],
                "covered_representative_names": [],
                "covered_direct_row_count": 0,
                "covered_reuse_row_count": 0,
                "reason_codes": [],
                "reasons": [],
            }
        group = grouped[key]
        append_unique(group["covered_target_names"], name)
        append_unique(
            group["covered_representative_names"],
            row.get("representative_canonical_beverage_name", ""),
        )
        group["covered_direct_row_count"] += int(row.get("direct_row_count") or 0)
        group["covered_reuse_row_count"] += int(row.get("reuse_row_count") or 0)
        append_unique(group["reason_codes"], reason_code)
        append_unique(group["reasons"], reason)

    output = []
    for group in grouped.values():
        output.append(
            {
                "run_target_id": group["run_target_id"],
                "run_target_name": group["run_target_name"],
                "representative_canonical_beverage_id": group[
                    "representative_canonical_beverage_id"
                ],
                "representative_canonical_beverage_name": group[
                    "representative_canonical_beverage_name"
                ],
                "covered_target_count": str(len(group["covered_target_names"])),
                "covered_direct_row_count": str(group["covered_direct_row_count"]),
                "covered_reuse_row_count": str(group["covered_reuse_row_count"]),
                "covered_target_names": json.dumps(
                    group["covered_target_names"], ensure_ascii=False
                ),
                "covered_representative_names": json.dumps(
                    group["covered_representative_names"], ensure_ascii=False
                ),
                "collapse_reason_code": "+".join(sorted(group["reason_codes"])),
                "collapse_reason": " | ".join(group["reasons"]),
            }
        )
    output.sort(key=lambda row: row["run_target_name"])
    summary = {
        "source_target_count": len(canonical_target_rows),
        "run_target_count": len(output),
        "reduction_count": len(canonical_target_rows) - len(output),
        "reduction_percent": round(
            ((len(canonical_target_rows) - len(output)) / len(canonical_target_rows) * 100)
            if canonical_target_rows
            else 0,
            2,
        ),
        "reason_code_counts": {},
        "largest_groups": [],
    }
    for row in output:
        for reason_code in row["collapse_reason_code"].split("+"):
            summary["reason_code_counts"][reason_code] = (
                summary["reason_code_counts"].get(reason_code, 0) + 1
            )
    summary["largest_groups"] = [
        {
            "run_target_name": row["run_target_name"],
            "covered_target_count": int(row["covered_target_count"]),
            "covered_target_names": parse_json_list(row["covered_target_names"]),
            "collapse_reason_code": row["collapse_reason_code"],
        }
        for row in sorted(output, key=lambda item: int(item["covered_target_count"]), reverse=True)[
            :25
        ]
        if int(row["covered_target_count"]) > 1
    ]
    return output, summary


def build_crunched_run_targets(
    *,
    canonical_targets_path: Path,
    out_path: Path,
    summary_path: Path,
    existing_final_targets_path: Path | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    existing_rows = (
        read_csv_rows(existing_final_targets_path)
        if existing_final_targets_path is not None and existing_final_targets_path.exists()
        else []
    )
    rows, summary = build_crunched_run_target_rows(
        canonical_target_rows=read_csv_rows(canonical_targets_path),
        existing_final_target_rows=existing_rows,
    )
    write_csv_rows(out_path, rows)
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))
    return rows, summary
