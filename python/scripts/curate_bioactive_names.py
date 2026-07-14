from __future__ import annotations

import csv
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Supabase Snippet Bioactive Names Lookup.csv"
ACCEPTED_OUTPUT = ROOT / "bioactive_names_accepted_for_review.csv"
REJECTED_OUTPUT = ROOT / "bioactive_names_rejected_for_review.csv"

ALIASES = {
    "(-)-Epicatechin": "Epicatechin",
    "(-)-Epigallocatechin": "Epigallocatechin",
    "(-)-Epigallocatechin 3-O-gallate": "Epigallocatechin gallate (EGCG)",
    "(+)-Catechin": "Catechin",
    "(+)-Gallocatechin 3-O-gallate": "Gallocatechin gallate",
    "[6]-Gingerol": "6-Gingerol",
    "Alpha Linolenic Acid": "Alpha-linolenic acid",
    "Anthocyanins, total": "Anthocyanins",
    "Beta Carotene": "Beta-carotene",
    "Beta Glucan": "Beta-glucan",
    "Docosahexaenoic Acid": "Docosahexaenoic acid (DHA)",
    "Eicosapentaenoic Acid": "Eicosapentaenoic acid (EPA)",
    "Gamma Linolenic Acid": "Gamma-linolenic acid",
    "Monounsaturated Fat": "Monounsaturated fat",
    "Omega 6 Fat": "Omega-6 fat",
    "Omega 9 Fat": "Omega-9 fat",
    "Omega-3 Fatty Acids": "Omega-3 fatty acids",
    "Polyphenols, total": "Polyphenols",
    "Polyunsaturated Fat": "Polyunsaturated fat",
    "Saturated Fat": "Saturated fat",
    "Vitamin A and Carotenoids": "Vitamin A and carotenoids",
}

HIGH_PRIORITY = {
    "(-)-Epicatechin",
    "(-)-Epigallocatechin 3-O-gallate",
    "(+)-Catechin",
    "[6]-Gingerol",
    "Alcohol",
    "Alpha Linolenic Acid",
    "Anthocyanins, total",
    "Arachidonic Acid",
    "Beta Carotene",
    "Beta Glucan",
    "Biotin",
    "Butyric Acid",
    "Caffeine",
    "Calcium",
    "Carbohydrates",
    "Casein",
    "Chloride",
    "Cholesterol",
    "Choline",
    "Chromium",
    "Copper",
    "Curcumin",
    "Docosahexaenoic Acid",
    "Eicosapentaenoic Acid",
    "Erythritol",
    "Fiber",
    "Fluoride",
    "Folate",
    "Fructose",
    "Glucose",
    "Gluten",
    "Insoluble Fiber",
    "Inulin",
    "Iodine",
    "Iron",
    "Lactose",
    "Linoleic Acid",
    "Magnesium",
    "Maltitol",
    "Maltodextrins",
    "Manganese",
    "Monounsaturated Fat",
    "Niacin",
    "Nitrate",
    "Nitrite",
    "Oleic Acid",
    "Omega 6 Fat",
    "Omega 9 Fat",
    "Omega-3 Fatty Acids",
    "Pantothenic Acid",
    "Phosphorus",
    "Polyphenols, total",
    "Polyunsaturated Fat",
    "Potassium",
    "Proteins",
    "Quercetin",
    "Resveratrol",
    "Riboflavin",
    "Salt",
    "Saturated Fat",
    "Selenium",
    "Sodium",
    "Soluble Fiber",
    "Starch",
    "Sucrose",
    "Sugars",
    "Taurine",
    "Thiamin",
    "Vitamin A and Carotenoids",
    "Vitamin B12",
    "Vitamin B6",
    "Vitamin B9",
    "Vitamin C",
    "Vitamin D",
    "Vitamin D2",
    "Vitamin D3",
    "Vitamin E",
    "Vitamin K",
    "Zinc",
}

MEDIUM_PRIORITY = {
    "(-)-Epigallocatechin",
    "(+)-Gallocatechin 3-O-gallate",
    "3-Caffeoylquinic acid",
    "5-Caffeoylquinic acid",
    "8-Prenylnaringenin",
    "Apigenin",
    "Cyanidin",
    "Cyanidin 3-O-glucoside",
    "Daidzein",
    "Dihydroquercetin",
    "Ellagic acid",
    "Eriocitrin",
    "Ferulic acid",
    "Gamma Linolenic Acid",
    "Genistein",
    "Hesperetin",
    "Hesperidin",
    "Hydroxytyrosol",
    "Isorhamnetin",
    "Kaempferol",
    "Lariciresinol",
    "Lauric Acid",
    "Luteolin",
    "Matairesinol",
    "Myricetin",
    "Myristic Acid",
    "Naringenin",
    "Naringin",
    "Narirutin",
    "Neohesperidin",
    "Nervonic Acid",
    "Nobiletin",
    "Oleuropein",
    "Palmitic Acid",
    "Phloridzin",
    "Piceatannol",
    "Pinoresinol",
    "Pterostilbene",
    "Rosmarinic acid",
    "Secoisolariciresinol",
    "Sesamin",
    "Stearic Acid",
    "Tangeretin",
    "Theaflavin",
    "Tyrosol",
    "Xanthohumol",
}

DUPLICATE_REJECTIONS = {
    "Nitrates": "Duplicate concept; review the canonical singular entry Nitrate.",
    "Nitrites": "Duplicate concept; review the canonical singular entry Nitrite.",
    "Polyphenols": "Duplicate/overlapping concept; use the database identity Polyphenols, total.",
}

SPECIFIC_LOW_EVIDENCE_PATTERN = re.compile(
    r"(?:"
    r"\bO-(?:glucoside|galactoside|rutinoside|arabinoside|rhamnoside|sambubioside|sophoroside|xyloside)"
    r"|\bglucosyl\b|\bgentiobiose\b|\bglucose\b.*(?:feruloyl|coumaroyl|caffeoyl)"
    r"|\bglucuronide\b|\bmalonyl\b|\bacetyl-galactoside\b|\bacetyl-glucoside\b"
    r"|\baglycone\b|\bferulate\b|\bmer(?:s)?\b|\bdimer\b|\btrimer\b"
    r")",
    flags=re.IGNORECASE,
)

METABOLITE_OR_ANALYTICAL_PATTERN = re.compile(
    r"(?:"
    r"\bglutathionyl\b|\baldehyde\b|\bacetophenone\b|\bmethylcatechol\b"
    r"|\bethylcatechol\b|\bethylguaiacol\b|\bvinylguaiacol\b|\bvinylphenol\b"
    r"|\bvinylsyringol\b|\bphenylacetic acid\b|\bdihydroxyphenylglycol\b"
    r"|\bDHPEA\b|\bHPEA\b"
    r")",
    flags=re.IGNORECASE,
)

VAGUE_OR_NON_INTERVENTION = {
    "04-06 mers",
    "07-10 mers",
    "1,4-Naphtoquinone",
    "Behenic Acid",
    "Benzoic acid",
    "Cerotic Acid",
    "Fat",
    "Mead Acid",
    "Melissic Acid",
    "Ph",
    "Phenol",
    "Pigment A",
    "Polymers (>10 mers)",
    "Serum Proteins",
    "Sulfate",
}

PREDOMINANTLY_NON_ORAL_OR_PRECLINICAL = {
    "1-Acetoxypinoresinol",
    "2-Hydroxybenzoic acid",
    "2-Methoxy-5-prop-1-enylphenol",
    "3-Methoxynobiletin",
    "3-Methoxysinensetin",
    "3,7-Dimethylquercetin",
    "4-Hydroxycoumarin",
    "6-Geranylnaringenin",
    "6-Hydroxyluteolin",
    "7-Hydroxymatairesinol",
    "7-Hydroxysecoisolariciresinol",
    "7-Oxomatairesinol",
    "7,3',4'-Trihydroxyflavone",
    "7,4'-Dihydroxyflavone",
    "Acetyl eugenol",
    "Anethole",
    "Anhydro-secoisolariciresinol",
    "Arbutin",
    "Arctigenin",
    "Baicalein",
    "Bergapten",
    "Biochanin A",
    "Bisdemethoxycurcumin",
    "Butein",
    "Caffeic acid",
    "Carnosic acid",
    "Carnosol",
    "Carvacrol",
    "Catechol",
    "Chicoric acid",
    "Chrysin",
    "Cinnamic acid",
    "Cinnamtannin A2",
    "Cirsilineol",
    "Cirsimaritin",
    "Conidendrin",
    "Coumarin",
    "Coumestrol",
    "Cyclolariciresinol",
    "d-Viniferin",
    "Demethoxycurcumin",
    "Demethyloleuropein",
    "Didymin",
    "Dihydrocaffeic acid",
    "Dimethylmatairesinol",
    "Diosmin",
    "Epirosmanol",
    "Eriodictyol",
    "Episesamin",
    "Episesaminol",
    "Esculetin",
    "Esculin",
    "Estragole",
    "Eugenol",
    "Eupatorin",
    "Formononetin",
    "Galangin",
    "Gallic acid",
    "Gardenin B",
    "Geraldone",
    "Glycitein",
    "Guaiacol",
    "Hispidulin",
    "Isoferulic acid",
    "Isohydroxymatairesinol",
    "Isopimpinellin",
    "Isorhoifolin",
    "Isoxanthohumol",
    "Jaceosidin",
    "Juglone",
    "Kaempferide",
    "Lambertianin C",
    "Ligstroside",
    "Mellein",
    "Methylgalangin",
    "Morin",
    "Neoeriocitrin",
    "Nepetin",
    "Nortrachelogenin",
    "Oleoside 11-methylester",
    "Oleoside dimethylester",
    "p-Anisaldehyde",
    "Pallidol",
    "Pebrellin",
    "Peonidin",
    "Phloretin",
    "Pinocembrin",
    "Pinosylvin",
    "Pinotin A",
    "Poncirin",
    "Protocatechuic acid",
    "Protocatechuic aldehyde",
    "Punicalagin",
    "Psoralen",
    "Pyrogallol",
    "Rhamnetin",
    "Rhoifolin",
    "Rosmadial",
    "Rosmanol",
    "Sakuranetin",
    "Sanguiin H-6",
    "Scopoletin",
    "Scutellarein",
    "Sesaminol",
    "Sesamol",
    "Sesamolin",
    "Sesamolinol",
    "Sinapaldehyde",
    "Sinapic acid",
    "Sinapine",
    "Sinensetin",
    "Syringaldehyde",
    "Syringaresinol",
    "Syringic acid",
    "Tetramethylscutellarein",
    "Theaflavin 3-O-gallate",
    "Theaflavin 3'-O-gallate",
    "Thymol",
    "Todolactol A",
    "Trachelogenin",
    "Umbelliferone",
    "Valoneic acid dilactone",
    "Vanillic acid",
    "Vanillin",
    "Verbascoside",
    "Vitisin A",
    "Xanthotoxin",
}


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()))


def accepted_reason(priority: str) -> str:
    if priority == "high":
        return "Substantial human oral-health, nutritional, or epidemiologic literature is expected."
    return "Meaningful but narrower human oral-health literature is expected; suitable for a focused review."


def rejection_reason(name: str) -> str:
    if name in DUPLICATE_REJECTIONS:
        return DUPLICATE_REJECTIONS[name]
    if name in VAGUE_OR_NON_INTERVENTION:
        return "Too broad, vague, analytical, or unsuitable as a distinct oral bioactive intervention."
    if SPECIFIC_LOW_EVIDENCE_PATTERN.search(name):
        return (
            "Highly specific conjugate, glycoside, oligomer, or ester; direct human oral-health "
            "evidence is likely too sparse and usually belongs under a parent compound review."
        )
    if METABOLITE_OR_ANALYTICAL_PATTERN.search(name):
        return (
            "Primarily an analytical/metabolic marker or minor derivative rather than a practical "
            "oral intervention with direct human outcome evidence."
        )
    if name in PREDOMINANTLY_NON_ORAL_OR_PRECLINICAL:
        return (
            "Available evidence is expected to be predominantly preclinical, topical, safety-focused, "
            "or pharmaceutical rather than human oral-health evidence."
        )
    return (
        "Direct human oral-health evidence is expected to be too sparse or indirect for a useful "
        "standalone review; consider grouping it under its parent compound or food class."
    )


def main() -> None:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        names = [row["bioactive_name"].strip() for row in csv.DictReader(handle)]

    accepted = []
    rejected = []
    for database_name in names:
        if database_name in HIGH_PRIORITY or database_name in MEDIUM_PRIORITY:
            priority = "high" if database_name in HIGH_PRIORITY else "medium"
            research_name = ALIASES.get(database_name, database_name)
            accepted.append(
                {
                    "bioactive_name": database_name,
                    "research_name": research_name,
                    "slug": slugify(research_name),
                    "review_priority": priority,
                    "curation_reason": accepted_reason(priority),
                }
            )
        else:
            rejected.append(
                {
                    "bioactive_name": database_name,
                    "rejection_reason": rejection_reason(database_name),
                }
            )

    with ACCEPTED_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bioactive_name",
                "research_name",
                "slug",
                "review_priority",
                "curation_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(accepted)

    with REJECTED_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bioactive_name", "rejection_reason"],
        )
        writer.writeheader()
        writer.writerows(rejected)

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()
