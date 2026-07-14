from __future__ import annotations

import csv
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "botanical_names_accepted_for_review.csv"


ITEMS = [
    ("Ashwagandha", "Ashwagandha root extract", "adaptogenic_botanical"),
    ("Rhodiola", "Rhodiola rosea extract", "adaptogenic_botanical"),
    ("Panax ginseng", "Panax ginseng extract", "adaptogenic_botanical"),
    ("American ginseng", "American ginseng extract", "adaptogenic_botanical"),
    ("Siberian ginseng / Eleuthero", "Eleuthero extract", "adaptogenic_botanical"),
    ("Holy basil / Tulsi", "Holy basil extract", "adaptogenic_botanical"),
    ("Schisandra", "Schisandra extract", "adaptogenic_botanical"),
    ("Maca", "Maca root extract", "adaptogenic_botanical"),
    ("Bacopa", "Bacopa monnieri extract", "nootropic_botanical"),
    ("Saffron", "Saffron extract", "mood_sleep_botanical"),
    ("Ginkgo biloba", "Ginkgo biloba extract", "nootropic_botanical"),
    ("Gotu kola", "Gotu kola extract", "nootropic_botanical"),
    ("Lemon balm", "Lemon balm extract", "mood_sleep_botanical"),
    ("Valerian root", "Valerian root extract", "mood_sleep_botanical"),
    ("Passionflower", "Passionflower extract", "mood_sleep_botanical"),
    ("Kava", "Kava extract", "mood_sleep_botanical"),
    ("Magnolia bark", "Magnolia bark extract", "mood_sleep_botanical"),
    ("Lavender extract", "Lavender extract", "mood_sleep_botanical"),
    ("Chamomile extract", "Chamomile extract", "mood_sleep_botanical"),
    ("Huperzia serrata / Huperzine A", "Huperzine A", "nootropic_botanical"),
    ("Polygala tenuifolia", "Polygala tenuifolia extract", "nootropic_botanical"),
    ("Lion's mane", "Lion's mane mushroom extract", "medicinal_mushroom"),
    ("Reishi", "Reishi mushroom extract", "medicinal_mushroom"),
    ("Cordyceps", "Cordyceps extract", "medicinal_mushroom"),
    ("Chaga", "Chaga mushroom extract", "medicinal_mushroom"),
    ("Turkey tail", "Turkey tail mushroom extract", "medicinal_mushroom"),
    ("Maitake", "Maitake mushroom extract", "medicinal_mushroom"),
    ("Shiitake extract", "Shiitake mushroom extract", "medicinal_mushroom"),
    ("Tremella", "Tremella mushroom extract", "medicinal_mushroom"),
    ("Agaricus blazei", "Agaricus blazei mushroom extract", "medicinal_mushroom"),
    ("Milk thistle", "Milk thistle extract", "liver_digestive_botanical"),
    ("Artichoke leaf", "Artichoke leaf extract", "liver_digestive_botanical"),
    ("Dandelion root", "Dandelion root extract", "liver_digestive_botanical"),
    ("Burdock root", "Burdock root extract", "liver_digestive_botanical"),
    ("Boswellia", "Boswellia extract", "joint_inflammation_botanical"),
    ("Ginger extract", "Ginger extract", "functional_spice_extract"),
    ("Garlic extract", "Garlic extract", "cardiometabolic_botanical"),
    ("Green tea extract / EGCG", "Green tea extract", "cardiometabolic_botanical"),
    ("Cinnamon extract", "Cinnamon extract", "metabolic_botanical"),
    ("Fenugreek", "Fenugreek extract", "metabolic_botanical"),
    ("Berberine", "Berberine", "metabolic_botanical"),
    ("Bitter melon", "Bitter melon extract", "metabolic_botanical"),
    ("Gymnema sylvestre", "Gymnema sylvestre extract", "metabolic_botanical"),
    ("Banaba leaf", "Banaba leaf extract", "metabolic_botanical"),
    ("Mulberry leaf", "Mulberry leaf extract", "metabolic_botanical"),
    ("Nopal cactus", "Nopal cactus extract", "metabolic_botanical"),
    ("Bergamot extract", "Bergamot extract", "cardiometabolic_botanical"),
    ("Psyllium husk", "Psyllium husk", "fiber_supplement"),
    ("Konjac / Glucomannan", "Glucomannan", "fiber_supplement"),
    ("Echinacea", "Echinacea extract", "immune_botanical"),
    ("Elderberry", "Elderberry extract", "immune_botanical"),
    ("Andrographis", "Andrographis extract", "immune_botanical"),
    ("Olive leaf extract", "Olive leaf extract", "immune_cardiometabolic_botanical"),
    ("Oregano oil", "Oregano oil", "immune_respiratory_botanical"),
    ("Thyme extract", "Thyme extract", "immune_respiratory_botanical"),
    ("Black seed / Nigella sativa", "Nigella sativa extract", "immune_metabolic_botanical"),
    ("Propolis", "Propolis", "immune_botanical"),
    ("Goldenseal", "Goldenseal extract", "immune_digestive_botanical"),
    ("Pelargonium sidoides", "Pelargonium sidoides extract", "immune_respiratory_botanical"),
    ("Marshmallow root", "Marshmallow root extract", "digestive_respiratory_botanical"),
    ("Mullein", "Mullein extract", "respiratory_botanical"),
    ("Peppermint oil", "Peppermint oil", "digestive_botanical"),
    ("Fennel seed", "Fennel seed extract", "digestive_botanical"),
    ("Slippery elm", "Slippery elm bark", "digestive_botanical"),
    ("Aloe vera", "Aloe vera", "digestive_botanical"),
    ("Triphala", "Triphala", "digestive_botanical"),
    ("Gentian root", "Gentian root extract", "digestive_botanical"),
    ("Saw palmetto", "Saw palmetto extract", "hormonal_botanical"),
    ("Tongkat ali", "Tongkat ali extract", "hormonal_botanical"),
    ("Tribulus terrestris", "Tribulus terrestris extract", "hormonal_botanical"),
    ("Shatavari", "Shatavari extract", "hormonal_botanical"),
    ("Vitex / Chasteberry", "Chasteberry extract", "hormonal_botanical"),
    ("Red clover", "Red clover extract", "hormonal_botanical"),
    ("Black cohosh", "Black cohosh extract", "hormonal_botanical"),
    ("Dong quai", "Dong quai extract", "hormonal_botanical"),
    ("Wild yam", "Wild yam extract", "hormonal_botanical"),
    ("Damiana", "Damiana extract", "hormonal_botanical"),
    ("Cistanche", "Cistanche extract", "hormonal_botanical"),
    ("Hawthorn berry", "Hawthorn berry extract", "cardiometabolic_botanical"),
    ("Beetroot extract", "Beetroot extract", "cardiometabolic_botanical"),
    ("Cocoa flavanols", "Cocoa flavanols", "cardiometabolic_polyphenol"),
    ("Grape seed extract", "Grape seed extract", "cardiometabolic_polyphenol"),
    ("Pine bark extract / Pycnogenol", "Pine bark extract", "cardiometabolic_polyphenol"),
    ("Hibiscus", "Hibiscus extract", "cardiometabolic_botanical"),
    ("Pomegranate extract", "Pomegranate extract", "cardiometabolic_polyphenol"),
    ("Arjuna bark", "Arjuna bark extract", "cardiometabolic_botanical"),
    ("Horse chestnut", "Horse chestnut extract", "vascular_botanical"),
    ("Butcher's broom", "Butcher's broom extract", "vascular_botanical"),
    ("Cayenne / Capsaicin", "Capsaicin", "functional_spice_extract"),
    ("Amla / Indian gooseberry", "Amla extract", "antioxidant_botanical"),
    ("Rosehip extract", "Rosehip extract", "antioxidant_botanical"),
    ("Sea buckthorn", "Sea buckthorn extract", "antioxidant_botanical"),
    ("Bilberry", "Bilberry extract", "antioxidant_polyphenol"),
    ("Acerola cherry", "Acerola cherry extract", "antioxidant_botanical"),
    ("Tart cherry extract", "Tart cherry extract", "antioxidant_polyphenol"),
    ("Sulforaphane / Broccoli sprout extract", "Broccoli sprout extract", "isothiocyanate_supplement"),
    ("Astaxanthin", "Astaxanthin", "carotenoid_supplement"),
    ("Fisetin", "Fisetin", "polyphenol_supplement"),
]


def slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()))


def priority_for(category: str) -> str:
    high_categories = {
        "adaptogenic_botanical",
        "cardiometabolic_botanical",
        "metabolic_botanical",
        "fiber_supplement",
        "immune_botanical",
        "cardiometabolic_polyphenol",
        "polyphenol_supplement",
    }
    return "high" if category in high_categories else "medium"


def main() -> None:
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bioactive_name",
                "research_name",
                "slug",
                "review_priority",
                "category",
                "curation_reason",
            ],
        )
        writer.writeheader()
        for bioactive_name, research_name, category in ITEMS:
            writer.writerow(
                {
                    "bioactive_name": bioactive_name,
                    "research_name": research_name,
                    "slug": slugify(research_name),
                    "review_priority": priority_for(category),
                    "category": category,
                    "curation_reason": (
                        "Supplement-level botanical, mushroom, extract, or isolated bioactive; "
                        "review oral human evidence separately from food-level nutrients."
                    ),
                }
            )
    print(f"Wrote {len(ITEMS)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
