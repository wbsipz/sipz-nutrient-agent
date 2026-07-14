from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, normalize_name, stable_uuid

ROOT = Path(__file__).resolve().parents[1]
BOTANICAL_QUEUE = ROOT / "botanical_names_accepted_for_review.csv"
LOOKUP = ROOT / "bioactive_health_evidence_rows.csv"


def main() -> None:
    with LOOKUP.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    with BOTANICAL_QUEUE.open("r", encoding="utf-8-sig", newline="") as handle:
        botanicals = list(csv.DictReader(handle))

    existing_names = {normalize_name(row.get("bioactive_name", "")) for row in rows}
    now = datetime.now(UTC).isoformat()
    added = []
    for botanical in botanicals:
        name = botanical["bioactive_name"].strip()
        research_name = botanical["research_name"].strip()
        if normalize_name(name) in existing_names or normalize_name(research_name) in existing_names:
            continue
        row = {column: "" for column in EVIDENCE_COLUMNS}
        row.update(
            {
                "id": stable_uuid("lookup-identity-row", "polyphenol", name),
                "bioactive_type": "polyphenol",
                "bioactive_id": stable_uuid("bioactive", "polyphenol", name),
                "bioactive_name": name,
                "review_status": "identity_lookup_only",
                "review_notes": (
                    "Identity-only lookup row for botanical supplement export; not an evidence row."
                ),
                "created_at": now,
                "updated_at": now,
            }
        )
        rows.append(row)
        existing_names.add(normalize_name(name))
        added.append(name)

    with LOOKUP.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Added {len(added)} botanical identity rows to {LOOKUP}")


if __name__ == "__main__":
    main()
