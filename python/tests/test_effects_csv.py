import csv
import json

from sipz_agent.core.artifacts import write_effects_csv
from sipz_agent.core.synthesis import build_effect_rows
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import SupportingQuote, ValidatedClaim
from sipz_agent.schemas.effects import EFFECT_CSV_COLUMNS, EffectRow


def test_accepted_claim_creates_valid_csv_row(tmp_path) -> None:
    claim = ValidatedClaim(
        effect_row_id="45929237-0046-4769-bd55-bd0c8de2199b",
        proposed_claim_id="55929237-0046-4769-bd55-bd0c8de2199b",
        citation_id="pmid:39362658",
        verdict="supported_with_limitations",
        support_level="human_systematic_review",
        claim_scope="human evidence",
        supporting_quotes=[
            SupportingQuote(
                quote="Water fluoridation probably reduces caries experience in children",
                section="Results",
                reason="Supports effect",
                match_status="exact",
            )
        ],
        limitations=[],
        accepted=True,
    )
    rows = build_effect_rows(
        nutrient_name="Fluoride",
        nutrient_id="0291c7ef-13e0-4740-a669-a1aca1b9655c",
        accepted_claims=[claim],
        citations=[
            CandidateCitation(
                id="pmid:39362658",
                title="Demo",
                url="https://pubmed.ncbi.nlm.nih.gov/39362658/",
                source="demo",
                retrieval_query="fluoride",
            )
        ],
    )

    assert len(rows) == 1
    path = tmp_path / "effects.csv"
    write_effects_csv(path, rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EFFECT_CSV_COLUMNS
        csv_rows = list(reader)

    assert len(csv_rows) == 1
    parsed = EffectRow.model_validate(
        {
            **csv_rows[0],
            "score": float(csv_rows[0]["score"]),
            "match_confidence": float(csv_rows[0]["match_confidence"]),
            "tags": json.loads(csv_rows[0]["tags"]),
            "sources": json.loads(csv_rows[0]["sources"]),
        }
    )
    assert parsed.effect_slug == "dental_caries_prevention"
