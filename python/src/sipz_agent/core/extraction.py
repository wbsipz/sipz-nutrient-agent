from uuid import uuid4

from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim


def extract_claims(nutrient_name: str, citations: list[CandidateCitation]) -> list[ProposedClaim]:
    if nutrient_name.lower() != "fluoride":
        return []

    claims: list[ProposedClaim] = []
    for citation in citations:
        if citation.id == "pmid:39362658":
            claims.append(
                ProposedClaim(
                    id=str(uuid4()),
                    nutrient_name="Fluoride",
                    citation_id=citation.id,
                    statement=(
                        "Human evidence supports that oral fluoride exposure from fluoridated "
                        "drinking water reduces dental caries risk, especially in children."
                    ),
                    proposed_effect_slug="dental_caries_prevention",
                    proposed_effect_label="Dental caries prevention",
                )
            )
        else:
            claims.append(
                ProposedClaim(
                    id=str(uuid4()),
                    nutrient_name="Fluoride",
                    citation_id=citation.id,
                    statement="Fluoride has a validated unsupported demo effect that should be rejected.",
                    proposed_effect_slug="unsupported_demo_effect",
                    proposed_effect_label="Unsupported demo effect",
                )
            )
    return claims
