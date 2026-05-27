from uuid import uuid4

from sipz_agent.core.quote_grounding import ground_quote
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, SupportingQuote, ValidatedClaim


def validate_claim(
    claim: ProposedClaim,
    citation: CandidateCitation,
    body_text_without_abstract: str,
) -> ValidatedClaim:
    if claim.proposed_effect_slug == "dental_caries_prevention":
        quote = "Water fluoridation probably reduces caries experience in children"
    else:
        quote = "This exact supporting quote is intentionally absent from the paper body"

    grounded = ground_quote(quote, body_text_without_abstract)
    accepted = claim.proposed_effect_slug == "dental_caries_prevention" and grounded.found

    return ValidatedClaim(
        effect_row_id=str(uuid4()),
        proposed_claim_id=claim.id,
        citation_id=citation.id,
        verdict="supported_with_limitations" if accepted else "quote_not_found",
        support_level="human_systematic_review" if accepted else "unsupported",
        claim_scope=(
            "Human evidence for dental caries prevention from oral fluoride exposure; "
            "benefit estimate varies by population and exposure context."
            if accepted
            else "Rejected because the validator quote could not be grounded in the body text."
        ),
        supporting_quotes=[
            SupportingQuote(
                quote=quote,
                section="Results" if accepted else "Unknown",
                reason=(
                    "Supports reduced caries risk."
                    if accepted
                    else "Demonstrates quote-grounding rejection."
                ),
                match_status=grounded.match_status,
            )
        ],
        limitations=(
            [
                "Effect size may differ between older and contemporary studies.",
                "Evidence depends on exposure route, population, and baseline fluoride availability.",
            ]
            if accepted
            else ["The required supporting quote was not found in the paper body."]
        ),
        accepted=accepted,
    )
