from typing import Literal

from pydantic import BaseModel, Field

QuoteMatchStatus = Literal[
    "exact",
    "normalized_whitespace",
    "dehyphenated_ligature_normalized",
    "not_found",
]


class ProposedClaim(BaseModel):
    id: str = Field(min_length=1)
    nutrient_name: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=1200)
    proposed_effect_slug: str = Field(min_length=1, max_length=120)
    proposed_effect_label: str = Field(min_length=1, max_length=200)


class SupportingQuote(BaseModel):
    quote: str = Field(min_length=1)
    section: str | None = None
    reason: str = Field(min_length=1)
    match_status: QuoteMatchStatus


class ValidatedClaim(BaseModel):
    effect_row_id: str = Field(min_length=1)
    proposed_claim_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    verdict: Literal[
        "supported",
        "supported_with_limitations",
        "unsupported",
        "over_scoped",
        "quote_not_found",
    ]
    support_level: str = Field(min_length=1)
    claim_scope: str = Field(min_length=1)
    supporting_quotes: list[SupportingQuote]
    limitations: list[str]
    accepted: bool
