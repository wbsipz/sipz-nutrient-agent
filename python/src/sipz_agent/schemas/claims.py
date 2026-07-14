from typing import Literal

from pydantic import BaseModel, Field, field_validator

QuoteMatchStatus = Literal[
    "exact",
    "normalized_whitespace",
    "dehyphenated_ligature_normalized",
    "not_found",
]

IntakeRoute = Literal["oral", "topical", "injection", "in_vitro", "unclear"]
ExposureCategory = Literal[
    "natural_food_level",
    "supplement_level",
    "pharmaceutical_level",
    "unclear",
]
ClaimDirection = Literal["beneficial", "harmful", "neutral", "mixed", "unclear"]
EvidenceType = Literal[
    "human_clinical",
    "human_observational",
    "human_mechanistic",
    "animal",
    "in_vitro",
    "mechanistic_theory",
    "review_author_interpretation",
    "composition_data",
    "unclear",
]


class ProposedClaim(BaseModel):
    id: str = Field(min_length=1)
    nutrient_name: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=1200)
    proposed_effect_slug: str = Field(min_length=1, max_length=120)
    proposed_effect_label: str = Field(min_length=1, max_length=200)
    compound: str | None = None
    effect: str | None = None
    direction: ClaimDirection = "unclear"
    population: str | None = None
    dose_or_exposure: str | None = None
    outcome: str | None = None
    study_type: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence_type: EvidenceType = "unclear"
    intake_route: IntakeRoute = "unclear"
    exposure_category: ExposureCategory = "unclear"
    natural_concentration_relevance: str | None = None
    supplement_level_relevance: str | None = None
    pharmaceutical_centered: bool = False
    concentration_notes: str | None = None

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator(
        "natural_concentration_relevance",
        "supplement_level_relevance",
        mode="before",
    )
    @classmethod
    def normalize_relevance_notes(cls, value: object) -> object:
        if isinstance(value, bool):
            return "relevant" if value else "not relevant"
        return value

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if normalized in {"beneficial", "harmful", "neutral", "mixed", "unclear"}:
            return normalized
        for direction in ("beneficial", "harmful", "neutral", "mixed", "unclear"):
            if normalized.startswith(direction):
                return direction
        return "unclear"

    @field_validator("intake_route", mode="before")
    @classmethod
    def normalize_intake_route(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"oral", "topical", "injection", "in_vitro", "unclear"}:
            return normalized
        if normalized in {"inhalation", "inhaled", "aromatherapy", "nasal"}:
            return "unclear"
        return value


class SupportingQuote(BaseModel):
    quote: str = Field(min_length=1)
    section: str | None = None
    reason: str = Field(min_length=1)
    match_status: QuoteMatchStatus
    evidence_role: Literal[
        "direct_result",
        "review_synthesis",
        "methods_or_context",
        "background_or_other",
    ] = "direct_result"


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
    validated_statement: str | None = None
    validator_reasoning: str | None = None
    supporting_quotes: list[SupportingQuote]
    limitations: list[str]
    accepted: bool
