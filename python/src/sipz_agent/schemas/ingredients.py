from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sipz_agent.schemas.artifacts import PacketCounts, PacketModel, StudyDepth
from sipz_agent.schemas.claims import (
    ClaimDirection,
    EvidenceType,
    SupportingQuote,
)


IngredientPreparationDecisionType = Literal[
    "research_direct",
    "reuse_group_evidence",
    "skip_low_value",
    "manual_review",
]

IngredientRelationship = Literal[
    "same_ingredient",
    "minimally_processed_form",
    "juice_or_beverage_form",
    "powder_or_concentrate_form",
    "extract_or_supplement_form",
    "sweetened_or_formulated_product",
    "unclear_relationship",
]


class IngredientLookupRow(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = Field(min_length=1)
    health_effect_positive: str = ""
    health_effect_negative: str = ""
    health_effect_positive_embedding: str = ""
    health_effect_negative_embedding: str = ""
    health_effect_positive_tags: str = ""
    health_effect_negative_tags: str = ""
    payload_nutrients_count: str = ""
    matched_nutrients_count: str = ""
    skipped_keys_count: str = ""
    missing_summary_count: str = ""
    missing_amount_count: str = ""
    source: str = ""
    embedding_model: str = ""
    embedded_at: str = ""
    created_at: str = ""
    updated_at: str = ""


class IngredientPreparationDecision(BaseModel):
    canonical_beverage_id: str = ""
    canonical_beverage_name: str = ""
    decision: IngredientPreparationDecisionType
    canonical_search_name: str = Field(default="", max_length=160)
    group_id: str = Field(default="", max_length=160)
    relationship: IngredientRelationship = "unclear_relationship"
    adaptation: str = Field(default="", max_length=500)
    reason: str = Field(min_length=1, max_length=700)
    confidence: float = Field(ge=0, le=1)

    @field_validator("canonical_search_name", "group_id", "adaptation", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("relationship", mode="before")
    @classmethod
    def default_relationship(cls, value: object) -> object:
        return "unclear_relationship" if value is None or value == "" else value

    @field_validator("reason", mode="before")
    @classmethod
    def default_reason(cls, value: object) -> str:
        if value is None or str(value).strip() == "":
            return "No reason provided by classifier."
        return str(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def default_confidence(cls, value: object) -> float:
        if value is None or value == "":
            return 0.5
        return float(value)  # type: ignore[arg-type]

    @field_validator("canonical_search_name", "group_id")
    @classmethod
    def normalize_optional_text(cls, value: str) -> str:
        return " ".join(value.strip().split())


class IngredientPreparationResponse(BaseModel):
    decisions: list[IngredientPreparationDecision]


class IngredientPreparationSummary(BaseModel):
    total_rows: int
    processed_rows: int
    research_direct: int
    reuse_group_evidence: int
    skip_low_value: int
    manual_review: int


IngredientExposureCategory = Literal[
    "whole_food",
    "juice_or_beverage",
    "powder_or_concentrate",
    "extract_or_supplement",
    "mixed_or_unclear",
]

IngredientGroupApplicability = Literal[
    "direct_only",
    "same_ingredient",
    "similar_forms",
    "form_specific",
    "do_not_propagate",
]


class IngredientEntityPlan(BaseModel):
    ingredient_name: str = Field(min_length=1)
    canonical_search_name: str = Field(min_length=1)
    canonical_beverage_id: str | None = None
    canonical_beverage_name: str | None = None
    ingredient_form: str = "mixed_or_unclear"
    relationship: IngredientRelationship = "unclear_relationship"
    aliases: list[str] = Field(default_factory=list)
    included_forms: list[str] = Field(default_factory=list)
    excluded_forms: list[str] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    excluded_query_terms: list[str] = Field(default_factory=list)
    search_warnings: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @field_validator(
        "aliases",
        "included_forms",
        "excluded_forms",
        "query_terms",
        "excluded_query_terms",
        "search_warnings",
        "retrieval_queries",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class IngredientEntityPlanEnrichmentResponse(BaseModel):
    aliases: list[str] = Field(default_factory=list, max_length=12)
    included_forms: list[str] = Field(default_factory=list, max_length=8)
    excluded_forms: list[str] = Field(default_factory=list, max_length=8)
    query_terms: list[str] = Field(default_factory=list, max_length=16)
    excluded_query_terms: list[str] = Field(default_factory=list, max_length=16)
    search_warnings: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", max_length=700)

    @field_validator(
        "aliases",
        "included_forms",
        "excluded_forms",
        "query_terms",
        "excluded_query_terms",
        "search_warnings",
        mode="before",
    )
    @classmethod
    def normalize_enrichment_lists(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class IngredientPacketInput(BaseModel):
    ingredient_name: str = Field(min_length=1)
    canonical_search_name: str = Field(min_length=1)
    ingredient_form: str = "mixed_or_unclear"
    canonical_beverage_id: str | None = None
    canonical_beverage_name: str | None = None
    depth: StudyDepth
    demo: bool
    retrieval_queries: list[str] = Field(default_factory=list)


class IngredientPacket(BaseModel):
    run_id: str = Field(min_length=1)
    input: IngredientPacketInput
    model: PacketModel
    status: Literal["completed", "failed"]
    created_at: str
    completed_at: str
    counts: PacketCounts


class ProposedIngredientClaim(BaseModel):
    id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    ingredient_form: str = Field(default="mixed_or_unclear", min_length=1, max_length=200)
    citation_id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=1200)
    proposed_effect_slug: str = Field(min_length=1, max_length=120)
    proposed_effect_label: str = Field(min_length=1, max_length=200)
    effect: str | None = None
    claim_direction: ClaimDirection = "unclear"
    population: str | None = None
    oral_exposure: str = Field(default="", max_length=500)
    dose_or_serving: str | None = None
    food_matrix: str | None = None
    outcome: str | None = None
    study_type: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence_type: EvidenceType = "unclear"
    exposure_category: IngredientExposureCategory = "mixed_or_unclear"
    concentration_notes: str | None = None
    claim_applies_to_group_members: IngredientGroupApplicability = "direct_only"

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator(
        "effect",
        "population",
        "oral_exposure",
        "dose_or_serving",
        "food_matrix",
        "outcome",
        "study_type",
        "concentration_notes",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return ""
        return str(value)

    @field_validator("claim_direction", mode="before")
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

    @field_validator("exposure_category", mode="before")
    @classmethod
    def normalize_exposure_category(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "juice": "juice_or_beverage",
            "beverage": "juice_or_beverage",
            "natural_food_level": "whole_food",
            "supplement_level": "extract_or_supplement",
            "powder": "powder_or_concentrate",
            "concentrate": "powder_or_concentrate",
            "unclear": "mixed_or_unclear",
        }
        return aliases.get(normalized, normalized)


class IngredientClaimProposalResponse(BaseModel):
    claims: list[ProposedIngredientClaim]
    skipped_reason: str | None = None


class ValidatedIngredientClaim(BaseModel):
    effect_row_id: str = Field(min_length=1)
    proposed_ingredient_claim_id: str = Field(min_length=1)
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
