from typing import Literal

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

BioactiveType = Literal["nutrient", "polyphenol"]
ExportEvidenceLevel = Literal["strong", "moderate", "limited"]
ExportExposureCategory = Literal[
    "natural_food_level",
    "supplement_level",
    "mixed",
    "unclear",
]


class ExportClaimReference(BaseModel):
    citation_id: str | None = Field(default=None, min_length=1)
    proposed_claim_id: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_reference(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"proposed_claim_id": value}
        if isinstance(value, dict) and "claim_id" in value and "proposed_claim_id" not in value:
            return {**value, "proposed_claim_id": value["claim_id"]}
        return value


class SynthesizedEffect(BaseModel):
    effect_slug: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    effect_label: str = Field(min_length=1, max_length=200)
    effect_description: str = Field(min_length=1, max_length=1000)
    description: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list)
    source_claims: list[ExportClaimReference] = Field(min_length=1)
    exposure_category: ExportExposureCategory = "unclear"
    dose_or_exposure: list[str] = Field(default_factory=list)
    concentration_notes: str | None = None
    review_notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_missing_description(cls, value: Any) -> Any:
        if isinstance(value, dict):
            fallback = (
                value.get("description")
                or value.get("effect_description")
                or value.get("review_notes")
                or value.get("effect_label")
            )
            updates = {}
            if not value.get("description") and fallback:
                updates["description"] = fallback
            if not value.get("effect_description") and fallback:
                updates["effect_description"] = fallback
            if updates:
                value = {**value, **updates}
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(tag.strip().lower() for tag in value if tag.strip()))
        if any(not tag.replace("_", "").isalnum() for tag in normalized):
            raise ValueError("tags_must_be_snake_case")
        return normalized


class ExcludedExportClaim(BaseModel):
    citation_id: str | None = Field(default=None, min_length=1)
    proposed_claim_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_excluded_claim(cls, value: Any) -> Any:
        if isinstance(value, dict) and "claim_id" in value and "proposed_claim_id" not in value:
            return {**value, "proposed_claim_id": value["claim_id"]}
        return value


class InternalSynthesisResponse(BaseModel):
    suggested_bioactive_type: BioactiveType = "nutrient"
    type_confidence: float = Field(default=0, ge=0, le=1)
    effects: list[SynthesizedEffect] = Field(default_factory=list)
    excluded_claims: list[ExcludedExportClaim] = Field(default_factory=list)


class ClaimFormattingResponse(BaseModel):
    suggested_bioactive_type: BioactiveType = "nutrient"
    type_confidence: float = Field(default=0, ge=0, le=1)
    effect: SynthesizedEffect

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_response(cls, value: Any) -> Any:
        if isinstance(value, dict) and "effect" not in value:
            effects = value.get("effects")
            if isinstance(effects, list) and len(effects) == 1:
                return {**value, "effect": effects[0]}
        return value


class ExportSource(BaseModel):
    title: str = Field(min_length=1)
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    year: int | None = None


class BioactiveHealthEvidenceExportRow(BaseModel):
    id: str = Field(min_length=1)
    bioactive_type: BioactiveType
    bioactive_id: str = Field(min_length=1)
    bioactive_name: str = Field(min_length=1)
    effect_slug: str = Field(min_length=1)
    effect_label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    evidence_level: ExportEvidenceLevel
    tags: list[str]
    sources: list[ExportSource]
    review_status: Literal["generated"] = "generated"
    review_notes: str | None = None
    created_at: str
    updated_at: str


class NewHealthEffectRow(BaseModel):
    id: str = Field(min_length=1)
    effect_slug: str = Field(min_length=1)
    effect_label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str]
    created_at: str
    updated_at: str


class NewBioactiveEntityRow(BaseModel):
    bioactive_type: BioactiveType
    bioactive_id: str = Field(min_length=1)
    bioactive_name: str = Field(min_length=1)
    created_at: str
    resolution_note: str = Field(min_length=1)


class ExposureContextRow(BaseModel):
    id: str = Field(min_length=1)
    evidence_row_id: str = Field(min_length=1)
    bioactive_type: BioactiveType
    bioactive_id: str = Field(min_length=1)
    bioactive_name: str = Field(min_length=1)
    effect_slug: str = Field(min_length=1)
    exposure_category: ExportExposureCategory
    dose_or_exposure: list[str]
    concentration_notes: str | None = None
    source_claims: list[ExportClaimReference]
    created_at: str
