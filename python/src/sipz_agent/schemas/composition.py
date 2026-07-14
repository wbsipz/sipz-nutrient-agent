from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


NutrimentSignificance = Literal["significant", "minor", "trace", "unknown_threshold"]
NutrimentClassificationStatus = Literal[
    "classified",
    "skipped_amount_quality_flag",
    "failed_llm_classification",
    "missing_reference_context",
]


class ReferenceIntakeRow(BaseModel):
    canonical_bioactive_name: str = Field(min_length=1)
    reference_amount: str = ""
    reference_unit: str = ""
    reference_type: str = ""
    caution_limit: str = ""
    caution_limit_unit: str = ""
    threshold_notes: str = ""
    source: str = ""
    review_status: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class NormalizedNutrimentProfile(BaseModel):
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    raw_amount: float
    raw_unit: str = ""
    display_amount: float
    display_unit: str = ""
    display_name: str = ""
    reference_amount: str = ""
    reference_unit: str = ""
    reference_type: str = ""
    caution_limit: str = ""
    caution_limit_unit: str = ""
    amount_quality_status: str = "ok"
    amount_quality_flags: list[str] = Field(default_factory=list)

    @field_validator(
        "raw_unit",
        "display_unit",
        "display_name",
        "reference_amount",
        "reference_unit",
        "reference_type",
        "caution_limit",
        "caution_limit_unit",
        "amount_quality_status",
        mode="before",
    )
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value

    @field_validator("amount_quality_flags", mode="before")
    @classmethod
    def normalize_quality_flags(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class IngredientNutrimentProfile(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = Field(min_length=1)
    canonical_category: str = ""
    canonical_slug: str = ""
    beverage_family: str = ""
    serving_basis: str = "100g"
    nutriments: list[NormalizedNutrimentProfile] = Field(default_factory=list)


class LlmNutrimentSignificanceClassification(BaseModel):
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    significance: NutrimentSignificance
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1, max_length=900)
    amount_context: str = Field(min_length=1, max_length=160)


class LlmNutrimentSignificanceResponse(BaseModel):
    classifications: list[LlmNutrimentSignificanceClassification]


class NutrimentSignificanceClassification(BaseModel):
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    raw_amount: float
    raw_unit: str = ""
    display_amount: float
    display_unit: str = ""
    amount_context: str = Field(min_length=1)
    reference_amount: str = ""
    reference_unit: str = ""
    reference_type: str = ""
    caution_limit: str = ""
    caution_limit_unit: str = ""
    significance: NutrimentSignificance
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)
    classification_status: NutrimentClassificationStatus
    used_llm: bool
    amount_quality_flags: list[str] = Field(default_factory=list)


class IngredientNutrimentSignificanceResult(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    serving_basis: str = "100g"
    classifications: list[NutrimentSignificanceClassification] = Field(default_factory=list)


class NutrimentSignificanceFailure(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class NutrimentSignificanceRunSummary(BaseModel):
    total_ingredients: int
    processed_ingredients: int
    failed_ingredients: int
    classified_nutriments: int
    skipped_invalid_nutriments: int
    significant: int
    minor: int
    trace: int
    unknown_threshold: int


class NutrimentEvidenceEffect(BaseModel):
    effect_slug: str = ""
    effect_label: str = ""
    description: str = ""
    score: float | None = None
    evidence_level: str = ""
    tags: list[Any] | str = Field(default_factory=list)
    sources: list[Any] | str = Field(default_factory=list)

    @field_validator("effect_slug", "effect_label", "description", "evidence_level", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class NutrimentEvidenceMatchRecord(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    serving_basis: str = "100g"
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    raw_amount: float | None = None
    raw_unit: str = ""
    display_amount: float | None = None
    display_unit: str = ""
    amount_context: str = Field(min_length=1)
    reference_amount: str = ""
    reference_unit: str = ""
    reference_type: str = ""
    caution_limit: str = ""
    caution_limit_unit: str = ""
    significance: NutrimentSignificance
    significance_confidence: float | None = Field(default=None, ge=0, le=1)
    significance_reasoning: str = ""
    classification_status: str = ""
    matched: bool = False
    match_count: int = 0
    evidence_matches: list[NutrimentEvidenceEffect] = Field(default_factory=list)

    @field_validator(
        "serving_basis",
        "raw_unit",
        "display_unit",
        "reference_amount",
        "reference_unit",
        "reference_type",
        "caution_limit",
        "caution_limit_unit",
        "significance_reasoning",
        "classification_status",
        mode="before",
    )
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class NutrimentEffectSummary(BaseModel):
    effect_slug: str = ""
    effect_label: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=900)
    evidence_level: str = ""
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("effect_slug", "evidence_level", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class SupplementLevelEffectSummary(BaseModel):
    effect_slug: str = ""
    effect_label: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=900)
    evidence_level: str = ""
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("effect_slug", "evidence_level", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class LlmNutrimentHealthSummary(BaseModel):
    canonical_bioactive_name: str = Field(min_length=1)
    amount_context: str = Field(min_length=1, max_length=160)
    food_level_relevance: str = Field(min_length=1, max_length=1200)
    strong_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    medium_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    low_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    supplement_level_relevance: list[SupplementLevelEffectSummary] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class NutrimentHealthSummaryRecord(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    serving_basis: str = "100g"
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    raw_amount: float | None = None
    raw_unit: str = ""
    display_amount: float | None = None
    display_unit: str = ""
    amount_context: str = Field(min_length=1)
    reference_amount: str = ""
    reference_unit: str = ""
    reference_type: str = ""
    caution_limit: str = ""
    caution_limit_unit: str = ""
    significance: NutrimentSignificance
    significance_confidence: float | None = Field(default=None, ge=0, le=1)
    significance_reasoning: str = ""
    evidence_match_count: int = 0
    food_level_relevance: str = Field(min_length=1)
    strong_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    medium_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    low_evidence_effects: list[NutrimentEffectSummary] = Field(default_factory=list)
    supplement_level_relevance: list[SupplementLevelEffectSummary] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    summary_status: Literal["summarized"] = "summarized"
    used_llm: bool = True
    model_provider: str = ""
    model_name: str = ""


class NutrimentHealthSummaryFailure(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    canonical_bioactive_name: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class NutrimentHealthSummaryRunSummary(BaseModel):
    total_evidence_match_rows: int
    eligible_matched_rows: int
    selected_matched_rows: int
    not_processed_due_to_limit: int
    processed_rows: int
    failed_rows: int
    skipped_unmatched_rows: int
    skipped_empty_summaries: int = 0
    llm_attempts: int
    strong_effects: int
    medium_effects: int
    low_effects: int
    supplement_level_effects: int


class IngredientLevelEffectSummary(BaseModel):
    effect_slug: str = ""
    effect_label: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=900)
    evidence_level: str = ""
    score: float | None = Field(default=None, ge=0, le=1)
    supporting_nutrients: list[str] = Field(default_factory=list)

    @field_validator("effect_slug", "evidence_level", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class IngredientDominantNutrient(BaseModel):
    canonical_bioactive_name: str = Field(min_length=1)
    amount_context: str = Field(min_length=1)
    dose_band: str = ""
    reason: str = Field(min_length=1, max_length=700)

    @field_validator("dose_band", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class IngredientIgnoredTraceNutrient(BaseModel):
    canonical_bioactive_name: str = Field(min_length=1)
    amount_context: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


class LlmIngredientHealthSummary(BaseModel):
    strong_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    medium_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    low_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    negative_or_cautionary_effects: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    dominant_nutrients: list[IngredientDominantNutrient] = Field(default_factory=list)
    supplement_level_only_effects: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    ignored_trace_nutrients: list[IngredientIgnoredTraceNutrient] = Field(default_factory=list)
    overall_summary: str = Field(min_length=1, max_length=1600)
    caveats: list[str] = Field(default_factory=list)


class IngredientHealthSummaryRecord(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = Field(min_length=1)
    ingredient_name: str = Field(min_length=1)
    serving_basis: str = "100g"
    summary_type: Literal["composition_based_literature_derived"] = (
        "composition_based_literature_derived"
    )
    strong_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    medium_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    low_evidence: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    negative_or_cautionary_effects: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    dominant_nutrients: list[IngredientDominantNutrient] = Field(default_factory=list)
    supplement_level_only_effects: list[IngredientLevelEffectSummary] = Field(default_factory=list)
    ignored_trace_nutrients: list[IngredientIgnoredTraceNutrient] = Field(default_factory=list)
    overall_summary: str = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)
    input_warnings: list[str] = Field(default_factory=list)
    summary_status: Literal["summarized"] = "summarized"
    used_llm: bool = True
    model_provider: str = ""
    model_name: str = ""


class IngredientHealthSummaryFailure(BaseModel):
    canonical_beverage_id: str = Field(min_length=1)
    canonical_beverage_name: str = ""
    ingredient_name: str = ""
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class IngredientHealthSummaryRunSummary(BaseModel):
    total_input_rows: int
    selected_input_rows: int
    not_processed_due_to_limit: int
    processed_rows: int
    failed_rows: int
    llm_attempts: int
    strong_effects: int
    medium_effects: int
    low_effects: int
    negative_or_cautionary_effects: int
    dominant_nutrients: int
    supplement_level_only_effects: int
    ignored_trace_nutrients: int
