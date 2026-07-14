from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class CandidateCitation(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    doi: str | None = None
    pmid: str | None = None
    year: int | None = None
    source: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    selection_reason: str | None = None
    abstract: str | None = None
    page_summary: str | None = None
    body_text: str | None = None


class SourceScreeningDecision(BaseModel):
    accepted: bool
    human_health_relevance: bool
    mentions_nutrient_or_bioactive: bool
    conclusiveness: Literal["conclusive", "needs_more_research", "unclear", "not_applicable"]
    relevance_class: Literal[
        "direct_human", "focused_review", "indirect_background", "reject"
    ] | None = None
    intervention_specificity: Literal[
        "isolated", "separable", "mixed_confounded", "mention_only", "not_applicable"
    ] | None = None
    publication_type: Literal[
        "primary_human_study",
        "systematic_review_meta_analysis",
        "focused_narrative_review",
        "case_report",
        "news_editorial",
        "broad_review",
        "preclinical",
        "other",
    ] | None = None
    entity_match: Literal[
        "exact_or_alias",
        "broader_class",
        "different_species",
        "different_entity",
        "unclear",
    ] = "unclear"
    rationale: str = Field(min_length=1)


SourceRejectionCode = Literal[
    "wrong_entity",
    "not_human",
    "not_oral",
    "preclinical_only",
    "not_health_effect",
    "background_review_only",
    "food_processing_only",
    "insufficient_metadata",
    "screening_error",
    "inconsistent_screening",
    "unclear",
]


class RejectedCitation(BaseModel):
    citation: CandidateCitation
    screening: SourceScreeningDecision
    rejection_code: SourceRejectionCode = "unclear"
    screening_confidence: float = Field(default=0.5, ge=0, le=1)
    matched_alias: str | None = None
    failed_requirement: str | None = None


class RetrievalQueryPlan(BaseModel):
    canonical_name: str = Field(min_length=1)
    is_niche: bool
    specific_synonyms: list[str] = Field(default_factory=list)
    source_terms: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
