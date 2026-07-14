from typing import Literal

from pydantic import BaseModel, Field

StudyDepth = Literal["light", "standard", "deep"]


class PacketInput(BaseModel):
    nutrient_name: str = Field(min_length=1)
    depth: StudyDepth
    demo: bool
    retrieval_queries: list[str] = Field(default_factory=list)


class PacketCounts(BaseModel):
    candidate_citations: int = Field(ge=0)
    raw_candidates_retrieved: int = Field(default=0, ge=0)
    unique_candidates: int = Field(default=0, ge=0)
    retrieval_pages_attempted: int = Field(default=0, ge=0)
    retrieval_stop_reason: str | None = None
    screened_sources: int = Field(default=0, ge=0)
    rejected_sources: int = Field(default=0, ge=0)
    proposed_claims: int = Field(ge=0)
    validated_claims: int = Field(ge=0)
    rejected_claims: int = Field(ge=0)
    effect_rows: int = Field(ge=0)
    full_text_found: int = Field(default=0, ge=0)
    abstract_only: int = Field(default=0, ge=0)
    page_summary_only: int = Field(default=0, ge=0)
    blocked: int = Field(default=0, ge=0)
    blocked_by_cloudflare: int = Field(default=0, ge=0)
    paywalled: int = Field(default=0, ge=0)
    pdf_parse_failed: int = Field(default=0, ge=0)
    no_oa_location: int = Field(default=0, ge=0)
    not_available: int = Field(default=0, ge=0)
    retrieval_error: int = Field(default=0, ge=0)


class PacketModel(BaseModel):
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)


class Packet(BaseModel):
    run_id: str = Field(min_length=1)
    input: PacketInput
    model: PacketModel
    status: Literal["completed", "failed"]
    created_at: str
    completed_at: str
    counts: PacketCounts
