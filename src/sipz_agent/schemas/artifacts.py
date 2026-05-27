from typing import Literal

from pydantic import BaseModel, Field

StudyDepth = Literal["light", "standard", "deep"]


class PacketInput(BaseModel):
    nutrient_name: str = Field(min_length=1)
    depth: StudyDepth
    demo: bool


class PacketCounts(BaseModel):
    candidate_citations: int = Field(ge=0)
    proposed_claims: int = Field(ge=0)
    validated_claims: int = Field(ge=0)
    rejected_claims: int = Field(ge=0)
    effect_rows: int = Field(ge=0)


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
