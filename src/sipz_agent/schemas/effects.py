from typing import Literal
import json

from pydantic import BaseModel, Field, HttpUrl


class EffectSource(BaseModel):
    url: HttpUrl
    type: str = Field(min_length=1)
    notes: str = ""


class EffectRow(BaseModel):
    id: str = Field(min_length=1)
    nutrient_id: str = Field(min_length=1)
    effect_slug: str = Field(min_length=1, max_length=120)
    effect_label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    score: float = Field(ge=0, le=1)
    evidence_level: Literal["strong", "moderate", "limited"]
    tags: list[str]
    sources: list[EffectSource]
    created_at: str
    updated_at: str
    nutrient_name: str = Field(min_length=1, max_length=200)
    match_status: str = Field(min_length=1)
    match_confidence: float = Field(ge=0, le=1)
    match_notes: str

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "id": self.id,
            "nutrient_id": self.nutrient_id,
            "effect_slug": self.effect_slug,
            "effect_label": self.effect_label,
            "description": self.description,
            "score": self.score,
            "evidence_level": self.evidence_level,
            "tags": json.dumps(self.tags),
            "sources": json.dumps([source.model_dump(mode="json") for source in self.sources]),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nutrient_name": self.nutrient_name,
            "match_status": self.match_status,
            "match_confidence": self.match_confidence,
            "match_notes": self.match_notes,
        }


EFFECT_CSV_COLUMNS = [
    "id",
    "nutrient_id",
    "effect_slug",
    "effect_label",
    "description",
    "score",
    "evidence_level",
    "tags",
    "sources",
    "created_at",
    "updated_at",
    "nutrient_name",
    "match_status",
    "match_confidence",
    "match_notes",
]
