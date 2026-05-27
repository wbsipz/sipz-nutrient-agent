from pathlib import Path
import json
import re

from pydantic import BaseModel

from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import CandidateCitation


class DemoNutrient(BaseModel):
    id: str
    name: str


class DemoCorpus(BaseModel):
    nutrient: DemoNutrient
    citations: list[CandidateCitation]


class CandidateFinderOutput(BaseModel):
    nutrient_id: str | None
    normalized_nutrient_name: str
    citations: list[CandidateCitation]


def slugify(value: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", value.strip().lower()))


def load_demo_corpus(nutrient_name: str) -> DemoCorpus:
    root = Path(__file__).resolve().parents[3]
    path = root / "examples" / "demo-corpus" / f"{slugify(nutrient_name)}.json"
    return DemoCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def find_candidate_papers(
    nutrient_name: str,
    depth: StudyDepth,
    demo: bool,
) -> CandidateFinderOutput:
    _ = depth
    if not demo:
        raise NotImplementedError("live_retrieval_not_implemented")

    corpus = load_demo_corpus(nutrient_name)
    return CandidateFinderOutput(
        nutrient_id=corpus.nutrient.id,
        normalized_nutrient_name=corpus.nutrient.name,
        citations=corpus.citations,
    )
