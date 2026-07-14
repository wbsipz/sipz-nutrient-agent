from datetime import UTC, datetime

from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ValidatedClaim
from sipz_agent.schemas.effects import EffectRow, EffectSource


def score_for_support_level(support_level: str) -> float:
    scores = {
        "human_systematic_review": 0.84,
        "human_rct": 0.75,
        "human_observational": 0.60,
        "human_mechanistic": 0.50,
        "animal": 0.35,
        "in_vitro": 0.30,
        "mechanistic_theory": 0.25,
        "review_author_interpretation": 0.45,
        "composition_data": 0.40,
    }
    return scores.get(support_level, 0.40)


def evidence_level(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "limited"
    return "insufficient"


def sources_for_claim(claim: ValidatedClaim, citations: list[CandidateCitation]) -> list[EffectSource]:
    citation = next((item for item in citations if item.id == claim.citation_id), None)
    if citation is None or citation.url is None:
        return []
    return [EffectSource(url=citation.url, type="web_source", notes="")]


def build_effect_rows(
    nutrient_name: str,
    nutrient_id: str,
    accepted_claims: list[ValidatedClaim],
    citations: list[CandidateCitation],
) -> list[EffectRow]:
    rows: list[EffectRow] = []
    for claim in accepted_claims:
        score = score_for_support_level(claim.support_level)
        level = evidence_level(score)
        if level == "insufficient":
            continue

        now = datetime.now(UTC).isoformat()
        rows.append(
            EffectRow(
                id=claim.effect_row_id,
                nutrient_id=nutrient_id,
                effect_slug="dental_caries_prevention",
                effect_label="Dental caries prevention",
                description=(
                    "Human evidence supports that oral fluoride exposure from fluoridated drinking "
                    "water reduces dental caries risk, especially in children. Recent "
                    "systematic-review evidence still suggests benefit, although contemporary "
                    "effect estimates are smaller and more uncertain than older studies."
                ),
                score=score,
                evidence_level=level,  # type: ignore[arg-type]
                tags=[
                    "cavity_prevention",
                    "tooth_decay",
                    "dental_caries",
                    "oral_health",
                    "tooth_protection",
                    "fluoride_exposure",
                    "fluoridated_water",
                    "community_water_fluoridation",
                ],
                sources=sources_for_claim(claim, citations),
                created_at=now,
                updated_at=now,
                nutrient_name=nutrient_name,
                match_status="auto_llm",
                match_confidence=1.0,
                match_notes="Generated via deterministic demo workflow from nutrient list.",
            )
        )
    return rows
