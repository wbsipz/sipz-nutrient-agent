from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re

from sipz_agent.core.artifacts import StudyArtifacts, write_study_artifacts
from sipz_agent.core.extraction import extract_claims
from sipz_agent.core.retrieval import find_candidate_papers
from sipz_agent.core.synthesis import build_effect_rows
from sipz_agent.core.validation import validate_claim
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, StudyDepth


class StudyResult:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id


def slugify(value: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", value.strip().lower()))


def timestamp_for_run_id() -> str:
    return datetime.now(UTC).isoformat().replace(":", "-").replace(".", "-")


def run_study(
    *,
    nutrient_name: str,
    depth: StudyDepth,
    demo: bool,
    out_dir: str | Path,
) -> StudyResult:
    created_at = datetime.now(UTC).isoformat()
    audit_events = [{"ts": created_at, "event": "study_started", "nutrient": nutrient_name}]

    found = find_candidate_papers(nutrient_name=nutrient_name, depth=depth, demo=demo)
    claims = extract_claims(
        nutrient_name=found.normalized_nutrient_name,
        citations=found.citations,
    )

    validated = []
    for claim in claims:
        citation = next((item for item in found.citations if item.id == claim.citation_id), None)
        if citation is None:
            continue
        validated.append(
            validate_claim(
                claim=claim,
                citation=citation,
                body_text_without_abstract=citation.body_text or "",
            )
        )

    accepted_claims = [claim for claim in validated if claim.accepted]
    rejected_claims = [claim for claim in validated if not claim.accepted]
    effect_rows = build_effect_rows(
        nutrient_name=found.normalized_nutrient_name,
        nutrient_id=found.nutrient_id or "00000000-0000-4000-8000-000000000000",
        accepted_claims=accepted_claims,
        citations=found.citations,
    )

    completed_at = datetime.now(UTC).isoformat()
    run_id = f"{timestamp_for_run_id()}_{slugify(found.normalized_nutrient_name)}"
    packet = Packet(
        run_id=run_id,
        input=PacketInput(nutrient_name=found.normalized_nutrient_name, depth=depth, demo=demo),
        status="completed",
        created_at=created_at,
        completed_at=completed_at,
        counts=PacketCounts(
            candidate_citations=len(found.citations),
            proposed_claims=len(claims),
            validated_claims=len(accepted_claims),
            rejected_claims=len(rejected_claims),
            effect_rows=len(effect_rows),
        ),
    )

    audit_events.append({"ts": completed_at, "event": "study_completed", "counts": packet.counts.model_dump()})
    summary_markdown = "\n".join(
        [
            f"# {found.normalized_nutrient_name} Research Summary",
            "",
            f"Accepted effects: {len(effect_rows)}",
            f"Rejected claims: {len(rejected_claims)}",
            "",
            "This is research support for data curation, not medical advice.",
            "",
        ]
    )

    run_dir = write_study_artifacts(
        out_dir=out_dir,
        run_id=run_id,
        artifacts=StudyArtifacts(
            packet=packet,
            sources=found.citations,
            effect_rows=effect_rows,
            validated_claims=accepted_claims,
            rejected_claims=rejected_claims,
            summary_markdown=summary_markdown,
            audit_events=audit_events,
        ),
    )
    return StudyResult(run_dir=run_dir, run_id=run_id)
