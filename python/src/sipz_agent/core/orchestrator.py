from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re

from sipz_agent.core.artifacts import StudyArtifacts, write_study_artifacts
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.extraction import extract_claims
from sipz_agent.core.full_text import retrieve_full_text_for_run
from sipz_agent.core.models import create_llm_provider
from sipz_agent.core.query_planning import fallback_query_plan, plan_retrieval_queries
from sipz_agent.core.retrieval import (
    CandidateFinderOutput,
    citation_key,
    deduplicate_citations,
    find_candidate_papers,
    find_live_candidate_page,
)
from sipz_agent.core.screening import SourceScreeningUnavailable, screen_sources
from sipz_agent.core.synthesis import build_effect_rows
from sipz_agent.core.validation import validate_claim
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel, StudyDepth
from sipz_agent.schemas.citations import CandidateCitation, RetrievalQueryPlan


class StudyResult:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id


LIVE_RETRIEVAL_PAGE_SIZE = 10
FALLBACK_RETRIEVAL_MAX_PAGES = 2
LIVE_RETRIEVAL_MAX_PAGES: dict[StudyDepth, int] = {
    "light": 5,
    "standard": 5,
    "deep": 10,
}
RETAINED_TARGETS: dict[StudyDepth, int] = {
    "light": 10,
    "standard": 25,
    "deep": 40,
}


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
    provider: str | None = None,
    model: str | None = None,
    retrieve_full_text: bool = False,
    full_text_workers: int = 4,
) -> StudyResult:
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    created_at = datetime.now(UTC).isoformat()
    audit_events = [
        {
            "ts": created_at,
            "event": "study_started",
            "nutrient": nutrient_name,
            "model_provider": model_config.provider,
            "model_name": model_config.model_name,
        }
    ]

    rejected_sources = []
    raw_candidate_count = 0
    unique_candidate_count = 0
    retrieval_pages_attempted = 0
    retrieval_stop_reason = "single_pass"
    query_plan: RetrievalQueryPlan | None = None

    if demo:
        found = find_candidate_papers(nutrient_name=nutrient_name, depth=depth, demo=True)
        raw_candidate_count = len(found.citations)
        unique_candidate_count = len(found.citations)
    else:
        normalized_name = nutrient_name.strip()
        query_plan = (
            fallback_query_plan(normalized_name, rationale="Heuristic provider uses the default query.")
            if model_config.provider == "heuristic"
            else plan_retrieval_queries(normalized_name, llm_provider)
        )
        target = RETAINED_TARGETS[depth]
        max_pages = LIVE_RETRIEVAL_MAX_PAGES[depth]
        unique_candidates: list[CandidateCitation] = []
        accepted_keys: set[str] = set()
        disabled_collectors: set[str] = set()
        disabled_metadata_sources: set[str] = set()
        screening_available = model_config.provider != "heuristic"
        retrieval_stop_reason = "max_pages"

        audit_events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "paginated_source_retrieval_started",
                "retained_target": target,
                "page_size_per_source": LIVE_RETRIEVAL_PAGE_SIZE,
                "max_pages": max_pages,
                "query_plan": query_plan.model_dump(mode="json"),
            }
        )

        for collector_mode, page_limit in [
            ("primary", max_pages),
            ("fallback", FALLBACK_RETRIEVAL_MAX_PAGES),
        ]:
            if len(accepted_keys) >= target:
                break
            if collector_mode == "fallback":
                audit_events.append(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "fallback_source_retrieval_started",
                        "sources": ["semantic_scholar", "firecrawl"],
                        "accepted_total": len(accepted_keys),
                        "retained_target": target,
                    }
                )
            for page_index in range(page_limit):
                page = find_live_candidate_page(
                normalized_name,
                depth,
                page_index=page_index,
                page_size=LIVE_RETRIEVAL_PAGE_SIZE,
                queries=query_plan.recommended_queries,
                disabled_collectors=disabled_collectors,
                disabled_metadata_sources=disabled_metadata_sources,
                collector_mode=collector_mode,
                )
                retrieval_pages_attempted += 1
                raw_candidate_count += page.raw_candidate_count
                previous_keys = {citation_key(item) for item in unique_candidates}
                unique_candidates = deduplicate_citations(unique_candidates + page.citations)
                new_candidates = [
                    item for item in unique_candidates if citation_key(item) not in previous_keys
                ]

                for failure in page.source_failures:
                    audit_events.append(
                        {
                            "ts": datetime.now(UTC).isoformat(),
                            "event": "candidate_source_page_failed",
                            "retrieval_tier": collector_mode,
                            "page": page_index + 1,
                            "failure": failure,
                        }
                    )

                accepted_this_page = 0
                rejected_this_page = 0
                if new_candidates and screening_available:
                    try:
                        screening = screen_sources(
                            nutrient_name=normalized_name,
                            citations=new_candidates,
                            provider=llm_provider,
                        )
                        accepted_keys.update(citation_key(item) for item in screening.accepted)
                        rejected_sources.extend(screening.rejected)
                        accepted_this_page = len(screening.accepted)
                        rejected_this_page = len(screening.rejected)
                    except SourceScreeningUnavailable as error:
                        screening_available = False
                        accepted_keys.update(citation_key(item) for item in new_candidates)
                        accepted_this_page = len(new_candidates)
                        audit_events.append(
                            {
                                "ts": datetime.now(UTC).isoformat(),
                                "event": "source_screening_skipped",
                                "reason": str(error),
                                "candidate_citations_retained": len(new_candidates),
                            }
                        )
                elif new_candidates:
                    accepted_keys.update(citation_key(item) for item in new_candidates)
                    accepted_this_page = len(new_candidates)

                audit_events.append(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "candidate_page_processed",
                        "retrieval_tier": collector_mode,
                        "page": page_index + 1,
                        "raw_candidates": page.raw_candidate_count,
                        "new_unique_candidates": len(new_candidates),
                        "accepted_this_page": accepted_this_page,
                        "rejected_this_page": rejected_this_page,
                        "accepted_total": len(accepted_keys),
                        "unique_total": len(unique_candidates),
                    }
                )

                if len(accepted_keys) >= target:
                    retrieval_stop_reason = "retained_target_reached"
                    break
                if not new_candidates:
                    retrieval_stop_reason = "no_new_unique_candidates"
                    break

        found = CandidateFinderOutput(
            nutrient_id=None,
            normalized_nutrient_name=normalized_name,
            citations=[
                item for item in unique_candidates if citation_key(item) in accepted_keys
            ],
        )
        unique_candidate_count = len(unique_candidates)

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
        input=PacketInput(
            nutrient_name=found.normalized_nutrient_name,
            depth=depth,
            demo=demo,
            retrieval_queries=query_plan.recommended_queries if query_plan else [],
        ),
        model=PacketModel(provider=model_config.provider, model_name=model_config.model_name),
        status="completed",
        created_at=created_at,
        completed_at=completed_at,
        counts=PacketCounts(
            candidate_citations=unique_candidate_count,
            raw_candidates_retrieved=raw_candidate_count,
            unique_candidates=unique_candidate_count,
            retrieval_pages_attempted=retrieval_pages_attempted,
            retrieval_stop_reason=retrieval_stop_reason,
            screened_sources=len(found.citations),
            rejected_sources=len(rejected_sources),
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
            rejected_sources=rejected_sources,
            query_plan=query_plan,
        ),
    )
    if retrieve_full_text:
        retrieve_full_text_for_run(run_dir, max_workers=full_text_workers)
    return StudyResult(run_dir=run_dir, run_id=run_id)
