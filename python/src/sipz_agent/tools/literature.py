from __future__ import annotations

from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, TypeAdapter

from sipz_agent.core.full_text import is_paywall_text, retrieve_full_text_for_source
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.models import create_llm_provider
from sipz_agent.core.retrieval import (
    citation_key,
    deduplicate_citations,
    enrich_missing_abstracts_from_metadata,
    find_live_candidate_page,
    normalize_doi,
)
from sipz_agent.core.screening import screen_sources
from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import (
    CandidateCitation,
    RejectedCitation,
    SourceScreeningDecision,
)
from sipz_agent.schemas.raw_texts import FullTextRetrievalAttempt, RawTextRecord
from sipz_agent.tools.progress import emit_progress


StopReason = Literal[
    "target_reached",
    "max_pages",
    "no_new_unique_candidates",
    "retrieval_error",
]


def _worker_default(name: str, fallback: int) -> int:
    try:
        return max(1, min(10, int(os.getenv(name, str(fallback)))))
    except ValueError:
        return fallback


class RetrieveCandidatesInput(BaseModel):
    substance: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    depth: StudyDepth = "standard"
    target_count: int = Field(default=25, ge=1, le=200)
    max_pages: int = Field(default=5, ge=1, le=20)
    page_size: int = Field(default=10, ge=1, le=50)
    output_path: Path | None = None


class RetrievalSourceCounts(BaseModel):
    raw: dict[str, int] = Field(default_factory=dict)
    unique: dict[str, int] = Field(default_factory=dict)


class RetrieveCandidatesOutput(BaseModel):
    substance: str
    aliases: list[str]
    queries: list[str]
    candidates: list[CandidateCitation]
    raw_candidate_count: int
    unique_candidate_count: int
    pages_attempted: int
    source_counts: RetrievalSourceCounts
    failures: list[str]
    stop_reason: StopReason
    output_path: str | None = None


class EnrichCandidateInput(BaseModel):
    id: str | None = None
    title: str = Field(min_length=1)
    doi: str | None = None
    pmid: str | None = None
    url: HttpUrl | None = None
    source: str = "manual"
    retrieval_query: str = "metadata enrichment"
    abstract: str | None = None


class EnrichCandidateOutput(BaseModel):
    candidate: CandidateCitation
    fields_added: list[str]
    enrichment_notes: str | None = None


class ScreenCandidatesInput(BaseModel):
    substance: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    candidates: list[CandidateCitation] = Field(default_factory=list)
    candidates_path: Path | None = None
    provider: str | None = None
    model: str | None = None
    output_dir: Path | None = None
    max_candidates: int | None = Field(default=None, ge=1, le=200)
    resume: bool = True
    max_workers: int = Field(
        default_factory=lambda: _worker_default("RESEARCH_MAX_LLM_WORKERS", 5), ge=1, le=10
    )


class ScreeningRecord(BaseModel):
    source_id: str
    title: str
    status: Literal["retained", "rejected"]
    relevance_class: str | None = None
    intervention_specificity: str | None = None
    publication_type: str | None = None
    decision: SourceScreeningDecision
    rejection_code: str | None = None


class ScreeningCounts(BaseModel):
    input: int
    classified: int
    retained: int
    rejected: int
    screening_errors: int


class ScreenCandidatesOutput(BaseModel):
    substance: str
    provider: str
    model: str
    counts: ScreeningCounts
    retained: list[CandidateCitation]
    rejected: list[RejectedCitation]
    records: list[ScreeningRecord]
    output_dir: str | None = None


class RetrieveFullTextInput(BaseModel):
    id: str | None = None
    title: str = Field(min_length=1)
    doi: str | None = None
    pmid: str | None = None
    url: HttpUrl | None = None
    source: str = "manual"
    retrieval_query: str = "full-text retrieval"
    abstract: str | None = None
    body_text: str | None = None
    output_dir: Path | None = None


class RetrieveFullTextOutput(BaseModel):
    record: RawTextRecord
    attempts: list[FullTextRetrievalAttempt]
    text_path: str | None = None
    body_text_preview: str | None = None


class RetrieveFullTextBatchInput(BaseModel):
    retained_sources_path: Path
    output_dir: Path
    resume: bool = True
    retry_failed: bool = False
    max_workers: int = Field(
        default_factory=lambda: _worker_default("RESEARCH_MAX_RETRIEVAL_WORKERS", 10), ge=1, le=10
    )


class RetrieveFullTextBatchOutput(BaseModel):
    input_count: int
    retrieved_count: int
    unavailable_count: int
    skipped_existing_count: int
    manifest_path: str
    attempts_path: str


class InspectRetrievalRunInput(BaseModel):
    run_path: Path


class InspectRetrievalRunOutput(BaseModel):
    run_path: str
    status: str | None = None
    nutrient_name: str | None = None
    retrieval_queries: list[str] = Field(default_factory=list)
    pages_attempted: int = 0
    raw_candidate_count: int = 0
    unique_candidate_count: int = 0
    retained_source_count: int = 0
    rejected_source_count: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    source_failures: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    full_text_status_counts: dict[str, int] = Field(default_factory=dict)
    missing_artifacts: list[str] = Field(default_factory=list)


def _clean_terms(substance: str, aliases: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            value.strip() for value in [substance, *aliases] if value and value.strip()
        )
    )


def _default_queries(substance: str, aliases: list[str]) -> list[str]:
    terms = _clean_terms(substance, aliases)
    entity = " OR ".join(f'"{term}"' for term in terms)
    return [
        f"({entity}) AND (human OR adults OR participants) AND (oral OR dietary OR consumed OR supplement)",
        f"({entity}) AND (randomized OR placebo OR clinical trial OR crossover)",
        f"({entity}) AND (systematic review OR meta-analysis OR review) AND human",
    ]


def _source_counts(citations: list[CandidateCitation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for citation in citations:
        counts[citation.source] = counts.get(citation.source, 0) + 1
    return counts


def retrieve_candidates(payload: RetrieveCandidatesInput) -> RetrieveCandidatesOutput:
    substance = payload.substance.strip()
    aliases = _clean_terms(substance, payload.aliases)[1:]
    queries = list(dict.fromkeys(query.strip() for query in payload.queries if query.strip()))
    if not queries:
        queries = _default_queries(substance, aliases)

    candidates: list[CandidateCitation] = []
    raw_candidates: list[CandidateCitation] = []
    failures: list[str] = []
    disabled_collectors: set[str] = set()
    disabled_metadata_sources: set[str] = set()
    pages_attempted = 0
    stop_reason: StopReason = "max_pages"

    for collector_mode, page_limit in [
        ("primary", payload.max_pages),
        ("fallback", min(2, payload.max_pages)),
    ]:
        if len(candidates) >= payload.target_count:
            break
        for page_index in range(page_limit):
            emit_progress(
                f"Searching {collector_mode} sources, page {page_index + 1}/{page_limit}...",
                stage="retrieval", mode=collector_mode, current=page_index + 1, total=page_limit,
            )
            page = find_live_candidate_page(
                substance,
                payload.depth,
                page_index=page_index,
                page_size=payload.page_size,
                queries=queries,
                disabled_collectors=disabled_collectors,
                disabled_metadata_sources=disabled_metadata_sources,
                collector_mode=collector_mode,
            )
            pages_attempted += 1
            raw_candidates.extend(page.citations)
            failures.extend(page.source_failures)
            previous_keys = {citation_key(item) for item in candidates}
            candidates = deduplicate_citations(candidates + page.citations)
            new_count = sum(citation_key(item) not in previous_keys for item in candidates)
            emit_progress(
                f"Retrieved page {pages_attempted}: {len(candidates)} unique candidates ({new_count} new).",
                stage="retrieval", current=pages_attempted, unique_candidates=len(candidates), new_candidates=new_count,
            )

            if len(candidates) >= payload.target_count:
                candidates = candidates[: payload.target_count]
                stop_reason = "target_reached"
                break
            if new_count == 0:
                stop_reason = "no_new_unique_candidates"
                break

    if not candidates and failures:
        stop_reason = "retrieval_error"

    output_path: str | None = None
    output = RetrieveCandidatesOutput(
        substance=substance,
        aliases=aliases,
        queries=queries,
        candidates=candidates,
        raw_candidate_count=len(raw_candidates),
        unique_candidate_count=len(candidates),
        pages_attempted=pages_attempted,
        source_counts=RetrievalSourceCounts(
            raw=_source_counts(raw_candidates),
            unique=_source_counts(candidates),
        ),
        failures=failures,
        stop_reason=stop_reason,
    )
    if payload.output_path:
        path = payload.output_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        output_path = str(path)
        output = output.model_copy(update={"output_path": output_path})
    return output


def screen_candidates(payload: ScreenCandidatesInput) -> ScreenCandidatesOutput:
    candidates = payload.candidates
    if payload.candidates_path:
        raw_candidates = _read_json(payload.candidates_path)
        if isinstance(raw_candidates, dict):
            raw_candidates = raw_candidates.get("candidates", [])
        candidates = TypeAdapter(list[CandidateCitation]).validate_python(raw_candidates)
    if not candidates:
        raise ValueError("screen_candidates_requires_candidates_or_candidates_path")
    if payload.max_candidates is not None:
        candidates = candidates[: payload.max_candidates]
    ids = [candidate.id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("screen_candidates_requires_unique_source_ids")
    model_config = resolve_model_config(
        provider=(
            payload.provider
            or os.getenv("WORKER_MODEL_PROVIDER")
            or os.getenv("RESEARCH_MODEL_PROVIDER")
        ),
        model=payload.model or os.getenv("WORKER_MODEL_ID") or os.getenv("RESEARCH_MODEL_ID"),
    )
    if model_config.provider == "heuristic":
        raise RuntimeError("screen_candidates_requires_llm_provider")
    provider = create_llm_provider(model_config)
    output_dir = payload.output_dir.expanduser().resolve() if payload.output_dir else None
    retained: list[CandidateCitation] = []
    rejected: list[RejectedCitation] = []
    records: list[ScreeningRecord] = []
    if payload.resume and output_dir:
        retained_path = output_dir / "retained_sources.json"
        rejected_path = output_dir / "rejected_sources.json"
        records_path = output_dir / "screening_decisions.json"
        if retained_path.exists():
            retained = TypeAdapter(list[CandidateCitation]).validate_json(retained_path.read_bytes())
        if rejected_path.exists():
            rejected = TypeAdapter(list[RejectedCitation]).validate_json(rejected_path.read_bytes())
        if records_path.exists():
            records = TypeAdapter(list[ScreeningRecord]).validate_json(records_path.read_bytes())
    # Persisted screening decisions are append-only during a resumed run. A caller may
    # request a smaller prefix before later expanding it; pruning here would erase prior
    # decisions and allow the same paper to receive a different LLM verdict on resume.
    requested_ids = {candidate.id for candidate in candidates}
    retained_by_id = {item.id: item for item in retained}
    rejected_by_id = {item.citation.id: item for item in rejected}
    records_by_id = {item.source_id: item for item in records}
    completed_ids = requested_ids & records_by_id.keys()

    def persisted_values(mapping: dict[str, Any]) -> list[Any]:
        requested = [mapping[source_id] for source_id in ids if source_id in mapping]
        prior = [value for source_id, value in mapping.items() if source_id not in requested_ids]
        return requested + prior

    def persist() -> None:
        if not output_dir:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "retained_sources.json").write_bytes(
            TypeAdapter(list[CandidateCitation]).dump_json(
                persisted_values(retained_by_id), indent=2
            )
        )
        (output_dir / "rejected_sources.json").write_bytes(
            TypeAdapter(list[RejectedCitation]).dump_json(
                persisted_values(rejected_by_id), indent=2
            )
        )
        (output_dir / "screening_decisions.json").write_bytes(
            TypeAdapter(list[ScreeningRecord]).dump_json(
                persisted_values(records_by_id), indent=2
            )
        )
        partial = ScreeningCounts(
            input=len(records_by_id),
            classified=len(records_by_id),
            retained=len(retained_by_id),
            rejected=len(rejected_by_id),
            screening_errors=sum(
                item.rejection_code == "screening_error"
                for item in rejected_by_id.values()
            ),
        )
        (output_dir / "screening_summary.json").write_text(
            partial.model_dump_json(indent=2), encoding="utf-8"
        )

    pending = [citation for citation in candidates if citation.id not in completed_ids]

    def screen_one(citation: CandidateCitation):
        return citation, screen_sources(
            nutrient_name=payload.substance.strip(),
            citations=[citation],
            provider=provider,
            aliases=payload.aliases,
        )

    with ThreadPoolExecutor(max_workers=min(payload.max_workers, len(pending) or 1)) as executor:
        futures = {executor.submit(screen_one, citation): citation for citation in pending}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            citation, screening = future.result()
            decision = screening.decisions.get(citation.id)
            if decision is None:
                raise RuntimeError(f"screening_decision_missing:{citation.id}")
            rejection = screening.rejected[0] if screening.rejected else None
            if screening.accepted:
                retained_by_id[citation.id] = screening.accepted[0]
                rejected_by_id.pop(citation.id, None)
            else:
                rejected_by_id[citation.id] = screening.rejected[0]
                retained_by_id.pop(citation.id, None)
            records_by_id[citation.id] = ScreeningRecord(
                    source_id=citation.id,
                    title=citation.title,
                    status="retained" if screening.accepted else "rejected",
                    relevance_class=decision.relevance_class,
                    intervention_specificity=decision.intervention_specificity,
                    publication_type=decision.publication_type,
                    decision=decision,
                    rejection_code=rejection.rejection_code if rejection else None,
                )
            persist()
            emit_progress(
                f"Screened {completed_index}/{len(pending)}: {citation.title}",
                stage="paper_screening", current=completed_index, total=len(pending), title=citation.title,
            )

    order = {citation.id: index for index, citation in enumerate(candidates)}
    retained = sorted(
        (item for source_id, item in retained_by_id.items() if source_id in requested_ids),
        key=lambda item: order[item.id],
    )
    rejected = sorted(
        (item for source_id, item in rejected_by_id.items() if source_id in requested_ids),
        key=lambda item: order[item.citation.id],
    )
    records = sorted(
        (item for source_id, item in records_by_id.items() if source_id in requested_ids),
        key=lambda item: order[item.source_id],
    )
    persist()

    classified = len(retained) + len(rejected)
    if classified != len(candidates):
        raise RuntimeError(
            f"screening_count_mismatch:input={len(candidates)}:classified={classified}"
        )
    counts = ScreeningCounts(
        input=len(candidates),
        classified=classified,
        retained=len(retained),
        rejected=len(rejected),
        screening_errors=sum(item.rejection_code == "screening_error" for item in rejected),
    )
    output = ScreenCandidatesOutput(
        substance=payload.substance.strip(),
        provider=model_config.provider,
        model=model_config.model_name,
        counts=counts,
        retained=retained,
        rejected=rejected,
        records=records,
    )
    if output_dir:
        persist()
        output = output.model_copy(update={"output_dir": str(output_dir)})
    return output


def _candidate_from_input(payload: EnrichCandidateInput | RetrieveFullTextInput) -> CandidateCitation:
    doi = normalize_doi(payload.doi)
    identifier = payload.id or (f"doi:{doi}" if doi else None) or (
        f"pmid:{payload.pmid}" if payload.pmid else None
    ) or f"manual:{abs(hash(payload.title))}"
    return CandidateCitation(
        id=identifier,
        title=payload.title.strip(),
        url=payload.url,
        doi=doi,
        pmid=payload.pmid,
        source=payload.source,
        retrieval_query=payload.retrieval_query,
        abstract=payload.abstract,
        body_text=getattr(payload, "body_text", None),
    )


def _normalized_title(value: str) -> str:
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def _title_match_score(expected: str, candidate: str) -> float:
    left = _normalized_title(expected)
    right = _normalized_title(candidate)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _resolve_candidate_by_title(candidate: CandidateCitation) -> CandidateCitation:
    if candidate.doi or candidate.pmid:
        return candidate
    page = find_live_candidate_page(
        candidate.title,
        "light",
        page_index=0,
        page_size=5,
        queries=[candidate.title],
    )
    ranked = sorted(
        page.citations,
        key=lambda item: _title_match_score(candidate.title, item.title),
        reverse=True,
    )
    if not ranked or _title_match_score(candidate.title, ranked[0].title) < 0.82:
        return candidate
    match = ranked[0]
    return candidate.model_copy(
        update={
            "doi": match.doi,
            "pmid": match.pmid,
            "url": candidate.url or match.url,
            "abstract": candidate.abstract or match.abstract,
            "selection_reason": (
                f"Matched title against {match.source} metadata before identifier enrichment."
            ),
        }
    )


def enrich_candidate(payload: EnrichCandidateInput) -> EnrichCandidateOutput:
    original = _candidate_from_input(payload)
    resolved = _resolve_candidate_by_title(original)
    enriched = enrich_missing_abstracts_from_metadata([resolved])[0]
    if enriched.doi and not enriched.url:
        enriched = enriched.model_copy(update={"url": f"https://doi.org/{enriched.doi}"})
    fields = [
        field
        for field in ("abstract", "doi", "pmid", "url")
        if not getattr(original, field) and getattr(enriched, field)
    ]
    return EnrichCandidateOutput(
        candidate=enriched,
        fields_added=fields,
        enrichment_notes=" ".join(
            part for part in [resolved.selection_reason, enriched.selection_reason] if part
        )
        or None,
    )


def retrieve_full_text(payload: RetrieveFullTextInput) -> RetrieveFullTextOutput:
    citation = _candidate_from_input(payload)
    result = retrieve_full_text_for_source(citation)
    text_path: str | None = None
    record = result.record
    if result.body_text and payload.output_dir:
        output_dir = payload.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = citation.id.replace(":", "_").replace("/", "_") + ".txt"
        path = output_dir / filename
        path.write_text(result.body_text, encoding="utf-8")
        text_path = str(path)
        record = record.model_copy(update={"text_path": text_path})
    return RetrieveFullTextOutput(
        record=record,
        attempts=result.attempts,
        text_path=text_path,
        body_text_preview=result.body_text[:1000] if result.body_text else None,
    )


def retrieve_full_text_batch(payload: RetrieveFullTextBatchInput) -> RetrieveFullTextBatchOutput:
    sources = TypeAdapter(list[CandidateCitation]).validate_json(
        payload.retained_sources_path.read_bytes()
    )
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = payload.output_dir / "papers"
    manifest_path = payload.output_dir / "manifest.json"
    attempts_path = payload.output_dir / "attempts.json"
    records: dict[str, RawTextRecord] = {}
    attempts: list[FullTextRetrievalAttempt] = []
    if payload.resume and manifest_path.exists():
        records = {
            item.source_id: item
            for item in TypeAdapter(list[RawTextRecord]).validate_json(manifest_path.read_bytes())
        }
    if payload.resume and attempts_path.exists():
        attempts = TypeAdapter(list[FullTextRetrievalAttempt]).validate_json(
            attempts_path.read_bytes()
        )
    requested_ids = {source.id for source in sources}
    records = {source_id: record for source_id, record in records.items() if source_id in requested_ids}
    attempts = [attempt for attempt in attempts if attempt.source_id in requested_ids]
    skipped = 0
    pending_sources: list[CandidateCitation] = []
    for citation in sources:
        existing = records.get(citation.id)
        if existing and existing.status == "full_text_found" and existing.text_path:
            stored_path = Path(existing.text_path)
            if not stored_path.exists():
                stored_path = papers_dir / stored_path.name
            if stored_path.exists() and is_paywall_text(
                stored_path.read_text(encoding="utf-8", errors="replace")
            ):
                attempt_index = 1 + max(
                    (
                        attempt.attempt_index
                        for attempt in attempts
                        if attempt.source_id == citation.id
                    ),
                    default=0,
                )
                attempts.append(
                    FullTextRetrievalAttempt(
                        source_id=citation.id,
                        attempt_index=attempt_index,
                        method=existing.retrieval_method,
                        url=existing.resolved_url or existing.url,
                        status="paywalled",
                        resolved_url=existing.resolved_url,
                        text_char_count=0,
                        notes=(
                            "Reclassified persisted text because it contains an explicit "
                            "subscription-preview marker."
                        ),
                    )
                )
                existing = existing.model_copy(
                    update={
                        "status": "paywalled",
                        "text_path": None,
                        "text_char_count": 0,
                        "notes": (
                            "Persisted publisher text was a subscription preview, not an "
                            "article body."
                        ),
                    }
                )
                records[citation.id] = existing
            elif stored_path.exists():
                skipped += 1
                continue
        if existing and not payload.retry_failed:
            skipped += 1
            continue
        pending_sources.append(citation)

    def retrieve_one(citation: CandidateCitation):
        return citation, retrieve_full_text(
            RetrieveFullTextInput(
                id=citation.id,
                title=citation.title,
                doi=citation.doi,
                pmid=citation.pmid,
                url=citation.url,
                source=citation.source,
                retrieval_query=citation.retrieval_query,
                abstract=citation.abstract,
                body_text=citation.body_text,
                output_dir=papers_dir,
            )
        )

    with ThreadPoolExecutor(
        max_workers=min(payload.max_workers, len(pending_sources) or 1)
    ) as executor:
        futures = {executor.submit(retrieve_one, citation): citation for citation in pending_sources}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            citation, result = future.result()
            records[citation.id] = result.record
            attempts.extend(result.attempts)
            ordered_records = [records[source.id] for source in sources if source.id in records]
            manifest_path.write_bytes(
                TypeAdapter(list[RawTextRecord]).dump_json(ordered_records, indent=2)
            )
            attempts_path.write_bytes(
                TypeAdapter(list[FullTextRetrievalAttempt]).dump_json(attempts, indent=2)
            )
            emit_progress(
                f"Full text {completed_index}/{len(pending_sources)}: {result.record.status} - {citation.title}",
                stage="full_text_retrieval", current=completed_index, total=len(pending_sources),
                title=citation.title, status=result.record.status,
            )
    values = [records[source.id] for source in sources if source.id in records]
    manifest_path.write_bytes(TypeAdapter(list[RawTextRecord]).dump_json(values, indent=2))
    attempts_path.write_bytes(
        TypeAdapter(list[FullTextRetrievalAttempt]).dump_json(attempts, indent=2)
    )
    return RetrieveFullTextBatchOutput(
        input_count=len(sources),
        retrieved_count=sum(item.status == "full_text_found" for item in values),
        unavailable_count=sum(item.status != "full_text_found" for item in values),
        skipped_existing_count=skipped,
        manifest_path=str(manifest_path.resolve()),
        attempts_path=str(attempts_path.resolve()),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_retrieval_run(payload: InspectRetrievalRunInput) -> InspectRetrievalRunOutput:
    run_path = payload.run_path.expanduser().resolve()
    missing: list[str] = []
    packet_data: dict[str, Any] = {}
    sources_data: list[dict[str, Any]] = []
    rejected_data: list[Any] = []
    raw_texts_data: list[dict[str, Any]] = []

    for filename, target in (
        ("packet.json", "packet"),
        ("sources.json", "sources"),
        ("rejected_sources.json", "rejected"),
        ("raw_texts.json", "raw_texts"),
    ):
        path = run_path / filename
        if not path.exists():
            missing.append(filename)
            continue
        value = _read_json(path)
        if target == "packet":
            packet_data = value
        elif target == "sources":
            sources_data = value
        elif target == "rejected":
            rejected_data = value
        else:
            raw_texts_data = value

    audit_path = run_path / "audit_log.jsonl"
    failures: list[str] = []
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "candidate_source_page_failed":
                failures.append(str(event.get("failure", "unknown retrieval failure")))
    else:
        missing.append("audit_log.jsonl")

    counts = packet_data.get("counts", {})
    packet_input = packet_data.get("input", {})
    status_counts: dict[str, int] = {}
    for record in raw_texts_data:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    source_models = TypeAdapter(list[CandidateCitation]).validate_python(sources_data)
    return InspectRetrievalRunOutput(
        run_path=str(run_path),
        status=packet_data.get("status"),
        nutrient_name=packet_input.get("nutrient_name"),
        retrieval_queries=packet_input.get("retrieval_queries", []),
        pages_attempted=counts.get("retrieval_pages_attempted", 0),
        raw_candidate_count=counts.get("raw_candidates_retrieved", 0),
        unique_candidate_count=counts.get("unique_candidates", len(source_models)),
        retained_source_count=len(source_models),
        rejected_source_count=counts.get("rejected_sources", len(rejected_data)),
        source_counts=_source_counts(source_models),
        source_failures=failures,
        stop_reason=counts.get("retrieval_stop_reason"),
        full_text_status_counts=status_counts,
        missing_artifacts=list(dict.fromkeys(missing)),
    )


TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "retrieve_candidates": RetrieveCandidatesInput,
    "screen_candidates": ScreenCandidatesInput,
    "enrich_candidate": EnrichCandidateInput,
    "retrieve_full_text": RetrieveFullTextInput,
    "retrieve_full_text_batch": RetrieveFullTextBatchInput,
    "inspect_retrieval_run": InspectRetrievalRunInput,
}


def execute_tool(name: str, raw_payload: dict[str, Any]) -> BaseModel:
    if name == "retrieve_candidates":
        return retrieve_candidates(RetrieveCandidatesInput.model_validate(raw_payload))
    if name == "screen_candidates":
        return screen_candidates(ScreenCandidatesInput.model_validate(raw_payload))
    if name == "enrich_candidate":
        return enrich_candidate(EnrichCandidateInput.model_validate(raw_payload))
    if name == "retrieve_full_text":
        return retrieve_full_text(RetrieveFullTextInput.model_validate(raw_payload))
    if name == "retrieve_full_text_batch":
        return retrieve_full_text_batch(RetrieveFullTextBatchInput.model_validate(raw_payload))
    if name == "inspect_retrieval_run":
        return inspect_retrieval_run(InspectRetrievalRunInput.model_validate(raw_payload))
    raise ValueError(f"unknown_literature_tool:{name}")


def tool_response(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    try:
        result = execute_tool(name, payload)
        return {
            "ok": True,
            "tool": name,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "result": result.model_dump(mode="json"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": name,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
