from pathlib import Path
import threading
import time
from types import SimpleNamespace

from pydantic import TypeAdapter

from sipz_agent.core.full_text import FullTextResult
from sipz_agent.core.retrieval import CandidatePageOutput
from sipz_agent.core.screening import SourceScreeningOutput
from sipz_agent.schemas.citations import CandidateCitation, SourceScreeningDecision
from sipz_agent.schemas.raw_texts import FullTextRetrievalAttempt, RawTextRecord
from sipz_agent.tools import literature


def citation(identifier: str, *, source: str = "pubmed") -> CandidateCitation:
    return CandidateCitation(
        id=identifier,
        title=f"Paper {identifier}",
        doi=identifier.removeprefix("doi:") if identifier.startswith("doi:") else None,
        source=source,
        retrieval_query="test query",
    )


def test_retrieve_candidates_paginates_and_deduplicates(monkeypatch) -> None:
    pages = [
        CandidatePageOutput(citations=[citation("doi:10.1/a"), citation("doi:10.1/b")], raw_candidate_count=2),
        CandidatePageOutput(citations=[citation("doi:10.1/b"), citation("doi:10.1/c")], raw_candidate_count=2),
    ]

    monkeypatch.setattr(literature, "find_live_candidate_page", lambda *args, **kwargs: pages[kwargs["page_index"]])
    result = literature.retrieve_candidates(
        literature.RetrieveCandidatesInput(substance="Test", target_count=3, max_pages=2)
    )

    assert result.stop_reason == "target_reached"
    assert result.pages_attempted == 2
    assert result.raw_candidate_count == 4
    assert result.unique_candidate_count == 3


def test_retrieve_candidates_uses_fallback_only_below_target(monkeypatch) -> None:
    modes: list[str] = []

    def page(*args, **kwargs):
        mode = kwargs["collector_mode"]
        modes.append(mode)
        item = citation("doi:10.1/primary" if mode == "primary" else "doi:10.1/fallback")
        return CandidatePageOutput(citations=[item], raw_candidate_count=1)

    monkeypatch.setattr(literature, "find_live_candidate_page", page)
    result = literature.retrieve_candidates(
        literature.RetrieveCandidatesInput(substance="Test", target_count=2, max_pages=1)
    )

    assert modes == ["primary", "fallback"]
    assert result.stop_reason == "target_reached"
    assert {item.doi for item in result.candidates} == {"10.1/primary", "10.1/fallback"}


def test_enrich_candidate_reports_added_fields(monkeypatch) -> None:
    def enrich(items):
        return [items[0].model_copy(update={"abstract": "An abstract", "pmid": "123"})]

    monkeypatch.setattr(literature, "enrich_missing_abstracts_from_metadata", enrich)
    result = literature.enrich_candidate(
        literature.EnrichCandidateInput(title="A paper", doi="10.1/a")
    )

    assert result.fields_added == ["abstract", "pmid", "url"]
    assert result.candidate.abstract == "An abstract"


def test_enrich_candidate_can_resolve_identifier_from_title(monkeypatch) -> None:
    match = citation("doi:10.1/a", source="crossref")
    match = match.model_copy(update={"title": "A precise paper title"})
    monkeypatch.setattr(
        literature,
        "find_live_candidate_page",
        lambda *args, **kwargs: CandidatePageOutput(citations=[match], raw_candidate_count=1),
    )
    monkeypatch.setattr(
        literature,
        "enrich_missing_abstracts_from_metadata",
        lambda items: items,
    )

    result = literature.enrich_candidate(
        literature.EnrichCandidateInput(title="A precise paper title")
    )

    assert result.candidate.doi == "10.1/a"
    assert result.candidate.url == "https://doi.org/10.1/a"
    assert "doi" in result.fields_added


def test_retrieve_full_text_writes_text(monkeypatch, tmp_path: Path) -> None:
    record = RawTextRecord(
        source_id="doi:10.1/a",
        title="A paper",
        doi="10.1/a",
        status="full_text_found",
        retrieval_method="publisher_page",
        text_char_count=9,
    )
    attempt = FullTextRetrievalAttempt(
        source_id="doi:10.1/a",
        attempt_index=1,
        method="publisher_page",
        status="full_text_found",
        text_char_count=9,
    )
    monkeypatch.setattr(
        literature,
        "retrieve_full_text_for_source",
        lambda _citation: FullTextResult(record=record, body_text="Body text", attempts=[attempt]),
    )

    result = literature.retrieve_full_text(
        literature.RetrieveFullTextInput(title="A paper", doi="10.1/a", output_dir=tmp_path)
    )

    assert result.text_path
    assert Path(result.text_path).read_text(encoding="utf-8") == "Body text"
    assert result.record.text_path == result.text_path


def test_retrieve_full_text_batch_runs_sources_in_parallel(monkeypatch, tmp_path: Path) -> None:
    sources = [citation(f"doi:10.1/{index}") for index in range(4)]
    sources_path = tmp_path / "retained.json"
    sources_path.write_bytes(TypeAdapter(list[CandidateCitation]).dump_json(sources))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def retrieve(item: CandidateCitation) -> FullTextResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        record = RawTextRecord(
            source_id=item.id,
            title=item.title,
            doi=item.doi,
            status="full_text_found",
            retrieval_method="publisher_page",
            text_char_count=9,
        )
        attempt = FullTextRetrievalAttempt(
            source_id=item.id,
            attempt_index=1,
            method="publisher_page",
            status="full_text_found",
            text_char_count=9,
        )
        return FullTextResult(record=record, body_text="Body text", attempts=[attempt])

    monkeypatch.setattr(literature, "retrieve_full_text_for_source", retrieve)
    result = literature.retrieve_full_text_batch(
        literature.RetrieveFullTextBatchInput(
            retained_sources_path=sources_path,
            output_dir=tmp_path / "full_text",
            max_workers=4,
        )
    )

    assert result.retrieved_count == 4
    assert max_active > 1


def test_retrieve_full_text_batch_reclassifies_persisted_subscription_preview(
    tmp_path: Path,
) -> None:
    source = citation("doi:10.1/preview")
    sources_path = tmp_path / "retained.json"
    sources_path.write_bytes(TypeAdapter(list[CandidateCitation]).dump_json([source]))
    output_dir = tmp_path / "full_text"
    papers_dir = output_dir / "papers"
    papers_dir.mkdir(parents=True)
    preview_path = papers_dir / "preview.txt"
    preview_path.write_text(
        "Abstract methods and results. This is a preview of subscription content. "
        + ("Reference methods and discussion. " * 100),
        encoding="utf-8",
    )
    record = RawTextRecord(
        source_id=source.id,
        title=source.title,
        doi=source.doi,
        status="full_text_found",
        retrieval_method="publisher_page",
        resolved_url="https://example.org/preview",
        text_path=str(preview_path),
        text_char_count=preview_path.stat().st_size,
    )
    (output_dir / "manifest.json").write_bytes(
        TypeAdapter(list[RawTextRecord]).dump_json([record], indent=2)
    )
    (output_dir / "attempts.json").write_bytes(
        TypeAdapter(list[FullTextRetrievalAttempt]).dump_json([], indent=2)
    )

    result = literature.retrieve_full_text_batch(
        literature.RetrieveFullTextBatchInput(
            retained_sources_path=sources_path,
            output_dir=output_dir,
            resume=True,
            retry_failed=False,
        )
    )

    persisted = TypeAdapter(list[RawTextRecord]).validate_json(
        (output_dir / "manifest.json").read_bytes()
    )
    attempts = TypeAdapter(list[FullTextRetrievalAttempt]).validate_json(
        (output_dir / "attempts.json").read_bytes()
    )
    assert result.retrieved_count == 0
    assert result.unavailable_count == 1
    assert persisted[0].status == "paywalled"
    assert persisted[0].text_path is None
    assert attempts[-1].status == "paywalled"


def test_screen_candidates_runs_llm_calls_in_parallel(monkeypatch, tmp_path: Path) -> None:
    sources = [
        citation(f"doi:10.2/{index}").model_copy(update={"abstract": "Human oral study."})
        for index in range(4)
    ]
    sources_path = tmp_path / "candidates.json"
    sources_path.write_bytes(TypeAdapter(list[CandidateCitation]).dump_json(sources))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def screen(*, citations, **kwargs):
        nonlocal active, max_active
        item = citations[0]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        decision = SourceScreeningDecision(
            accepted=True,
            human_health_relevance=True,
            mentions_nutrient_or_bioactive=True,
            conclusiveness="conclusive",
            relevance_class="direct_human",
            intervention_specificity="isolated",
            publication_type="primary_human_study",
            rationale="Direct human oral study.",
        )
        return SourceScreeningOutput(
            accepted=[item], rejected=[], decisions={item.id: decision}
        )

    monkeypatch.setattr(literature, "screen_sources", screen)
    monkeypatch.setattr(
        literature,
        "resolve_model_config",
        lambda **kwargs: SimpleNamespace(provider="test", model_name="test-model"),
    )
    monkeypatch.setattr(literature, "create_llm_provider", lambda _: object())

    result = literature.screen_candidates(
        literature.ScreenCandidatesInput(
            substance="Test",
            candidates_path=sources_path,
            output_dir=tmp_path / "screening",
            max_workers=4,
        )
    )

    assert result.counts.retained == 4
    assert max_active > 1


def test_screen_candidates_resume_does_not_prune_decisions_outside_smaller_window(
    monkeypatch, tmp_path: Path
) -> None:
    sources = [citation(f"pmid:{index}") for index in range(1, 4)]
    sources_path = tmp_path / "candidates.json"
    sources_path.write_bytes(TypeAdapter(list[CandidateCitation]).dump_json(sources))
    calls: list[str] = []

    def screen(*, citations, **kwargs):
        item = citations[0]
        calls.append(item.id)
        decision = SourceScreeningDecision(
            accepted=True,
            human_health_relevance=True,
            mentions_nutrient_or_bioactive=True,
            conclusiveness="conclusive",
            relevance_class="direct_human",
            intervention_specificity="isolated",
            publication_type="primary_human_study",
            rationale="Direct human oral study.",
        )
        return SourceScreeningOutput(
            accepted=[item], rejected=[], decisions={item.id: decision}
        )

    monkeypatch.setattr(literature, "screen_sources", screen)
    monkeypatch.setattr(
        literature,
        "resolve_model_config",
        lambda **kwargs: SimpleNamespace(provider="test", model_name="test-model"),
    )
    monkeypatch.setattr(literature, "create_llm_provider", lambda _: object())
    output_dir = tmp_path / "screening"

    first = literature.screen_candidates(
        literature.ScreenCandidatesInput(
            substance="Test",
            candidates_path=sources_path,
            output_dir=output_dir,
            max_candidates=3,
            resume=True,
        )
    )
    resumed = literature.screen_candidates(
        literature.ScreenCandidatesInput(
            substance="Test",
            candidates_path=sources_path,
            output_dir=output_dir,
            max_candidates=2,
            resume=True,
        )
    )

    persisted = TypeAdapter(list[CandidateCitation]).validate_json(
        (output_dir / "retained_sources.json").read_bytes()
    )
    assert first.counts.retained == 3
    assert resumed.counts.input == resumed.counts.retained == 2
    assert calls == ["pmid:1", "pmid:2", "pmid:3"]
    assert [item.id for item in persisted] == ["pmid:1", "pmid:2", "pmid:3"]


def test_inspect_retrieval_run_reads_artifacts(tmp_path: Path) -> None:
    (tmp_path / "packet.json").write_text(
        '{"status":"completed","input":{"nutrient_name":"Test","retrieval_queries":["q"]},'
        '"counts":{"raw_candidates_retrieved":4,"unique_candidates":2,'
        '"retrieval_pages_attempted":1,"retrieval_stop_reason":"target_reached","rejected_sources":1}}',
        encoding="utf-8",
    )
    (tmp_path / "sources.json").write_text(
        '[{"id":"pmid:1","title":"Paper","source":"pubmed","retrieval_query":"q"}]',
        encoding="utf-8",
    )
    (tmp_path / "raw_texts.json").write_text(
        '[{"source_id":"pmid:1","title":"Paper","status":"full_text_found",'
        '"retrieval_method":"pubmed_central","text_char_count":100}]',
        encoding="utf-8",
    )
    (tmp_path / "audit_log.jsonl").write_text(
        '{"event":"candidate_source_page_failed","failure":"Crossref: timeout"}\n',
        encoding="utf-8",
    )

    result = literature.inspect_retrieval_run(
        literature.InspectRetrievalRunInput(run_path=tmp_path)
    )

    assert result.nutrient_name == "Test"
    assert result.source_counts == {"pubmed": 1}
    assert result.full_text_status_counts == {"full_text_found": 1}
    assert result.source_failures == ["Crossref: timeout"]
