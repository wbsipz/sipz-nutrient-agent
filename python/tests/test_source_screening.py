import json

from sipz_agent.core.orchestrator import run_study
from sipz_agent.core.retrieval import CandidatePageOutput
from sipz_agent.core.screening import SourceScreeningUnavailable, screen_source, screen_sources
from sipz_agent.schemas.citations import CandidateCitation


class FakeScreeningProvider:
    def __init__(self, decisions: list[dict] | None = None, fail: bool = False) -> None:
        self.decisions = decisions or []
        self.fail = fail
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("provider unavailable")
        if "retrieval query planner" in prompt:
            return adapter.validate_python(
                {
                    "canonical_name": "magnesium",
                    "is_niche": False,
                    "specific_synonyms": [],
                    "source_terms": [],
                    "recommended_queries": ["magnesium health human"],
                    "rationale": "Common nutrient; no expansion needed.",
                }
            )
        return adapter.validate_python(self.decisions.pop(0))


class PaymentRequiredProvider:
    def complete_json(self, prompt, adapter):
        raise RuntimeError("llm_provider_payment_required")


def citation(
    *,
    id: str = "pmid:1",
    title: str = "Magnesium and blood pressure in adults",
    abstract: str | None = "A human study of magnesium supplementation and blood pressure.",
) -> CandidateCitation:
    pmid = id.removeprefix("pmid:")
    return CandidateCitation(
        id=id,
        title=title,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        pmid=pmid,
        source="pubmed",
        retrieval_query="magnesium health human",
        selection_reason="Selected by PubMed.",
        abstract=abstract,
    )


def accepted_decision() -> dict:
    return {
        "accepted": True,
        "human_health_relevance": True,
        "mentions_nutrient_or_bioactive": True,
        "conclusiveness": "needs_more_research",
        "rationale": "Human magnesium-health abstract; authors call for more research.",
    }


def rejected_decision() -> dict:
    return {
        "accepted": False,
        "human_health_relevance": False,
        "mentions_nutrient_or_bioactive": True,
        "conclusiveness": "not_applicable",
        "rationale": "Animal-only study.",
    }


def test_screen_source_sends_abstract_to_llm_and_records_conclusiveness() -> None:
    provider = FakeScreeningProvider([accepted_decision()])

    decision = screen_source(
        nutrient_name="magnesium",
        citation=citation(),
        provider=provider,
    )

    assert decision.accepted is True
    assert decision.conclusiveness == "needs_more_research"
    assert "A human study of magnesium supplementation" in provider.prompts[0]
    assert "Requested nutrient/bioactive: magnesium" in provider.prompts[0]


def test_screen_source_rejects_missing_abstract_without_llm_call() -> None:
    provider = FakeScreeningProvider([accepted_decision()])

    decision = screen_source(
        nutrient_name="magnesium",
        citation=citation(abstract=None),
        provider=provider,
    )

    assert decision.accepted is False
    assert decision.conclusiveness == "not_applicable"
    assert "no abstract was available" in decision.rationale
    assert provider.prompts == []


def test_screen_source_uses_page_summary_when_abstract_missing() -> None:
    provider = FakeScreeningProvider([accepted_decision()])
    paper = citation(abstract=None)
    paper = paper.model_copy(update={"page_summary": "Visible page text about magnesium in humans."})

    decision = screen_source(
        nutrient_name="magnesium",
        citation=paper,
        provider=provider,
    )

    assert decision.accepted is True
    assert "Screening context type: visible landing-page summary" in provider.prompts[0]
    assert "Visible page text about magnesium in humans" in provider.prompts[0]


def test_screen_sources_rejects_provider_errors() -> None:
    output = screen_sources(
        nutrient_name="magnesium",
        citations=[citation()],
        provider=FakeScreeningProvider(fail=True),
    )

    assert output.accepted == []
    assert len(output.rejected) == 1
    assert output.rejected[0].screening.rationale.startswith("screening_error:")
    assert output.rejected[0].rejection_code == "screening_error"
    assert output.rejected[0].failed_requirement == "screening_execution"


def test_screen_sources_raises_when_provider_payment_required() -> None:
    try:
        screen_sources(
            nutrient_name="magnesium",
            citations=[citation()],
            provider=PaymentRequiredProvider(),
        )
    except SourceScreeningUnavailable as error:
        assert "llm_provider_payment_required" in str(error)
    else:
        raise AssertionError("expected SourceScreeningUnavailable")


def test_screen_sources_rejects_inconsistent_acceptance_flags() -> None:
    decision = accepted_decision()
    decision["human_health_relevance"] = False

    output = screen_sources(
        nutrient_name="magnesium",
        citations=[citation()],
        provider=FakeScreeningProvider([decision]),
    )

    assert output.accepted == []
    assert output.rejected[0].screening.accepted is False
    assert "internally inconsistent" in output.rejected[0].screening.rationale
    assert output.rejected[0].rejection_code == "inconsistent_screening"
    assert output.rejected[0].failed_requirement == "screening_consistency"


def test_screen_sources_rejects_related_species_as_wrong_entity() -> None:
    decision = accepted_decision()
    decision["entity_match"] = "different_species"
    decision["rationale"] = (
        "The requested target is Valeriana officinalis, but this paper studies Valeriana edulis."
    )

    output = screen_sources(
        nutrient_name="Valeriana officinalis",
        citations=[
            citation(
                title="Valeriana edulis and sleep in children",
                abstract="Children orally consumed Valeriana edulis root powder.",
            )
        ],
        provider=FakeScreeningProvider([decision]),
    )

    assert output.accepted == []
    assert output.rejected[0].rejection_code == "wrong_entity"
    assert output.rejected[0].screening.entity_match == "different_species"
    assert output.rejected[0].failed_requirement == "entity_match"


def test_screen_sources_records_normalized_rejection_metadata() -> None:
    output = screen_sources(
        nutrient_name="magnesium",
        citations=[
            citation(
                title="Magnesium response in isolated cells",
                abstract="An in vitro cell line study of magnesium exposure.",
            )
        ],
        provider=FakeScreeningProvider([rejected_decision()]),
        aliases=["Mg"],
    )

    rejected = output.rejected[0]

    assert rejected.rejection_code == "preclinical_only"
    assert rejected.failed_requirement == "human_evidence"
    assert rejected.screening_confidence == 0.9
    assert rejected.matched_alias == "magnesium"


def test_live_study_writes_screened_and_rejected_sources(monkeypatch, tmp_path) -> None:
    first_page = CandidatePageOutput(
        citations=[
            citation(id="pmid:accepted"),
            citation(
                id="pmid:rejected",
                title="Magnesium response in isolated cells",
                abstract="An in vitro cell study of magnesium exposure.",
            ),
        ],
        raw_candidate_count=2,
    )
    provider = FakeScreeningProvider([accepted_decision(), rejected_decision()])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "sipz_agent.core.orchestrator.find_live_candidate_page",
        lambda *_, page_index, **__: (
            first_page if page_index == 0 else CandidatePageOutput(citations=[])
        ),
    )
    monkeypatch.setattr("sipz_agent.core.orchestrator.create_llm_provider", lambda _: provider)

    result = run_study(
        nutrient_name="magnesium",
        depth="light",
        demo=False,
        out_dir=tmp_path,
        provider="deepseek",
    )

    sources = json.loads((result.run_dir / "sources.json").read_text(encoding="utf-8"))
    rejected = json.loads((result.run_dir / "rejected_sources.json").read_text(encoding="utf-8"))
    packet = json.loads((result.run_dir / "packet.json").read_text(encoding="utf-8"))

    assert [source["id"] for source in sources] == ["pmid:accepted"]
    assert rejected[0]["citation"]["id"] == "pmid:rejected"
    assert rejected[0]["screening"]["rationale"] == "Animal-only study."
    assert rejected[0]["rejection_code"] == "preclinical_only"
    assert rejected[0]["screening_confidence"] == 0.9
    assert rejected[0]["matched_alias"] == "magnesium"
    assert rejected[0]["failed_requirement"] == "human_evidence"
    assert packet["counts"]["candidate_citations"] == 2
    assert packet["counts"]["screened_sources"] == 1
    assert packet["counts"]["rejected_sources"] == 1
    assert (result.run_dir / "rejected_sources.md").exists()


def test_live_study_retains_sources_when_screening_provider_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    first_page = CandidatePageOutput(
        citations=[citation(id="pmid:retained")],
        raw_candidate_count=1,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "sipz_agent.core.orchestrator.find_live_candidate_page",
        lambda *_, page_index, **__: (
            first_page if page_index == 0 else CandidatePageOutput(citations=[])
        ),
    )
    monkeypatch.setattr(
        "sipz_agent.core.orchestrator.create_llm_provider",
        lambda _: PaymentRequiredProvider(),
    )

    result = run_study(
        nutrient_name="magnesium",
        depth="light",
        demo=False,
        out_dir=tmp_path,
        provider="deepseek",
    )

    sources = json.loads((result.run_dir / "sources.json").read_text(encoding="utf-8"))
    rejected = json.loads((result.run_dir / "rejected_sources.json").read_text(encoding="utf-8"))
    audit_log = (result.run_dir / "audit_log.jsonl").read_text(encoding="utf-8")

    assert [source["id"] for source in sources] == ["pmid:retained"]
    assert rejected == []
    assert "source_screening_skipped" in audit_log


def test_live_study_pages_until_retained_target_and_keeps_full_final_page(
    monkeypatch,
    tmp_path,
) -> None:
    pages = [
        CandidatePageOutput(
            citations=[citation(id=f"pmid:{page * 10 + index}") for index in range(10)],
            raw_candidate_count=10,
        )
        for page in range(3)
    ]
    requested_pages: list[tuple[str, int]] = []

    def fake_page(*_, page_index, collector_mode, **__):
        requested_pages.append((collector_mode, page_index))
        return pages[page_index]

    provider = FakeScreeningProvider([accepted_decision() for _ in range(30)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("sipz_agent.core.orchestrator.find_live_candidate_page", fake_page)
    monkeypatch.setattr("sipz_agent.core.orchestrator.create_llm_provider", lambda _: provider)

    result = run_study(
        nutrient_name="magnesium",
        depth="standard",
        demo=False,
        out_dir=tmp_path,
        provider="deepseek",
    )

    packet = json.loads((result.run_dir / "packet.json").read_text(encoding="utf-8"))
    sources = json.loads((result.run_dir / "sources.json").read_text(encoding="utf-8"))

    assert requested_pages == [("primary", 0), ("primary", 1), ("primary", 2)]
    assert len(sources) == 30
    assert len(provider.prompts) == 31
    assert packet["counts"]["retrieval_pages_attempted"] == 3
    assert packet["counts"]["retrieval_stop_reason"] == "retained_target_reached"
    assert packet["counts"]["raw_candidates_retrieved"] == 30
    assert packet["counts"]["unique_candidates"] == 30


def test_live_study_stops_when_next_page_has_no_new_unique_candidates(
    monkeypatch,
    tmp_path,
) -> None:
    repeated = CandidatePageOutput(
        citations=[citation(id=f"pmid:{index}") for index in range(10)],
        raw_candidate_count=10,
    )
    requested_pages: list[tuple[str, int]] = []

    def fake_page(*_, page_index, collector_mode, **__):
        requested_pages.append((collector_mode, page_index))
        return repeated

    provider = FakeScreeningProvider([accepted_decision() for _ in range(10)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("sipz_agent.core.orchestrator.find_live_candidate_page", fake_page)
    monkeypatch.setattr("sipz_agent.core.orchestrator.create_llm_provider", lambda _: provider)

    result = run_study(
        nutrient_name="magnesium",
        depth="standard",
        demo=False,
        out_dir=tmp_path,
        provider="deepseek",
    )

    packet = json.loads((result.run_dir / "packet.json").read_text(encoding="utf-8"))

    assert requested_pages == [("primary", 0), ("primary", 1), ("fallback", 0)]
    assert len(provider.prompts) == 11
    assert packet["counts"]["retrieval_stop_reason"] == "no_new_unique_candidates"
    assert packet["counts"]["unique_candidates"] == 10


def test_deep_study_can_attempt_ten_primary_pages_then_fallback(monkeypatch, tmp_path) -> None:
    requested_pages: list[tuple[str, int]] = []

    def fake_page(*_, page_index, collector_mode, **__):
        requested_pages.append((collector_mode, page_index))
        return CandidatePageOutput(
            citations=[citation(id=f"pmid:{page_index}")],
            raw_candidate_count=1,
        )

    provider = FakeScreeningProvider([rejected_decision() for _ in range(10)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("sipz_agent.core.orchestrator.find_live_candidate_page", fake_page)
    monkeypatch.setattr("sipz_agent.core.orchestrator.create_llm_provider", lambda _: provider)

    result = run_study(
        nutrient_name="magnesium",
        depth="deep",
        demo=False,
        out_dir=tmp_path,
        provider="deepseek",
    )

    packet = json.loads((result.run_dir / "packet.json").read_text(encoding="utf-8"))
    audit_events = [
        json.loads(line)
        for line in (result.run_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retrieval_started = next(
        event for event in audit_events if event["event"] == "paginated_source_retrieval_started"
    )

    assert requested_pages == [
        *[("primary", page) for page in range(10)],
        ("fallback", 0),
    ]
    assert retrieval_started["max_pages"] == 10
    assert packet["counts"]["retrieval_pages_attempted"] == 11
    assert packet["counts"]["retrieval_stop_reason"] == "no_new_unique_candidates"
    assert any(event["event"] == "fallback_source_retrieval_started" for event in audit_events)
