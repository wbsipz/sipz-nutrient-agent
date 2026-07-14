import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import TypeAdapter

from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, SupportingQuote, ValidatedClaim
from sipz_agent.schemas.raw_texts import RawTextRecord
from sipz_agent.tools import workflow
from sipz_agent.tools import literature


def _write(path: Path, adapter: TypeAdapter, values: list[object]) -> None:
    path.write_bytes(adapter.dump_json(values, indent=2))


def _artifacts(tmp_path: Path, *, grounded: bool = True) -> dict[str, Path]:
    source = CandidateCitation(
        id="pmid:1",
        title="Human oral study",
        pmid="1",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        source="pubmed",
        retrieval_query="test",
    )
    proposed = ProposedClaim(
        id="claim-1",
        nutrient_name="Vitamin C",
        citation_id=source.id,
        statement="Oral vitamin C reduced symptom duration.",
        proposed_effect_slug="cold_symptom_duration",
        proposed_effect_label="Cold symptom duration",
        direction="beneficial",
        evidence_type="human_clinical",
        intake_route="oral",
        exposure_category="supplement_level",
    )
    validated = ValidatedClaim(
        effect_row_id="effect-1",
        proposed_claim_id=proposed.id,
        citation_id=source.id,
        verdict="supported",
        support_level="human_rct",
        claim_scope="Adults taking oral vitamin C supplements",
        validated_statement=proposed.statement,
        supporting_quotes=[
            SupportingQuote(
                quote="Symptom duration was reduced.",
                reason="Direct result",
                match_status="exact" if grounded else "not_found",
            )
        ],
        limitations=[],
        accepted=True,
    )
    paths = {
        "sources": tmp_path / "sources.json",
        "proposed": tmp_path / "proposed.json",
        "validated": tmp_path / "validated.json",
        "rejected": tmp_path / "rejected.json",
    }
    _write(paths["sources"], TypeAdapter(list[CandidateCitation]), [source])
    _write(paths["proposed"], TypeAdapter(list[ProposedClaim]), [proposed])
    _write(paths["validated"], TypeAdapter(list[ValidatedClaim]), [validated])
    _write(paths["rejected"], TypeAdapter(list[ValidatedClaim]), [])
    return paths


def test_export_includes_validator_accepted_claims(tmp_path: Path, monkeypatch) -> None:
    paths = _artifacts(tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ORCHESTRATOR_MODEL_ID", "claude-opus-4-8")
    monkeypatch.setenv("ORCHESTRATOR_THINKING", "medium")
    monkeypatch.setenv("WORKER_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("WORKER_MODEL_ID", "gpt-5-mini")

    result = workflow.export_research_report(
        workflow.ExportResearchReportInput(
            substance="Vitamin C",
            proposed_claims_path=paths["proposed"],
            validated_claims_path=paths["validated"],
            rejected_claims_path=paths["rejected"],
            sources_path=paths["sources"],
            output_dir=tmp_path / "report",
        )
    )

    assert result.accepted_count == 1
    assert result.rejected_count == result.held_count == result.failure_count == 0
    assert len((tmp_path / "report" / "claims_report.csv").read_text().splitlines()) == 2
    report = json.loads((tmp_path / "report" / "claims_report.json").read_text())
    assert report["counts"]["validated_claims"] == 1
    assert report["counts"]["retained_sources"] == 1
    assert report["counts"]["papers_reviewed_from_full_text"] == 1
    assert report["validated_claims"][0]["health_effect"] == "Cold symptom duration"
    assert report["model_provenance"]["orchestrator"]["model"] == "claude-opus-4-8"
    assert report["model_provenance"]["worker"] == {
        "provider": "openai",
        "model": "gpt-5-mini",
    }
    manifest = json.loads((tmp_path / "report" / "report_manifest.json").read_text())
    assert manifest["model_provenance"] == report["model_provenance"]
    assert "| Cold symptom duration | Beneficial |" in result.terminal_table_markdown
    assert "https://pubmed.ncbi.nlm.nih.gov/1/" in result.terminal_table_markdown
    assert result.report_link_markdown.startswith("[Open the full claims report](")
    assert result.report_directory_link_markdown.startswith("[Open the report folder](")
    assert result.report_location_markdown == (
        f"**Saved report folder:** `{(tmp_path / 'report').resolve()}`\n\n"
        f"**Main report file:** `{(tmp_path / 'report' / 'claims_report.md').resolve()}`"
    )
    assert result.terminal_response_markdown == (
        f"{result.terminal_summary_markdown}\n\n"
        f"{result.terminal_table_markdown}\n\n"
        f"{result.report_location_markdown}\n\n"
        f"{result.report_link_markdown}\n\n"
        f"{result.report_directory_link_markdown}\n\n"
        f"**JSON:** `{(tmp_path / 'report' / 'claims_report.json').resolve()}`\n\n"
        f"**CSV:** `{(tmp_path / 'report' / 'claims_report.csv').resolve()}`"
    )
    assert "Symptom duration was reduced." in (tmp_path / "report" / "claims_report.md").read_text()
    assert "Sources retained after screening: 1" in result.terminal_summary_markdown
    assert "Papers reviewed from usable full text: 1" in result.terminal_summary_markdown
    assert not (tmp_path / "report" / "effects.csv").exists()


def test_public_report_source_link_prefers_doi_then_pmid_then_url() -> None:
    base = {"id": "source", "title": "Paper", "source": "test", "retrieval_query": "q"}
    assert workflow._public_source_url(
        CandidateCitation(**base, doi="10.1000/example", pmid="123", url="https://example.com")
    ) == "https://doi.org/10.1000/example"
    assert workflow._public_source_url(
        CandidateCitation(**base, pmid="123", url="https://example.com")
    ) == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert workflow._public_source_url(
        CandidateCitation(**base, url="https://example.com")
    ) == "https://example.com/"


def test_export_reports_retrieval_target_shortfall(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    expansion = tmp_path / "retrieval_expansion.json"
    expansion.write_text(
        json.dumps(
            {
                "requested_count": 6,
                "target_met": False,
                "stop_reason": "max_rounds",
                "totals": {"candidates": 20, "retained": 4, "usable_full_texts": 3},
            }
        ),
        encoding="utf-8",
    )

    result = workflow.export_research_report(
        workflow.ExportResearchReportInput(
            substance="Vitamin C",
            proposed_claims_path=paths["proposed"],
            validated_claims_path=paths["validated"],
            rejected_claims_path=paths["rejected"],
            sources_path=paths["sources"],
            output_dir=tmp_path / "report",
            retrieval_expansion_path=expansion,
        )
    )

    assert "Requested usable full texts: 6" in result.terminal_summary_markdown
    assert "Usable full texts retrieved: 3" in result.terminal_summary_markdown
    assert "Retrieval target met: no" in result.terminal_summary_markdown
    report = json.loads((tmp_path / "report" / "claims_report.json").read_text())
    assert report["retrieval_coverage"]["stop_reason"] == "max_rounds"


def test_bridge_tool_registry_contains_complete_workflow() -> None:
    assert set(workflow.TOOL_INPUTS) == {
        "advance_research_pipeline",
        "inspect_research_state",
        "extract_claims",
        "validate_claims",
        "export_research_report",
        "run_research_pipeline",
    }


def test_complete_pipeline_output_preserves_preassembled_terminal_response() -> None:
    exported = workflow.ExportResearchReportOutput(
        accepted_count=1,
        rejected_count=0,
        held_count=0,
        failure_count=0,
        source_count=1,
        report_dir="/tmp/report",
        markdown_path="/tmp/report/claims_report.md",
        json_path="/tmp/report/claims_report.json",
        csv_path="/tmp/report/claims_report.csv",
        manifest_path="/tmp/report/report_manifest.json",
        terminal_summary_markdown="summary",
        terminal_table_markdown="table",
        report_link_markdown="report link",
        report_directory_link_markdown="folder link",
        report_location_markdown="location",
        terminal_response_markdown="exact final response",
    )
    result = workflow.RunResearchPipelineOutput(
        run_id="run",
        run_dir="/tmp/run",
        status="completed",
        stage_counts={},
        terminal_response_markdown=exported.terminal_response_markdown,
    )

    assert result.terminal_response_markdown == "exact final response"


def test_compatibility_pipeline_uses_adaptive_controller_until_closed(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[workflow.AdvanceResearchPipelineInput] = []

    def advance(payload):
        calls.append(payload)
        common = {
            "substance": payload.substance,
            "target_stage": payload.target_stage,
            "run_dir": str(payload.run_dir_override),
            "candidates_count": 6 if len(calls) == 1 else 18,
            "retained_count": 2 if len(calls) == 1 else 8,
            "rejected_count": 4 if len(calls) == 1 else 10,
            "full_text_retrieved_count": 1 if len(calls) == 1 else 6,
            "retained_sources_path": str(tmp_path / "selected.json"),
            "requested_count": payload.requested_count,
            "expansion_round": len(calls),
            "max_expansion_rounds": 5,
            "expansion_state_path": str(tmp_path / "retrieval_expansion.json"),
        }
        if len(calls) == 1:
            return workflow.AdvanceResearchPipelineOutput(
                **common,
                expansion_recommended=True,
                stop_reason="expansion_required",
            )
        return workflow.AdvanceResearchPipelineOutput(
            **common,
            target_met=True,
            stop_reason="target_met",
            proposed_claims_path=str(tmp_path / "proposed.json"),
            validated_claims_path=str(tmp_path / "validated.json"),
            rejected_claims_path=str(tmp_path / "rejected.json"),
            held_claims_path=str(tmp_path / "held.json"),
            body_adequacy_path=str(tmp_path / "adequacy.json"),
            validation_failures_path=str(tmp_path / "failures.json"),
        )

    exported = workflow.ExportResearchReportOutput(
        accepted_count=1,
        rejected_count=0,
        held_count=0,
        failure_count=0,
        source_count=1,
        report_dir=str(tmp_path / "report"),
        markdown_path=str(tmp_path / "report.md"),
        json_path=str(tmp_path / "report.json"),
        csv_path=str(tmp_path / "report.csv"),
        manifest_path=str(tmp_path / "manifest.json"),
        terminal_summary_markdown="summary",
        terminal_table_markdown="table",
        report_link_markdown="report link",
        report_directory_link_markdown="folder link",
        report_location_markdown="location",
        terminal_response_markdown="final response",
    )
    monkeypatch.setattr(workflow, "advance_research_pipeline", advance)
    monkeypatch.setattr(workflow, "export_research_report", lambda payload: exported)

    result = workflow.run_research_pipeline(
        workflow.RunResearchPipelineInput(
            substance="Valerian root",
            output_root=tmp_path / "runs",
            run_id="valerian",
            target_count=6,
        )
    )

    assert len(calls) == 2
    assert calls[0].requested_count == 6
    assert calls[0].expansion_queries == []
    assert calls[1].expansion_queries
    assert result.stage_counts["usable_full_texts"] == 6
    assert result.terminal_response_markdown == "final response"


def test_inspect_research_state_finds_retained_sources(tmp_path: Path) -> None:
    screening = tmp_path / "runs" / "vitamin-c" / "screening"
    screening.mkdir(parents=True)
    retained = screening / "retained_sources.json"
    retained.write_text("[]", encoding="utf-8")

    result = workflow.inspect_research_state(
        workflow.InspectResearchStateInput(
            substance="Vitamin C",
            workspace_root=tmp_path,
        )
    )

    assert result.retained_sources_path == str(retained)
    assert result.suggested_action == "retrieve_full_text"


def test_workspace_root_normalizes_repository_root(tmp_path: Path) -> None:
    (tmp_path / "agent-home").mkdir()
    screening = tmp_path / "workspace" / "runs" / "vitamin-c" / "screening"
    screening.mkdir(parents=True)
    retained = screening / "retained_sources.json"
    retained.write_text("[]", encoding="utf-8")

    result = workflow.inspect_research_state(
        workflow.InspectResearchStateInput(
            substance="Vitamin C",
            workspace_root=tmp_path,
        )
    )

    assert result.retained_sources_path == str(retained)


def test_workflow_provider_uses_research_model_environment(monkeypatch) -> None:
    captured = {}
    monkeypatch.delenv("WORKER_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("WORKER_MODEL_ID", raising=False)

    def resolve(**kwargs):
        captured.update(kwargs)
        return type(
            "Config",
            (),
            {"provider": "deepseek", "model_name": "deepseek-v4-pro"},
        )()

    monkeypatch.setenv("RESEARCH_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("RESEARCH_MODEL_ID", "deepseek-v4-pro")
    monkeypatch.setattr(workflow, "resolve_model_config", resolve)
    monkeypatch.setattr(workflow, "create_llm_provider", lambda _: object())

    _, provider, model = workflow._provider(None, None)

    assert captured == {"provider": "deepseek", "model": "deepseek-v4-pro"}
    assert provider == "deepseek"
    assert model == "deepseek-v4-pro"


def test_workflow_provider_prefers_worker_model_environment(monkeypatch) -> None:
    captured = {}

    def resolve(**kwargs):
        captured.update(kwargs)
        return type("Config", (), {"provider": "openai", "model_name": "gpt-5-mini"})()

    monkeypatch.setenv("RESEARCH_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("RESEARCH_MODEL_ID", "deepseek-v4-pro")
    monkeypatch.setenv("WORKER_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("WORKER_MODEL_ID", "gpt-5-mini")
    monkeypatch.setattr(workflow, "resolve_model_config", resolve)
    monkeypatch.setattr(workflow, "create_llm_provider", lambda _: object())

    _, provider, model = workflow._provider(None, None)

    assert captured == {"provider": "openai", "model": "gpt-5-mini"}
    assert (provider, model) == ("openai", "gpt-5-mini")


def test_validation_retry_policy_is_limited_to_recoverable_execution_errors() -> None:
    assert workflow._retryable_validation_exception(
        RuntimeError("llm_provider_invalid_json")
    )
    assert workflow._retryable_validation_exception(TimeoutError("timed out"))
    assert not workflow._retryable_validation_exception(
        RuntimeError("unsupported human oral claim")
    )
    assert not workflow._retryable_validation_exception(
        RuntimeError("llm_provider_payment_required")
    )


def test_extraction_records_paper_failure_and_continues(monkeypatch, tmp_path: Path) -> None:
    sources = [
        CandidateCitation(id="pmid:1", title="Good", source="pubmed", retrieval_query="q"),
        CandidateCitation(id="pmid:2", title="Bad", source="pubmed", retrieval_query="q"),
    ]
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "one.txt").write_text("Human oral study body text.", encoding="utf-8")
    (papers / "two.txt").write_text("Another human oral study body text.", encoding="utf-8")
    records = [
        RawTextRecord(
            source_id="pmid:1", title="Good", status="full_text_found",
            retrieval_method="publisher_page", text_path="one.txt", text_char_count=27,
        ),
        RawTextRecord(
            source_id="pmid:2", title="Bad", status="full_text_found",
            retrieval_method="publisher_page", text_path="two.txt", text_char_count=35,
        ),
    ]
    sources_path = tmp_path / "sources.json"
    manifest_path = tmp_path / "manifest.json"
    _write(sources_path, TypeAdapter(list[CandidateCitation]), sources)
    _write(manifest_path, TypeAdapter(list[RawTextRecord]), records)

    class Provider:
        def complete_json(self, prompt, adapter):
            if "Title: Bad" in prompt:
                raise RuntimeError("malformed response")
            return adapter.validate_python(
                {
                    "claims": [
                        {
                            "id": "temporary",
                            "nutrient_name": "Vitamin C",
                            "citation_id": "pmid:1",
                            "statement": "Oral vitamin C supported a human outcome.",
                            "proposed_effect_slug": "human_outcome",
                            "proposed_effect_label": "Human outcome",
                            "evidence_type": "human_clinical",
                            "intake_route": "oral",
                            "exposure_category": "supplement_level",
                        }
                    ]
                }
            )

    monkeypatch.setattr(workflow, "_provider", lambda *_: (Provider(), "test", "model"))
    result = workflow.extract_claims(
        workflow.ExtractClaimsInput(
            substance="Vitamin C",
            sources_path=sources_path,
            raw_texts_manifest_path=manifest_path,
            raw_texts_dir=papers,
            output_dir=tmp_path / "claims",
            resume=False,
        )
    )

    assert result.proposed_claim_count == 1
    assert result.failure_count == 1


def test_validation_runs_one_call_per_claim_and_writes_resume_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    source = CandidateCitation(
        id="pmid:validation",
        title="Human oral vitamin C trial",
        abstract="The abstract reports a human oral trial.",
        source="pubmed",
        retrieval_query="q",
    )
    claims = [
        ProposedClaim(
            id=f"claim-{index}",
            nutrient_name="Vitamin C",
            citation_id=source.id,
            statement=f"Oral vitamin C supported outcome {index}.",
            proposed_effect_slug=f"outcome_{index}",
            proposed_effect_label=f"Outcome {index}",
            evidence_type="human_clinical",
            intake_route="oral",
            exposure_category="supplement_level",
        )
        for index in (1, 2)
    ]
    body = (
        f"{source.title} Abstract {source.abstract} "
        "1. Introduction Background material. "
        "Methods Adults consumed oral vitamin C daily in a randomized trial. "
        "Results Symptom duration was reduced in the vitamin C group compared with placebo. "
        "Discussion The result applies to the tested supplement dose. "
        + ("Additional discussion text about the human trial. " * 20)
        + "\nConclusion\nThe abstract conclusion must not be validator evidence."
        + "\nReferences\nExample A, et al."
    )
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "paper.txt").write_text(body, encoding="utf-8")
    record = RawTextRecord(
        source_id=source.id,
        title=source.title,
        status="full_text_found",
        retrieval_method="publisher_page",
        text_path="paper.txt",
        text_char_count=len(body),
    )
    sources_path = tmp_path / "sources.json"
    proposed_path = tmp_path / "proposed.json"
    manifest_path = tmp_path / "manifest.json"
    _write(sources_path, TypeAdapter(list[CandidateCitation]), [source])
    _write(proposed_path, TypeAdapter(list[ProposedClaim]), claims)
    _write(manifest_path, TypeAdapter(list[RawTextRecord]), [record])

    class Provider:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.transient_failures_remaining = 1

        def complete_json(self, prompt, adapter):
            self.prompts.append(prompt)
            if adapter is workflow.BODY_ADEQUACY_ADAPTER:
                return adapter.validate_python(
                    {
                        "status": "adequate",
                        "coverage": {
                            "human_population": True,
                            "oral_exposure": True,
                            "intervention_or_exposure": True,
                            "study_design": True,
                            "comparator_when_applicable": True,
                            "substantive_results": True,
                            "dose_or_duration": True,
                            "limitations_or_uncertainty": True,
                        },
                        "reason_codes": [],
                        "reasoning": "Methods and results are present.",
                        "diagnostic_excerpts": [
                            {
                                "quote": "Methods Adults consumed oral vitamin C daily in a randomized trial.",
                                "section": "Methods",
                                "reason": "Describes the human oral intervention.",
                            },
                            {
                                "quote": "Results Symptom duration was reduced in the vitamin C group compared with placebo.",
                                "section": "Results",
                                "reason": "Reports the substantive result.",
                            },
                        ],
                    }
                )
            if self.transient_failures_remaining:
                self.transient_failures_remaining -= 1
                raise RuntimeError("llm_provider_invalid_json")
            claim_id = "claim-1" if '"claim-1"' in prompt else "claim-2"
            return adapter.validate_python(
                {
                    "decisions": [
                        {
                            "proposed_claim_id": claim_id,
                            "verdict": "supported_with_limitations",
                            "validated_statement": "At the tested dose, oral vitamin C reduced symptom duration.",
                            "support_level": "human_rct",
                            "entity_match": "exact_or_alias",
                            "claim_scope": "Adults receiving the tested oral supplement dose.",
                            "supporting_quotes": [
                                {
                                    "quote": "Symptom duration was reduced in the vitamin C group compared with placebo.",
                                    "section": "Results",
                                    "reason": "Direct result.",
                                }
                            ],
                            "limitations": ["Applies to the tested dose."],
                            "reasoning": "The body directly supports a narrower claim.",
                        }
                    ]
                }
            )

    provider = Provider()
    monkeypatch.setattr(workflow, "_provider", lambda *_: (provider, "test", "model"))
    output_dir = tmp_path / "validation"
    result = workflow.validate_claims(
        workflow.ValidateClaimsInput(
            proposed_claims_path=proposed_path,
            sources_path=sources_path,
            raw_texts_manifest_path=manifest_path,
            raw_texts_dir=papers,
            output_dir=output_dir,
            resume=False,
            max_workers=2,
        )
    )

    assert result.accepted_count == 2
    assert result.completed_claim_count == 2
    assert len(provider.prompts) == 4
    assert all(
        sum(f'"{claim.id}"' in prompt for claim in claims) == 1
        for prompt in provider.prompts
        if '"proposed_claim_id"' in prompt
    )
    contexts = json.loads((output_dir / "validation_contexts.json").read_text())
    assert contexts[source.id]["excluded_sections"] == [
        "title", "abstract", "conclusion", "references"
    ]
    assert "abstract conclusion" not in workflow.sanitize_paper_body(
        body_text=body, citation=source
    ).lower()
    ids = {
        item.effect_row_id
        for item in TypeAdapter(list[ValidatedClaim]).validate_json(
            (output_dir / "validated_claims.json").read_bytes()
        )
    }
    attempts = sorted(
        item["attempts"]
        for item in json.loads((output_dir / "validation_status.json").read_text()).values()
    )
    assert attempts == [1, 2]

    resumed = workflow.validate_claims(
        workflow.ValidateClaimsInput(
            proposed_claims_path=proposed_path,
            sources_path=sources_path,
            raw_texts_manifest_path=manifest_path,
            raw_texts_dir=papers,
            output_dir=output_dir,
            resume=True,
        )
    )
    assert resumed.pending_claim_count == 0
    assert len(provider.prompts) == 4
    assert ids == {
        item.effect_row_id
        for item in TypeAdapter(list[ValidatedClaim]).validate_json(
            (output_dir / "validated_claims.json").read_bytes()
        )
    }


def test_inspect_research_state_reports_completed_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "vitamin-c"
    validation = run_dir / "validation"
    validation.mkdir(parents=True)
    claims_dir = run_dir / "claims"
    claims_dir.mkdir()
    source_paths = _artifacts(tmp_path)
    (claims_dir / "proposed_claims.json").write_bytes(source_paths["proposed"].read_bytes())
    (validation / "validated_claims.json").write_text("[]", encoding="utf-8")
    (validation / "rejected_claims.json").write_text("[]", encoding="utf-8")
    (validation / "validation_status.json").write_text(
        json.dumps({"pmid:1::claim-1": {"status": "accepted"}}), encoding="utf-8"
    )

    result = workflow.inspect_research_state(
        workflow.InspectResearchStateInput(substance="Vitamin C", workspace_root=tmp_path)
    )

    assert result.suggested_action == "export_report"
    assert result.validation_status_counts == {"accepted": 1}
    assert result.validated_claims_path is not None


def test_limited_body_holds_claim_without_calling_claim_validator(monkeypatch, tmp_path: Path) -> None:
    source = CandidateCitation(
        id="pmid:limited",
        title="Short publisher text",
        abstract="Abstract summary.",
        source="pubmed",
        retrieval_query="q",
    )
    claim = ProposedClaim(
        id="claim-limited",
        nutrient_name="Vitamin C",
        citation_id=source.id,
        statement="Oral vitamin C improves an outcome.",
        proposed_effect_slug="outcome",
        proposed_effect_label="Outcome",
        evidence_type="human_clinical",
        intake_route="oral",
        exposure_category="supplement_level",
    )
    body = (
        f"{source.title} {source.abstract} 1. Introduction Background. "
        "Methods Adults consumed oral vitamin C. "
        "Results A favorable result was reported. "
        + ("Brief publisher summary text. " * 25)
    )
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "limited.txt").write_text(body, encoding="utf-8")
    record = RawTextRecord(
        source_id=source.id,
        title=source.title,
        status="full_text_found",
        retrieval_method="publisher_page",
        text_path="limited.txt",
        text_char_count=len(body),
    )
    sources_path = tmp_path / "sources.json"
    proposed_path = tmp_path / "proposed.json"
    manifest_path = tmp_path / "manifest.json"
    _write(sources_path, TypeAdapter(list[CandidateCitation]), [source])
    _write(proposed_path, TypeAdapter(list[ProposedClaim]), [claim])
    _write(manifest_path, TypeAdapter(list[RawTextRecord]), [record])

    class Provider:
        calls = 0

        def complete_json(self, prompt, adapter):
            self.calls += 1
            assert adapter is workflow.BODY_ADEQUACY_ADAPTER
            return adapter.validate_python(
                {
                    "status": "limited",
                    "coverage": {"human_population": True, "oral_exposure": True, "substantive_results": True},
                    "reason_codes": ["methods_and_results_too_brief"],
                    "reasoning": "The text lacks enough design and limitation detail.",
                    "diagnostic_excerpts": [],
                }
            )

    provider = Provider()
    monkeypatch.setattr(workflow, "_provider", lambda *_: (provider, "test", "model"))
    output_dir = tmp_path / "validation"
    result = workflow.validate_claims(
        workflow.ValidateClaimsInput(
            proposed_claims_path=proposed_path,
            sources_path=sources_path,
            raw_texts_manifest_path=manifest_path,
            raw_texts_dir=papers,
            output_dir=output_dir,
            resume=False,
        )
    )

    assert provider.calls == 1
    assert result.held_claim_count == 1
    assert result.limited_paper_count == 1
    assert result.accepted_count == result.rejected_count == result.pending_claim_count == 0
    assert json.loads((output_dir / "validated_claims.json").read_text()) == []
    assert json.loads((output_dir / "rejected_claims.json").read_text()) == []
    assert json.loads((output_dir / "held_claims.json").read_text())[0]["status"] == "held_insufficient_body"
    assert json.loads((output_dir / "body_retrieval_queue.json").read_text())[0]["current_status"] == "limited"


def test_inspect_research_state_recommends_better_text_for_held_claims(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "vitamin-c"
    claims_dir = run_dir / "claims"
    validation_dir = run_dir / "validation"
    claims_dir.mkdir(parents=True)
    validation_dir.mkdir()
    paths = _artifacts(tmp_path)
    (claims_dir / "proposed_claims.json").write_bytes(paths["proposed"].read_bytes())
    (validation_dir / "validation_status.json").write_text(
        json.dumps({"pmid:1::claim-1": {"status": "held_insufficient_body"}}), encoding="utf-8"
    )
    (validation_dir / "body_adequacy.json").write_text(
        json.dumps({"pmid:1": {"status": "limited"}}), encoding="utf-8"
    )

    result = workflow.inspect_research_state(
        workflow.InspectResearchStateInput(substance="Vitamin C", workspace_root=tmp_path)
    )

    assert result.suggested_action == "retrieve_better_body_text"
    assert result.held_claim_count == 1
    assert result.body_adequacy_status_counts == {"limited": 1}


def _retrieval_output(
    substance: str,
    queries: list[str],
    candidates: list[CandidateCitation],
) -> literature.RetrieveCandidatesOutput:
    return literature.RetrieveCandidatesOutput(
        substance=substance,
        aliases=[],
        queries=queries or [f'"{substance}" AND human'],
        candidates=candidates,
        raw_candidate_count=len(candidates),
        unique_candidate_count=len(candidates),
        pages_attempted=1,
        source_counts=literature.RetrievalSourceCounts(
            raw={"pubmed": len(candidates)}, unique={"pubmed": len(candidates)}
        ),
        failures=[],
        stop_reason="target_reached" if candidates else "no_new_unique_candidates",
    )


def test_adaptive_pipeline_expands_until_usable_full_text_target(
    monkeypatch, tmp_path: Path
) -> None:
    first = [
        CandidateCitation(id="pmid:1", title="First", source="pubmed", retrieval_query="q"),
        CandidateCitation(id="pmid:2", title="Blocked", source="pubmed", retrieval_query="q"),
    ]
    expanded = [
        CandidateCitation(id="pmid:3", title="Sleep trial", source="pubmed", retrieval_query="sleep")
    ]
    retrieval_calls: list[list[str]] = []

    def retrieve(payload):
        retrieval_calls.append(payload.queries)
        return _retrieval_output(
            payload.substance,
            payload.queries,
            first if len(retrieval_calls) == 1 else expanded,
        )

    def screen(payload):
        records = [
            SimpleNamespace(
                publication_type="primary_human_study", rejection_code=None
            )
            for _ in payload.candidates
        ]
        return SimpleNamespace(
            retained=payload.candidates,
            rejected=[],
            records=records,
            counts=SimpleNamespace(
                input=len(payload.candidates),
                classified=len(payload.candidates),
                retained=len(payload.candidates),
                rejected=0,
                screening_errors=0,
            ),
        )

    def full_text(payload):
        sources = TypeAdapter(list[CandidateCitation]).validate_json(
            payload.retained_sources_path.read_bytes()
        )
        records = []
        for source in sources:
            found = source.id in {"pmid:1", "pmid:3"}
            records.append(
                RawTextRecord(
                    source_id=source.id,
                    title=source.title,
                    status="full_text_found" if found else "blocked_by_cloudflare",
                    retrieval_method="pubmed_central" if found else "publisher_page",
                    text_path=f"{source.id}.txt" if found else None,
                    text_char_count=2000 if found else 0,
                )
            )
        payload.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = payload.output_dir / "manifest.json"
        attempts = payload.output_dir / "attempts.json"
        manifest.write_bytes(TypeAdapter(list[RawTextRecord]).dump_json(records, indent=2))
        attempts.write_text("[]", encoding="utf-8")
        return SimpleNamespace(
            retrieved_count=sum(item.status == "full_text_found" for item in records),
            unavailable_count=sum(item.status != "full_text_found" for item in records),
            skipped_existing_count=0,
            manifest_path=str(manifest),
            attempts_path=str(attempts),
        )

    monkeypatch.setattr(literature, "retrieve_candidates", retrieve)
    monkeypatch.setattr(literature, "screen_candidates", screen)
    monkeypatch.setattr(literature, "retrieve_full_text_batch", full_text)

    initial = workflow.advance_research_pipeline(
        workflow.AdvanceResearchPipelineInput(
            substance="Valerian root",
            target_stage="full_text",
            requested_count=2,
            workspace_root=tmp_path,
        )
    )
    expanded_result = workflow.advance_research_pipeline(
        workflow.AdvanceResearchPipelineInput(
            substance="Valerian root",
            target_stage="full_text",
            requested_count=2,
            expansion_queries=[
                '("Valeriana officinalis" OR valerian) AND (sleep OR insomnia) AND human'
            ],
            workspace_root=tmp_path,
        )
    )

    assert initial.full_text_retrieved_count == 1
    assert initial.expansion_recommended is True
    assert initial.stop_reason == "expansion_required"
    assert expanded_result.full_text_retrieved_count == 2
    assert expanded_result.target_met is True
    assert expanded_result.stop_reason == "target_met"
    assert expanded_result.new_candidate_count == 1
    assert expanded_result.new_usable_full_text_count == 1
    assert expanded_result.expansion_round == 2
    state = json.loads(Path(expanded_result.expansion_state_path).read_text())
    assert len(state["rounds"]) == 2
    assert "sleep OR insomnia" in state["rounds"][1]["queries"][0]


def test_adaptive_pipeline_stops_after_two_zero_yield_rounds(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        literature,
        "retrieve_candidates",
        lambda payload: _retrieval_output(payload.substance, payload.queries, []),
    )

    initial = workflow.advance_research_pipeline(
        workflow.AdvanceResearchPipelineInput(
            substance="Sparse compound",
            target_stage="screening",
            requested_count=3,
            workspace_root=tmp_path,
        )
    )
    second = workflow.advance_research_pipeline(
        workflow.AdvanceResearchPipelineInput(
            substance="Sparse compound",
            target_stage="screening",
            requested_count=3,
            expansion_queries=['"Sparse compound" AND randomized human trial'],
            workspace_root=tmp_path,
        )
    )

    assert initial.expansion_recommended is True
    assert second.expansion_recommended is False
    assert second.target_met is False
    assert second.stop_reason == "no_new_usable_papers"
    assert second.expansion_round == 2


def test_screening_target_counts_retained_papers_without_full_text(
    monkeypatch, tmp_path: Path
) -> None:
    candidates = [
        CandidateCitation(id=f"pmid:{index}", title=f"Paper {index}", source="pubmed", retrieval_query="q")
        for index in range(3)
    ]
    monkeypatch.setattr(
        literature,
        "retrieve_candidates",
        lambda payload: _retrieval_output(payload.substance, payload.queries, candidates),
    )
    monkeypatch.setattr(
        literature,
        "screen_candidates",
        lambda payload: SimpleNamespace(
            retained=payload.candidates,
            rejected=[],
            records=[
                SimpleNamespace(publication_type="review", rejection_code=None)
                for _ in payload.candidates
            ],
            counts=SimpleNamespace(rejected=0),
        ),
    )
    monkeypatch.setattr(
        literature,
        "retrieve_full_text_batch",
        lambda payload: (_ for _ in ()).throw(AssertionError("full text should not run")),
    )

    result = workflow.advance_research_pipeline(
        workflow.AdvanceResearchPipelineInput(
            substance="Vitamin C",
            target_stage="screening",
            requested_count=2,
            workspace_root=tmp_path,
        )
    )

    assert result.target_metric == "retained_papers"
    assert result.target_met is True
    assert result.retained_count == 2
