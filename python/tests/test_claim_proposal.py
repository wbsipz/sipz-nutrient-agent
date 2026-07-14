import json
from pathlib import Path

from typer.testing import CliRunner

from sipz_agent.cli import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.claim_proposal import (
    build_claim_extraction_context,
    accepted_proposed_claim,
    claim_proposal_prompt,
    propose_claims_from_raw_texts,
)
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim
from sipz_agent.schemas.raw_texts import RawTextRecord


class FakeClaimProvider:
    def __init__(self, claims: list[dict]) -> None:
        self.claims = claims
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python({"claims": self.claims, "skipped_reason": None})


def test_claim_context_prioritizes_study_sections_and_removes_references() -> None:
    text = (
        "Abstract This abstract should be excluded.\nIntroduction\n" + "Intro. " * 2000
        + "\nMethods\n" + "Methods evidence. " * 1000
        + "\nResults\n" + "Results evidence. " * 1000
        + "\nDiscussion\n" + "Discussion evidence. " * 1000
        + "\nReferences\n" + "Reference material. " * 1000
    )

    context, metadata = build_claim_extraction_context(text)

    assert len(context) <= 45_000
    assert "Methods evidence" in context
    assert "Results evidence" in context
    assert "Reference material" not in context
    assert metadata["strategy"] == "section_priority"
    assert metadata["truncated"] is True


def claim_payload(**overrides) -> dict:
    payload = {
        "id": "claim-1",
        "nutrient_name": "quercetin",
        "citation_id": "pmid:1",
        "statement": "Oral quercetin intake may support inflammatory balance in adults.",
        "proposed_effect_slug": "inflammatory_balance",
        "proposed_effect_label": "Inflammatory balance",
        "compound": "quercetin",
        "effect": "inflammatory balance",
        "direction": "beneficial",
        "population": "adults",
        "dose_or_exposure": "500 mg/day oral supplement",
        "outcome": "inflammatory markers",
        "study_type": "human review",
        "limitations": ["review-level interpretation"],
        "evidence_type": "review_author_interpretation",
        "intake_route": "oral",
        "exposure_category": "supplement_level",
        "natural_concentration_relevance": "unclear",
        "supplement_level_relevance": "directly relevant to supplement intake",
        "pharmaceutical_centered": False,
        "concentration_notes": "Dose appears supplement-level.",
    }
    payload.update(overrides)
    return payload


def citation() -> CandidateCitation:
    return CandidateCitation(
        id="pmid:1",
        title="Quercetin and human health",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        doi="10.1000/test",
        pmid="1",
        source="test",
        retrieval_query="quercetin health human",
        abstract="Abstract should not be used for proposal evidence.",
    )


def write_claim_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw_texts"
    raw_dir.mkdir(parents=True)
    write_json(run_dir / "sources.json", [citation().model_dump(mode="json")])
    write_json(
        run_dir / "raw_texts.json",
        [
            RawTextRecord(
                source_id="pmid:1",
                title="Quercetin and human health",
                doi="10.1000/test",
                pmid="1",
                url="https://pubmed.ncbi.nlm.nih.gov/1/",
                status="full_text_found",
                retrieval_method="pmc_oa_xml",
                text_path="raw_texts/001_pmid_1.txt",
                text_char_count=2000,
            ).model_dump(mode="json")
        ],
    )
    (raw_dir / "001_pmid_1.txt").write_text(
        "Introduction Methods Results Discussion References "
        + ("Oral quercetin supplementation in adults was discussed. " * 80),
        encoding="utf-8",
    )
    packet = Packet(
        run_id="run",
        input=PacketInput(nutrient_name="quercetin", depth="light", demo=False),
        model=PacketModel(provider="deepseek", model_name="deepseek-chat"),
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
        counts=PacketCounts(
            candidate_citations=1,
            screened_sources=1,
            proposed_claims=0,
            validated_claims=0,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "packet.json", packet.model_dump(mode="json"))
    (run_dir / "audit_log.jsonl").write_text("", encoding="utf-8")
    return run_dir


def test_claim_proposal_prompt_contains_oral_and_concentration_constraints() -> None:
    prompt = claim_proposal_prompt(
        nutrient_name="quercetin",
        citation=citation(),
        body_text="Introduction Results Discussion " + ("oral intake " * 200),
    )

    assert "oral consumption" in prompt
    assert "natural ingredient/food concentrations" in prompt
    assert "supplement-level concentrations" in prompt
    assert "pharmaceutical use" in prompt
    assert "Reject animal-only and in-vitro-only evidence" in prompt
    assert "extrapolated from animals" in prompt
    assert "empty claims array" in prompt
    assert "Paper body text, not abstract" in prompt


def test_accepted_proposed_claim_filters_non_oral_and_pharmaceutical_claims() -> None:
    assert accepted_proposed_claim(ProposedClaim.model_validate(claim_payload())) is True
    assert (
        accepted_proposed_claim(
            ProposedClaim.model_validate(claim_payload(intake_route="topical"))
        )
        is False
    )
    assert (
        accepted_proposed_claim(
            ProposedClaim.model_validate(
                claim_payload(exposure_category="pharmaceutical_level", pharmaceutical_centered=True)
            )
        )
        is False
    )
    inhalation = ProposedClaim.model_validate(claim_payload(intake_route="inhalation"))
    assert inhalation.intake_route == "unclear"
    assert accepted_proposed_claim(inhalation) is False
    assert (
        accepted_proposed_claim(
            ProposedClaim.model_validate(claim_payload(evidence_type="in_vitro"))
        )
        is False
    )


def test_proposed_claim_normalizes_single_limitation_string() -> None:
    claim = ProposedClaim.model_validate(
        claim_payload(limitations="Human evidence remains limited.")
    )

    assert claim.limitations == ["Human evidence remains limited."]


def test_proposed_claim_normalizes_boolean_relevance_notes() -> None:
    claim = ProposedClaim.model_validate(
        claim_payload(
            natural_concentration_relevance=True,
            supplement_level_relevance=False,
        )
    )

    assert claim.natural_concentration_relevance == "relevant"
    assert claim.supplement_level_relevance == "not relevant"


def test_proposed_claim_normalizes_qualified_direction() -> None:
    claim = ProposedClaim.model_validate(
        claim_payload(direction="beneficial (potential)")
    )

    assert claim.direction == "beneficial"


def test_proposed_claim_normalizes_unknown_direction_to_unclear() -> None:
    claim = ProposedClaim.model_validate(
        claim_payload(direction="possibly supportive")
    )

    assert claim.direction == "unclear"


def test_propose_claims_from_raw_texts_writes_artifacts_and_updates_packet(tmp_path) -> None:
    run_dir = write_claim_run(tmp_path)
    provider = FakeClaimProvider([claim_payload()])
    progress_events = []

    claims = propose_claims_from_raw_texts(
        nutrient_name="quercetin",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        update_run_packet=True,
        progress=lambda index, total, source: progress_events.append(
            (index, total, source.title)
        ),
    )

    packet = json.loads((run_dir / "packet.json").read_text(encoding="utf-8"))
    audit_log = (run_dir / "audit_log.jsonl").read_text(encoding="utf-8")

    assert len(claims) == 1
    assert (run_dir / "proposed_claims.json").exists()
    assert (run_dir / "proposed_claims.md").exists()
    assert packet["counts"]["proposed_claims"] == 1
    assert "claim_proposal_started" in audit_log
    assert "claim_proposal_completed" in audit_log
    assert progress_events == [(1, 1, "Quercetin and human health")]


def test_propose_claims_from_raw_texts_skips_abstract_only_records(tmp_path) -> None:
    run_dir = write_claim_run(tmp_path)
    records = json.loads((run_dir / "raw_texts.json").read_text(encoding="utf-8"))
    records[0]["status"] = "abstract_only"
    records[0]["text_path"] = None
    write_json(run_dir / "raw_texts.json", records)
    provider = FakeClaimProvider([claim_payload()])

    claims = propose_claims_from_raw_texts(
        nutrient_name="quercetin",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
    )

    assert claims == []
    assert provider.prompts == []


def test_cli_propose_claims_run_mode(monkeypatch, tmp_path) -> None:
    run_dir = write_claim_run(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("sipz_agent.cli.create_llm_provider", lambda _: FakeClaimProvider([claim_payload()]))
    runner = CliRunner()

    result = runner.invoke(app, ["propose-claims", "--run", str(run_dir), "--provider", "deepseek"])

    assert result.exit_code == 0
    assert "[1/1] Analyzing Quercetin and human health (DOI: 10.1000/test)" in result.stdout
    assert "Proposed 1 claims" in result.stdout
    assert (run_dir / "proposed_claims.json").exists()


def test_cli_propose_claims_isolated_mode(monkeypatch, tmp_path) -> None:
    run_dir = write_claim_run(tmp_path)
    out_dir = tmp_path / "isolated-output"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("sipz_agent.cli.create_llm_provider", lambda _: FakeClaimProvider([claim_payload()]))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "propose-claims",
            "--raw-texts-dir",
            str(run_dir / "raw_texts"),
            "--raw-texts-manifest",
            str(run_dir / "raw_texts.json"),
            "--sources",
            str(run_dir / "sources.json"),
            "--nutrient",
            "quercetin",
            "--out",
            str(out_dir),
            "--provider",
            "deepseek",
        ],
    )

    assert result.exit_code == 0
    assert (out_dir / "proposed_claims.json").exists()
    assert (out_dir / "claim_proposal_packet.json").exists()
