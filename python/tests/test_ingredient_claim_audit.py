import json
from pathlib import Path

from typer.testing import CliRunner

from sipz_agent.cli_ingredients import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.ingredient_claim_audit import audit_ingredient_claims, audit_prompt
from sipz_agent.core.models import HeuristicProvider
from sipz_agent.schemas.artifacts import PacketCounts, PacketModel
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import SupportingQuote
from sipz_agent.schemas.ingredients import (
    IngredientPacket,
    IngredientPacketInput,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)


class FakeAuditProvider:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python(self.response)


def ingredient_packet() -> IngredientPacket:
    return IngredientPacket(
        run_id="run",
        input=IngredientPacketInput(
            ingredient_name="acai",
            canonical_search_name="acai",
            ingredient_form="whole_food",
            canonical_beverage_id="ingredient-1",
            canonical_beverage_name="acai",
            depth="standard",
            demo=False,
            retrieval_queries=["acai human health"],
        ),
        model=PacketModel(provider="deepseek", model_name="deepseek-chat"),
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
        counts=PacketCounts(
            candidate_citations=1,
            screened_sources=1,
            proposed_claims=1,
            validated_claims=1,
            rejected_claims=0,
            effect_rows=0,
        ),
    )


def source() -> CandidateCitation:
    return CandidateCitation(
        id="pmid:1",
        title="Acai pulp intake and oxidative stress markers",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        doi="10.1000/acai",
        pmid="1",
        year=2025,
        source="pubmed",
        retrieval_query="acai human health",
        abstract="Human participants consumed acai pulp.",
    )


def proposed_claim(**overrides) -> ProposedIngredientClaim:
    payload = {
        "id": "claim-1",
        "ingredient_name": "acai",
        "ingredient_form": "pulp",
        "citation_id": "pmid:1",
        "statement": "Acai pulp may improve oxidative stress markers.",
        "proposed_effect_slug": "oxidative_stress",
        "proposed_effect_label": "Oxidative stress",
        "effect": "oxidative stress",
        "claim_direction": "beneficial",
        "population": "healthy adults",
        "oral_exposure": "consumed orally as acai pulp",
        "dose_or_serving": "200 g/day",
        "food_matrix": "pulp",
        "outcome": "oxidative stress markers",
        "study_type": "human observational",
        "limitations": ["Small human study."],
        "evidence_type": "human_observational",
        "exposure_category": "whole_food",
        "concentration_notes": "Pulp food form.",
        "claim_applies_to_group_members": "direct_only",
    }
    payload.update(overrides)
    return ProposedIngredientClaim.model_validate(payload)


def validated_claim(**overrides) -> ValidatedIngredientClaim:
    payload = {
        "effect_row_id": "effect-1",
        "proposed_ingredient_claim_id": "claim-1",
        "citation_id": "pmid:1",
        "verdict": "supported_with_limitations",
        "support_level": "human_observational",
        "claim_scope": "Healthy adults consuming acai pulp.",
        "validated_statement": "Acai pulp may improve oxidative stress markers.",
        "validator_reasoning": "The body supports a limited oral human claim.",
        "supporting_quotes": [
            SupportingQuote(
                quote="Participants consumed acai pulp and oxidative stress markers improved.",
                section="Results",
                reason="Reports the outcome.",
                match_status="exact",
            )
        ],
        "limitations": ["Small human study."],
        "accepted": True,
    }
    payload.update(overrides)
    return ValidatedIngredientClaim.model_validate(payload)


def write_run(
    tmp_path: Path,
    *,
    proposed: list[ProposedIngredientClaim] | None = None,
    validated: list[ValidatedIngredientClaim] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "ingredient_packet.json", ingredient_packet().model_dump(mode="json"))
    write_json(run_dir / "sources.json", [source().model_dump(mode="json")])
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [claim.model_dump(mode="json") for claim in (proposed or [proposed_claim()])],
    )
    write_json(
        run_dir / "validated_ingredient_claims.json",
        [claim.model_dump(mode="json") for claim in (validated or [validated_claim()])],
    )
    (run_dir / "audit_log.jsonl").write_text("", encoding="utf-8")
    return run_dir


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_audit_skips_zero_accepted_claims(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, validated=[validated_claim(accepted=False, verdict="unsupported")])

    result = audit_ingredient_claims(run_dir=run_dir, provider=HeuristicProvider())

    assert result.skipped is True
    assert result.skip_reason == "no_accepted_claims"
    assert not (run_dir / "audited_ingredient_claims.json").exists()


def test_audit_writes_passed_claims_with_heuristic_provider(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)

    result = audit_ingredient_claims(run_dir=run_dir, provider=HeuristicProvider())

    audited = read_json(run_dir / "audited_ingredient_claims.json")
    rejected = read_json(run_dir / "rejected_audited_ingredient_claims.json")
    findings = read_json(run_dir / "ingredient_claim_audit_findings.json")

    assert len(result.audited_claims) == 1
    assert audited[0]["proposed_ingredient_claim_id"] == "claim-1"
    assert rejected == []
    assert findings[0]["verdict"] == "pass"
    assert findings[0]["paper_support_check"] == "validated_and_quote_grounded"
    assert findings[0]["report_fit_check"] == "suitable_with_caveats"
    assert findings[0]["modern_evidence_check"] == "not_assessed"
    assert (run_dir / "ingredient_claim_audit_summary.md").exists()


def test_audit_rejects_topical_oral_hygiene_claim_deterministically(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path,
        proposed=[
            proposed_claim(
                statement="Coconut oil pulling may reduce dental plaque.",
                oral_exposure="used as a mouthwash for oil pulling",
                outcome="dental plaque",
                food_matrix="mouth rinse",
            )
        ],
        validated=[
            validated_claim(
                validated_statement="Coconut oil pulling may reduce dental plaque.",
                claim_scope="Adults using coconut oil as an oral hygiene rinse.",
            )
        ],
    )

    result = audit_ingredient_claims(run_dir=run_dir, provider=HeuristicProvider())

    findings = read_json(run_dir / "ingredient_claim_audit_findings.json")
    assert result.audited_claims == []
    assert len(result.rejected_claims) == 1
    assert "topical_or_oral_hygiene_evidence" in findings[0]["issue_categories"]
    assert findings[0]["paper_support_check"] == "validated_and_quote_grounded"
    assert findings[0]["report_fit_check"] == "not_suitable"


def test_audit_llm_rejects_implausible_or_outdated_claim(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    provider = FakeAuditProvider(
        {
            "decisions": [
                {
                    "proposed_ingredient_claim_id": "claim-1",
                    "citation_id": "pmid:1",
                    "verdict": "reject_for_synthesis",
                    "paper_support_check": "validated_and_quote_grounded",
                    "report_fit_check": "not_suitable",
                    "modern_evidence_check": "implausible_or_outdated",
                    "issue_categories": ["implausible_or_outdated_claim"],
                    "reasoning": "The claim conflicts with current scientific understanding.",
                    "paper_support_reasoning": "The validator artifact is accepted and quote-grounded.",
                    "report_fit_reasoning": "The claim should not appear in the report.",
                    "modern_evidence_reasoning": "The claim is outdated based on model knowledge.",
                    "suggested_scope": None,
                    "confidence": 0.8,
                }
            ]
        }
    )

    result = audit_ingredient_claims(run_dir=run_dir, provider=provider)

    assert provider.prompts
    assert result.audited_claims == []
    assert len(result.rejected_claims) == 1
    assert read_json(run_dir / "ingredient_claim_audit_findings.json")[0]["verdict"] == (
        "reject_for_synthesis"
    )


def test_audit_prompt_treats_combination_evidence_as_caveat() -> None:
    prompt = audit_prompt(
        packet=ingredient_packet(),
        source=source(),
        items=[
            (
                validated_claim(
                    validated_statement=(
                        "A food group including acai was associated with improved markers."
                    ),
                    limitations=["The independent effect of acai was not isolated."],
                ),
                proposed_claim(
                    statement="A food group including acai may improve markers.",
                    food_matrix="mixed food group",
                    claim_applies_to_group_members="form_specific",
                ),
            )
        ],
    )

    assert "This is NOT paper validation" in prompt
    assert "This audit has three responsibilities" in prompt
    assert "Paper support sanity" in prompt
    assert "Report fit" in prompt
    assert "Modern model-knowledge check" in prompt
    assert "Do not re-judge whether the supplied paper actually found the effect" in prompt
    assert "Do not reject a claim solely because the ingredient was studied as part of" in prompt
    assert "combination_or_food_group_evidence as a caveat" in prompt
    assert "paper_support_check" in prompt
    assert "report_fit_check" in prompt
    assert "modern_evidence_check" in prompt
    assert "small_sample_size" in prompt
    assert "no_control_group" in prompt
    assert "composition_data_not_human_trial" in prompt


def test_audit_llm_needs_review_is_excluded_from_audited_claims(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    provider = FakeAuditProvider(
        {
            "decisions": [
                {
                    "proposed_ingredient_claim_id": "claim-1",
                    "citation_id": "pmid:1",
                    "verdict": "needs_review",
                    "paper_support_check": "validated_and_quote_grounded",
                    "report_fit_check": "needs_review",
                    "modern_evidence_check": "limited_or_uncertain",
                    "issue_categories": ["dose_or_serving_missing"],
                    "reasoning": "The serving is unclear for final report synthesis.",
                    "paper_support_reasoning": "The validator artifact is accepted and quote-grounded.",
                    "report_fit_reasoning": "Report wording needs serving caveats.",
                    "modern_evidence_reasoning": "The claim is plausible but evidence may be limited.",
                    "confidence": 0.7,
                }
            ]
        }
    )

    result = audit_ingredient_claims(run_dir=run_dir, provider=provider)

    audited = read_json(run_dir / "audited_ingredient_claims.json")
    rejected = read_json(run_dir / "rejected_audited_ingredient_claims.json")
    assert result.audited_claims == []
    assert audited == []
    assert rejected[0]["proposed_ingredient_claim_id"] == "claim-1"


def test_audit_claims_cli_writes_artifacts(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "audit-claims",
            "--run",
            str(run_dir),
            "--provider",
            "heuristic",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Ingredient claim audit complete: 1 passed, 0 excluded." in result.output
    assert (run_dir / "audited_ingredient_claims.json").exists()
