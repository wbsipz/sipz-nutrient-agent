import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sipz_agent.cli import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.validation import (
    BodySanitizationError,
    sanitize_paper_body,
    validate_proposed_claims,
)
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim
from sipz_agent.schemas.raw_texts import RawTextRecord


class FakeValidationProvider:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python(self.responses.pop(0))


def citation(
    *,
    source_id: str = "pmid:1",
    title: str = "Quercetin and vascular health",
    abstract: str | None = "Oral quercetin may improve endothelial function in adults.",
) -> CandidateCitation:
    return CandidateCitation(
        id=source_id,
        title=title,
        url=f"https://example.com/{source_id}",
        doi=f"10.1000/{source_id.replace(':', '-')}",
        source="test",
        retrieval_query="quercetin human health",
        abstract=abstract,
    )


def proposed_claim(
    *,
    claim_id: str = "claim-1",
    source_id: str = "pmid:1",
    statement: str = "Oral quercetin improves endothelial function in adults.",
) -> ProposedClaim:
    return ProposedClaim(
        id=claim_id,
        nutrient_name="quercetin",
        citation_id=source_id,
        statement=statement,
        proposed_effect_slug="endothelial-function",
        proposed_effect_label="Endothelial function",
        compound="quercetin",
        effect="endothelial function",
        direction="beneficial",
        population="adults",
        dose_or_exposure="500 mg/day",
        outcome="flow-mediated dilation",
        study_type="randomized trial",
        evidence_type="human_clinical",
        intake_route="oral",
        exposure_category="supplement_level",
    )


def decision(
    claim_id: str,
    *,
    verdict: str = "supported",
    quote: str = "Flow-mediated dilation increased after oral quercetin supplementation.",
) -> dict:
    return {
        "proposed_claim_id": claim_id,
        "verdict": verdict,
        "validated_statement": "Oral quercetin may improve flow-mediated dilation in adults.",
        "support_level": "human_rct" if verdict != "unsupported" else "unsupported",
        "entity_match": "exact_or_alias",
        "claim_scope": "Adults receiving oral supplemental quercetin.",
        "supporting_quotes": (
            [{"quote": quote, "section": "Results", "reason": "Reports the measured outcome."}]
            if quote
            else []
        ),
        "limitations": ["Small sample size."],
        "reasoning": "The body reports the stated human outcome.",
    }


def body(title: str, abstract: str) -> str:
    return (
        f"{title}\n{abstract}\nKeywords quercetin vascular health\n"
        "1. Introduction Quercetin is a dietary flavonoid. "
        "Methods Adults received oral quercetin. "
        "Results Flow-mediated dilation increased after oral quercetin supplementation. "
        "Discussion The sample was small and confirmation is required. "
        + ("Additional body evidence. " * 30)
    )


def write_validation_inputs(
    tmp_path: Path,
    *,
    citations: list[CandidateCitation] | None = None,
    claims: list[ProposedClaim] | None = None,
    bodies: dict[str, str] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw_texts"
    raw_dir.mkdir(parents=True)
    citations = citations or [citation()]
    claims = claims or [proposed_claim()]
    bodies = bodies or {
        citations[0].id: body(citations[0].title, citations[0].abstract or "")
    }
    records = []
    for index, source in enumerate(citations, start=1):
        text = bodies[source.id]
        filename = f"{index:03d}.txt"
        (raw_dir / filename).write_text(text, encoding="utf-8")
        records.append(
            RawTextRecord(
                source_id=source.id,
                title=source.title,
                doi=source.doi,
                url=str(source.url),
                status="full_text_found",
                retrieval_method="publisher_page",
                text_path=f"raw_texts/{filename}",
                text_char_count=len(text),
            )
        )
    write_json(run_dir / "sources.json", [item.model_dump(mode="json") for item in citations])
    write_json(run_dir / "proposed_claims.json", [item.model_dump(mode="json") for item in claims])
    write_json(run_dir / "raw_texts.json", [item.model_dump(mode="json") for item in records])
    return run_dir


def test_sanitize_paper_body_removes_normalized_title_and_abstract() -> None:
    source = citation(abstract="Oral quercetin improves vascular function.")
    text = (
        "Quercetin and vascular health "
        "Oral quer-\ncetin improves vascular function. "
        "1. Introduction Body evidence Quercetin and vascular health "
        + ("continues " * 100)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert source.title.lower() not in sanitized.lower()
    assert "oral quer-" not in sanitized.lower()
    assert "1. Introduction" in sanitized


def test_sanitize_paper_body_removes_references_section() -> None:
    source = citation(abstract="Oral quercetin improves vascular function.")
    text = (
        "Quercetin and vascular health "
        "Oral quercetin improves vascular function. "
        "1. Introduction Body evidence. "
        "Methods Adults received oral quercetin. "
        "Results Flow-mediated dilation increased after oral quercetin supplementation. "
        "Discussion The result requires confirmation. "
        + ("Additional body evidence. " * 40)
        + "References Smith AB, Example paper title. "
        + ("Reference-only animal evidence. " * 40)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert "Flow-mediated dilation increased" in sanitized
    assert "References Smith AB" not in sanitized
    assert "Reference-only animal evidence" not in sanitized


def test_sanitize_paper_body_removes_terminal_conclusions_section() -> None:
    source = citation()
    text = (
        f"{source.title}\nAbstract\n{source.abstract}\nIntroduction\n"
        + "Background, methods, and results evidence. " * 60
        + "\nConclusions\nThis conclusion must not be validator evidence."
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert "methods, and results evidence" in sanitized
    assert "must not be validator evidence" not in sanitized


def test_sanitize_paper_body_uses_introduction_fallback() -> None:
    source = citation(abstract="Metadata abstract not present in body.")
    text = "Publisher preamble and author names 1. Introduction " + ("Body evidence. " * 80)

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith(" 1. Introduction") or sanitized.startswith("1. Introduction")
    assert "Publisher preamble" not in sanitized


def test_sanitize_paper_body_uses_first_numbered_section_fallback() -> None:
    source = citation(abstract="Metadata abstract not present in body.")
    text = (
        "Publisher metadata and an unmatched abstract summary. "
        "1. Neuroinflammation Toxicity and Neuroprotection "
        + ("Body evidence about neuroinflammation. " * 40)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith("1. Neuroinflammation Toxicity and Neuroprotection")
    assert "unmatched abstract summary" not in sanitized


def test_sanitize_paper_body_uses_polish_wstep_fallback() -> None:
    source = citation(abstract="English metadata abstract not present verbatim.")
    text = (
        "Streszczenie Polski tekst abstraktu. Summary English abstract text. "
        "Słowa kluczowe: Keywords: kwercetyna "
        "Wstęp Hipokrates opisywał znaczenie żywności. "
        + ("Treść artykułu o kwercetynie. " * 40)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith("Wstęp")
    assert "Streszczenie" not in sanitized
    assert "Summary English abstract" not in sanitized


def test_sanitize_paper_body_uses_structured_background_fallback() -> None:
    source = citation(abstract="Metadata abstract not present verbatim.")
    text = (
        "Publisher metadata and journal navigation. "
        + ("Preamble text. " * 140)
        + "Background Tart cherry studies require body validation. "
        + ("Background body evidence. " * 80)
        + "Methods Adults consumed oral tart cherry concentrate. "
        + ("Methods body evidence. " * 80)
        + "Results Sleep and recovery outcomes were measured. "
        + ("Results body evidence. " * 80)
        + "Discussion Results require cautious interpretation. "
        + ("Discussion body evidence. " * 80)
        + "References Smith AB, Reference-only text. "
        + ("Reference evidence. " * 40)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith("Background")
    assert "Publisher metadata" not in sanitized
    assert "Reference-only text" not in sanitized


def test_sanitize_paper_body_removes_abstract_heading_without_exact_abstract_match() -> None:
    source = citation(abstract="Metadata abstract does not match extracted body text.")
    text = (
        "Journal header Abstract This extracted abstract has broken wording about adults consuming acai. "
        "Keywords acai polyphenols "
        "Introduction The article begins after metadata. "
        "Methods Adults consumed oral acai pulp daily. "
        "Results Biomarkers were measured after dietary acai intake. "
        "Discussion The authors interpreted the human findings cautiously. "
        + ("Body evidence. " * 80)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith("Keywords") or sanitized.startswith("Introduction")
    assert "Journal header Abstract" not in sanitized
    assert "Adults consumed oral acai pulp" in sanitized


def test_sanitize_paper_body_allows_methods_results_discussion_fragment() -> None:
    source = citation(abstract="Metadata abstract not present verbatim.")
    text = (
        "PDF extraction header, article metadata, clipped abstract, and navigation. "
        + ("Noisy preamble. " * 60)
        + "2 M E T H O D S Participants consumed tart cherry juice orally for seven days. "
        + ("Methods fragment with participants and outcome measurement. " * 30)
        + "3 R E S U L T S Sleep duration and soreness outcomes were reported. "
        + ("Results fragment for human participants. " * 30)
        + "4 D I S C U S S I O N The authors discussed cautious interpretation. "
        + ("Discussion fragment. " * 30)
        + "References Smith AB, Reference-only material. "
        + ("Reference fragment. " * 30)
    )

    sanitized = sanitize_paper_body(body_text=text, citation=source)

    assert sanitized.startswith("M E T H O D S")
    assert "clipped abstract" not in sanitized
    assert "Reference-only material" not in sanitized


def test_sanitize_paper_body_rejects_subscription_preview_abstract() -> None:
    source = citation(abstract="Metadata abstract not present verbatim.")
    text = (
        "Article page Abstract Background Tart cherries may affect sleep. "
        "Methods This abstract describes volunteers consuming juice. "
        "Results This abstract reports improved sleep. "
        "This is a preview of subscription content, log in via an institution to check access. "
        "References Smith AB, Example reference. "
        + ("Reference evidence. " * 40)
    )

    with pytest.raises(BodySanitizationError, match="body_sanitization_failed"):
        sanitize_paper_body(body_text=text, citation=source)


def test_sanitize_paper_body_fails_without_abstract_match_or_introduction() -> None:
    with pytest.raises(BodySanitizationError, match="body_sanitization_failed"):
        sanitize_paper_body(
            body_text="Unstructured publisher text without a reliable section boundary.",
            citation=citation(abstract="An unmatched abstract."),
        )


def test_validation_groups_claims_by_paper_and_grounds_quotes(tmp_path) -> None:
    claims = [
        proposed_claim(claim_id="claim-1"),
        proposed_claim(claim_id="claim-2", statement="Quercetin prevents all vascular disease."),
    ]
    run_dir = write_validation_inputs(tmp_path, claims=claims)
    provider = FakeValidationProvider(
        [
            {
                "decisions": [
                    decision("claim-1"),
                    decision("claim-2", verdict="unsupported", quote=""),
                ]
            }
        ]
    )

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert len(provider.prompts) == 1
    assert citation().title not in provider.prompts[0]
    assert citation().abstract not in provider.prompts[0]
    assert [item.proposed_claim_id for item in accepted] == ["claim-1"]
    assert accepted[0].supporting_quotes[0].match_status == "exact"
    assert [item.proposed_claim_id for item in rejected] == ["claim-2"]
    assert failures == []


def test_validator_prompt_excludes_references_and_requires_preclinical_rejection(tmp_path) -> None:
    source = citation()
    paper_body = (
        body(source.title, source.abstract or "")
        + "\nReferences\n"
        + ("Animal-only reference title. " * 40)
    )
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[source],
        bodies={source.id: paper_body},
    )
    provider = FakeValidationProvider([{"decisions": [decision("claim-1")]}])

    validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert "Animal-only reference title" not in provider.prompts[0]
    assert "verdict must be\n  unsupported" in provider.prompts[0]
    assert "Do not use supported_with_limitations" in provider.prompts[0]


def test_narrowed_claim_is_accepted_and_missing_quote_is_rejected(tmp_path) -> None:
    claims = [
        proposed_claim(claim_id="claim-narrow"),
        proposed_claim(claim_id="claim-bad-quote"),
    ]
    run_dir = write_validation_inputs(tmp_path, claims=claims)
    provider = FakeValidationProvider(
        [
            {
                "decisions": [
                    decision("claim-narrow", verdict="supported_with_limitations"),
                    decision("claim-bad-quote", quote="This quote does not occur in the paper."),
                ]
            }
        ]
    )

    accepted, rejected, _ = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted[0].verdict == "supported_with_limitations"
    assert accepted[0].validated_statement
    assert rejected[0].verdict == "quote_not_found"
    assert rejected[0].accepted is False


def test_quote_repair_accepts_exact_copied_quote_after_failed_grounding(tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    provider = FakeValidationProvider(
        [
            {"decisions": [decision("claim-1", quote="Flow mediated dilation improved with quercetin.")]},
            {
                "supporting_quotes": [
                    {
                        "quote": "Flow-mediated dilation increased after oral quercetin supplementation.",
                        "section": "Results",
                        "reason": "Exact copied result sentence.",
                    }
                ]
            },
        ]
    )

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert rejected == []
    assert failures == []
    assert accepted[0].accepted is True
    assert accepted[0].supporting_quotes[0].quote == (
        "Flow-mediated dilation increased after oral quercetin supplementation."
    )
    assert accepted[0].supporting_quotes[0].match_status == "exact"
    assert len(provider.prompts) == 2
    assert "Return exact copied quote from this body excerpt only" in provider.prompts[1]


def test_quote_repair_still_rejects_when_repair_quote_is_not_exact(tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    provider = FakeValidationProvider(
        [
            {"decisions": [decision("claim-1", quote="Flow mediated dilation improved with quercetin.")]},
            {
                "supporting_quotes": [
                    {
                        "quote": "Flow mediated dilation improved after quercetin.",
                        "section": "Results",
                        "reason": "Still paraphrased.",
                    }
                ]
            },
        ]
    )

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "quote_not_found"
    assert rejected[0].supporting_quotes[0].match_status == "not_found"
    assert len(provider.prompts) == 2


def test_title_like_quote_is_not_enough_to_accept_supported_claim(tmp_path) -> None:
    source = citation(
        title=(
            "Tart cherry intake and serum uric acid: Meta-analysis of randomized "
            "controlled trials and evidence from network pharmacology."
        ),
        abstract="This meta-analysis evaluated tart cherry intake in adults.",
    )
    claim = proposed_claim(
        statement="Tart cherry intake modestly reduces serum uric acid in adults.",
    )
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[source],
        claims=[claim],
        bodies={
            source.id: (
                "Publisher page metadata. "
                "This meta-analysis evaluated tart cherry intake in adults. "
                "Tart cherry intake and serum uric acid: Meta-analysis of randomized "
                "controlled trials "
                "1. Introduction Tart cherry has been studied in humans. "
                "Methods The review identified randomized controlled trials. "
                "Results The body text in this fixture does not report the effect estimate. "
                "Discussion Additional research is needed. "
                + ("Additional body evidence. " * 35)
            )
        },
    )
    provider = FakeValidationProvider(
        [
            {
                "decisions": [
                    decision(
                        "claim-1",
                        verdict="supported_with_limitations",
                        quote=(
                            "Tart cherry intake and serum uric acid: Meta-analysis of "
                            "randomized controlled trials"
                        ),
                    )
                ]
            }
        ]
    )

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "quote_not_found"
    assert "paper title or article header" in rejected[0].limitations[-1]


def test_exact_context_quote_is_not_enough_to_prove_health_effect(tmp_path) -> None:
    source = citation(title="Valerian for sleep")
    claim = proposed_claim(
        statement="Oral valerian improves subjective sleep quality in adults."
    )
    context_quote = (
        "Only 6 of the 16 identified studies reported a dichotomous outcome measure of sleep."
    )
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[source],
        claims=[claim],
        bodies={
            source.id: (
                f"1. Introduction Valerian is used for sleep. Methods {context_quote} "
                "Results Effect estimates were unavailable in this fixture. "
                + ("Additional body evidence. " * 35)
            )
        },
    )
    supported = decision("claim-1", verdict="supported_with_limitations", quote=context_quote)
    supported["supporting_quotes"][0]["evidence_role"] = "methods_or_context"
    provider = FakeValidationProvider([{"decisions": [supported]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "quote_not_found"
    assert rejected[0].supporting_quotes[0].match_status == "exact"
    assert "directly reported the claimed result" in rejected[0].limitations[-1]


def test_validator_rejects_claim_supported_by_different_species(tmp_path) -> None:
    source = citation(title="Valeriana species and sleep")
    claim = proposed_claim(
        statement="Valeriana edulis root improves sleep latency in children."
    ).model_copy(
        update={"nutrient_name": "Valeriana officinalis", "compound": "Valeriana edulis"}
    )
    exact_result = "Valeriana edulis root significantly reduced sleep latency in five children."
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[source],
        claims=[claim],
        bodies={
            source.id: (
                f"1. Introduction Species differ. Results {exact_result} "
                + ("Additional body evidence. " * 35)
            )
        },
    )
    supported = decision("claim-1", quote=exact_result)
    supported["entity_match"] = "different_species"
    provider = FakeValidationProvider([{"decisions": [supported]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "unsupported"
    assert "different species or entity" in rejected[0].limitations[-1]


def test_title_like_quote_does_not_block_real_body_quote(tmp_path) -> None:
    source = citation(
        title=(
            "Tart cherry intake and serum uric acid: Meta-analysis of randomized "
            "controlled trials and evidence from network pharmacology."
        ),
        abstract="This meta-analysis evaluated tart cherry intake in adults.",
    )
    claim = proposed_claim(
        statement="Tart cherry intake modestly reduces serum uric acid in adults.",
    )
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[source],
        claims=[claim],
        bodies={
            source.id: (
                "Publisher page metadata. "
                "This meta-analysis evaluated tart cherry intake in adults. "
                "Tart cherry intake and serum uric acid: Meta-analysis of randomized "
                "controlled trials "
                "1. Introduction Tart cherry has been studied in humans. "
                "Methods The review identified randomized controlled trials. "
                "Results Tart cherry intake was associated with a modest reduction in serum "
                "uric acid compared with control. "
                "Discussion Additional research is needed. "
                + ("Additional body evidence. " * 35)
            )
        },
    )
    supported = decision(
        "claim-1",
        verdict="supported_with_limitations",
        quote=(
            "Tart cherry intake and serum uric acid: Meta-analysis of randomized "
            "controlled trials"
        ),
    )
    supported["supporting_quotes"].append(
        {
            "quote": (
                "Tart cherry intake was associated with a modest reduction in serum "
                "uric acid compared with control."
            ),
            "section": "Results",
            "reason": "Reports the body result.",
        }
    )
    supported["supporting_quotes"].append(
        {
            "quote": "Tart cherry reduced serum urate in the pooled analysis.",
            "section": "Results",
            "reason": "Paraphrased result that should not be retained.",
        }
    )
    provider = FakeValidationProvider([{"decisions": [supported]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert [item.proposed_claim_id for item in accepted] == ["claim-1"]
    assert rejected == []
    assert failures == []
    assert [quote.quote for quote in accepted[0].supporting_quotes] == [
        "Tart cherry intake was associated with a modest reduction in serum "
        "uric acid compared with control."
    ]
    assert accepted[0].supporting_quotes[0].match_status == "exact"


def test_rejected_decision_can_have_blank_statement_and_scope(tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    unsupported = decision("claim-1", verdict="unsupported", quote="")
    unsupported["validated_statement"] = ""
    unsupported["claim_scope"] = ""
    provider = FakeValidationProvider([{"decisions": [unsupported]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "unsupported"
    assert rejected[0].validated_statement == (
        "Rejected because the paper body did not support this proposed claim."
    )
    assert rejected[0].claim_scope == (
        "Rejected because the paper body did not support a validated claim scope."
    )


def test_rejected_decision_can_have_null_statement_and_scope(tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    over_scoped = decision("claim-1", verdict="over_scoped", quote="")
    over_scoped["validated_statement"] = None
    over_scoped["claim_scope"] = None
    provider = FakeValidationProvider([{"decisions": [over_scoped]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].verdict == "over_scoped"


def test_supported_decision_with_blank_statement_is_rejected_not_accepted(tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    supported = decision("claim-1")
    supported["validated_statement"] = ""
    provider = FakeValidationProvider([{"decisions": [supported]}])

    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert accepted == []
    assert failures == []
    assert rejected[0].accepted is False
    assert "validated statement" in rejected[0].limitations[-1]


def test_validation_failure_does_not_abort_later_paper_and_resume_skips_completed(tmp_path) -> None:
    bad_source = citation(source_id="pmid:bad", title="Bad title", abstract="Missing abstract.")
    good_source = citation(source_id="pmid:good", title="Good title", abstract="Good abstract.")
    claims = [
        proposed_claim(claim_id="bad-claim", source_id=bad_source.id),
        proposed_claim(claim_id="good-claim", source_id=good_source.id),
    ]
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[bad_source, good_source],
        claims=claims,
        bodies={
            bad_source.id: "No abstract and no section boundary.",
            good_source.id: body(good_source.title, good_source.abstract or ""),
        },
    )
    provider = FakeValidationProvider([{"decisions": [decision("good-claim")]}])

    accepted, _, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=False,
    )

    assert [item.proposed_claim_id for item in accepted] == ["good-claim"]
    assert failures[0]["citation_id"] == "pmid:bad"
    assert failures[0]["failure_code"] == "body_sanitization_failed"
    preview_path = run_dir / failures[0]["sanitized_body_preview_path"]
    assert preview_path.name == "sanitized_body_preview.txt"
    assert "No abstract and no section boundary" in preview_path.read_text(encoding="utf-8")
    assert len(provider.prompts) == 1

    validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=provider,
        resume=True,
    )
    assert len(provider.prompts) == 1


def test_resume_uses_citation_and_claim_id_when_claim_ids_are_reused(tmp_path) -> None:
    first_source = citation(source_id="pmid:first", title="First title", abstract="First abstract.")
    second_source = citation(source_id="pmid:second", title="Second title", abstract="Second abstract.")
    claims = [
        proposed_claim(claim_id="claim-1", source_id=first_source.id),
        proposed_claim(claim_id="claim-1", source_id=second_source.id),
    ]
    run_dir = write_validation_inputs(
        tmp_path,
        citations=[first_source, second_source],
        claims=claims,
        bodies={
            first_source.id: "No abstract and no section boundary.",
            second_source.id: body(second_source.title, second_source.abstract or ""),
        },
    )
    first_provider = FakeValidationProvider([{"decisions": [decision("claim-1")]}])

    validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=first_provider,
        resume=False,
    )
    assert len(first_provider.prompts) == 1

    first_record = json.loads((run_dir / "raw_texts.json").read_text(encoding="utf-8"))[0]
    (run_dir / first_record["text_path"]).write_text(
        body(first_source.title, first_source.abstract or ""),
        encoding="utf-8",
    )
    second_provider = FakeValidationProvider([{"decisions": [decision("claim-1")]}])
    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        raw_texts_manifest_path=run_dir / "raw_texts.json",
        raw_texts_dir=run_dir / "raw_texts",
        out_dir=run_dir,
        provider=second_provider,
        resume=True,
    )

    assert len(second_provider.prompts) == 1
    assert len(accepted) == 2
    assert rejected == []
    assert failures == []
    assert {claim.citation_id for claim in accepted} == {"pmid:first", "pmid:second"}


def test_cli_validate_claims_infers_sibling_artifacts(monkeypatch, tmp_path) -> None:
    run_dir = write_validation_inputs(tmp_path)
    provider = FakeValidationProvider([{"decisions": [decision("claim-1")]}])
    monkeypatch.setattr("sipz_agent.cli.resolve_model_config", lambda **_: object())
    monkeypatch.setattr("sipz_agent.cli.create_llm_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "validate-claims",
            "--proposed-claims",
            str(run_dir / "proposed_claims.json"),
            "--no-resume",
        ],
    )

    assert result.exit_code == 0
    assert "[1/1] Validating 1 claim (DOI:" in result.stdout
    assert "1 accepted, 0 rejected, 0 paper failures" in result.stdout
    assert json.loads((run_dir / "validated_claims.json").read_text())[0][
        "proposed_claim_id"
    ] == "claim-1"
