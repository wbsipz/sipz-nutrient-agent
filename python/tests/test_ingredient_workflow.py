import csv
import json
from pathlib import Path

import orjson

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.ingredient_synthesis import (
    INGREDIENT_HEALTH_REPORT_COLUMNS,
    synthesize_ingredient_report,
)
from sipz_agent.core.ingredient_workflow import (
    build_entity_plan,
    legacy_from_ingredient_claim,
    propose_ingredient_claims_from_raw_texts,
    rank_ingredient_candidates,
    run_ingredient_study,
    validate_ingredient_claims,
    write_ingredient_packet,
)
from sipz_agent.core.models import HeuristicProvider
from sipz_agent.core.retrieval import CandidatePageOutput
from sipz_agent.schemas.artifacts import PacketCounts, PacketModel
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import SupportingQuote
from sipz_agent.schemas.ingredients import (
    IngredientPacket,
    IngredientPacketInput,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)
from sipz_agent.schemas.raw_texts import RawTextRecord


class FakeIngredientClaimProvider:
    def __init__(self, claims: list[dict]) -> None:
        self.claims = claims
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python({"claims": self.claims, "skipped_reason": None})


class FakeIngredientValidationProvider:
    def __init__(self, decisions: list[dict]) -> None:
        self.responses = list(decisions)
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if "decisions" in response or ("supporting_quotes" in response and "proposed_claim_id" not in response):
            return adapter.validate_python(response)
        return adapter.validate_python({"decisions": [response]})


def lookup_row(
    *,
    row_id: str = "ingredient-1",
    name: str = "tart cherry juice",
) -> dict[str, str]:
    return {
        "canonical_beverage_id": row_id,
        "canonical_beverage_name": name,
        "health_effect_positive": "",
        "health_effect_negative": "",
        "health_effect_positive_embedding": "",
        "health_effect_negative_embedding": "",
        "health_effect_positive_tags": "",
        "health_effect_negative_tags": "",
        "payload_nutrients_count": "0",
        "matched_nutrients_count": "0",
        "skipped_keys_count": "0",
        "missing_summary_count": "0",
        "missing_amount_count": "0",
        "source": "",
        "embedding_model": "",
        "embedded_at": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def write_lookup(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INGREDIENT_HEALTH_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def citation() -> CandidateCitation:
    return CandidateCitation(
        id="pmid:1",
        title="Tart cherry juice and exercise recovery",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        doi="10.1000/tart",
        pmid="1",
        year=2025,
        source="pubmed",
        retrieval_query="tart cherry juice human health",
        abstract="Abstract text should not be used for validation.",
    )


def ingredient_packet() -> IngredientPacket:
    return IngredientPacket(
        run_id="run",
        input=IngredientPacketInput(
            ingredient_name="tart cherry juice",
            canonical_search_name="tart cherry",
            ingredient_form="juice_or_beverage",
            canonical_beverage_id="ingredient-1",
            canonical_beverage_name="tart cherry juice",
            depth="standard",
            demo=False,
            retrieval_queries=["tart cherry juice human health"],
        ),
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


def claim_payload(**overrides) -> dict:
    payload = {
        "id": "claim-1",
        "ingredient_name": "tart cherry juice",
        "ingredient_form": "juice",
        "citation_id": "pmid:1",
        "statement": "Oral tart cherry juice may reduce muscle soreness after exercise.",
        "proposed_effect_slug": "exercise_recovery",
        "proposed_effect_label": "Exercise recovery",
        "effect": "muscle soreness",
        "claim_direction": "beneficial",
        "population": "healthy adults",
        "oral_exposure": "consumed orally as tart cherry juice",
        "dose_or_serving": "240 mL/day",
        "food_matrix": "juice",
        "outcome": "muscle soreness",
        "study_type": "randomized trial",
        "limitations": ["Small human trial."],
        "evidence_type": "human_clinical",
        "exposure_category": "juice",
        "concentration_notes": "Juice form.",
        "claim_applies_to_group_members": "direct_only",
    }
    payload.update(overrides)
    return payload


def proposed_claim() -> ProposedIngredientClaim:
    return ProposedIngredientClaim.model_validate(claim_payload())


def validated_claim() -> ValidatedIngredientClaim:
    return ValidatedIngredientClaim(
        effect_row_id="effect-1",
        proposed_ingredient_claim_id="claim-1",
        citation_id="pmid:1",
        verdict="supported_with_limitations",
        support_level="human_rct",
        claim_scope="Healthy adults consuming tart cherry juice after exercise.",
        validated_statement="Tart cherry juice may reduce muscle soreness after exercise.",
        validator_reasoning="The body reports the stated oral human outcome.",
        supporting_quotes=[
            SupportingQuote(
                quote="Muscle soreness was lower after oral tart cherry juice intake.",
                section="Results",
                reason="Reports the outcome.",
                match_status="exact",
            )
        ],
        limitations=["Evidence is specific to juice intake after exercise."],
        accepted=True,
    )


def write_ingredient_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw_texts"
    raw_dir.mkdir(parents=True)
    write_ingredient_packet(run_dir, ingredient_packet())
    write_json(run_dir / "sources.json", [citation().model_dump(mode="json")])
    body = (
        "1. Introduction Tart cherry juice was evaluated in adults. "
        "Methods Healthy adults consumed oral tart cherry juice. "
        "Results Muscle soreness was lower after oral tart cherry juice intake. "
        "Discussion The result requires confirmation. "
        + ("Additional body evidence. " * 60)
    )
    (raw_dir / "001.txt").write_text(body, encoding="utf-8")
    record = RawTextRecord(
        source_id="pmid:1",
        title=citation().title,
        doi=citation().doi,
        pmid=citation().pmid,
        url=str(citation().url),
        status="full_text_found",
        retrieval_method="publisher_page",
        text_path="raw_texts/001.txt",
        text_char_count=len(body),
    )
    write_json(run_dir / "raw_texts.json", [record.model_dump(mode="json")])
    (run_dir / "audit_log.jsonl").write_text("", encoding="utf-8")
    return run_dir


def test_ingredient_study_writes_first_class_and_compat_artifacts(monkeypatch, tmp_path: Path) -> None:
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])
    retrieval_query_calls = []

    def fake_candidate_page(*args, **kwargs):
        page_index = kwargs["page_index"]
        retrieval_query_calls.append(list(kwargs["queries"]))
        return CandidatePageOutput(
            citations=[citation()] if page_index == 0 else [],
            raw_candidate_count=1 if page_index == 0 else 0,
        )

    monkeypatch.setattr(
        "sipz_agent.core.ingredient_workflow.find_live_candidate_page",
        fake_candidate_page,
    )

    result = run_ingredient_study(
        ingredient_name="tart cherry juice",
        lookup_path=lookup,
        depth="light",
        out_dir=tmp_path / "ingredient_runs",
        provider="heuristic",
    )

    run_dir = result.run_dir
    packet = json.loads((run_dir / "ingredient_packet.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "ingredient_entity_plan.json").read_text(encoding="utf-8"))

    assert run_dir.parent.name == "ingredient_runs"
    assert packet["input"]["ingredient_name"] == "tart cherry juice"
    assert packet["input"]["canonical_search_name"] == "tart cherry"
    assert packet["input"]["ingredient_form"] == "juice_or_beverage"
    assert packet["input"]["canonical_beverage_id"] == "ingredient-1"
    assert "sour cherry" in plan["aliases"]
    assert "Montmorency cherry" in plan["aliases"]
    assert "Prunus cerasus" in plan["aliases"]
    assert "juice" in plan["included_forms"]
    assert "concentrate" in plan["included_forms"]
    assert "capsule extract unless explicitly ingredient-relevant" in plan["excluded_forms"]
    assert "tart cherry concentrate" in plan["query_terms"]
    assert "in vitro" in plan["excluded_query_terms"]
    assert any("Montmorency cherry" in query for query in plan["retrieval_queries"])
    assert any("tart cherry concentrate" in query for query in plan["retrieval_queries"])
    assert any("NOT" in query for query in plan["retrieval_queries"])
    assert any("critical review" in query and "NOT" not in query for query in plan["retrieval_queries"])
    assert retrieval_query_calls[0] == plan["retrieval_queries"]
    assert packet["input"]["retrieval_queries"] == plan["retrieval_queries"]
    assert plan["retrieval_queries"]
    assert (run_dir / "packet.json").exists()
    assert (run_dir / "sources.json").exists()
    assert (run_dir / "effects.csv").exists()


def test_ingredient_study_can_search_as_run_target_with_representative_row(
    monkeypatch, tmp_path: Path
) -> None:
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row(row_id="cheese-1", name="panela cheese")])
    retrieval_query_calls = []

    def fake_candidate_page(*args, **kwargs):
        retrieval_query_calls.append(list(kwargs["queries"]))
        return CandidatePageOutput(citations=[], raw_candidate_count=0)

    monkeypatch.setattr(
        "sipz_agent.core.ingredient_workflow.find_live_candidate_page",
        fake_candidate_page,
    )

    result = run_ingredient_study(
        ingredient_name="cheese",
        lookup_path=lookup,
        depth="light",
        out_dir=tmp_path / "ingredient_runs",
        provider="heuristic",
        canonical_beverage_id="cheese-1",
        search_as_ingredient_name=True,
    )

    packet = json.loads(
        (result.run_dir / "ingredient_packet.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (result.run_dir / "ingredient_entity_plan.json").read_text(encoding="utf-8")
    )

    assert packet["input"]["ingredient_name"] == "cheese"
    assert packet["input"]["canonical_search_name"] == "cheese"
    assert packet["input"]["canonical_beverage_id"] == "cheese-1"
    assert packet["input"]["canonical_beverage_name"] == "panela cheese"
    assert plan["canonical_search_name"] == "cheese"
    assert "panela cheese" in plan["query_terms"]
    assert retrieval_query_calls[0] == plan["retrieval_queries"]


def test_acai_entity_plan_includes_spelling_species_and_food_forms() -> None:
    plan = build_entity_plan(
        ingredient_name="acai",
        canonical_search_name_value="acai",
        row=lookup_row(name="acai"),
        relationship="same_ingredient",
        ingredient_form="whole_food",
        provider=HeuristicProvider(),
        use_llm_enrichment=False,
    )

    assert "açaí" in plan.aliases
    assert "Euterpe oleracea" in plan.aliases
    assert "pulp" in plan.included_forms
    assert "juice" in plan.included_forms
    assert "acai pulp" in plan.query_terms
    assert "acai juice" in plan.query_terms
    assert "in vitro" in plan.excluded_query_terms
    assert "mouse" in plan.excluded_query_terms
    assert any("Euterpe oleracea" in query for query in plan.retrieval_queries)
    assert any("acai pulp" in query for query in plan.retrieval_queries)
    assert any("NOT" in query for query in plan.retrieval_queries)
    review_queries = [query for query in plan.retrieval_queries if "critical review" in query]
    assert review_queries
    assert all("NOT" not in query for query in review_queries)
    assert any("review" in query for query in review_queries)
    assert any("Euterpe oleracea" in query for query in review_queries)


def test_acai_review_query_does_not_exclude_mixed_preclinical_review_terms() -> None:
    plan = build_entity_plan(
        ingredient_name="acai",
        canonical_search_name_value="acai",
        row=lookup_row(name="acai juice"),
        relationship="juice_or_beverage_form",
        ingredient_form="juice_or_beverage",
        provider=HeuristicProvider(),
        use_llm_enrichment=False,
    )

    review_query = next(query for query in plan.retrieval_queries if "critical review" in query)

    assert '"acai"' in review_query
    assert '"Euterpe oleracea"' in review_query
    assert '"human"' in review_query
    assert '"health"' in review_query
    assert '"critical review"' in review_query
    assert 'NOT ("in vitro"' not in review_query
    assert '"in vitro"' not in review_query


def test_ingredient_candidate_ranking_boosts_human_reviews_and_downranks_irrelevant_sources() -> None:
    chemistry = CandidateCitation(
        id="doi:chemistry",
        title="Chemical composition and extraction processing of acai seed phytochemicals",
        doi="10.1000/chemistry",
        source="crossref",
        retrieval_query="acai",
        abstract="This paper evaluates extraction and chemical composition.",
    )
    animal = CandidateCitation(
        id="doi:animal",
        title="Acai extract effects in rat animal models",
        doi="10.1000/animal",
        source="openalex",
        retrieval_query="acai",
        abstract="Rat and mouse animal model data are reported.",
    )
    review = CandidateCitation(
        id="pmid:review",
        title="Systematic review and meta-analysis of acai consumption in humans",
        pmid="123",
        source="pubmed",
        retrieval_query="acai",
        abstract="A systematic review of human dietary acai intake and clinical outcomes.",
    )

    ranked = rank_ingredient_candidates([chemistry, animal, review])

    assert ranked[0].id == "pmid:review"
    assert ranked[-1].id in {"doi:chemistry", "doi:animal"}
    assert "Ingredient retrieval ranking:" in ranked[0].selection_reason
    assert "boosts=systematic_review" in ranked[0].selection_reason
    assert any("penalties=" in citation.selection_reason for citation in ranked[1:])


def test_ingredient_claim_proposal_writes_native_and_compat_artifacts(tmp_path: Path) -> None:
    run_dir = write_ingredient_run(tmp_path)
    provider = FakeIngredientClaimProvider([claim_payload()])

    claims = propose_ingredient_claims_from_raw_texts(
        run_dir=run_dir,
        provider=provider,
        workers=2,
    )

    native = json.loads((run_dir / "proposed_ingredient_claims.json").read_text(encoding="utf-8"))
    legacy = json.loads((run_dir / "proposed_claims.json").read_text(encoding="utf-8"))
    packet = json.loads((run_dir / "ingredient_packet.json").read_text(encoding="utf-8"))

    assert len(claims) == 1
    assert native[0]["ingredient_name"] == "tart cherry juice"
    assert native[0]["oral_exposure"] == "consumed orally as tart cherry juice"
    assert native[0]["dose_or_serving"] == "240 mL/day"
    assert native[0]["food_matrix"] == "juice"
    assert native[0]["exposure_category"] == "juice_or_beverage"
    assert native[0]["claim_applies_to_group_members"] == "similar_forms"
    assert legacy[0]["nutrient_name"] == "tart cherry juice"
    assert legacy[0]["intake_route"] == "oral"
    assert packet["counts"]["proposed_claims"] == 1


def test_ingredient_claim_proposal_filters_preclinical_review_claims(tmp_path: Path) -> None:
    run_dir = write_ingredient_run(tmp_path)
    provider = FakeIngredientClaimProvider(
        [
            claim_payload(
                statement=(
                    "Acai juice polyphenols may reduce intracellular lipid accumulation "
                    "in adipocytes, suggesting potential anti-obesity effects."
                ),
                study_type="in vitro",
                evidence_type="human_mechanistic",
                oral_exposure="frozen concentrated acai juice polyphenols",
                dose_or_serving="2.5, 5, and 10 ug GAE/mL",
                population="general population (in vitro study using 3T3-L1 adipocytes)",
                outcome="intracellular lipid accumulation in adipocytes",
                limitations=[
                    "in vitro study using mouse cell line",
                    "doses may not reflect human dietary exposure",
                ],
            )
        ]
    )

    claims = propose_ingredient_claims_from_raw_texts(run_dir=run_dir, provider=provider)

    native = json.loads((run_dir / "proposed_ingredient_claims.json").read_text(encoding="utf-8"))
    legacy = json.loads((run_dir / "proposed_claims.json").read_text(encoding="utf-8"))
    packet = json.loads((run_dir / "ingredient_packet.json").read_text(encoding="utf-8"))

    assert claims == []
    assert native == []
    assert legacy == []
    assert packet["counts"]["proposed_claims"] == 0
    assert "review only supports a statement with in vitro" in provider.prompts[0]


def test_ingredient_claim_proposal_tolerates_boolean_oral_exposure(
    tmp_path: Path,
) -> None:
    run_dir = write_ingredient_run(tmp_path)
    provider = FakeIngredientClaimProvider(
        [
            claim_payload(
                oral_exposure=False,
                dose_or_serving=False,
                food_matrix=False,
            )
        ]
    )

    claims = propose_ingredient_claims_from_raw_texts(run_dir=run_dir, provider=provider)

    native = json.loads((run_dir / "proposed_ingredient_claims.json").read_text(encoding="utf-8"))
    legacy = json.loads((run_dir / "proposed_claims.json").read_text(encoding="utf-8"))

    assert claims == []
    assert native == []
    assert legacy == []


def test_ingredient_workflow_legacy_conversion_treats_pulp_as_food_level() -> None:
    claim = ProposedIngredientClaim.model_validate(
        claim_payload(
            ingredient_name="acai",
            ingredient_form="pulp",
            statement="Acai pulp may improve oxidative stress markers.",
            oral_exposure="consumed orally as acai pulp",
            dose_or_serving="200 g per day",
            food_matrix="acai pulp",
            exposure_category="powder_or_concentrate",
            concentration_notes="Pulp food form.",
        )
    )

    legacy = legacy_from_ingredient_claim(claim)

    assert legacy.exposure_category == "natural_food_level"
    assert legacy.natural_concentration_relevance == "Ingredient-level evidence."
    assert legacy.supplement_level_relevance is None


def test_ingredient_validation_writes_grounded_native_and_compat_artifacts(tmp_path: Path) -> None:
    run_dir = write_ingredient_run(tmp_path)
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [proposed_claim().model_dump(mode="json")],
    )
    provider = FakeIngredientValidationProvider(
        [
            {
                "proposed_claim_id": "claim-1",
                "verdict": "supported_with_limitations",
                "validated_statement": "Tart cherry juice may reduce muscle soreness after exercise.",
                "support_level": "human_rct",
                "claim_scope": "Healthy adults consuming tart cherry juice after exercise.",
                "supporting_quotes": [
                    {
                        "quote": "Muscle soreness was lower after oral tart cherry juice intake.",
                        "section": "Results",
                        "reason": "Reports the outcome.",
                    }
                ],
                "limitations": ["Small human trial."],
                "reasoning": "The body reports the stated human outcome.",
            }
        ]
    )

    accepted, rejected, failures = validate_ingredient_claims(
        run_dir=run_dir,
        provider=provider,
        workers=2,
    )

    native = json.loads((run_dir / "validated_ingredient_claims.json").read_text(encoding="utf-8"))
    legacy = json.loads((run_dir / "validated_claims.json").read_text(encoding="utf-8"))
    packet = json.loads((run_dir / "ingredient_packet.json").read_text(encoding="utf-8"))

    assert len(accepted) == 1
    assert rejected == []
    assert failures == []
    assert native[0]["proposed_ingredient_claim_id"] == "claim-1"
    assert native[0]["supporting_quotes"][0]["match_status"] == "exact"
    assert legacy[0]["proposed_claim_id"] == "claim-1"
    assert packet["counts"]["validated_claims"] == 1


def test_ingredient_validation_keeps_only_grounded_quotes_for_accepted_claim(
    tmp_path: Path,
) -> None:
    run_dir = write_ingredient_run(tmp_path)
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [proposed_claim().model_dump(mode="json")],
    )
    provider = FakeIngredientValidationProvider(
        [
            {
                "proposed_claim_id": "claim-1",
                "verdict": "supported_with_limitations",
                "validated_statement": "Tart cherry juice may reduce muscle soreness after exercise.",
                "support_level": "human_rct",
                "claim_scope": "Healthy adults consuming tart cherry juice after exercise.",
                "supporting_quotes": [
                    {
                        "quote": "Muscle soreness improved after tart cherry juice.",
                        "section": "Results",
                        "reason": "Paraphrased result.",
                    },
                    {
                        "quote": "Muscle soreness was lower after oral tart cherry juice intake.",
                        "section": "Results",
                        "reason": "Exact copied result sentence.",
                    },
                ],
                "limitations": ["Small human trial."],
                "reasoning": "The body reports the stated human outcome.",
            }
        ]
    )

    accepted, rejected, failures = validate_ingredient_claims(run_dir=run_dir, provider=provider)

    native = json.loads((run_dir / "validated_ingredient_claims.json").read_text(encoding="utf-8"))
    legacy = json.loads((run_dir / "validated_claims.json").read_text(encoding="utf-8"))

    assert len(accepted) == 1
    assert rejected == []
    assert failures == []
    assert [quote.quote for quote in accepted[0].supporting_quotes] == [
        "Muscle soreness was lower after oral tart cherry juice intake."
    ]
    assert native[0]["supporting_quotes"] == legacy[0]["supporting_quotes"]
    assert native[0]["supporting_quotes"][0]["match_status"] == "exact"


def test_ingredient_validation_repairs_failed_quote_with_exact_body_copy(tmp_path: Path) -> None:
    run_dir = write_ingredient_run(tmp_path)
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [proposed_claim().model_dump(mode="json")],
    )
    provider = FakeIngredientValidationProvider(
        [
            {
                "proposed_claim_id": "claim-1",
                "verdict": "supported_with_limitations",
                "validated_statement": "Tart cherry juice may reduce muscle soreness after exercise.",
                "support_level": "human_rct",
                "claim_scope": "Healthy adults consuming tart cherry juice after exercise.",
                "supporting_quotes": [
                    {
                        "quote": "Muscle soreness improved after tart cherry juice.",
                        "section": "Results",
                        "reason": "Paraphrased result.",
                    }
                ],
                "limitations": ["Small human trial."],
                "reasoning": "The body reports the stated human outcome.",
            },
            {
                "supporting_quotes": [
                    {
                        "quote": "Muscle soreness was lower after oral tart cherry juice intake.",
                        "section": "Results",
                        "reason": "Exact copied result sentence.",
                    }
                ]
            },
        ]
    )

    accepted, rejected, failures = validate_ingredient_claims(run_dir=run_dir, provider=provider)

    assert rejected == []
    assert failures == []
    assert accepted[0].accepted is True
    assert accepted[0].supporting_quotes[0].quote == (
        "Muscle soreness was lower after oral tart cherry juice intake."
    )
    assert accepted[0].supporting_quotes[0].match_status == "exact"
    assert len(provider.prompts) == 2
    assert "Return exact copied quote from this body excerpt only" in provider.prompts[1]


def test_ingredient_validation_records_sanitization_failure_preview(tmp_path: Path) -> None:
    run_dir = write_ingredient_run(tmp_path)
    (run_dir / "raw_texts" / "001.txt").write_text(
        "Unstructured article shell without abstract match or reliable body sections.",
        encoding="utf-8",
    )
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [proposed_claim().model_dump(mode="json")],
    )
    provider = FakeIngredientValidationProvider([])

    accepted, rejected, failures = validate_ingredient_claims(run_dir=run_dir, provider=provider)

    assert accepted == []
    assert rejected == []
    assert failures[0]["failure_code"] == "body_sanitization_failed"
    preview_path = run_dir / failures[0]["sanitized_body_preview_path"]
    assert preview_path.name == "sanitized_body_preview.txt"
    assert "Unstructured article shell" in preview_path.read_text(encoding="utf-8")


def test_synthesis_prefers_ingredient_artifacts_without_legacy_claim_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_ingredient_packet(run_dir, ingredient_packet())
    write_json(run_dir / "sources.json", [citation().model_dump(mode="json")])
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [proposed_claim().model_dump(mode="json")],
    )
    write_json(
        run_dir / "validated_ingredient_claims.json",
        [validated_claim().model_dump(mode="json")],
    )
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "export",
        provider=HeuristicProvider(),
        canonical_beverage_id="ingredient-1",
    )

    export_path = tmp_path / "export" / "ingredient_health_report_rows.updated.csv"
    with export_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    notes = orjson.loads((tmp_path / "export" / "ingredient_synthesis_notes.json").read_bytes())

    assert result.updated_row["health_effect_positive"].startswith("Tart cherry juice may reduce")
    assert rows[0]["canonical_beverage_id"] == "ingredient-1"
    assert notes["packet_input_name"] == "tart cherry juice"
    assert result.claim_sources[0]["proposed_claim_id"] == "claim-1"
