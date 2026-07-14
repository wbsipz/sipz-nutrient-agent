import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from sipz_agent.cli_ingredients import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.ingredient_synthesis import (
    INGREDIENT_HEALTH_REPORT_COLUMNS,
    legacy_from_ingredient_claim,
    synthesize_ingredient_report,
)
from sipz_agent.core.models import HeuristicProvider
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, SupportingQuote, ValidatedClaim
from sipz_agent.schemas.ingredients import (
    IngredientPacket,
    IngredientPacketInput,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)


class FailingPolishProvider:
    def complete_json(self, prompt, adapter):
        _ = prompt
        _ = adapter
        raise RuntimeError("polish_failed")


class FakePolishProvider:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python(self.response)


def lookup_row(
    *,
    row_id: str = "ingredient-1",
    name: str = "tart cherry juice",
) -> dict[str, str]:
    return {
        "canonical_beverage_id": row_id,
        "canonical_beverage_name": name,
        "health_effect_positive": "Existing positive summary.",
        "health_effect_negative": "Existing negative summary.",
        "health_effect_positive_embedding": "positive-embedding",
        "health_effect_negative_embedding": "negative-embedding",
        "health_effect_positive_tags": json.dumps(
            [{"tag": "existing_positive", "score": 0.4, "confidence": 0.5}]
        ),
        "health_effect_negative_tags": json.dumps(
            [{"tag": "existing_negative", "score": 0.4, "confidence": 0.5}]
        ),
        "payload_nutrients_count": "1",
        "matched_nutrients_count": "1",
        "skipped_keys_count": "0",
        "missing_summary_count": "0",
        "missing_amount_count": "0",
        "source": "",
        "embedding_model": "embedding-model",
        "embedded_at": "2026-01-01T00:00:00+00:00",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def write_lookup(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INGREDIENT_HEALTH_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def proposed_claim(
    *,
    claim_id: str = "claim-1",
    direction: str = "beneficial",
    effect_slug: str = "exercise_recovery",
) -> ProposedClaim:
    return ProposedClaim(
        id=claim_id,
        nutrient_name="tart cherry",
        citation_id="pmid:1",
        statement="Tart cherry juice may support recovery after exercise.",
        proposed_effect_slug=effect_slug,
        proposed_effect_label="Exercise recovery",
        compound="tart cherry",
        effect="exercise recovery",
        direction=direction,
        population="healthy adults",
        dose_or_exposure="60 mL concentrate daily",
        outcome="muscle soreness",
        study_type="randomized trial",
        limitations=["Small study."],
        evidence_type="human_clinical",
        intake_route="oral",
        exposure_category="natural_food_level",
        natural_concentration_relevance="Relevant to tart cherry concentrate.",
        pharmaceutical_centered=False,
        concentration_notes="Juice concentrate form.",
    )


def validated_claim(
    *,
    claim_id: str = "claim-1",
    accepted: bool = True,
) -> ValidatedClaim:
    return ValidatedClaim(
        effect_row_id=f"effect-{claim_id}",
        proposed_claim_id=claim_id,
        citation_id="pmid:1",
        verdict="supported_with_limitations" if accepted else "unsupported",
        support_level="human_rct" if accepted else "unsupported",
        claim_scope="Healthy adults consuming tart cherry concentrate after exercise.",
        validated_statement=(
            "Tart cherry concentrate may reduce muscle soreness after exercise in healthy adults."
        ),
        validator_reasoning="The body supports a limited human oral claim.",
        supporting_quotes=[
            SupportingQuote(
                quote="Muscle soreness was lower after tart cherry concentrate.",
                section="Results",
                reason="Reports the outcome.",
                match_status="exact",
            )
        ],
        limitations=["Evidence is specific to concentrate and exercise recovery."],
        accepted=accepted,
    )


def proposed_ingredient_claim(**overrides) -> ProposedIngredientClaim:
    payload = {
        "id": "claim-1",
        "ingredient_name": "acai",
        "ingredient_form": "pulp",
        "citation_id": "pmid:1",
        "statement": "Acai pulp may improve oxidative stress markers.",
        "proposed_effect_slug": "reduced_oxidative_stress",
        "proposed_effect_label": "Reduced oxidative stress",
        "effect": "oxidative stress",
        "claim_direction": "beneficial",
        "population": "healthy adults",
        "oral_exposure": "consumed orally as acai pulp",
        "dose_or_serving": "200 g per day",
        "food_matrix": "acai pulp",
        "outcome": "oxidative stress markers",
        "study_type": "human observational",
        "limitations": ["Small study."],
        "evidence_type": "human_observational",
        "exposure_category": "powder_or_concentrate",
        "concentration_notes": "Pulp food form.",
        "claim_applies_to_group_members": "direct_only",
    }
    payload.update(overrides)
    return ProposedIngredientClaim.model_validate(payload)


def validated_ingredient_claim(**overrides) -> ValidatedIngredientClaim:
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
                quote="Acai pulp improved oxidative stress markers.",
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


def source() -> CandidateCitation:
    return CandidateCitation(
        id="pmid:1",
        title="Tart cherry and exercise recovery",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        doi="10.1000/tart",
        pmid="1",
        year=2025,
        source="pubmed",
        retrieval_query="tart cherry human",
    )


def write_run(
    tmp_path: Path,
    *,
    proposed: list[ProposedClaim] | None = None,
    validated: list[ValidatedClaim] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    packet = Packet(
        run_id="run",
        input=PacketInput(nutrient_name="tart cherry", depth="standard", demo=False),
        model=PacketModel(provider="deepseek", model_name="deepseek-chat"),
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
        counts=PacketCounts(
            candidate_citations=1,
            proposed_claims=1,
            validated_claims=1,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "packet.json", packet.model_dump(mode="json"))
    write_json(
        run_dir / "proposed_claims.json",
        [item.model_dump(mode="json") for item in (proposed or [proposed_claim()])],
    )
    write_json(
        run_dir / "validated_claims.json",
        [item.model_dump(mode="json") for item in (validated or [validated_claim()])],
    )
    write_json(run_dir / "sources.json", [source().model_dump(mode="json")])
    return run_dir


def write_ingredient_packet_for_run(
    run_dir: Path,
    *,
    ingredient_name: str = "acai",
    canonical_beverage_id: str | None = "ingredient-1",
    canonical_beverage_name: str | None = "acai juice",
) -> None:
    packet = IngredientPacket(
        run_id="run",
        input=IngredientPacketInput(
            ingredient_name=ingredient_name,
            canonical_search_name=ingredient_name,
            ingredient_form="juice_or_beverage",
            canonical_beverage_id=canonical_beverage_id,
            canonical_beverage_name=canonical_beverage_name,
            depth="standard",
            demo=False,
            retrieval_queries=[f"{ingredient_name} human health"],
        ),
        model=PacketModel(provider="deepseek", model_name="deepseek-chat"),
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
        counts=PacketCounts(
            candidate_citations=1,
            proposed_claims=1,
            validated_claims=1,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "ingredient_packet.json", packet.model_dump(mode="json"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_synthesize_ingredient_report_writes_lookup_shaped_export(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])
    out_dir = tmp_path / "out"

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=out_dir,
        provider=HeuristicProvider(),
        canonical_beverage_id="ingredient-1",
    )

    columns, rows = read_csv(out_dir / "ingredient_health_report_rows.updated.csv")
    assert columns == INGREDIENT_HEALTH_REPORT_COLUMNS
    assert rows[0]["canonical_beverage_id"] == "ingredient-1"
    assert rows[0]["health_effect_positive"].startswith("Tart cherry concentrate may reduce")
    assert rows[0]["health_effect_negative"] == "Existing negative summary."
    assert rows[0]["health_effect_positive_embedding"] == "positive-embedding"
    assert rows[0]["source"] == "literature_agent_v1"
    assert json.loads(rows[0]["health_effect_positive_tags"])[0]["tag"] == "exercise_recovery"
    assert result.claim_sources[0]["supporting_quotes"][0]["match_status"] == "exact"
    assert (out_dir / "ingredient_claim_sources.json").exists()
    assert (out_dir / "ingredient_synthesis_notes.json").exists()
    assert (out_dir / "ingredient_rejected_synthesis_items.json").exists()


def test_legacy_from_ingredient_claim_treats_pulp_as_food_level() -> None:
    claim = proposed_ingredient_claim()

    legacy = legacy_from_ingredient_claim(claim)

    assert legacy.exposure_category == "natural_food_level"
    assert legacy.natural_concentration_relevance == "Ingredient-level evidence."
    assert legacy.supplement_level_relevance is None


def test_synthesize_ingredient_report_uses_only_accepted_claims(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path,
        proposed=[
            proposed_claim(claim_id="claim-1", effect_slug="exercise_recovery"),
            proposed_claim(claim_id="claim-2", effect_slug="unsupported_claim"),
        ],
        validated=[
            validated_claim(claim_id="claim-1", accepted=True),
            validated_claim(claim_id="claim-2", accepted=False),
        ],
    )
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=HeuristicProvider(),
        canonical_beverage_id="ingredient-1",
    )

    assert len(result.claim_sources) == 1
    assert result.claim_sources[0]["proposed_claim_id"] == "claim-1"
    assert "unsupported_claim" not in result.updated_row["health_effect_positive_tags"]


def test_synthesize_ingredient_report_routes_harmful_claims_to_negative(
    tmp_path: Path,
) -> None:
    run_dir = write_run(
        tmp_path,
        proposed=[
            proposed_claim(
                claim_id="claim-1",
                direction="harmful",
                effect_slug="digestive_discomfort",
            )
        ],
        validated=[validated_claim(claim_id="claim-1", accepted=True)],
    )
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=HeuristicProvider(),
        canonical_beverage_id="ingredient-1",
    )

    assert result.updated_row["health_effect_positive"] == "Existing positive summary."
    assert result.updated_row["health_effect_negative"].startswith(
        "Tart cherry concentrate may reduce"
    )
    assert json.loads(result.updated_row["health_effect_negative_tags"])[0]["tag"] == (
        "digestive_discomfort"
    )


def test_synthesize_ingredient_report_requires_disambiguation_for_broad_name(
    tmp_path: Path,
) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    write_lookup(
        lookup,
        [
            lookup_row(row_id="ingredient-1", name="tart cherry juice"),
            lookup_row(row_id="ingredient-2", name="tart cherry juice drink"),
        ],
    )

    result = CliRunner().invoke(
        app,
        [
            "synthesize-report",
            "--run",
            str(run_dir),
            "--lookup",
            str(lookup),
            "--provider",
            "heuristic",
        ],
    )

    assert result.exit_code != 0
    assert "ingredient_lookup_row_not_found:tart cherry" in result.output
    assert "tart cherry juice" in result.output


def test_synthesize_ingredient_report_uses_packet_canonical_id_by_default(
    tmp_path: Path,
) -> None:
    run_dir = write_run(tmp_path)
    write_ingredient_packet_for_run(
        run_dir,
        ingredient_name="acai",
        canonical_beverage_id="ingredient-1",
        canonical_beverage_name="acai juice",
    )
    lookup = tmp_path / "lookup.csv"
    write_lookup(
        lookup,
        [
            lookup_row(row_id="ingredient-1", name="acai juice"),
            lookup_row(row_id="ingredient-2", name="acai smoothie"),
        ],
    )

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=HeuristicProvider(),
    )

    assert result.updated_row["canonical_beverage_id"] == "ingredient-1"
    assert result.notes["target_resolution"] == {
        "method": "canonical_beverage_id",
        "value": "ingredient-1",
    }


def test_synthesize_ingredient_report_cli_writes_artifacts(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    out_dir = tmp_path / "export"
    write_lookup(lookup, [lookup_row()])

    result = CliRunner().invoke(
        app,
        [
            "synthesize-report",
            "--run",
            str(run_dir),
            "--lookup",
            str(lookup),
            "--out",
            str(out_dir),
            "--canonical-beverage-id",
            "ingredient-1",
            "--provider",
            "heuristic",
        ],
    )

    assert result.exit_code == 0
    assert "Ingredient report synthesis complete." in result.output
    assert (out_dir / "ingredient_health_report_rows.updated.csv").exists()
    assert (out_dir / "ingredient_claim_sources.json").exists()


def test_synthesize_ingredient_report_cli_defaults_to_run_directory(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])

    result = CliRunner().invoke(
        app,
        [
            "synthesize-report",
            "--run",
            str(run_dir),
            "--lookup",
            str(lookup),
            "--canonical-beverage-id",
            "ingredient-1",
            "--provider",
            "heuristic",
        ],
    )

    assert result.exit_code == 0
    assert (run_dir / "ingredient_health_report_rows.updated.csv").exists()
    assert (run_dir / "ingredient_claim_sources.json").exists()


def test_synthesize_ingredient_report_falls_back_when_polish_fails(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=FailingPolishProvider(),
        canonical_beverage_id="ingredient-1",
    )

    assert result.notes["llm_polish"]["used"] is False
    assert "polish_failed" in result.notes["llm_polish"]["reason"]
    assert result.updated_row["health_effect_positive"].startswith(
        "Tart cherry concentrate may reduce"
    )


def test_synthesize_ingredient_report_uses_valid_polish(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row()])
    provider = FakePolishProvider(
        {
            "health_effect_positive": "Polished positive tart cherry recovery summary.",
            "health_effect_negative": "Existing negative summary.",
            "health_effect_positive_tags": [
                {"tag": "muscle_recovery", "score": 0.75, "confidence": 0.8}
            ],
            "health_effect_negative_tags": [
                {"tag": "existing_negative", "score": 0.4, "confidence": 0.5}
            ],
            "synthesis_notes": "Polished without adding claims.",
        }
    )

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=provider,
        canonical_beverage_id="ingredient-1",
    )

    assert provider.prompts
    assert result.notes["llm_polish"]["used"] is True
    assert result.updated_row["health_effect_positive"] == (
        "Polished positive tart cherry recovery summary."
    )


def test_synthesize_ingredient_report_prefers_audited_ingredient_claims(
    tmp_path: Path,
) -> None:
    run_dir = write_run(tmp_path)
    write_ingredient_packet_for_run(
        run_dir,
        ingredient_name="acai",
        canonical_beverage_id="ingredient-1",
        canonical_beverage_name="acai juice",
    )
    write_json(
        run_dir / "proposed_ingredient_claims.json",
        [
            proposed_ingredient_claim(
                id="claim-1",
                proposed_effect_slug="oxidative_stress",
                proposed_effect_label="Oxidative stress",
            ).model_dump(mode="json"),
            proposed_ingredient_claim(
                id="claim-2",
                statement="Acai extract may affect cholesterol markers.",
                proposed_effect_slug="cholesterol_markers",
                proposed_effect_label="Cholesterol markers",
                effect="cholesterol markers",
            ).model_dump(mode="json"),
        ],
    )
    write_json(
        run_dir / "validated_ingredient_claims.json",
        [
            validated_ingredient_claim(
                proposed_ingredient_claim_id="claim-1",
                effect_row_id="effect-1",
            ).model_dump(mode="json"),
            validated_ingredient_claim(
                proposed_ingredient_claim_id="claim-2",
                effect_row_id="effect-2",
                validated_statement="Acai extract may affect cholesterol markers.",
            ).model_dump(mode="json"),
        ],
    )
    write_json(
        run_dir / "audited_ingredient_claims.json",
        [
            validated_ingredient_claim(
                proposed_ingredient_claim_id="claim-1",
                effect_row_id="effect-1",
            ).model_dump(mode="json")
        ],
    )
    lookup = tmp_path / "lookup.csv"
    write_lookup(lookup, [lookup_row(row_id="ingredient-1", name="acai juice")])

    result = synthesize_ingredient_report(
        run_dir=run_dir,
        lookup_path=lookup,
        out_dir=tmp_path / "out",
        provider=HeuristicProvider(),
    )

    assert len(result.claim_sources) == 1
    assert result.claim_sources[0]["proposed_claim_id"] == "claim-1"
    assert result.notes["used_audited_claims"] is True
    assert result.notes["claim_source_artifact"] == "audited_ingredient_claims.json"
    assert "cholesterol_markers" not in result.updated_row["health_effect_positive_tags"]
