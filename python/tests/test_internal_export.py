import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from sipz_agent.cli_internal import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.internal_export import (
    EVIDENCE_COLUMNS,
    NEW_EFFECT_COLUMNS,
    export_health_evidence,
)
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, SupportingQuote, ValidatedClaim


class FakeInternalProvider:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete_json(self, prompt, adapter):
        self.prompts.append(prompt)
        return adapter.validate_python(self.response)


def proposed_claim(
    *,
    claim_id: str = "claim-1",
    citation_id: str = "pmid:1",
    entity_name: str = "quercetin",
) -> ProposedClaim:
    return ProposedClaim(
        id=claim_id,
        nutrient_name=entity_name,
        citation_id=citation_id,
        statement="Oral quercetin may support blood pressure in adults.",
        proposed_effect_slug="blood-pressure-support",
        proposed_effect_label="Blood pressure support",
        effect="blood pressure",
        population="adults",
        dose_or_exposure="500 mg/day",
        outcome="systolic blood pressure",
        study_type="human trial",
        evidence_type="human_clinical",
        intake_route="oral",
        exposure_category="supplement_level",
        concentration_notes="The studied dose is supplement-level.",
    )


def validated_claim(
    *,
    claim_id: str = "claim-1",
    citation_id: str = "pmid:1",
    support_level: str = "human_rct",
) -> ValidatedClaim:
    return ValidatedClaim(
        effect_row_id="effect-row-1",
        proposed_claim_id=claim_id,
        citation_id=citation_id,
        verdict="supported_with_limitations",
        support_level=support_level,
        claim_scope="Supplement-level oral intake in adults.",
        validated_statement="Oral quercetin may modestly support blood pressure in adults.",
        validator_reasoning="A human trial supports a limited statement.",
        supporting_quotes=[
            SupportingQuote(
                quote="Blood pressure was lower after supplementation.",
                section="Results",
                reason="Supports the outcome.",
                match_status="exact",
            )
        ],
        limitations=["Small trial."],
        accepted=True,
    )


def source(citation_id: str = "pmid:1") -> CandidateCitation:
    return CandidateCitation(
        id=citation_id,
        title="Quercetin and blood pressure",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        doi="10.1000/quercetin",
        pmid="1",
        year=2024,
        source="pubmed",
        retrieval_query="quercetin health human",
    )


def synthesis_response(
    *,
    effect_slug: str = "blood_lipids_cholesterol_profile",
    source_claim_id: str = "claim-1",
    citation_id: str = "pmid:1",
) -> dict:
    return {
        "suggested_bioactive_type": "polyphenol",
        "type_confidence": 0.99,
        "effects": [
            {
                "effect_slug": effect_slug,
                "effect_label": "Blood pressure and vascular function",
                "effect_description": "Effects on blood pressure and vascular function.",
                "description": (
                    "Supplement-level oral quercetin may modestly support selected vascular "
                    "markers in adults, although evidence is limited by small studies."
                ),
                "tags": ["blood_pressure", "vascular_health"],
                "source_claims": [
                    {
                        "citation_id": citation_id,
                        "proposed_claim_id": source_claim_id,
                    }
                ],
                "exposure_category": "supplement_level",
                "dose_or_exposure": ["500 mg/day"],
                "concentration_notes": "The evidence is supplement-level.",
                "review_notes": "Human evidence remains limited.",
            }
        ],
        "excluded_claims": [],
    }


def synthesis_response_without_description() -> dict:
    response = synthesis_response(effect_slug="new_effect_without_description")
    del response["effects"][0]["description"]
    return response


def write_inputs(
    tmp_path: Path,
    *,
    entity_name: str = "quercetin",
    lookup_rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "validated_claims.json",
        [validated_claim().model_dump(mode="json")],
    )
    write_json(
        run_dir / "proposed_claims.json",
        [proposed_claim(entity_name=entity_name).model_dump(mode="json")],
    )
    write_json(run_dir / "sources.json", [source().model_dump(mode="json")])
    packet = Packet(
        run_id="run",
        input=PacketInput(nutrient_name=entity_name, depth="standard", demo=False),
        model=PacketModel(provider="openai-compatible", model_name="test-model"),
        status="completed",
        created_at="2026-06-07T00:00:00+00:00",
        completed_at="2026-06-07T00:00:01+00:00",
        counts=PacketCounts(
            candidate_citations=1,
            proposed_claims=1,
            validated_claims=1,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "packet.json", packet.model_dump(mode="json"))

    lookup_path = tmp_path / "lookup.csv"
    rows = lookup_rows or [
        {
            "id": "07b8452b-93fc-42a4-8639-efea557eabba",
            "bioactive_type": "polyphenol",
            "bioactive_id": "31be4ac3-e940-42f9-9b7f-d2bd0a9cd8c3",
            "bioactive_name": "Quercetin",
            "effect_slug": "blood_lipids_cholesterol_profile",
            "effect_label": "Blood lipids and cholesterol profile",
            "description": "Existing description",
            "score": "0.35",
            "evidence_level": "mixed",
            "tags": "",
            "sources": "",
            "review_status": "generated",
            "review_notes": "",
            "created_at": "2025-12-14 17:38:45+00",
            "updated_at": "2025-12-14 17:38:45+00",
        }
    ]
    with lookup_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_dir, lookup_path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_export_skips_existing_effect_for_same_entity(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    out_dir = tmp_path / "out"

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="quercetin",
        out_dir=out_dir,
        provider=FakeInternalProvider(synthesis_response()),
    )

    assert result.evidence_rows == []
    assert result.rejected == [
        {
            "type": "existing_entity_effect_skipped",
            "effect_slug": "blood_lipids_cholesterol_profile",
            "bioactive_type": "polyphenol",
            "bioactive_id": "31be4ac3-e940-42f9-9b7f-d2bd0a9cd8c3",
            "citation_id": "pmid:1",
            "proposed_claim_id": "claim-1",
        }
    ]
    columns, csv_rows = read_csv(out_dir / "bioactive_health_evidence_rows.generated.csv")
    assert columns == EVIDENCE_COLUMNS
    assert csv_rows == []
    assert result.new_entities == []
    assert result.new_effects == []


def test_export_creates_new_effect_and_exposure_sidecar(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=out_dir,
        provider=FakeInternalProvider(
            synthesis_response(effect_slug="blood_pressure_and_vascular_support")
        ),
    )

    assert len(result.new_effects) == 1
    columns, effects = read_csv(out_dir / "health_effects_new.csv")
    assert columns == NEW_EFFECT_COLUMNS
    assert effects[0]["effect_slug"] == "blood_pressure_and_vascular_support"
    assert json.loads(effects[0]["tags"]) == ["blood_pressure", "vascular_health"]
    _, exposures = read_csv(
        out_dir / "bioactive_health_evidence_exposure_context.csv"
    )
    assert exposures[0]["evidence_row_id"] == result.evidence_rows[0].id
    assert exposures[0]["exposure_category"] == "supplement_level"
    assert json.loads(exposures[0]["dose_or_exposure"]) == ["500 mg/day"]
    row = result.evidence_rows[0]
    assert row.bioactive_type == "polyphenol"
    assert row.bioactive_id == "31be4ac3-e940-42f9-9b7f-d2bd0a9cd8c3"
    assert row.score == 0.75
    assert row.evidence_level == "strong"
    assert row.sources[0].pmid == "1"


def test_export_can_use_database_identity_name_different_from_research_name(tmp_path) -> None:
    lookup_rows = [
        {
            "id": "existing-anthocyanin-row",
            "bioactive_type": "polyphenol",
            "bioactive_id": "5fe1cec1-7963-4c24-9406-5baa9bb6e21f",
            "bioactive_name": "Anthocyanins, total",
            "effect_slug": "existing_effect",
            "effect_label": "Existing effect",
            "description": "",
            "score": "",
            "evidence_level": "",
            "tags": "",
            "sources": "",
            "review_status": "generated",
            "review_notes": "",
            "created_at": "",
            "updated_at": "",
        }
    ]
    run_dir, lookup_path = write_inputs(
        tmp_path,
        entity_name="Anthocyanins",
        lookup_rows=lookup_rows,
    )
    provider = FakeInternalProvider(
        synthesis_response(effect_slug="vascular_function_support")
    )

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Anthocyanins",
        lookup_entity_name="Anthocyanins, total",
        out_dir=tmp_path / "out",
        provider=provider,
    )

    assert len(result.evidence_rows) == 1
    assert result.evidence_rows[0].bioactive_id == "5fe1cec1-7963-4c24-9406-5baa9bb6e21f"
    assert result.evidence_rows[0].bioactive_name == "Anthocyanins, total"
    assert result.new_entities == []
    assert "Entity: Anthocyanins" in provider.prompts[0]


def test_export_fails_when_explicit_database_identity_is_missing(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path, entity_name="Anthocyanins")

    try:
        export_health_evidence(
            validated_claims_path=run_dir / "validated_claims.json",
            proposed_claims_path=run_dir / "proposed_claims.json",
            sources_path=run_dir / "sources.json",
            lookup_path=lookup_path,
            entity_name="Anthocyanins",
            lookup_entity_name="Anthocyanins, total",
            out_dir=tmp_path / "out",
            provider=FakeInternalProvider(synthesis_response()),
        )
    except ValueError as exc:
        assert str(exc) == "bioactive_lookup_name_not_found:Anthocyanins, total"
    else:
        raise AssertionError("expected explicit lookup identity failure")


def test_export_allows_exact_slug_used_by_another_entity(tmp_path) -> None:
    lookup_rows = [
        {
            "id": "quercetin-row",
            "bioactive_type": "polyphenol",
            "bioactive_id": "31be4ac3-e940-42f9-9b7f-d2bd0a9cd8c3",
            "bioactive_name": "Quercetin",
            "effect_slug": "quercetin_existing_effect",
            "effect_label": "Quercetin existing effect",
            "description": "",
            "score": "",
            "evidence_level": "",
            "tags": "",
            "sources": "",
            "review_status": "generated",
            "review_notes": "",
            "created_at": "",
            "updated_at": "",
        },
        {
            "id": "magnesium-row",
            "bioactive_type": "nutrient",
            "bioactive_id": "magnesium-id",
            "bioactive_name": "Magnesium",
            "effect_slug": "shared_effect_slug",
            "effect_label": "Shared effect",
            "description": "",
            "score": "",
            "evidence_level": "",
            "tags": "",
            "sources": "",
            "review_status": "generated",
            "review_notes": "",
            "created_at": "",
            "updated_at": "",
        },
    ]
    run_dir, lookup_path = write_inputs(tmp_path, lookup_rows=lookup_rows)

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=FakeInternalProvider(
            synthesis_response(effect_slug="shared_effect_slug")
        ),
    )

    assert len(result.evidence_rows) == 1
    assert result.new_effects == []
    assert result.rejected == []


def test_export_records_unformatted_llm_response_as_rejection(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=FakeInternalProvider(
            {
                "suggested_bioactive_type": "polyphenol",
                "type_confidence": 0.99,
                "effects": [],
                "excluded_claims": [],
            }
        ),
    )

    assert result.evidence_rows == []
    assert result.rejected[0]["type"] == "invalid_csv_format"
    assert result.rejected[0]["citation_id"] == "pmid:1"


def test_export_formats_each_accepted_claim_separately(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    claims = [
        validated_claim(claim_id=f"claim-{index}", citation_id=f"pmid:{index}")
        for index in range(1, 7)
    ]
    proposed = [
        proposed_claim(claim_id=f"claim-{index}", citation_id=f"pmid:{index}")
        for index in range(1, 7)
    ]
    sources = [source(citation_id=f"pmid:{index}") for index in range(1, 7)]
    write_json(
        run_dir / "validated_claims.json",
        [claim.model_dump(mode="json") for claim in claims],
    )
    write_json(
        run_dir / "proposed_claims.json",
        [claim.model_dump(mode="json") for claim in proposed],
    )
    write_json(
        run_dir / "sources.json",
        [citation.model_dump(mode="json") for citation in sources],
    )
    provider = FakeInternalProvider(
        {
            "suggested_bioactive_type": "polyphenol",
            "type_confidence": 0.99,
            "effects": [],
            "excluded_claims": [],
        }
    )
    progress: list[tuple[int, int]] = []

    export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=provider,
        progress=lambda current, total: progress.append((current, total)),
    )

    assert len(provider.prompts) == 6
    assert progress == [(index, 6) for index in range(1, 7)]


def test_export_accepts_effect_description_as_description_fallback(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=FakeInternalProvider(synthesis_response_without_description()),
    )

    assert result.evidence_rows[0].description == (
        "Oral quercetin may modestly support blood pressure in adults."
    )


def test_export_normalizes_loose_llm_reference_shapes(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    response = synthesis_response(effect_slug="loose_reference_effect")
    response["effects"][0]["source_claims"] = ["claim-1"]
    del response["effects"][0]["effect_description"]
    del response["effects"][0]["description"]
    response["excluded_claims"] = [
        {"claim_id": "excluded-claim", "reason": "Not a health effect."}
    ]

    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=FakeInternalProvider(response),
    )

    assert len(result.evidence_rows) == 1
    assert result.evidence_rows[0].description == (
        "Oral quercetin may modestly support blood pressure in adults."
    )
    assert result.exposure_rows[0].source_claims[0].citation_id == "pmid:1"
    assert result.rejected == []


def test_export_creates_stable_missing_entity_defaulting_to_nutrient(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path, entity_name="Novel Bioactive")
    out_one = tmp_path / "out-one"
    out_two = tmp_path / "out-two"
    response = synthesis_response(effect_slug="novel_health_support")
    response["suggested_bioactive_type"] = "polyphenol"
    response["type_confidence"] = 0.5

    first = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Novel Bioactive",
        out_dir=out_one,
        provider=FakeInternalProvider(response),
    )
    second = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Novel Bioactive",
        out_dir=out_two,
        provider=FakeInternalProvider(response),
    )

    assert first.new_entities[0].bioactive_type == "nutrient"
    assert first.new_entities[0].bioactive_id == second.new_entities[0].bioactive_id


def test_export_overrides_llm_claim_reference_with_current_validated_claim(tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    result = export_health_evidence(
        validated_claims_path=run_dir / "validated_claims.json",
        proposed_claims_path=run_dir / "proposed_claims.json",
        sources_path=run_dir / "sources.json",
        lookup_path=lookup_path,
        entity_name="Quercetin",
        out_dir=tmp_path / "out",
        provider=FakeInternalProvider(
            synthesis_response(
                effect_slug="new_effect_with_invalid_reference",
                source_claim_id="not-accepted",
            )
        ),
    )

    assert len(result.evidence_rows) == 1
    assert result.rejected == []
    assert result.exposure_rows[0].source_claims[0].proposed_claim_id == "claim-1"


def test_internal_cli_run_mode_writes_private_export(monkeypatch, tmp_path) -> None:
    run_dir, lookup_path = write_inputs(tmp_path)
    provider = FakeInternalProvider(
        synthesis_response(effect_slug="blood_pressure_and_vascular_support")
    )
    monkeypatch.setattr("sipz_agent.cli_internal.resolve_model_config", lambda **_: object())
    monkeypatch.setattr("sipz_agent.cli_internal.create_llm_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export-health-evidence",
            "--run",
            str(run_dir),
            "--lookup",
            str(lookup_path),
        ],
    )

    assert result.exit_code == 0
    assert "Created 1 database evidence rows" in result.stdout
    assert "health_effects_new.csv into beverage.health_effects first" in result.stdout
    assert (run_dir / "internal_export" / "internal_export_summary.json").exists()
    instructions = run_dir / "internal_export" / "IMPORT_INSTRUCTIONS.md"
    assert instructions.exists()
    assert "beverage.bioactive_health_evidence" in instructions.read_text()


def test_internal_cli_run_mode_accepts_lookup_entity_name(monkeypatch, tmp_path) -> None:
    lookup_rows = [
        {
            "id": "existing-anthocyanin-row",
            "bioactive_type": "polyphenol",
            "bioactive_id": "5fe1cec1-7963-4c24-9406-5baa9bb6e21f",
            "bioactive_name": "Anthocyanins, total",
            "effect_slug": "existing_effect",
            "effect_label": "Existing effect",
            "description": "",
            "score": "",
            "evidence_level": "",
            "tags": "",
            "sources": "",
            "review_status": "generated",
            "review_notes": "",
            "created_at": "",
            "updated_at": "",
        }
    ]
    run_dir, lookup_path = write_inputs(
        tmp_path,
        entity_name="Anthocyanins",
        lookup_rows=lookup_rows,
    )
    provider = FakeInternalProvider(
        synthesis_response(effect_slug="vascular_function_support")
    )
    monkeypatch.setattr("sipz_agent.cli_internal.resolve_model_config", lambda **_: object())
    monkeypatch.setattr("sipz_agent.cli_internal.create_llm_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export-health-evidence",
            "--run",
            str(run_dir),
            "--lookup",
            str(lookup_path),
            "--lookup-entity-name",
            "Anthocyanins, total",
        ],
    )

    assert result.exit_code == 0
    _, rows = read_csv(
        run_dir / "internal_export" / "bioactive_health_evidence_rows.generated.csv"
    )
    assert rows[0]["bioactive_name"] == "Anthocyanins, total"
