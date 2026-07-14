import json

from sipz_agent.core.artifacts import StudyArtifacts, write_study_artifacts
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel
from sipz_agent.schemas.citations import CandidateCitation


def test_write_study_artifacts_writes_sources_markdown_with_selection_reasons(tmp_path) -> None:
    packet = Packet(
        run_id="run-1",
        input=PacketInput(nutrient_name="magnesium", depth="light", demo=False),
        model=PacketModel(provider="heuristic", model_name="heuristic-demo"),
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        counts=PacketCounts(
            candidate_citations=1,
            proposed_claims=0,
            validated_claims=0,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    source = CandidateCitation(
        id="pmid:1",
        title="Magnesium and human health",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
        pmid="1",
        source="pubmed",
        retrieval_query="magnesium health human",
        selection_reason="Selected because PubMed returned it for a human-health magnesium query.",
    )

    run_dir = write_study_artifacts(
        tmp_path,
        "run-1",
        StudyArtifacts(
            packet=packet,
            sources=[source],
            effect_rows=[],
            validated_claims=[],
            rejected_claims=[],
            summary_markdown="summary",
            audit_events=[],
        ),
    )

    sources = json.loads((run_dir / "sources.json").read_text(encoding="utf-8"))
    markdown = (run_dir / "sources.md").read_text(encoding="utf-8")

    assert sources[0]["selection_reason"] == source.selection_reason
    assert "Magnesium and human health" in markdown
    assert "https://pubmed.ncbi.nlm.nih.gov/1/" in markdown
    assert "Selected because PubMed returned it" in markdown
