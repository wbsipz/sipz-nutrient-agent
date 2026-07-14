import csv
import json

from sipz_agent.core.audit import audit_run
from sipz_agent.core.orchestrator import run_study


def test_demo_study_writes_artifacts_and_passes_audit(tmp_path) -> None:
    result = run_study(
        nutrient_name="fluoride",
        depth="standard",
        demo=True,
        out_dir=tmp_path,
    )

    expected = {
        "effects.csv",
        "validated_claims.json",
        "rejected_claims.json",
        "sources.json",
        "sources.md",
        "rejected_sources.json",
        "rejected_sources.md",
        "packet.json",
        "summary.md",
        "audit_log.jsonl",
    }
    assert expected == {path.name for path in result.run_dir.iterdir()}

    with (result.run_dir / "effects.csv").open("r", encoding="utf-8", newline="") as handle:
        effects = list(csv.DictReader(handle))
    validated = json.loads((result.run_dir / "validated_claims.json").read_text(encoding="utf-8"))
    rejected = json.loads((result.run_dir / "rejected_claims.json").read_text(encoding="utf-8"))

    assert len(effects) == 1
    assert len(validated) == 1
    assert len(rejected) == 1
    assert audit_run(result.run_dir).ok is True
