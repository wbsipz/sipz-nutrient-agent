import csv
from pathlib import Path
from threading import Lock
import time

import orjson

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.batch import (
    AcceptedBioactive,
    BatchStageError,
    BatchItemResult,
    BatchSession,
    has_valid_internal_export,
    is_completed,
    read_accepted_bioactives,
    resume_batch_internal_exports,
    run_batch,
    select_batch_entries,
    write_session_artifacts,
)
from sipz_agent.core.internal_export import EVIDENCE_COLUMNS
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel


def write_queue(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bioactive_name",
                "research_name",
                "slug",
                "review_priority",
            ],
        )
        writer.writeheader()
        for bioactive_name, research_name, slug in rows:
            writer.writerow(
                {
                    "bioactive_name": bioactive_name,
                    "research_name": research_name,
                    "slug": slug,
                    "review_priority": "high",
                }
            )


def write_packet(run_dir: Path, nutrient_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    packet = Packet(
        run_id=run_dir.name,
        input=PacketInput(
            nutrient_name=nutrient_name,
            depth="standard",
            demo=False,
        ),
        model=PacketModel(provider="heuristic", model_name="heuristic-demo"),
        status="completed",
        created_at="2026-06-08T00:00:00+00:00",
        completed_at="2026-06-08T00:00:01+00:00",
        counts=PacketCounts(
            candidate_citations=0,
            proposed_claims=0,
            validated_claims=0,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "packet.json", packet.model_dump(mode="json"))


def write_complete_export(run_dir: Path) -> None:
    export_dir = run_dir / "internal_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        export_dir / "internal_export_summary.json",
        {
            "evidence_rows": 0,
            "new_health_effects": 0,
            "new_entities": 0,
            "exposure_rows": 0,
            "rejected": 0,
        },
    )
    with (
        export_dir / "bioactive_health_evidence_rows.generated.csv"
    ).open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=EVIDENCE_COLUMNS).writeheader()


def test_read_queue_rejects_duplicate_slugs(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    write_queue(
        queue,
        [
            ("First", "First", "duplicate"),
            ("Second", "Second", "duplicate"),
        ],
    )

    try:
        read_accepted_bioactives(queue)
    except ValueError as exc:
        assert str(exc) == "accepted_bioactive_duplicate_slugs:duplicate"
    else:
        raise AssertionError("Expected duplicate slug validation to fail.")


def test_completion_requires_valid_latest_internal_export(tmp_path: Path) -> None:
    entry = AcceptedBioactive(
        bioactive_name="Anthocyanins, total",
        research_name="Anthocyanins",
        slug="anthocyanins",
    )
    older = tmp_path / "2026-06-07T00-00-00+00-00_anthocyanins"
    newer = tmp_path / "2026-06-08T00-00-00+00-00_anthocyanins"
    write_packet(older, "Anthocyanins")
    write_complete_export(older)
    write_packet(newer, "Anthocyanins")

    assert has_valid_internal_export(older) is True
    assert is_completed(entry, tmp_path) is False

    write_complete_export(newer)
    assert is_completed(entry, tmp_path) is True


def test_select_entries_skips_completed_and_preserves_order(tmp_path: Path) -> None:
    entries = [
        AcceptedBioactive(bioactive_name="A", research_name="A", slug="a"),
        AcceptedBioactive(bioactive_name="B", research_name="B", slug="b"),
        AcceptedBioactive(bioactive_name="C", research_name="C", slug="c"),
    ]
    completed = tmp_path / "2026-06-08T00-00-00+00-00_a"
    write_packet(completed, "A")
    write_complete_export(completed)

    selected, skipped = select_batch_entries(entries, count=2, runs_dir=tmp_path)

    assert [entry.research_name for entry in selected] == ["B", "C"]
    assert skipped == 1


def test_batch_continues_after_cancelled_item_and_limits_workers(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    lookup = tmp_path / "lookup.csv"
    runs_dir = tmp_path / "research_runs"
    batch_dir = tmp_path / "batch_runs"
    lookup.write_text("bioactive_name\n", encoding="utf-8")
    write_queue(
        queue,
        [
            ("Alpha DB", "Alpha", "alpha"),
            ("Beta DB", "Beta", "beta"),
            ("Gamma DB", "Gamma", "gamma"),
        ],
    )

    concurrency_lock = Lock()
    active = 0
    peak = 0

    def executor(entry: AcceptedBioactive, progress) -> Path:
        nonlocal active, peak
        progress("study", None)
        with concurrency_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.03)
            if entry.research_name == "Beta":
                raise BatchStageError("claim_validation", ValueError("invalid response"))
            run_dir = runs_dir / f"run_{entry.slug}"
            run_dir.mkdir(parents=True)
            progress("internal_export", run_dir)
            return run_dir
        finally:
            with concurrency_lock:
                active -= 1

    result = run_batch(
        count=3,
        workers=2,
        depth="standard",
        provider="heuristic",
        model=None,
        input_path=queue,
        lookup_path=lookup,
        runs_dir=runs_dir,
        batch_runs_dir=batch_dir,
        executor=executor,
    )

    assert peak == 2
    assert [item.status for item in result.session.items] == [
        "completed",
        "cancelled",
        "completed",
    ]
    failed = result.session.items[1]
    assert failed.current_stage == "claim_validation"
    assert failed.error_type == "ValueError"
    assert failed.error_message == "invalid response"
    assert (result.session_dir / "batch.json").exists()
    assert (result.session_dir / "batch_results.csv").exists()
    assert (result.session_dir / "batch_log.jsonl").exists()

    persisted = orjson.loads((result.session_dir / "batch.json").read_bytes())
    assert persisted["items"][1]["status"] == "cancelled"


def test_cancelled_item_is_eligible_in_next_batch(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("bioactive_name\n", encoding="utf-8")
    write_queue(queue, [("Alpha", "Alpha", "alpha")])

    def failing_executor(entry: AcceptedBioactive, progress) -> Path:
        progress("study", None)
        raise BatchStageError("study", RuntimeError("temporary failure"))

    first = run_batch(
        count=1,
        workers=1,
        depth="standard",
        provider="heuristic",
        model=None,
        input_path=queue,
        lookup_path=lookup,
        runs_dir=tmp_path / "research_runs",
        batch_runs_dir=tmp_path / "batch_runs",
        executor=failing_executor,
    )
    assert first.session.items[0].status == "cancelled"

    attempted: list[str] = []

    def succeeding_executor(entry: AcceptedBioactive, progress) -> Path:
        attempted.append(entry.research_name)
        run_dir = tmp_path / "research_runs" / "run_alpha"
        run_dir.mkdir(parents=True)
        return run_dir

    second = run_batch(
        count=1,
        workers=1,
        depth="standard",
        provider="heuristic",
        model=None,
        input_path=queue,
        lookup_path=lookup,
        runs_dir=tmp_path / "research_runs",
        batch_runs_dir=tmp_path / "batch_runs",
        executor=succeeding_executor,
    )

    assert attempted == ["Alpha"]
    assert second.session.items[0].status == "completed"


def test_resume_batch_internal_exports_updates_only_export_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lookup = tmp_path / "lookup.csv"
    lookup.write_text("bioactive_name\n", encoding="utf-8")
    batch_dir = tmp_path / "batch"
    run_dir = tmp_path / "research_runs" / "ashwagandha"
    run_dir.mkdir(parents=True)
    for artifact in ["validated_claims.json", "proposed_claims.json", "sources.json"]:
        (run_dir / artifact).write_text("[]", encoding="utf-8")
    session = BatchSession(
        batch_id="batch",
        created_at="2026-06-15T00:00:00+00:00",
        requested_count=2,
        workers=1,
        depth="standard",
        provider="heuristic",
        model="heuristic-demo",
        input_path="queue.csv",
        lookup_path=str(lookup),
        runs_dir=str(tmp_path / "research_runs"),
        items=[
            BatchItemResult(
                bioactive_name="Ashwagandha",
                research_name="Ashwagandha root extract",
                slug="ashwagandha-root-extract",
                status="cancelled",
                current_stage="internal_export",
                run_dir=str(run_dir),
                error_type="ValueError",
                error_message="bioactive_lookup_name_not_found:Ashwagandha",
            ),
            BatchItemResult(
                bioactive_name="Lavender extract",
                research_name="Lavender extract",
                slug="lavender-extract",
                status="cancelled",
                current_stage="claim_proposal",
                error_type="ValidationError",
                error_message="invalid intake route",
            ),
        ],
    )
    write_session_artifacts(batch_dir, session)

    def fake_export(**kwargs):
        assert kwargs["lookup_entity_name"] == "Ashwagandha"
        write_complete_export(kwargs["out_dir"].parent)

    monkeypatch.setattr("sipz_agent.core.batch.resolve_model_config", lambda **_: object())
    monkeypatch.setattr("sipz_agent.core.batch.create_llm_provider", lambda _: object())
    monkeypatch.setattr("sipz_agent.core.batch.export_health_evidence", fake_export)

    result = resume_batch_internal_exports(
        batch_dir=batch_dir,
        lookup_path=lookup,
        provider="heuristic",
        model=None,
    )

    assert result.attempted == 1
    assert result.completed == 1
    assert result.cancelled == 0
    assert result.session.items[0].status == "completed"
    assert result.session.items[1].status == "cancelled"
