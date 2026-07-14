from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Literal

import orjson
from pydantic import BaseModel, Field, TypeAdapter

from sipz_agent.core.claim_proposal import propose_claims_from_raw_texts
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.internal_export import EVIDENCE_COLUMNS, export_health_evidence
from sipz_agent.core.models import create_llm_provider
from sipz_agent.core.orchestrator import run_study
from sipz_agent.core.validation import validate_proposed_claims
from sipz_agent.schemas.artifacts import Packet, StudyDepth

BatchItemStatus = Literal["queued", "running", "completed", "cancelled"]
BatchProgress = Callable[[int, int, str, str], None]
StageProgress = Callable[[str, Path | None], None]


class AcceptedBioactive(BaseModel):
    bioactive_name: str = Field(min_length=1)
    research_name: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    review_priority: str | None = None


class BatchItemResult(BaseModel):
    bioactive_name: str
    research_name: str
    slug: str
    status: BatchItemStatus = "queued"
    current_stage: str = "queued"
    run_dir: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class BatchSession(BaseModel):
    batch_id: str
    created_at: str
    completed_at: str | None = None
    requested_count: int = Field(gt=0)
    workers: int = Field(ge=1, le=5)
    depth: StudyDepth
    provider: str
    model: str
    input_path: str
    lookup_path: str
    runs_dir: str
    skipped_existing: int = Field(default=0, ge=0)
    items: list[BatchItemResult]


ACCEPTED_ADAPTER = TypeAdapter(list[AcceptedBioactive])
PACKET_ADAPTER = TypeAdapter(Packet)


@dataclass(frozen=True)
class BatchRunResult:
    session_dir: Path
    session: BatchSession


@dataclass(frozen=True)
class BatchExportResumeResult:
    session_dir: Path
    session: BatchSession
    attempted: int
    completed: int
    cancelled: int


class BatchStageError(RuntimeError):
    def __init__(self, stage: str, error: Exception, run_dir: Path | None = None) -> None:
        super().__init__(str(error) or type(error).__name__)
        self.stage = stage
        self.error = error
        self.run_dir = run_dir


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def timestamp_id() -> str:
    return utc_now().replace(":", "-").replace(".", "-")


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def read_accepted_bioactives(path: Path) -> list[AcceptedBioactive]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"bioactive_name", "research_name", "slug"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("accepted_bioactive_columns_missing:" + ",".join(missing))
        rows = ACCEPTED_ADAPTER.validate_python(list(reader))
    duplicate_slugs = sorted(
        slug for slug in {row.slug for row in rows} if sum(item.slug == slug for item in rows) > 1
    )
    if duplicate_slugs:
        raise ValueError("accepted_bioactive_duplicate_slugs:" + ",".join(duplicate_slugs))
    return rows


def matching_run_dirs(entry: AcceptedBioactive, runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    expected_name = normalize_name(entry.research_name)
    matches: list[Path] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        packet_path = run_dir / "packet.json"
        matched_packet = False
        if packet_path.exists():
            try:
                packet = PACKET_ADAPTER.validate_python(orjson.loads(packet_path.read_bytes()))
                matched_packet = normalize_name(packet.input.nutrient_name) == expected_name
            except Exception:
                matched_packet = False
        if matched_packet or run_dir.name.endswith(f"_{entry.slug}"):
            matches.append(run_dir)
    return sorted(matches, key=lambda path: path.name, reverse=True)


def has_valid_internal_export(run_dir: Path) -> bool:
    summary_path = run_dir / "internal_export" / "internal_export_summary.json"
    evidence_path = (
        run_dir / "internal_export" / "bioactive_health_evidence_rows.generated.csv"
    )
    if not summary_path.exists() or not evidence_path.exists():
        return False
    try:
        summary = orjson.loads(summary_path.read_bytes())
        if not isinstance(summary, dict):
            return False
        required_summary = {
            "evidence_rows",
            "new_health_effects",
            "new_entities",
            "exposure_rows",
            "rejected",
        }
        if not required_summary.issubset(summary):
            return False
        with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        return set(EVIDENCE_COLUMNS).issubset(fieldnames)
    except (OSError, ValueError, TypeError, orjson.JSONDecodeError):
        return False


def latest_matching_run(entry: AcceptedBioactive, runs_dir: Path) -> Path | None:
    matches = matching_run_dirs(entry, runs_dir)
    return matches[0] if matches else None


def is_completed(entry: AcceptedBioactive, runs_dir: Path) -> bool:
    latest = latest_matching_run(entry, runs_dir)
    return latest is not None and has_valid_internal_export(latest)


def select_batch_entries(
    entries: list[AcceptedBioactive],
    *,
    count: int,
    runs_dir: Path,
) -> tuple[list[AcceptedBioactive], int]:
    selected: list[AcceptedBioactive] = []
    skipped = 0
    for entry in entries:
        if is_completed(entry, runs_dir):
            skipped += 1
            continue
        selected.append(entry)
        if len(selected) == count:
            break
    return selected, skipped


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_session_artifacts(session_dir: Path, session: BatchSession) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        session_dir / "batch.json",
        orjson.dumps(
            session.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
        ),
    )
    csv_path = session_dir / "batch_results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fields = list(BatchItemResult.model_fields)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in session.items:
            writer.writerow(item.model_dump(mode="json"))
    temporary.replace(csv_path)


def append_batch_event(session_dir: Path, event: dict[str, object]) -> None:
    with (session_dir / "batch_log.jsonl").open("ab") as handle:
        handle.write(orjson.dumps(event, option=orjson.OPT_APPEND_NEWLINE))


def read_batch_session(batch_dir: Path) -> BatchSession:
    return BatchSession.model_validate_json(
        (batch_dir / "batch.json").read_text(encoding="utf-8")
    )


def item_needs_internal_export_resume(item: BatchItemResult) -> bool:
    return (
        item.status == "cancelled"
        and item.current_stage == "internal_export"
        and bool(item.run_dir)
    )


def resume_batch_internal_exports(
    *,
    batch_dir: Path,
    lookup_path: Path,
    provider: str | None,
    model: str | None,
    progress: BatchProgress | None = None,
) -> BatchExportResumeResult:
    if not lookup_path.exists():
        raise ValueError(f"lookup_file_not_found:{lookup_path}")

    session = read_batch_session(batch_dir)
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    indexes = [
        index for index, item in enumerate(session.items) if item_needs_internal_export_resume(item)
    ]
    append_batch_event(
        batch_dir,
        {
            "ts": utc_now(),
            "event": "internal_export_resume_started",
            "attempted": len(indexes),
            "lookup_path": str(lookup_path.resolve()),
        },
    )

    completed = 0
    cancelled = 0
    total = len(indexes)
    for position, index in enumerate(indexes, start=1):
        item = session.items[index]
        run_dir = Path(item.run_dir or "")
        started_at = item.started_at or utc_now()
        session.items[index] = item.model_copy(
            update={
                "status": "running",
                "current_stage": "internal_export",
                "started_at": started_at,
                "error_type": None,
                "error_message": None,
            }
        )
        write_session_artifacts(batch_dir, session)
        append_batch_event(
            batch_dir,
            {
                "ts": utc_now(),
                "event": "item_status_changed",
                "item": session.items[index].model_dump(mode="json"),
            },
        )
        if progress:
            progress(position, total, item.research_name, "internal_export")
        try:
            export_health_evidence(
                validated_claims_path=run_dir / "validated_claims.json",
                proposed_claims_path=run_dir / "proposed_claims.json",
                sources_path=run_dir / "sources.json",
                lookup_path=lookup_path,
                entity_name=item.research_name,
                lookup_entity_name=item.bioactive_name,
                out_dir=run_dir / "internal_export",
                provider=llm_provider,
            )
            if not has_valid_internal_export(run_dir):
                raise ValueError("internal_export_artifacts_invalid")
            session.items[index] = session.items[index].model_copy(
                update={
                    "status": "completed",
                    "current_stage": "completed",
                    "completed_at": utc_now(),
                    "error_type": None,
                    "error_message": None,
                }
            )
            completed += 1
            if progress:
                progress(position, total, item.research_name, "completed")
        except Exception as exc:
            session.items[index] = session.items[index].model_copy(
                update={
                    "status": "cancelled",
                    "current_stage": "internal_export",
                    "completed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc) or type(exc).__name__,
                }
            )
            cancelled += 1
            if progress:
                progress(position, total, item.research_name, f"cancelled: {exc}")
        write_session_artifacts(batch_dir, session)
        append_batch_event(
            batch_dir,
            {
                "ts": utc_now(),
                "event": "item_status_changed",
                "item": session.items[index].model_dump(mode="json"),
            },
        )

    session.completed_at = utc_now()
    write_session_artifacts(batch_dir, session)
    append_batch_event(
        batch_dir,
        {
            "ts": session.completed_at,
            "event": "internal_export_resume_completed",
            "attempted": len(indexes),
            "completed": completed,
            "cancelled": cancelled,
        },
    )
    return BatchExportResumeResult(
        session_dir=batch_dir,
        session=session,
        attempted=len(indexes),
        completed=completed,
        cancelled=cancelled,
    )


def execute_full_pipeline(
    entry: AcceptedBioactive,
    *,
    depth: StudyDepth,
    provider: str | None,
    model: str | None,
    runs_dir: Path,
    lookup_path: Path,
    full_text_workers: int,
    stage_progress: StageProgress,
) -> Path:
    run_dir: Path | None = None
    current_stage = "initialization"

    def stage(name: str) -> None:
        nonlocal current_stage
        current_stage = name
        stage_progress(name, run_dir)

    try:
        stage("study")
        study_result = run_study(
            nutrient_name=entry.research_name,
            depth=depth,
            demo=False,
            out_dir=runs_dir,
            provider=provider,
            model=model,
            retrieve_full_text=True,
            full_text_workers=full_text_workers,
        )
        run_dir = study_result.run_dir

        model_config = resolve_model_config(provider=provider, model=model)
        llm_provider = create_llm_provider(model_config)

        stage("claim_proposal")
        propose_claims_from_raw_texts(
            nutrient_name=entry.research_name,
            sources_path=run_dir / "sources.json",
            raw_texts_manifest_path=run_dir / "raw_texts.json",
            raw_texts_dir=run_dir / "raw_texts",
            out_dir=run_dir,
            provider=llm_provider,
            update_run_packet=True,
        )

        stage("claim_validation")
        validate_proposed_claims(
            proposed_claims_path=run_dir / "proposed_claims.json",
            sources_path=run_dir / "sources.json",
            raw_texts_manifest_path=run_dir / "raw_texts.json",
            raw_texts_dir=run_dir / "raw_texts",
            out_dir=run_dir,
            provider=llm_provider,
            update_run_packet=True,
            resume=True,
        )

        stage("internal_export")
        export_health_evidence(
            validated_claims_path=run_dir / "validated_claims.json",
            proposed_claims_path=run_dir / "proposed_claims.json",
            sources_path=run_dir / "sources.json",
            lookup_path=lookup_path,
            entity_name=entry.research_name,
            lookup_entity_name=entry.bioactive_name,
            out_dir=run_dir / "internal_export",
            provider=llm_provider,
        )
        if not has_valid_internal_export(run_dir):
            raise ValueError("internal_export_artifacts_invalid")
        return run_dir
    except Exception as exc:
        raise BatchStageError(
            stage=current_stage,
            error=exc,
            run_dir=run_dir,
        ) from exc


def run_batch(
    *,
    count: int,
    workers: int,
    depth: StudyDepth,
    provider: str | None,
    model: str | None,
    input_path: Path,
    lookup_path: Path,
    runs_dir: Path,
    batch_runs_dir: Path = Path("batch_runs"),
    full_text_workers: int = 2,
    progress: BatchProgress | None = None,
    executor: Callable[[AcceptedBioactive, StageProgress], Path] | None = None,
) -> BatchRunResult:
    if count < 1:
        raise ValueError("batch_count_must_be_positive")
    if not 1 <= workers <= 5:
        raise ValueError("batch_workers_must_be_between_1_and_5")
    if full_text_workers < 1:
        raise ValueError("full_text_workers_must_be_positive")
    if not lookup_path.exists():
        raise ValueError(f"lookup_file_not_found:{lookup_path}")

    model_config = resolve_model_config(provider=provider, model=model)
    entries = read_accepted_bioactives(input_path)
    selected, skipped = select_batch_entries(entries, count=count, runs_dir=runs_dir)
    batch_id = timestamp_id()
    session_dir = batch_runs_dir.resolve() / batch_id
    session = BatchSession(
        batch_id=batch_id,
        created_at=utc_now(),
        requested_count=count,
        workers=workers,
        depth=depth,
        provider=model_config.provider,
        model=model_config.model_name,
        input_path=str(input_path.resolve()),
        lookup_path=str(lookup_path.resolve()),
        runs_dir=str(runs_dir.resolve()),
        skipped_existing=skipped,
        items=[
            BatchItemResult(
                bioactive_name=entry.bioactive_name,
                research_name=entry.research_name,
                slug=entry.slug,
            )
            for entry in selected
        ],
    )
    write_session_artifacts(session_dir, session)
    append_batch_event(
        session_dir,
        {
            "ts": utc_now(),
            "event": "batch_started",
            "requested_count": count,
            "selected_count": len(selected),
            "skipped_existing": skipped,
        },
    )

    lock = Lock()

    def update_item(index: int, **updates: object) -> None:
        with lock:
            item_data = session.items[index].model_dump()
            item_data.update(updates)
            session.items[index] = BatchItemResult.model_validate(item_data)
            write_session_artifacts(session_dir, session)
            append_batch_event(
                session_dir,
                {
                    "ts": utc_now(),
                    "event": "item_status_changed",
                    "item": session.items[index].model_dump(mode="json"),
                },
            )

    def run_one(index: int, entry: AcceptedBioactive) -> None:
        started_at = utc_now()
        update_item(index, status="running", current_stage="starting", started_at=started_at)

        def report_stage(stage: str, run_dir: Path | None) -> None:
            update_item(
                index,
                current_stage=stage,
                run_dir=str(run_dir) if run_dir else session.items[index].run_dir,
            )
            if progress:
                progress(index + 1, len(selected), entry.research_name, stage)

        try:
            if executor is None:
                completed_run = execute_full_pipeline(
                    entry,
                    depth=depth,
                    provider=provider,
                    model=model,
                    runs_dir=runs_dir,
                    lookup_path=lookup_path,
                    full_text_workers=full_text_workers,
                    stage_progress=report_stage,
                )
            else:
                completed_run = executor(entry, report_stage)
            update_item(
                index,
                status="completed",
                current_stage="completed",
                run_dir=str(completed_run),
                completed_at=utc_now(),
            )
            if progress:
                progress(index + 1, len(selected), entry.research_name, "completed")
        except BatchStageError as exc:
            update_item(
                index,
                status="cancelled",
                current_stage=exc.stage,
                run_dir=str(exc.run_dir) if exc.run_dir else session.items[index].run_dir,
                completed_at=utc_now(),
                error_type=type(exc.error).__name__,
                error_message=str(exc.error) or type(exc.error).__name__,
            )
            if progress:
                progress(index + 1, len(selected), entry.research_name, f"cancelled: {exc}")
        except Exception as exc:
            update_item(
                index,
                status="cancelled",
                completed_at=utc_now(),
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )
            if progress:
                progress(index + 1, len(selected), entry.research_name, f"cancelled: {exc}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_one, index, entry) for index, entry in enumerate(selected)
        ]
        for future in as_completed(futures):
            future.result()

    session.completed_at = utc_now()
    write_session_artifacts(session_dir, session)
    append_batch_event(
        session_dir,
        {
            "ts": session.completed_at,
            "event": "batch_completed",
            "completed": sum(item.status == "completed" for item in session.items),
            "cancelled": sum(item.status == "cancelled" for item in session.items),
        },
    )
    return BatchRunResult(session_dir=session_dir, session=session)
