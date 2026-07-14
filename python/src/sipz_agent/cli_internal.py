from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console

from sipz_agent.core.batch import resume_batch_internal_exports, run_batch
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.internal_export import export_health_evidence
from sipz_agent.core.models import create_llm_provider
from sipz_agent.schemas.artifacts import Packet, StudyDepth

app = typer.Typer(help="Private Sipz database export utilities.")
console = Console()


@app.callback()
def internal_cli() -> None:
    """Private Sipz database export utilities."""


@app.command("batch-health-evidence")
def batch_health_evidence_command(
    count: Annotated[
        int,
        typer.Argument(help="Number of uncompleted bioactives to attempt."),
    ],
    workers: Annotated[
        int,
        typer.Option(help="Parallel nutrient pipelines; must be between 1 and 5."),
    ] = 2,
    depth: Annotated[
        str,
        typer.Option(help="Research depth: light, standard, or deep."),
    ] = "standard",
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    input: Annotated[
        Path,
        typer.Option(help="Accepted bioactive queue CSV."),
    ] = Path("bioactive_names_accepted_for_review.csv"),
    lookup: Annotated[
        Path,
        typer.Option(help="Existing bioactive health evidence lookup CSV."),
    ] = Path("bioactive_health_evidence_rows.csv"),
    runs_dir: Annotated[
        Path,
        typer.Option(help="Research run output and completion-scan directory."),
    ] = Path("research_runs"),
    full_text_workers: Annotated[
        int,
        typer.Option(help="Parallel paper retrieval workers within each nutrient pipeline."),
    ] = 2,
) -> None:
    if count < 1:
        raise typer.BadParameter("count must be greater than zero")
    if not 1 <= workers <= 5:
        raise typer.BadParameter("--workers must be between 1 and 5")
    if depth not in {"light", "standard", "deep"}:
        raise typer.BadParameter("--depth must be light, standard, or deep")
    if full_text_workers < 1:
        raise typer.BadParameter("--full-text-workers must be greater than zero")

    result = run_batch(
        count=count,
        workers=workers,
        depth=cast(StudyDepth, depth),
        provider=provider,
        model=model,
        input_path=input,
        lookup_path=lookup,
        runs_dir=runs_dir,
        full_text_workers=full_text_workers,
        progress=lambda current, total, name, stage: console.print(
            f"[{current}/{total}][{name}] {stage}"
        ),
    )
    completed = sum(item.status == "completed" for item in result.session.items)
    cancelled = sum(item.status == "cancelled" for item in result.session.items)
    console.print(
        f"Batch complete: {completed} completed, {cancelled} cancelled, "
        f"{result.session.skipped_existing} existing runs skipped."
    )
    console.print(f"Batch artifacts: {result.session_dir}")


@app.command("export-health-evidence")
def export_health_evidence_command(
    run: Annotated[
        Path | None,
        typer.Option(help="Research run containing validated claims and source artifacts."),
    ] = None,
    validated_claims: Annotated[
        Path | None,
        typer.Option(help="Path to validated_claims.json for isolated mode."),
    ] = None,
    proposed_claims: Annotated[
        Path | None,
        typer.Option(help="Path to proposed_claims.json for isolated mode."),
    ] = None,
    sources: Annotated[
        Path | None,
        typer.Option(help="Path to sources.json for isolated mode."),
    ] = None,
    entity_name: Annotated[
        str | None,
        typer.Option(help="Canonical nutrient/bioactive name for isolated mode."),
    ] = None,
    lookup_entity_name: Annotated[
        str | None,
        typer.Option(
            help=(
                "Existing database identity name to resolve in --lookup when it differs "
                "from the research name."
            )
        ),
    ] = None,
    lookup: Annotated[
        Path,
        typer.Option(help="Existing bioactive_health_evidence CSV used for identity/effect lookup."),
    ] = Path("bioactive_health_evidence_rows.csv"),
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to <run>/internal_export."),
    ] = None,
    bioactive_type: Annotated[
        str,
        typer.Option(help="Routing type for a missing identity: auto, nutrient, or polyphenol."),
    ] = "auto",
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
) -> None:
    if bioactive_type not in {"auto", "nutrient", "polyphenol"}:
        raise typer.BadParameter("--bioactive-type must be auto, nutrient, or polyphenol.")
    if run is not None:
        if any(value is not None for value in [validated_claims, proposed_claims, sources, entity_name]):
            raise typer.BadParameter(
                "Use either --run or isolated artifact options, not both."
            )
        packet_path = run / "packet.json"
        if not packet_path.exists():
            raise typer.BadParameter(f"Missing required artifact: {packet_path}")
        packet = Packet.model_validate_json(packet_path.read_text(encoding="utf-8"))
        validated_path = run / "validated_claims.json"
        proposed_path = run / "proposed_claims.json"
        sources_path = run / "sources.json"
        resolved_entity_name = packet.input.nutrient_name
        out_dir = out or run / "internal_export"
    else:
        if not validated_claims or not proposed_claims or not sources or not entity_name:
            raise typer.BadParameter(
                "Provide --run, or provide --validated-claims, --proposed-claims, "
                "--sources, and --entity-name."
            )
        validated_path = validated_claims
        proposed_path = proposed_claims
        sources_path = sources
        resolved_entity_name = entity_name
        out_dir = out or validated_claims.parent / "internal_export"

    required = [validated_path, proposed_path, sources_path, lookup]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required artifacts: " + ", ".join(missing))

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    result = export_health_evidence(
        validated_claims_path=validated_path,
        proposed_claims_path=proposed_path,
        sources_path=sources_path,
        lookup_path=lookup,
        entity_name=resolved_entity_name,
        lookup_entity_name=lookup_entity_name,
        out_dir=out_dir,
        provider=llm_provider,
        bioactive_type=cast(
            Literal["auto", "nutrient", "polyphenol"],
            bioactive_type,
        ),
        progress=lambda current, total: console.print(
            f"Formatting validated claim {current}/{total}..."
        ),
    )
    console.print(f"Created {len(result.evidence_rows)} database evidence rows.")
    console.print(f"Created {len(result.new_effects)} new health-effect references.")
    console.print(f"Created {len(result.new_entities)} new entity references.")
    console.print(f"Rejected {len(result.rejected)} synthesis items.")
    console.print(f"Wrote internal export artifacts under: {out_dir}")
    console.print(
        "Import health_effects_new.csv into beverage.health_effects first."
    )
    console.print(
        "Then import bioactive_health_evidence_rows.generated.csv into "
        "beverage.bioactive_health_evidence."
    )


@app.command("resume-batch-export")
def resume_batch_export_command(
    batch: Annotated[
        Path,
        typer.Option(help="Batch run directory containing batch.json."),
    ],
    lookup: Annotated[
        Path,
        typer.Option(help="Existing bioactive_health_evidence CSV used for identity/effect lookup."),
    ] = Path("bioactive_health_evidence_rows.csv"),
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
) -> None:
    if not (batch / "batch.json").exists():
        raise typer.BadParameter(f"Missing required artifact: {batch / 'batch.json'}")
    result = resume_batch_internal_exports(
        batch_dir=batch,
        lookup_path=lookup,
        provider=provider,
        model=model,
        progress=lambda current, total, name, stage: console.print(
            f"[{current}/{total}][{name}] {stage}"
        ),
    )
    console.print(
        f"Batch export resume complete: {result.completed} completed, "
        f"{result.cancelled} cancelled, {result.attempted} attempted."
    )
    console.print(f"Batch artifacts: {result.session_dir}")


if __name__ == "__main__":
    app()
