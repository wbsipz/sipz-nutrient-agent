from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from sipz_agent.core.audit import audit_run
from sipz_agent.core.claim_proposal import propose_claims_from_raw_texts
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.full_text import raw_text_counts, retrieve_full_text_for_run
from sipz_agent.core.models import create_llm_provider
from sipz_agent.core.orchestrator import run_study
from sipz_agent.core.validation import validate_proposed_claims
from sipz_agent.core.artifacts import write_json
from sipz_agent.schemas.artifacts import Packet
from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import CandidateCitation
app = typer.Typer(help="Sipz nutrient and bioactive research agent.")
console = Console()


def print_claim_progress(index: int, total: int, citation: CandidateCitation) -> None:
    identifier = f"DOI: {citation.doi}" if citation.doi else f"Source: {citation.id}"
    console.print(f"[{index}/{total}] Analyzing {citation.title} ({identifier})")


def print_validation_progress(
    index: int,
    total: int,
    citation: CandidateCitation,
    claim_count: int,
) -> None:
    identifier = f"DOI: {citation.doi}" if citation.doi else f"Source: {citation.id}"
    noun = "claim" if claim_count == 1 else "claims"
    console.print(f"[{index}/{total}] Validating {claim_count} {noun} ({identifier})")


@app.command()
def study(
    nutrient: Annotated[str, typer.Argument(help="Nutrient, bioactive, or category to study.")],
    demo: Annotated[bool, typer.Option(help="Use bundled demo corpus.")] = False,
    depth: Annotated[str, typer.Option(help="light, standard, or deep.")] = "standard",
    provider: Annotated[str | None, typer.Option(help="LLM provider: heuristic, deepseek, openai, anthropic, or openai-compatible.")] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("research_runs"),
    retrieve_full_text: Annotated[
        bool,
        typer.Option(
            "--retrieve-full-text",
            help="After source discovery, retrieve available paper body text into raw_texts artifacts.",
        ),
    ] = False,
    full_text_workers: Annotated[
        int,
        typer.Option(help="Parallel workers for --retrieve-full-text."),
    ] = 4,
) -> None:
    if depth not in {"light", "standard", "deep"}:
        raise typer.BadParameter("depth must be one of: light, standard, deep")
    result = run_study(
        nutrient_name=nutrient,
        depth=cast(StudyDepth, depth),
        demo=demo,
        out_dir=out,
        provider=provider,
        model=model,
        retrieve_full_text=retrieve_full_text,
        full_text_workers=full_text_workers,
    )
    console.print(f"Created research run: {result.run_dir}")
    if retrieve_full_text:
        console.print(
            "Raw text retrieval artifacts: raw_texts.json, raw_texts.md, "
            "full_text_retrieval_attempts.json, raw_texts/"
        )


@app.command("retrieve-text")
def retrieve_text(
    run: Annotated[Path, typer.Option(help="Research run directory with sources.json.")],
    max_workers: Annotated[int, typer.Option(help="Parallel retrieval workers.")] = 4,
) -> None:
    records = retrieve_full_text_for_run(run, max_workers=max_workers)
    counts = raw_text_counts(records)
    console.print(f"Processed {len(records)} sources.")
    for status, count in counts.items():
        if count:
            console.print(f"- {status}: {count}")
    console.print(f"Wrote raw text retrieval artifacts under: {run}")
    console.print("Attempt details: full_text_retrieval_attempts.json")


@app.command("propose-claims")
def propose_claims(
    run: Annotated[Path | None, typer.Option(help="Research run directory with sources.json and raw_texts.json.")] = None,
    raw_texts_dir: Annotated[Path | None, typer.Option(help="Directory containing extracted raw text files.")] = None,
    raw_texts_manifest: Annotated[Path | None, typer.Option(help="Path to raw_texts.json for isolated mode.")] = None,
    sources: Annotated[Path | None, typer.Option(help="Path to sources.json for isolated mode.")] = None,
    nutrient: Annotated[str | None, typer.Option(help="Nutrient/bioactive name for isolated mode.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output directory for isolated mode.")] = None,
    provider: Annotated[str | None, typer.Option(help="LLM provider: deepseek, openai, anthropic, or openai-compatible.")] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
) -> None:
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)

    if run is not None:
        packet = Packet.model_validate_json((run / "packet.json").read_text(encoding="utf-8"))
        claims = propose_claims_from_raw_texts(
            nutrient_name=packet.input.nutrient_name,
            sources_path=run / "sources.json",
            raw_texts_manifest_path=run / "raw_texts.json",
            raw_texts_dir=run / "raw_texts",
            out_dir=run,
            provider=llm_provider,
            update_run_packet=True,
            progress=print_claim_progress,
        )
        console.print(f"Proposed {len(claims)} claims from raw texts.")
        console.print(f"Wrote proposed claim artifacts under: {run}")
        return

    if not raw_texts_dir or not sources or not nutrient or not out:
        raise typer.BadParameter(
            "Either provide --run, or provide --raw-texts-dir, --sources, --nutrient, and --out."
        )
    manifest_path = raw_texts_manifest or raw_texts_dir.parent / "raw_texts.json"
    claims = propose_claims_from_raw_texts(
        nutrient_name=nutrient,
        sources_path=sources,
        raw_texts_manifest_path=manifest_path,
        raw_texts_dir=raw_texts_dir,
        out_dir=out,
        provider=llm_provider,
        update_run_packet=False,
        progress=print_claim_progress,
    )
    write_json(out / "claim_proposal_packet.json", {"nutrient_name": nutrient, "proposed_claims": len(claims)})
    console.print(f"Proposed {len(claims)} claims from raw texts.")
    console.print(f"Wrote proposed claim artifacts under: {out}")


@app.command("validate-claims")
def validate_claims(
    run: Annotated[
        Path | None,
        typer.Option(help="Research run directory containing proposed claims and raw text artifacts."),
    ] = None,
    proposed_claims: Annotated[
        Path | None,
        typer.Option(help="Path to proposed_claims.json; sibling artifacts are inferred."),
    ] = None,
    sources: Annotated[Path | None, typer.Option(help="Override path to sources.json.")] = None,
    raw_texts_manifest: Annotated[
        Path | None,
        typer.Option(help="Override path to raw_texts.json."),
    ] = None,
    raw_texts_dir: Annotated[
        Path | None,
        typer.Option(help="Override directory containing extracted paper text files."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output directory for validation artifacts.")] = None,
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek, openai, anthropic, or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    max_body_chars: Annotated[
        int,
        typer.Option(help="Maximum sanitized paper characters accepted without truncation."),
    ] = 750_000,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse completed claim validations."),
    ] = True,
) -> None:
    if run is not None and proposed_claims is not None:
        raise typer.BadParameter("Provide either --run or --proposed-claims, not both.")
    if run is None and proposed_claims is None:
        raise typer.BadParameter("Provide either --run or --proposed-claims.")
    if max_body_chars < 1:
        raise typer.BadParameter("--max-body-chars must be greater than zero.")

    base_dir = run if run is not None else proposed_claims.parent
    claims_path = proposed_claims or base_dir / "proposed_claims.json"
    sources_path = sources or base_dir / "sources.json"
    manifest_path = raw_texts_manifest or base_dir / "raw_texts.json"
    texts_dir = raw_texts_dir or base_dir / "raw_texts"
    out_dir = out or base_dir
    required = [claims_path, sources_path, manifest_path, texts_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required validation artifacts: " + ", ".join(missing))

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    accepted, rejected, failures = validate_proposed_claims(
        proposed_claims_path=claims_path,
        sources_path=sources_path,
        raw_texts_manifest_path=manifest_path,
        raw_texts_dir=texts_dir,
        out_dir=out_dir,
        provider=llm_provider,
        update_run_packet=run is not None,
        progress=print_validation_progress,
        max_body_chars=max_body_chars,
        resume=resume,
    )
    console.print(
        f"Validation complete: {len(accepted)} accepted, "
        f"{len(rejected)} rejected, {len(failures)} paper failures."
    )
    console.print(f"Wrote validation artifacts under: {out_dir}")


@app.command()
def audit(
    run: Annotated[Path, typer.Option(help="Research run directory.")],
) -> None:
    result = audit_run(run)
    if result.ok:
        console.print("Audit passed.")
        return

    console.print("Audit failed:")
    for issue in result.issues:
        console.print(f"- {issue}")
    raise typer.Exit(code=1)


@app.command()
def export(
    run: Annotated[Path, typer.Option(help="Research run directory.")],
    format: Annotated[str, typer.Option(help="Export format.")] = "csv",
) -> None:
    if format != "csv":
        raise typer.BadParameter("Only csv export is supported.")
    console.print((run / "effects.csv").read_text(encoding="utf-8").rstrip())
