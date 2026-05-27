from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sipz_agent.core.audit import audit_run
from sipz_agent.core.orchestrator import run_study
app = typer.Typer(help="Sipz nutrient and bioactive research agent.")
console = Console()


@app.command()
def study(
    nutrient: Annotated[str, typer.Argument(help="Nutrient, bioactive, or category to study.")],
    demo: Annotated[bool, typer.Option(help="Use bundled demo corpus.")] = False,
    depth: Annotated[str, typer.Option(help="light, standard, or deep.")] = "standard",
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("research_runs"),
) -> None:
    if depth not in {"light", "standard", "deep"}:
        raise typer.BadParameter("depth must be one of: light, standard, deep")
    result = run_study(nutrient_name=nutrient, depth=depth, demo=demo, out_dir=out)
    console.print(f"Created research run: {result.run_dir}")


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
