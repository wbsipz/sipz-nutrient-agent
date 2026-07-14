from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.composition_ingredient_health_summary import (
    run_ingredient_health_summary_generation,
)
from sipz_agent.core.composition_nutriment_summary import (
    run_nutriment_health_summary_generation,
)
from sipz_agent.core.composition_ingredient_summary_inputs import (
    run_ingredient_summary_input_assembly,
)
from sipz_agent.core.composition_significance import run_nutriment_significance_classification
from sipz_agent.core.full_text import ingest_manual_full_text_queue
from sipz_agent.core.final_export_inputs import assemble_final_export_inputs
from sipz_agent.core.final_main_csv import (
    write_final_main_csv,
    write_final_search_companion_csvs,
)
from sipz_agent.core.final_export_merge import run_final_health_summary_merge
from sipz_agent.core.final_export_normalization import normalize_final_export_summaries
from sipz_agent.core.fallback_composition import (
    build_fallback_composition_inputs,
    combine_composition_summaries,
    complete_final_health_summary_merges,
)
from sipz_agent.core.ingredient_claim_audit import audit_ingredient_claims
from sipz_agent.core.ingredient_preparation import prepare_ingredients
from sipz_agent.core.ingredient_synthesis import synthesize_ingredient_report
from sipz_agent.core.ingredient_workflow import (
    propose_ingredient_claims_from_raw_texts,
    run_ingredient_study,
    validate_ingredient_claims,
)
from sipz_agent.core.models import create_llm_provider
from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import CandidateCitation

app = typer.Typer(help="Sipz ingredient health report research utilities.")
console = Console()


@app.callback()
def ingredient_cli() -> None:
    """Ingredient health report research utilities."""


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


def print_audit_progress(
    index: int,
    total: int,
    citation: CandidateCitation,
    claim_count: int,
) -> None:
    identifier = f"DOI: {citation.doi}" if citation.doi else f"Source: {citation.id}"
    noun = "claim" if claim_count == 1 else "claims"
    console.print(f"[{index}/{total}] Auditing {claim_count} {noun} ({identifier})")


def print_significance_progress(index: int, total: int, profile) -> None:
    console.print(
        f"[{index}/{total}] Classifying nutriment significance for "
        f"{profile.canonical_beverage_name}"
    )


def print_nutriment_summary_progress(index: int, total: int, record) -> None:
    console.print(
        f"[{index}/{total}] Summarizing {record.canonical_bioactive_name} for "
        f"{record.ingredient_name}"
    )


def print_ingredient_summary_progress(index: int, total: int, row) -> None:
    name = row.get("canonical_beverage_name") or row.get("ingredient_name")
    console.print(f"[{index}/{total}] Summarizing ingredient health for {name}")


def print_final_merge_progress(index: int, total: int, row) -> None:
    name = row.get("canonical_beverage_name")
    console.print(f"[{index}/{total}] Merging final health summary for {name}")


@app.command("assemble-final-export-inputs")
def assemble_final_export_inputs_command(
    legacy_v1: Annotated[
        Path,
        typer.Option(help="Legacy health_reports_ingredients_v1_rows.csv."),
    ],
    legacy_v2: Annotated[
        Path,
        typer.Option(help="Legacy health_reports_ingredients_v2_rows.csv."),
    ],
    claim_audit: Annotated[
        Path,
        typer.Option(help="Audited legacy claim CSV."),
    ],
    combined_map: Annotated[
        Path,
        typer.Option(help="Combined ingredient-to-research-target map CSV."),
    ],
    included_map: Annotated[
        Path,
        typer.Option(help="Included ingredient-to-research-target map CSV."),
    ],
    composition_summaries: Annotated[
        Path,
        typer.Option(help="Composition ingredient health summaries JSONL."),
    ],
    ingredient_runs: Annotated[
        Path,
        typer.Option(help="Directory containing direct-literature ingredient runs."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for final export merge input artifacts."),
    ],
) -> None:
    required = [
        legacy_v1,
        legacy_v2,
        claim_audit,
        combined_map,
        included_map,
        composition_summaries,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing final export input files: " + ", ".join(missing))
    if not ingredient_runs.exists():
        raise typer.BadParameter(f"Missing ingredient runs directory: {ingredient_runs}")

    try:
        result = assemble_final_export_inputs(
            legacy_v1_path=legacy_v1,
            legacy_v2_path=legacy_v2,
            claim_audit_path=claim_audit,
            combined_map_path=combined_map,
            included_map_path=included_map,
            composition_summaries_path=composition_summaries,
            ingredient_runs_dir=ingredient_runs,
            out_dir=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Final export merge input assembly complete.")
    console.print(f"- combined map rows: {summary['combined_map_rows']}")
    console.print(f"- written rows: {summary['written_rows']}")
    console.print(f"- direct-literature rows: {summary['direct_literature_rows']}")
    console.print(f"- composition-based rows: {summary['composition_based_rows']}")
    console.print(f"- both-source rows: {summary['both_sources_rows']}")
    console.print(f"- skipped rows: {summary['skipped_rows']}")
    console.print(f"- legacy audit claims attached: {summary['legacy_audit_claims_attached']}")
    console.print(f"- resolver warnings: {summary['resolver_warnings']}")
    console.print(f"Wrote final export merge input artifacts under: {out}")


@app.command("normalize-final-export-summaries")
def normalize_final_export_summaries_command(
    inputs: Annotated[
        Path,
        typer.Option(help="Final export merge inputs JSONL from Step 1."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for normalized final export summary artifacts."),
    ],
) -> None:
    if not inputs.exists():
        raise typer.BadParameter(f"Missing final export merge inputs JSONL: {inputs}")

    try:
        result = normalize_final_export_summaries(
            inputs_path=inputs,
            out_dir=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Final export summary normalization complete.")
    console.print(f"- input rows: {summary['input_rows']}")
    console.print(f"- normalized rows: {summary['normalized_rows']}")
    console.print(f"- failed rows: {summary['failed_rows']}")
    console.print(f"- direct-literature rows: {summary['direct_literature_rows']}")
    console.print(f"- composition-based rows: {summary['composition_based_rows']}")
    console.print(f"- strong effects: {summary['strong_effects']}")
    console.print(f"- medium effects: {summary['medium_effects']}")
    console.print(f"- low effects: {summary['low_effects']}")
    console.print(f"- negative/cautionary effects: {summary['negative_or_cautionary_effects']}")
    console.print(f"- supplement-level-only effects: {summary['supplement_level_only_effects']}")
    console.print(
        f"- rows with warnings: {summary['rows_with_normalization_warnings']}"
    )
    console.print(f"Wrote normalized final export summary artifacts under: {out}")


@app.command("merge-final-health-summaries")
def merge_final_health_summaries_command(
    inputs: Annotated[
        Path,
        typer.Option(help="Normalized final export summaries JSONL from Step 2."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for final LLM merge artifacts."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = "deepseek",
    model: Annotated[
        str | None,
        typer.Option(help="Provider model name, e.g. deepseek-v4-pro."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel ingredient workers. Maximum: 30."),
    ] = 10,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N normalized rows for testing."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse completed final merge rows."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate all final merge rows and replace artifacts."),
    ] = False,
    llm_attempts: Annotated[
        int,
        typer.Option(help="LLM attempts per ingredient before recording a row-level failure."),
    ] = 3,
) -> None:
    if not inputs.exists():
        raise typer.BadParameter(f"Missing normalized summaries JSONL: {inputs}")
    if workers < 1 or workers > 30:
        raise typer.BadParameter("--workers must be between 1 and 30.")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero.")
    if llm_attempts < 1:
        raise typer.BadParameter("--llm-attempts must be greater than zero.")

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    try:
        result = run_final_health_summary_merge(
            inputs_path=inputs,
            out_dir=out,
            provider=llm_provider,
            model_provider=model_config.provider,
            model_name=model_config.model_name,
            workers=workers,
            limit=limit,
            resume=resume,
            force=force,
            llm_attempts=llm_attempts,
            progress=print_final_merge_progress,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Final health summary merge complete.")
    console.print(f"- selected input rows: {summary.selected_input_rows}")
    console.print(f"- not processed due to limit: {summary.not_processed_due_to_limit}")
    console.print(f"- processed rows: {summary.processed_rows}")
    console.print(f"- failed rows: {summary.failed_rows}")
    console.print(f"- llm attempts: {summary.llm_attempts}")
    console.print(f"- strong effects: {summary.strong_effects}")
    console.print(f"- medium effects: {summary.medium_effects}")
    console.print(f"- low effects: {summary.low_effects}")
    console.print(f"- negative/cautionary effects: {summary.negative_or_cautionary_effects}")
    console.print(f"- supplement-level-only effects: {summary.supplement_level_only_effects}")
    console.print(f"- legacy claims used: {summary.legacy_claims_used}")
    console.print(f"- legacy claims excluded: {summary.legacy_claims_excluded}")
    console.print(f"- rows with warnings: {summary.rows_with_warnings}")
    console.print(f"Wrote final health summary merge artifacts under: {out}")


@app.command("write-final-main-csv")
def write_final_main_csv_command(
    merges: Annotated[
        Path,
        typer.Option(help="Final health summary merges JSONL from Step 4."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output CSV path for health_reports_ingredients_final_rows.csv."),
    ] = Path("health_reports_ingredients_final_rows.csv"),
) -> None:
    if not merges.exists():
        raise typer.BadParameter(f"Missing final health summary merges JSONL: {merges}")

    try:
        result = write_final_main_csv(
            merges_path=merges,
            out_path=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("Final main health reports CSV written.")
    console.print(f"- rows written: {result.rows_written}")
    console.print(f"- direct-literature rows: {result.direct_literature_rows}")
    console.print(f"- composition-based rows: {result.composition_based_rows}")
    console.print(f"- unknown rows: {result.unknown_rows}")
    console.print(f"- created_at: {result.created_at}")
    console.print(f"Wrote final CSV: {result.out_path}")


@app.command("build-fallback-composition-inputs")
def build_fallback_composition_inputs_command(
    skipped: Annotated[
        Path,
        typer.Option(help="Skipped/uncovered ingredients CSV from final export assembly."),
    ],
    canonical: Annotated[
        Path,
        typer.Option(help="Canonical beverage category CSV with avg_nutriments."),
    ],
    alias_map: Annotated[
        Path,
        typer.Option(help="Reviewed nutriment alias map CSV."),
    ],
    reference: Annotated[
        Path,
        typer.Option(help="Reference-intake table CSV."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for fallback composition input artifacts."),
    ],
) -> None:
    missing = [str(path) for path in [skipped, canonical, alias_map, reference] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing fallback composition input files: " + ", ".join(missing))

    try:
        result = build_fallback_composition_inputs(
            skipped_path=skipped,
            canonical_path=canonical,
            alias_path=alias_map,
            reference_path=reference,
            out_dir=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Fallback composition inputs built.")
    console.print(f"- skipped input rows: {summary['skipped_input_rows']}")
    console.print(f"- fallback profile rows: {summary['fallback_profile_rows']}")
    console.print(f"- unknown placeholder rows: {summary['unknown_placeholder_rows']}")
    console.print(f"- nutriments in profiles: {summary['nutriments_in_profiles']}")
    console.print(f"- profiles with quality flags: {summary['profiles_with_quality_flags']}")
    console.print(f"Wrote fallback artifacts under: {result.out_dir}")


@app.command("combine-composition-summaries")
def combine_composition_summaries_command(
    existing: Annotated[
        Path,
        typer.Option(help="Existing composition ingredient summaries JSONL."),
    ],
    fallback: Annotated[
        Path,
        typer.Option(help="Fallback composition ingredient summaries JSONL."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Combined composition summaries JSONL output path."),
    ],
) -> None:
    missing = [str(path) for path in [existing, fallback] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing composition summary files: " + ", ".join(missing))

    try:
        result = combine_composition_summaries(
            existing_path=existing,
            fallback_path=fallback,
            out_path=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Composition summaries combined.")
    console.print(f"- existing rows: {summary['existing_rows']}")
    console.print(f"- fallback rows: {summary['fallback_rows']}")
    console.print(f"- combined rows: {summary['combined_rows']}")
    console.print(f"Wrote combined summaries: {result.out_path}")


@app.command("complete-final-health-summary-merges")
def complete_final_health_summary_merges_command(
    merges: Annotated[
        Path,
        typer.Option(help="Final health summary merges JSONL after fallback merge."),
    ],
    unknown_placeholders: Annotated[
        Path,
        typer.Option(help="Unknown placeholder JSONL from fallback composition input build."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Complete final health summary merges JSONL output path."),
    ],
) -> None:
    missing = [str(path) for path in [merges, unknown_placeholders] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing final completion files: " + ", ".join(missing))

    try:
        result = complete_final_health_summary_merges(
            merges_path=merges,
            placeholders_path=unknown_placeholders,
            out_path=out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Complete final health summary merges written.")
    console.print(f"- input merge rows: {summary['input_merge_rows']}")
    console.print(f"- placeholder rows: {summary['placeholder_rows']}")
    console.print(f"- complete rows: {summary['complete_rows']}")
    console.print(f"- source type counts: {summary['source_type_counts']}")
    console.print(f"Wrote complete merges: {result.out_path}")


@app.command("write-final-search-companion-csvs")
def write_final_search_companion_csvs_command(
    merges: Annotated[
        Path,
        typer.Option(help="Final health summary merges JSONL from Step 4."),
    ],
    effects_out: Annotated[
        Path,
        typer.Option(help="Output CSV path for ingredient effect rows."),
    ] = Path("health_reports_ingredient_effects_rows.csv"),
    tags_out: Annotated[
        Path,
        typer.Option(help="Output CSV path for ingredient tag rows."),
    ] = Path("health_reports_ingredient_tags_rows.csv"),
) -> None:
    if not merges.exists():
        raise typer.BadParameter(f"Missing final health summary merges JSONL: {merges}")

    try:
        result = write_final_search_companion_csvs(
            merges_path=merges,
            effects_out_path=effects_out,
            tags_out_path=tags_out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("Final search companion CSVs written.")
    console.print(f"- ingredient rows: {result.ingredient_rows}")
    console.print(f"- effect rows written: {result.effect_rows_written}")
    console.print(f"- tag rows written: {result.tag_rows_written}")
    console.print(f"- created_at: {result.created_at}")
    console.print(f"Wrote effects CSV: {result.effects_path}")
    console.print(f"Wrote tags CSV: {result.tags_path}")


@app.command("prepare")
def prepare_command(
    lookup: Annotated[
        Path,
        typer.Option(help="Ingredient health report lookup CSV."),
    ] = Path("health_reports_ingredients_v1_rows.csv"),
    out: Annotated[
        Path,
        typer.Option(help="Output directory for preparation artifacts."),
    ] = Path("ingredient_preparation"),
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: heuristic, deepseek, or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N rows for testing."),
    ] = None,
    chunk_size: Annotated[
        int,
        typer.Option(help="Rows per LLM classification batch."),
    ] = 25,
) -> None:
    if not lookup.exists():
        raise typer.BadParameter(f"Missing lookup CSV: {lookup}")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero")
    if chunk_size < 1 or chunk_size > 100:
        raise typer.BadParameter("--chunk-size must be between 1 and 100")

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    result = prepare_ingredients(
        lookup_path=lookup,
        out_dir=out,
        provider=llm_provider,
        limit=limit,
        chunk_size=chunk_size,
        progress=lambda current, total: console.print(
            f"Classifying ingredient batch {current}/{total}..."
        ),
    )
    summary = result.summary
    console.print(f"Processed {summary.processed_rows} ingredient rows.")
    console.print(f"- research_direct: {summary.research_direct}")
    console.print(f"- reuse_group_evidence: {summary.reuse_group_evidence}")
    console.print(f"- skip_low_value: {summary.skip_low_value}")
    console.print(f"- manual_review: {summary.manual_review}")
    console.print(f"Wrote ingredient preparation artifacts under: {out}")


@app.command("classify-significance")
def classify_significance_command(
    profiles: Annotated[
        Path,
        typer.Option(help="Normalized ingredient profiles JSONL from step 4."),
    ],
    reference: Annotated[
        Path,
        typer.Option(help="Reference-intake table CSV from step 3."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for significance artifacts."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = "deepseek",
    model: Annotated[
        str | None,
        typer.Option(help="Provider model name, e.g. the exact DeepSeek v4 pro identifier."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel ingredient workers. Maximum: 30."),
    ] = 10,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N ingredient profiles for testing."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse completed ingredient classifications."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reclassify all profiles and replace existing artifacts."),
    ] = False,
) -> None:
    missing = [str(path) for path in [profiles, reference] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required significance inputs: " + ", ".join(missing))
    if workers < 1 or workers > 30:
        raise typer.BadParameter("--workers must be between 1 and 30.")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero.")

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    try:
        result = run_nutriment_significance_classification(
            profiles_path=profiles,
            reference_path=reference,
            out_dir=out,
            provider=llm_provider,
            workers=workers,
            limit=limit,
            resume=resume,
            force=force,
            progress=print_significance_progress,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Nutriment significance classification complete.")
    console.print(f"- processed ingredients: {summary.processed_ingredients}")
    console.print(f"- failed ingredients: {summary.failed_ingredients}")
    console.print(f"- classified nutriments: {summary.classified_nutriments}")
    console.print(f"- skipped invalid nutriments: {summary.skipped_invalid_nutriments}")
    console.print(f"- significant: {summary.significant}")
    console.print(f"- minor: {summary.minor}")
    console.print(f"- trace: {summary.trace}")
    console.print(f"- unknown_threshold: {summary.unknown_threshold}")
    console.print(f"Wrote nutriment significance artifacts under: {out}")


@app.command("summarize-nutriments")
def summarize_nutriments_command(
    evidence_matches: Annotated[
        Path,
        typer.Option(help="Nutriment evidence matches JSONL from step 6."),
    ],
    reference: Annotated[
        Path,
        typer.Option(help="Reference-intake table CSV from step 3."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for nutriment summary artifacts."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = "deepseek",
    model: Annotated[
        str | None,
        typer.Option(help="Provider model name, e.g. the exact DeepSeek v4 pro identifier."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel nutriment workers. Maximum: 30."),
    ] = 10,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N matched nutriment records for testing."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse completed nutriment summaries."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate all summaries and replace existing artifacts."),
    ] = False,
    llm_attempts: Annotated[
        int,
        typer.Option(help="LLM attempts per nutriment before recording a row-level failure."),
    ] = 3,
) -> None:
    missing = [str(path) for path in [evidence_matches, reference] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required nutriment summary inputs: " + ", ".join(missing))
    if workers < 1 or workers > 30:
        raise typer.BadParameter("--workers must be between 1 and 30.")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero.")
    if llm_attempts < 1:
        raise typer.BadParameter("--llm-attempts must be greater than zero.")

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    try:
        result = run_nutriment_health_summary_generation(
            evidence_matches_path=evidence_matches,
            reference_path=reference,
            out_dir=out,
            provider=llm_provider,
            model_provider=model_config.provider,
            model_name=model_config.model_name,
            workers=workers,
            limit=limit,
            resume=resume,
            force=force,
            llm_attempts=llm_attempts,
            progress=print_nutriment_summary_progress,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Nutriment health summary generation complete.")
    console.print(f"- eligible matched rows: {summary.eligible_matched_rows}")
    console.print(f"- selected matched rows: {summary.selected_matched_rows}")
    console.print(f"- not processed due to limit: {summary.not_processed_due_to_limit}")
    console.print(f"- processed rows: {summary.processed_rows}")
    console.print(f"- failed rows: {summary.failed_rows}")
    console.print(f"- skipped unmatched rows: {summary.skipped_unmatched_rows}")
    console.print(f"- skipped empty summaries: {summary.skipped_empty_summaries}")
    console.print(f"- llm attempts: {summary.llm_attempts}")
    console.print(f"- strong effects: {summary.strong_effects}")
    console.print(f"- medium effects: {summary.medium_effects}")
    console.print(f"- low effects: {summary.low_effects}")
    console.print(f"- supplement-level effects: {summary.supplement_level_effects}")
    console.print(f"Wrote nutriment summary artifacts under: {out}")


@app.command("build-ingredient-summary-inputs")
def build_ingredient_summary_inputs_command(
    profiles: Annotated[
        Path,
        typer.Option(help="Normalized ingredient profiles JSONL from step 4."),
    ],
    significance: Annotated[
        Path,
        typer.Option(help="Nutriment significance JSONL from step 5."),
    ],
    nutriment_summaries: Annotated[
        Path,
        typer.Option(help="Cleaned nutriment summaries JSONL from step 7."),
    ],
    dose_band_table: Annotated[
        Path,
        typer.Option(help="Reviewed nutriment dose-band table CSV."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for ingredient summary input artifacts."),
    ],
    significance_failures: Annotated[
        Path | None,
        typer.Option(help="Optional nutriment significance failures JSONL."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N ingredient profiles for testing."),
    ] = None,
) -> None:
    required = [profiles, significance, nutriment_summaries, dose_band_table]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required ingredient summary inputs: " + ", ".join(missing))
    if significance_failures is not None and not significance_failures.exists():
        raise typer.BadParameter(f"Missing significance failures JSONL: {significance_failures}")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero.")

    try:
        result = run_ingredient_summary_input_assembly(
            profiles_path=profiles,
            significance_path=significance,
            nutriment_summaries_path=nutriment_summaries,
            dose_band_table_path=dose_band_table,
            significance_failures_path=significance_failures,
            out_dir=out,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Ingredient summary input assembly complete.")
    console.print(f"- ingredient rows: {summary['ingredient_rows']}")
    console.print(f"- ingredients with warnings: {summary['ingredients_with_input_warnings']}")
    console.print(f"- dose-banded nutriments: {summary['dose_banded_nutriments']}")
    console.print(f"- food-level summary nutrients: {summary['food_level_summary_nutrients']}")
    console.print(f"- supplement-level summary nutrients: {summary['supplement_level_summary_nutrients']}")
    console.print(f"- ignored trace nutrients: {summary['ignored_trace_nutrients']}")
    console.print(f"- background nutrients: {summary['minor_or_trace_background_nutrients']}")
    console.print(f"- skipped no-effect summaries: {summary['skipped_no_effect_summary_nutrients']}")
    console.print(f"Wrote ingredient summary input artifacts under: {out}")


@app.command("summarize-ingredients")
def summarize_ingredients_command(
    inputs: Annotated[
        Path,
        typer.Option(help="Ingredient summary input JSONL from step 8a."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for ingredient health summary artifacts."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = "deepseek",
    model: Annotated[
        str | None,
        typer.Option(help="Provider model name, e.g. deepseek-v4-pro."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel ingredient workers. Maximum: 30."),
    ] = 10,
    limit: Annotated[
        int | None,
        typer.Option(help="Only process the first N ingredient summary inputs for testing."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reuse completed ingredient summaries."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate all ingredient summaries and replace artifacts."),
    ] = False,
    llm_attempts: Annotated[
        int,
        typer.Option(help="LLM attempts per ingredient before recording a row-level failure."),
    ] = 3,
) -> None:
    if not inputs.exists():
        raise typer.BadParameter(f"Missing ingredient summary input JSONL: {inputs}")
    if workers < 1 or workers > 30:
        raise typer.BadParameter("--workers must be between 1 and 30.")
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be greater than zero.")
    if llm_attempts < 1:
        raise typer.BadParameter("--llm-attempts must be greater than zero.")

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    try:
        result = run_ingredient_health_summary_generation(
            inputs_path=inputs,
            out_dir=out,
            provider=llm_provider,
            model_provider=model_config.provider,
            model_name=model_config.model_name,
            workers=workers,
            limit=limit,
            resume=resume,
            force=force,
            llm_attempts=llm_attempts,
            progress=print_ingredient_summary_progress,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = result.summary
    console.print("Ingredient health summary generation complete.")
    console.print(f"- selected input rows: {summary.selected_input_rows}")
    console.print(f"- not processed due to limit: {summary.not_processed_due_to_limit}")
    console.print(f"- processed rows: {summary.processed_rows}")
    console.print(f"- failed rows: {summary.failed_rows}")
    console.print(f"- llm attempts: {summary.llm_attempts}")
    console.print(f"- strong effects: {summary.strong_effects}")
    console.print(f"- medium effects: {summary.medium_effects}")
    console.print(f"- low effects: {summary.low_effects}")
    console.print(f"- negative/cautionary effects: {summary.negative_or_cautionary_effects}")
    console.print(f"- dominant nutrients: {summary.dominant_nutrients}")
    console.print(f"- supplement-level-only effects: {summary.supplement_level_only_effects}")
    console.print(f"- ignored trace nutrients: {summary.ignored_trace_nutrients}")
    console.print(f"Wrote ingredient health summary artifacts under: {out}")


@app.command("study")
def study_command(
    ingredient: Annotated[str, typer.Argument(help="Ingredient to study.")],
    lookup: Annotated[
        Path,
        typer.Option(help="Ingredient health report lookup CSV."),
    ] = Path("health_reports_ingredients_v1_rows.csv"),
    canonical_beverage_id: Annotated[
        str | None,
        typer.Option(help="Resolve the target lookup row by canonical_beverage_id."),
    ] = None,
    search_as_ingredient_name: Annotated[
        bool,
        typer.Option(
            "--search-as-ingredient-name",
            help=(
                "Use the ingredient argument as the canonical search name even when "
                "--canonical-beverage-id resolves a representative lookup row."
            ),
        ),
    ] = False,
    depth: Annotated[str, typer.Option(help="light, standard, or deep.")] = "standard",
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: heuristic, deepseek, or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    out: Annotated[
        Path,
        typer.Option(help="Output directory for ingredient runs."),
    ] = Path("ingredient_runs"),
    retrieve_full_text: Annotated[
        bool,
        typer.Option(
            "--retrieve-full-text",
            help="Retrieve available paper body text into raw_texts artifacts.",
        ),
    ] = False,
    full_text_workers: Annotated[
        int,
        typer.Option(help="Parallel workers for --retrieve-full-text."),
    ] = 4,
) -> None:
    if depth not in {"light", "standard", "deep"}:
        raise typer.BadParameter("depth must be one of: light, standard, deep")
    if not lookup.exists():
        raise typer.BadParameter(f"Missing lookup CSV: {lookup}")
    if full_text_workers < 1:
        raise typer.BadParameter("--full-text-workers must be greater than zero")
    try:
        result = run_ingredient_study(
            ingredient_name=ingredient,
            lookup_path=lookup,
            depth=cast(StudyDepth, depth),
            out_dir=out,
            provider=provider,
            model=model,
            canonical_beverage_id=canonical_beverage_id,
            search_as_ingredient_name=search_as_ingredient_name,
            retrieve_full_text=retrieve_full_text,
            full_text_workers=full_text_workers,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Created ingredient run: {result.run_dir}")
    if retrieve_full_text:
        console.print("Raw text retrieval artifacts: raw_texts.json, raw_texts.md, raw_texts/")


@app.command("ingest-manual-full-text")
def ingest_manual_full_text_command(
    run: Annotated[
        Path,
        typer.Option(help="Ingredient run directory with a manual_full_text_queue."),
    ],
) -> None:
    required = [
        run / "ingredient_packet.json",
        run / "sources.json",
        run / "raw_texts.json",
        run / "manual_full_text_queue" / "manual_full_text_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required manual ingestion artifacts: " + ", ".join(missing))

    try:
        result = ingest_manual_full_text_queue(run)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"Manual full-text ingestion complete for: {result.run_dir}")
    console.print(f"- ingested PDFs: {len(result.ingested)}")
    console.print(f"- missing PDFs marked paywalled/unavailable: {len(result.missing_assumed_paywalled)}")
    console.print(f"- parse failures: {len(result.failed)}")
    console.print(f"- unchanged already-full-text sources: {len(result.unchanged)}")
    console.print(f"- full_text_found: {result.counts.get('full_text_found', 0)}")
    console.print(f"- paywalled: {result.counts.get('paywalled', 0)}")
    console.print(f"- pdf_parse_failed: {result.counts.get('pdf_parse_failed', 0)}")


@app.command("propose-claims")
def propose_claims_command(
    run: Annotated[
        Path,
        typer.Option(help="Ingredient run directory with sources.json and raw_texts.json."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel paper workers for claim proposal."),
    ] = 1,
) -> None:
    required = [
        run / "ingredient_packet.json",
        run / "sources.json",
        run / "raw_texts.json",
        run / "raw_texts",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required proposal artifacts: " + ", ".join(missing))
    if workers < 1:
        raise typer.BadParameter("--workers must be greater than zero.")
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    claims = propose_ingredient_claims_from_raw_texts(
        run_dir=run,
        provider=llm_provider,
        progress=print_claim_progress,
        workers=workers,
    )
    console.print(f"Proposed {len(claims)} ingredient claims from raw texts.")
    console.print(f"Wrote proposed ingredient claim artifacts under: {run}")


@app.command("validate-claims")
def validate_claims_command(
    run: Annotated[
        Path,
        typer.Option(help="Ingredient run directory containing proposed claims and raw texts."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: deepseek or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    max_body_chars: Annotated[
        int,
        typer.Option(help="Maximum sanitized paper characters accepted without truncation."),
    ] = 750_000,
    workers: Annotated[
        int,
        typer.Option(help="Parallel paper workers for claim validation."),
    ] = 1,
) -> None:
    required = [
        run / "ingredient_packet.json",
        run / "proposed_ingredient_claims.json",
        run / "sources.json",
        run / "raw_texts.json",
        run / "raw_texts",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required validation artifacts: " + ", ".join(missing))
    if max_body_chars < 1:
        raise typer.BadParameter("--max-body-chars must be greater than zero.")
    if workers < 1:
        raise typer.BadParameter("--workers must be greater than zero.")
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    accepted, rejected, failures = validate_ingredient_claims(
        run_dir=run,
        provider=llm_provider,
        progress=print_validation_progress,
        max_body_chars=max_body_chars,
        workers=workers,
    )
    console.print(
        f"Validation complete: {len(accepted)} accepted, "
        f"{len(rejected)} rejected, {len(failures)} paper failures."
    )
    console.print(f"Wrote validation artifacts under: {run}")


@app.command("audit-claims")
def audit_claims_command(
    run: Annotated[
        Path,
        typer.Option(help="Ingredient run directory containing validated ingredient claims."),
    ],
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider: heuristic, deepseek, or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    workers: Annotated[
        int,
        typer.Option(help="Parallel paper workers for claim auditing."),
    ] = 1,
) -> None:
    required = [
        run / "ingredient_packet.json",
        run / "proposed_ingredient_claims.json",
        run / "validated_ingredient_claims.json",
        run / "sources.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing required audit artifacts: " + ", ".join(missing))
    if workers < 1:
        raise typer.BadParameter("--workers must be greater than zero.")
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    result = audit_ingredient_claims(
        run_dir=run,
        provider=llm_provider,
        workers=workers,
        progress=print_audit_progress,
    )
    if result.skipped:
        console.print(f"Ingredient claim audit skipped: {result.skip_reason}")
        return
    console.print(
        f"Ingredient claim audit complete: {len(result.audited_claims)} passed, "
        f"{len(result.rejected_claims)} excluded."
    )
    console.print(f"Wrote ingredient claim audit artifacts under: {run}")


@app.command("synthesize-report")
def synthesize_report_command(
    run: Annotated[
        Path,
        typer.Option(help="Research run containing proposed and validated claim artifacts."),
    ],
    lookup: Annotated[
        Path,
        typer.Option(help="Ingredient health report lookup CSV."),
    ] = Path("health_reports_ingredients_v1_rows.csv"),
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to the ingredient run directory."),
    ] = None,
    ingredient_name: Annotated[
        str | None,
        typer.Option(
            help="Override packet input name for ingredient row matching and synthesis labeling."
        ),
    ] = None,
    canonical_beverage_id: Annotated[
        str | None,
        typer.Option(help="Resolve the target lookup row by canonical_beverage_id."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(help="LLM provider for optional polish: heuristic, deepseek, or openai-compatible."),
    ] = None,
    model: Annotated[str | None, typer.Option(help="Provider model name.")] = None,
    replace_existing: Annotated[
        bool,
        typer.Option(
            "--replace-existing/--preserve-existing",
            help="Replace existing positive/negative fields when generated text is available.",
        ),
    ] = True,
) -> None:
    packet_artifacts = [run / "ingredient_packet.json", run / "packet.json"]
    proposed_artifacts = [run / "proposed_ingredient_claims.json", run / "proposed_claims.json"]
    validated_artifacts = [
        run / "audited_ingredient_claims.json",
        run / "validated_ingredient_claims.json",
        run / "validated_claims.json",
    ]
    required = [run / "sources.json", lookup]
    missing = [str(path) for path in required if not path.exists()]
    if not any(path.exists() for path in packet_artifacts):
        missing.append("ingredient_packet.json or packet.json")
    if not any(path.exists() for path in proposed_artifacts):
        missing.append("proposed_ingredient_claims.json or proposed_claims.json")
    if not any(path.exists() for path in validated_artifacts):
        missing.append("validated_ingredient_claims.json or validated_claims.json")
    if missing:
        raise typer.BadParameter("Missing required synthesis artifacts: " + ", ".join(missing))

    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    out_dir = out or run
    try:
        result = synthesize_ingredient_report(
            run_dir=run,
            lookup_path=lookup,
            out_dir=out_dir,
            provider=llm_provider,
            ingredient_name=ingredient_name,
            canonical_beverage_id=canonical_beverage_id,
            replace_existing=replace_existing,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("Ingredient report synthesis complete.")
    console.print(
        "Updated row: "
        f"{result.updated_row.get('canonical_beverage_name')} "
        f"({result.updated_row.get('canonical_beverage_id')})"
    )
    console.print(f"Used accepted claims: {len(result.claim_sources)}")
    console.print(f"Rejected synthesis items: {len(result.rejected_items)}")
    console.print(f"Wrote ingredient export artifacts under: {out_dir}")


if __name__ == "__main__":
    app()
