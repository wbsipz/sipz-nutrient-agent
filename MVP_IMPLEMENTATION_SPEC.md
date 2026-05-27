# MVP Implementation Spec

## Objective

Build a CLI-only Python proof of concept for the Sipz Nutrient Research Agent.

The MVP demonstrates the research shape without production database writes:

```txt
Input nutrient or bioactive name
  -> find or load candidate papers
  -> extract candidate health claims
  -> validate claims against paper body text
  -> verify validator quotes with string matching
  -> produce Supabase-ready effects.csv
  -> produce auditable research artifacts
```

## Non-Goals

Do not build these in the MVP:

- web UI
- Supabase writes
- Sipz admin import endpoint
- background job queue
- authentication
- multi-user state
- paid model-specific assumptions
- autonomous medical advice or dosage recommendations

## Stack

Use:

- Python 3.11+
- uv
- Pydantic
- Typer
- Rich
- pytest
- orjson

## Project Structure

```txt
.
  pyproject.toml
  .env.example
  README.md
  docs/
    architecture.md
    evidence-grading.md
    artifact-format.md
  src/sipz_agent/
    cli.py
    __main__.py
    schemas/
      artifacts.py
      effects.py
      claims.py
      citations.py
    core/
      orchestrator.py
      retrieval.py
      extraction.py
      validation.py
      quote_grounding.py
      synthesis.py
      audit.py
      artifacts.py
      models.py
      config.py
  tests/
    test_quote_grounding.py
    test_effects_csv.py
    test_demo_run.py
  examples/demo-corpus/
    chlorogenic-acids.json
    fluoride.json
  research_runs/
    .gitkeep
```

## CLI Commands

Primary command:

```bash
uv run sipz-agent study "fluoride" --demo
```

Supported flags:

```txt
--demo
  Use bundled demo corpus instead of live web search.

--depth light|standard|deep
  Defaults to standard.

--out <dir>
  Defaults to research_runs.
```

Audit command:

```bash
uv run sipz-agent audit --run research_runs/<run_id>
```

Export command:

```bash
uv run sipz-agent export --run research_runs/<run_id> --format csv
```

## Output Artifacts

Each run creates:

```txt
research_runs/<timestamp>_fluoride/
  effects.csv
  validated_claims.json
  rejected_claims.json
  sources.json
  packet.json
  summary.md
  audit_log.jsonl
```

`effects.csv` is the primary output for Supabase import.

## Environment Variables

`.env.example` should include:

```bash
MODEL_PROVIDER=heuristic
MODEL_NAME=heuristic-demo
OPENAI_COMPATIBLE_BASE_URL=
OPENAI_COMPATIBLE_API_KEY=
FIRECRAWL_API_KEY=
NCBI_EMAIL=
```

The MVP must work with no API keys when `--demo` is used.

## CSV Contract

`effects.csv` must use exactly these columns:

```txt
id,nutrient_id,effect_slug,effect_label,description,score,evidence_level,tags,sources,created_at,updated_at,nutrient_name,match_status,match_confidence,match_notes
```

Rules:

- `tags` is a JSON string array in one CSV cell.
- `sources` is a JSON string array in one CSV cell.
- timestamps are ISO 8601 UTC strings.
- only accepted, quote-grounded claims appear in `effects.csv`.
- rejected or ungrounded claims appear only in `rejected_claims.json`.

## Pipeline Contracts

Candidate finder:

- demo mode loads fixtures from `examples/demo-corpus`.
- live mode raises `live_retrieval_not_implemented`.

Claim extractor:

- demo mode uses deterministic fixture claims.
- future mode may call an LLM provider.

Validator:

- receives paper body text, not abstract text.
- accepted claims require at least one supporting quote.
- unsupported or over-scoped claims go to `rejected_claims.json`.

Quote grounding:

```txt
exact
normalized_whitespace
dehyphenated_ligature_normalized
not_found
```

Effect CSV builder:

- only accepted, quote-grounded claims become CSV rows.
- rows validate with Pydantic before writing.
- `insufficient` evidence is not exported.

## Tests

Minimum tests:

- exact quote match passes.
- normalized whitespace quote match passes.
- dehyphenated quote match passes.
- missing quote rejects claim.
- accepted claim produces a valid CSV row.
- rejected claim is excluded from `effects.csv`.
- demo `study` command writes all expected artifacts.
- `audit` fails if an accepted claim has no grounded quote.

## Definition Of Done

The MVP is done when:

- `uv sync` works.
- `uv run pytest` passes.
- `uv run sipz-agent study "fluoride" --demo` creates a complete research run.
- `uv run sipz-agent audit --run <created-run>` passes.
- `effects.csv` contains at least one Supabase-ready row.
- `validated_claims.json` links back to that row by `effect_row_id`.
- every accepted validator quote has a successful quote-grounding status.
