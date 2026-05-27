# AI Build Handoff

## Role

You are implementing the Sipz Nutrient Research Agent.

Read these files before coding:

1. `SIPZ_NUTRIENT_RESEARCH_AGENT_PLAN.md`
2. `MVP_IMPLEMENTATION_SPEC.md`
3. `AGENTS.md`

The MVP is a Python CLI that writes a Supabase-ready CSV plus quote-grounding audit artifacts.

## Build Target

The core command is:

```bash
uv run sipz-agent study "fluoride" --demo
```

It must produce:

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

Then this command must pass:

```bash
uv run sipz-agent audit --run research_runs/<timestamp>_fluoride
```

## Implementation Priorities

Build in this order:

1. Pydantic schemas
2. Demo corpus loader
3. Quote grounding utilities
4. Deterministic demo claim extractor
5. Deterministic demo claim validator
6. Effect CSV row builder
7. Artifact writer
8. `study` CLI command
9. `audit` CLI command
10. Tests
11. README/docs

Do not begin with live web search or Firecrawl. The MVP must first work in deterministic demo mode.

## Required Technical Choices

Use:

- Python 3.11+
- uv
- Pydantic
- Typer
- pytest

## Critical Product Rules

### LLMs Can Propose, Not Prove

```txt
LLM can propose.
LLM cannot prove.
Proof must be grounded in retrievable paper text.
```

For the MVP, deterministic fixtures stand in for LLM output.

### Validator Context Excludes Abstracts

The validator must operate on paper body text, not abstract text.

### Accepted Claims Need Grounded Quotes

An accepted claim must have at least one supporting quote whose text is found in the paper body.

If a quote is not found, the claim must be rejected or fail audit.

### CSV Is The Primary Export

`effects.csv` must include:

```txt
id,nutrient_id,effect_slug,effect_label,description,score,evidence_level,tags,sources,created_at,updated_at,nutrient_name,match_status,match_confidence,match_notes
```

`tags` and `sources` are JSON strings inside CSV cells.

Quote evidence must not be mixed into `effects.csv`. Store it in `validated_claims.json`, linked by `effect_row_id`.

## Do Not Build Yet

Do not build these until the deterministic MVP passes:

- real Firecrawl search
- PDF download/extraction
- Supabase writes
- web UI
- background workers
- Docker deployment
- deep research scheduler

## Final Verification

Before stopping, run:

```bash
uv run pytest
uv run sipz-agent study "fluoride" --demo
uv run sipz-agent audit --run <created-run-dir>
```

If any command cannot be run, state exactly why in the final response.
