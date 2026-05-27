# Agent Instructions

This repository is the Sipz Nutrient Research Agent. It is intended to be worked on by AI coding agents and human reviewers.

## Read First

Before making implementation changes, read:

1. `SIPZ_NUTRIENT_RESEARCH_AGENT_PLAN.md`
2. `MVP_IMPLEMENTATION_SPEC.md`
3. `AI_BUILD_HANDOFF.md`

The plan explains the product. The MVP spec is the implementation contract. The handoff file is the short execution guide.

## Current Build Target

The first milestone is a deterministic Python CLI demo:

```bash
uv run sipz-agent study "fluoride" --demo
uv run sipz-agent audit --run research_runs/<created-run-dir>
```

The demo must produce:

```txt
effects.csv
validated_claims.json
rejected_claims.json
sources.json
packet.json
summary.md
audit_log.jsonl
```

## Core Rules

- Keep the MVP CLI-first.
- `effects.csv` is the primary output for Supabase import.
- Do not add a web UI until the CLI demo is complete.
- Do not write to Supabase in the MVP.
- Do not add live Firecrawl or PDF retrieval until demo mode passes tests.
- Keep all generated claims auditable.
- Accepted claims must have grounded body-text quotes.
- Validator context must exclude abstracts.
- Quote evidence belongs in `validated_claims.json`, linked by `effect_row_id`.

## Project Skills

Use the local skill docs when working on specific areas:

- `skills/nutrient-research/SKILL.md`
- `skills/quote-grounding/SKILL.md`
- `skills/sipz-export/SKILL.md`

These are lightweight project skills, not external dependencies.

## Preferred Workflow

1. Make small, focused changes.
2. Add or update tests for behavior changes.
3. Run:

```bash
uv run pytest
```

4. If implementing CLI behavior, also run the relevant CLI command.

## Language And Tooling

This project uses Python because the hard parts are research automation, paper text handling, future PDF extraction, source retrieval, and data curation.

This project uses Pydantic because every generated artifact needs explicit runtime validation before it is written.

This project uses pytest because the most important early tests are pure functions: quote grounding, schema validation, CSV generation, and artifact auditing.
