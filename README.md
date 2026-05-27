# Sipz Nutrient Research Agent

A local-first Python research agent that studies nutrients and bioactives, validates health-effect claims against paper body text, and writes a Supabase-ready CSV.

The project is built around a simple rule:

```txt
LLM can propose.
LLM cannot prove.
Proof must be grounded in retrievable paper text.
```

## Current Status

This repository implements the deterministic CLI MVP described in:

- `SIPZ_NUTRIENT_RESEARCH_AGENT_PLAN.md`
- `MVP_IMPLEMENTATION_SPEC.md`
- `AI_BUILD_HANDOFF.md`

The MVP uses demo fixtures and does not call Firecrawl, download PDFs, call LLMs, or write to Supabase.

## Setup

```bash
uv sync
```

## Demo

```bash
uv run sipz-agent study "fluoride" --demo
uv run sipz-agent audit --run research_runs/<created-run-dir>
uv run sipz-agent export --run research_runs/<created-run-dir> --format csv
```

Each study run writes:

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

`effects.csv` is the primary artifact intended for Supabase import.

## Test

```bash
uv run pytest
```

## Safety Boundary

This project supports research and data curation. It does not provide medical advice, dosage instructions, diagnosis, or treatment recommendations.
