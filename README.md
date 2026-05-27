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

The package also exposes the longer command alias:

```bash
uv run sipz-nutrient-agent study "fluoride" --demo
```

## LLM Provider Scaffolding

The deterministic MVP still uses bundled demo fixtures for retrieval/extraction, but the CLI now accepts provider configuration so live LLM-backed steps can be added behind the same interface.

For DeepSeek, create a local `.env` file or export the key in your shell:

```bash
DEEPSEEK_API_KEY=sk-...
```

Then run with:

```bash
uv run sipz-nutrient-agent study "fluoride" --demo --provider deepseek
```

Useful model overrides:

```bash
uv run sipz-nutrient-agent study "fluoride" --demo --provider deepseek --model deepseek-chat
uv run sipz-nutrient-agent study "fluoride" --demo --provider deepseek --model deepseek-reasoner
```

The provider metadata is recorded in `packet.json`. Live candidate-paper retrieval is now implemented for PubMed, Europe PMC, OpenAlex, Semantic Scholar, Crossref, and optional Firecrawl supplemental search. Arbitrary nutrients such as `magnesium` can collect real `sources.json` records, but LLM-backed claim extraction/validation is still a later step, so live runs may produce zero `effects.csv` rows until that stage is implemented.

Each study run writes:

```txt
research_runs/<timestamp>_fluoride/
  effects.csv
  validated_claims.json
  rejected_claims.json
  sources.json
  sources.md
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
