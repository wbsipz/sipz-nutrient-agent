# Agent Home Blueprint

> Status: implemented reference. This document records the original harness design; runtime rules
> live in `SYSTEM.md` and `AGENTS.md`, and all workflow skills listed below are installed.

This blueprint describes what each `agent-home` file should do for the Sipz nutrient research agent. It is based on `nutrientagentplan.txt`, whose core design is a single orchestrator that loads focused skills, retrieves and filters papers, runs parallel claim extraction and validation subagents, and writes auditable outputs.

## Product Role

The agent researches the human-health effects of orally ingested nutrients, bioactives, extracts, additives, or related substances. It should not behave like a general coding agent unless explicitly asked. Its primary job is to search sources, read papers, extract candidate health claims, validate those claims against body text, and write structured research outputs.

The core query shape is:

```txt
What is the effect of <substance> on human health when orally ingested?
```

The workflow should preserve a clear chain from search query to paper to claim to validator quote to final pass/fail decision.

## `SYSTEM.md`

Purpose: define the agent's highest-level identity and hard behavioral rules.

What it should contain:

- The agent is a research workflow agent, not a coding assistant.
- The agent researches orally ingested effects on human health.
- The agent must never cite a source it has not read.
- The agent must separate claims, evidence, interpretation, and uncertainty.
- The agent must reject or flag claims that are not grounded in paper body text.
- The agent must treat title, abstract, and conclusion as screening context, not validator proof.
- The validator stage must use paper body/results/methods text, excluding title, abstract, and conclusion where possible.
- The agent must keep research artifacts auditable.
- The agent must write final outputs under `workspace/reports` and scratch work under `workspace/notes`.

Recommended expansion:

```md
You are the Sipz Nutrient Research Agent.

Your mission is to answer:
"What is the effect of <substance> on human health when orally ingested?"

You operate as a staged research workflow:
1. Resolve substance aliases and oral-ingestion framing.
2. Generate retrieval queries.
3. Retrieve candidate papers.
4. Filter by title and abstract.
5. Extract candidate claims from eligible papers.
6. Validate claims against body text only.
7. Write auditable outputs.

Hard rules:
- Never cite a source you have not read.
- Never accept a claim without a grounded quote.
- Do not use abstracts as validator evidence.
- Reject claims that do not support the oral-ingestion human-health query.
- Prefer rejection over overstating evidence.
```

## `AGENTS.md`

Purpose: provide operational context, workspace map, and the canonical workflow for the TUI/session.

What it should contain:

- Directory meanings for `workspace/sources`, `workspace/extracted`, `workspace/notes`, and `workspace/reports`.
- The loop structure from `nutrientagentplan.txt`.
- The roles of orchestrator, retriever, screener, researcher, and validator.
- Rules for session/run isolation between substances.
- Expected final artifacts.

Recommended sections:

```md
# Sipz Research Agent Operating Context

## Workspace
- `workspace/sources`: original PDFs, HTML, metadata, citations, and retrieval payloads.
- `workspace/extracted`: extracted body text and cleaned paper sections.
- `workspace/notes`: scratch notes, retrieval failures, unresolved questions, and manual review queues.
- `workspace/reports`: final research packets and cited summaries.

## Canonical Workflow
1. Resolve aliases and substance class.
2. Generate search queries for oral-ingestion human-health effects.
3. Retrieve candidate papers until target count is reached.
4. Filter by title for plausibility.
5. Filter by abstract for relevance.
6. For each retained paper, extract candidate claims.
7. Validate each claim against body text stripped of title, abstract, and conclusion.
8. Retry only recoverable provider/schema failures, with the original claim unchanged.
9. Save accepted and rejected outputs.

## Acceptance Criteria
- Accepted claims need a quote found in body text.
- The validator must reverse-engineer the original query or a semantically equivalent query.
- Claims that do not support oral ingestion in humans fail.
- Evidence limitations must be preserved.
```

## `settings.json`

Purpose: store stable local Pi/runtime preferences.

Current intended content:

```json
{
  "defaultModelProvider": "deepseek",
  "defaultModelId": "deepseek-v4-pro",
  "workspaceRoot": "workspace",
  "theme": "dark"
}
```

Runtime model selection is role-specific. `ORCHESTRATOR_MODEL_PROVIDER`,
`ORCHESTRATOR_MODEL_ID`, and `ORCHESTRATOR_THINKING` configure Pi. `WORKER_MODEL_PROVIDER` and
`WORKER_MODEL_ID` configure screening, extraction, body-adequacy, and validation calls. The
orchestrator must not select worker models in tool arguments.

Recommended future additions:

- `retrievalTargetPapers`: target number of candidate papers before filtering.
- `maxValidationAttempts`: default `3`, limited to recoverable execution failures.
- `validatorExcludesSections`: `["title", "abstract", "conclusion"]`.
- `acceptedEvidenceRequiresQuote`: `true`.
- `finalOutputDir`: `workspace/reports`.

Do not store secrets here. API keys belong in `.env`, shell environment, or Pi auth storage.

## `sessions/`

Purpose: persisted Pi session history for each research run.

Use sessions for per-run state, conversation history, compaction, resume/debug, and isolation between nutrients. Start a fresh session for each new substance so one completed run does not leak assumptions into the next run.

Stable operating instructions belong in `SYSTEM.md`, `AGENTS.md`, and focused skill docs. Research outputs belong in `workspace/reports`.

## `skills/web-fetch/SKILL.md`

Purpose: define how retrieval/search should work once live web retrieval is enabled.

It should evolve from a placeholder into the retrieval skill loaded by the orchestrator.

What it should contain:

- Supported sources: PubMed, Europe PMC, Semantic Scholar, Crossref, OpenAlex, clinical trial registries, and eventually PDF/full-text sources.
- Query generation requirements.
- Search result metadata schema.
- Retrieval loop rules.
- When to stop searching.
- How to write retrieved metadata into `workspace/sources`.

Recommended content outline:

```md
# Web Fetch / Literature Retrieval Skill

## Goal
Retrieve candidate literature for:
"What is the effect of <substance> on human health when orally ingested?"

## Inputs
- substance name
- aliases
- parent class
- oral-ingestion framing
- target candidate paper count

## Query Strategy
- Generate direct substance queries.
- Generate alias queries.
- Generate parent-class queries when direct evidence is sparse.
- Include human, oral, ingestion, dietary, supplement, trial, review terms.

## Candidate Record Schema
- source_id
- title
- authors
- year
- doi
- pmid
- abstract
- source_url
- retrieval_query
- retrieval_source

## Stop Criteria
- target candidate count reached, or
- no new relevant candidates after N query expansions.

## Output
Write source metadata under `workspace/sources/<run-id>/`.
```

## `skills/send-email/SKILL.md`

Purpose: should remain disabled for the MVP unless explicit human approval and an email integration are added.

Recommended role:

- Draft emails only.
- Never send automatically.
- Store drafts in `workspace/notes`.
- Require explicit approval and future tool integration.

This skill is not part of the nutrient research loop and should stay inert.

## Implemented Workflow Skills

The following workflow skills are implemented under `agent-home/skills/`:

- `agent-help`: explains supported subjects, workflow stages, usage examples, outputs, safeguards,
  and limitations without starting research implicitly.

```txt
agent-home/skills/agent-help/
  SKILL.md
agent-home/skills/retrieval/
  SKILL.md
agent-home/skills/paper-screening/
  SKILL.md
agent-home/skills/claim-extraction/
  SKILL.md
agent-home/skills/claim-validation/
  SKILL.md
agent-home/skills/report-export/
  SKILL.md
```

### `skills/retrieval/SKILL.md`

Defines alias generation, oral-ingestion framing, search query JSON schema, target paper count, and retrieval loop memory updates.

### `skills/paper-screening/SKILL.md`

Defines title and abstract filters:

- include plausible oral ingestion
- include human health outcomes
- exclude irrelevant exposure routes
- exclude marketing/supplement sales pages
- exclude unverifiable or off-topic papers

### `skills/claim-extraction/SKILL.md`

Defines the researcher subagent:

- receives a retained paper
- extracts health claims tied to oral ingestion in humans
- records study design, population, dose/exposure, outcome, limitations, and source reference
- does not decide final acceptance

### `skills/claim-validation/SKILL.md`

Defines the blind validator:

- receives candidate claim and body text with title, abstract, and conclusion removed
- must find supporting quotes in body text
- must reconstruct the original query
- must reject unsupported or mismatched claims

### `skills/report-export/SKILL.md`

Defines final artifacts:

- accepted claims
- rejected claims
- source registry
- quote registry
- run summary
- audit trail

## Cross-Run State

Do not keep automatic cross-run memory in `agent-home`. Per-run state belongs in `sessions/`, and final research artifacts belong in `workspace/reports`.

If the project later needs structured global data, add it deliberately with a concrete schema and clear read/write rules. Do not store per-substance findings there.

## `sessions/`

Purpose: Pi interactive session storage.

Do not hand-edit session JSONL files except for debugging. Treat them as runtime state, not product configuration.

## `auth.json`

Purpose: Pi credential storage if created by the TUI or login flow.

Do not commit secrets. Prefer `.env` for local development and keep `agent-home/auth.json` ignored or local-only.

## `bin/`

Purpose: runtime helper binaries installed or discovered by Pi.

Do not use this directory for project source code. Treat it as generated/runtime support.

## Recommended Final Agent-Home Shape

```txt
agent-home/
  SYSTEM.md
  AGENTS.md
  BLUEPRINT.md
  settings.json
  sessions/
  skills/
    retrieval/
      SKILL.md
    paper-screening/
      SKILL.md
    claim-extraction/
      SKILL.md
    claim-validation/
      SKILL.md
    report-export/
      SKILL.md
    send-email/
      SKILL.md
    web-fetch/
      SKILL.md
```

## Implementation History

The original blueprint milestones are complete: `SYSTEM.md` contains the proof policy,
`AGENTS.md` defines the staged workflow, session state is isolated, and the retrieval, screening,
claim-extraction, claim-validation, and report-export skills are installed. `send-email` remains
deliberately inert and outside the research pipeline.
