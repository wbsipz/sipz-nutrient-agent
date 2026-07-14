# Architecture

## Design Goal

Sipz Nutrient Agent separates probabilistic research decisions from deterministic evidence and
artifact controls. Models decide what to search, which records appear relevant, and what claims a
paper may support. Code controls schemas, state, retries, text boundaries, quote matching, stopping,
and export.

## Runtime Components

| Component | Responsibility |
|---|---|
| Pi session | Conversational orchestration, intent routing, and adaptive query selection |
| Skills | Stage-specific operating instructions and acceptance criteria |
| TypeScript tools | Typed Pi-facing contracts, progress events, and Python bridge execution |
| Python workflow | Stage ordering, persistence, resumption, concurrency, and report assembly |
| Worker model | Screening, claim extraction, body adequacy, and claim validation |
| Structured APIs | Candidate metadata and open-access location discovery |
| Workspace | Per-run source, text, claim, validation, and report artifacts |

## Stage State Machine

```text
missing
  -> candidates_retrieved
  -> sources_screened
  -> full_text_complete
  -> claims_extracted
  -> claims_validated
  -> report_exported
```

`inspect_research_state` verifies artifacts before a transition. `advance_research_pipeline`
executes only missing prerequisites through a requested terminal stage. If usable full-text
coverage is low, it returns a structured checkpoint instead of silently lowering the target. The
orchestrator can then submit untried queries for another bounded round.

## Trust Boundaries

### Model-owned decisions

- query and alias suggestions;
- title/abstract relevance;
- candidate claim formulation;
- body-text adequacy assessment;
- claim scope, limitations, and evidence interpretation.

### Code-owned decisions

- Pydantic schema validation;
- source deduplication and identifier reconciliation;
- maximum rounds, workers, retries, and candidates;
- required artifact existence and resumption;
- title/abstract/conclusion/reference removal;
- exact and normalized quote location;
- rejection of quotes labeled as context rather than result evidence;
- report counts, paths, formats, and provenance.

## Validation Contract

The claim validator receives the proposed claim and sanitized body text. It must identify the
requested entity, exposure route, population, dose/form, outcome, direction, and limitations. A
supported decision requires at least one exact quote that reports a result or review synthesis.
Quotes that only establish study count, methods, background, or article identity cannot prove a
health outcome. Different species, mixed interventions, and non-oral routes are ineligible unless
the research target explicitly names that entity or formulation.

## Failure Handling

Each stage writes status and failure artifacts. Transient provider and malformed-schema failures
are retried with bounded attempts. Substantive validation failures are not repeatedly sent to a
model because doing so can pressure the model to invent support. Full-text failures retain attempted
URLs and normalized statuses such as `paywalled`, `abstract_only`, or `blocked_by_cloudflare`.

## Concurrency

Network retrieval and independent model work are parallelized with separate worker limits. Claims
from one paper do not share model context with another paper. Results are reconciled and written in
stable order after workers complete.
