---
name: claim-extraction
description: Extracts structured candidate health claims from retained paper body text while preserving oral exposure, population, dose, outcome, study design, and limitations. Use after relevant papers have been screened and full text has been retrieved, or when the user asks to propose claims from one or more papers.
compatibility: Requires retained paper metadata and usable source text.
---

# Claim Extraction Skill

Use this skill for each retained paper after body text or usable source text has been retrieved.

## Goal

Extract candidate health-effect claims tied to oral ingestion in humans. This skill proposes claims only; it does not prove them.

## Inputs

- retained paper metadata
- abstract for orientation only
- paper body text when available
- original substance query
- substance aliases and exposure framing

## Extraction Rules

- Extract only claims related to oral consumption and human health.
- Preserve study type, population, exposure form, dose or serving, duration, comparator, outcome, direction, and limitations.
- Separate food-level, beverage/ingredient-level, and supplement-level exposure.
- Keep uncertainty and "further research needed" language when present.
- Do not convert mechanistic background into a user-facing health claim.
- Do not generalize isolated-compound findings to whole ingredients unless the paper supports that bridge.
- Do not treat related biological species as aliases. Extract a claim only when its evidence applies
  to the requested species or to an explicitly requested broader genus/class.
- Do not generalize supplement-level results to normal food levels unless the paper supports that bridge.

## Candidate Claim Fields

Each candidate claim should include:

- `claim_id`
- `source_id`
- `substance`
- `health_effect`
- `claim_text`
- `direction`: beneficial, harmful, neutral, mixed, or unclear
- `study_type`
- `population`
- `oral_exposure`
- `dose_or_serving`
- `duration`
- `outcome_measures`
- `natural_food_level_relevance`
- `supplement_level_relevance`
- `limitations`
- `extractor_confidence`

## Output Contract

Call `extract_claims` with the retained source registry, full-text manifest, paper-text directory, and
run claim directory. Do not extract claims manually in conversation. Every claim must point back to
a source. Zero claims from a retained paper is valid. Final acceptance belongs to validation.

For a subject-level request without explicit paths, call `advance_research_pipeline` with
`target_stage=claim_extraction`. It must reuse available full texts, run only missing prerequisites,
and stop before validation.

Run one LLM call per eligible paper with 1-10 bounded workers (default 5). Use the section-aware
context builder rather than a raw body prefix. Persist `extraction_status.json`,
`extraction_failures.json`, and `extraction_contexts.json` after every completed paper. A completed
paper with zero claims is successful and must be skipped on resume.
