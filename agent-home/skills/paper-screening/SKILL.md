---
name: paper-screening
description: Screens candidate papers by title and abstract for oral-ingestion human-health relevance before full-text retrieval and claim extraction. Use after literature discovery, when filtering source lists, diagnosing irrelevant candidates, or deciding which papers should proceed to full-text retrieval.
compatibility: Requires candidate paper metadata and preferably abstracts.
---

# Paper Screening Skill

Use this skill after retrieval and before claim extraction.

## Goal

Filter candidate papers into a retained set likely to answer:

```txt
What is the effect of <substance> on human health when orally ingested?
```

## Inputs

- candidate source metadata
- title
- abstract
- DOI/PMID/source URL
- retrieval query and source
- substance aliases and exposure framing

## Required Tool

Call `screen_candidates` for the complete candidate set. Do not substitute a conversational
screening table. Supply an output directory so retained, rejected, decision, and summary artifacts
are persisted. Verify `counts.input == counts.classified` before proceeding.
Use bounded parallel workers for independent candidates. Persist completed decisions from the
coordinating thread and restore original candidate ordering in final artifacts.

## Screening Stages

1. Title screen:
   - keep plausible oral-ingestion human-health papers
   - keep likely reviews, trials, observational studies, and human mechanistic studies
   - reject obvious non-health, non-human, non-oral, or off-topic records

2. Abstract screen:
   - confirm population, exposure, outcome, and relevance
   - identify study type
   - classify evidence strength and whether the paper should proceed to body-text retrieval

3. Specificity screen:
   - retain direct human studies only when the requested substance is isolated or separately interpretable
   - retain reviews only when the requested substance is a central subject
   - reject mixed interventions when the substance's contribution cannot be separated
   - reject broad multi-nutrient papers that merely mention the substance
   - reject news briefs, editorials, table fragments, protocols without results, and case reports from the retained evidence set

## Include Categories

- `include_direct_human`
- `include_indirect_human`
- `include_review`
- `include_mechanistic_human`
- `include_food_composition_background`

## Exclude Categories

- `exclude_irrelevant`
- `exclude_non_oral`
- `exclude_non_human_direct_claim`
- `exclude_in_vitro_only`
- `exclude_animal_only`
- `exclude_marketing`
- `exclude_unverifiable`
- `exclude_composition_only`

## Evidence Priority

Prioritize:

1. systematic reviews and meta-analyses
2. human randomized trials
3. human observational studies
4. human mechanistic studies
5. animal studies as background only
6. in-vitro studies as background only

## Output Contract

Write `retained_sources.json`, `rejected_sources.json`, `screening_decisions.json`, and
`screening_summary.json`. Every input source must have exactly one retained or rejected decision.
Do not create final health claims in this skill.
