---
name: agent-help
description: Explains the Sipz Nutrient Research Agent's capabilities, supported research subjects, workflow stages, evidence safeguards, output files, and correct natural-language usage. Use when the user asks what the agent can do, how to use it, what prompts or commands it understands, what substances it supports, how stages resume, where results are saved, or what its limitations are.
compatibility: Provides guidance only and does not require model-provider credentials, network access, or research artifacts.
---

# Agent Help

Answer questions about this agent from the runtime capabilities below. Do not start a research run,
call workflow tools, or inspect research state unless the user also asks to perform research.

## Capabilities

The agent investigates effects on human health from oral consumption of nutrients, bioactives,
phytochemicals, botanicals, extracts, additives, foods, and food ingredients. It can:

1. find candidate papers and resolve useful aliases;
2. screen titles and abstracts for direct human oral-health relevance;
3. retrieve and clean available full texts;
4. extract structured candidate claims one paper at a time;
5. validate claims against sanitized paper body text and grounded quotes; and
6. export auditable Markdown, JSON, and CSV reports.

The stages are independently resumable. A request for a later stage causes the agent to inspect
existing artifacts, reuse valid prerequisites, and run only the missing stages through the requested
boundary.

## How To Use It

Users interact in natural language after starting `sipz-nutrient`. Give examples matched to the
question, such as:

```text
Find 15 candidate papers about oral magnesium and human health. Screen them, but stop before full-text retrieval.
```

```text
Retrieve full text for the retained magnesium papers. Reuse the existing run.
```

```text
Extract claims from the available magnesium papers, but stop before validation.
```

```text
Validate the magnesium claims and create the final report.
```

```text
Run the complete workflow for valerian root with a target of 10 usable full-text papers.
```

Explain count semantics when relevant: screening counts target retained papers; full-text,
extraction, validation, and complete-workflow counts target usable full texts.

## Evidence Safeguards

- Eligible claims concern humans and oral exposure.
- Animal-only, in-vitro-only, topical, inhaled, and injected evidence is excluded.
- Titles and abstracts support screening but cannot prove claims.
- Accepted claims require a result-bearing quote grounded in sanitized paper body text.
- Dose, form, population, study design, limitations, and food-versus-supplement exposure are
  preserved where available.
- Related species and isolated compounds are not automatically generalized to a whole ingredient.

## Outputs

Final reports are written under `workspace/reports/<subject>/` and include:

- `claims_report.md`
- `claims_report.json`
- `claims_report.csv`
- `report_manifest.json`

Research-stage artifacts are stored under `workspace/runs/<subject>/`. Completed report responses
show the claims table, absolute report directory, and links or paths to the generated files.

## Boundaries

The agent supports evidence research and curation, not medical diagnosis, treatment advice,
personalized dosing, guaranteed exhaustive systematic reviews, paywall bypassing, or automatic
database writes. Full text may remain unavailable because of publisher access controls.

When answering a general capability question, be concise but include capabilities, supported
subjects, one complete-workflow example, output location, and the medical-advice boundary. Add
stage-specific examples or configuration details only when they help answer the user's question.
