---
name: web-fetch
description: Recovers a known paper landing page or full-text source from a DOI, PMID, PMCID, or URL when structured literature and open-access retrieval routes are incomplete. Use for blocked publisher pages, missing full text, DOI resolution, Firecrawl fallback, or manual web recovery after candidate discovery; do not use as the primary literature-search workflow.
compatibility: Requires network access and may require configured publisher or Firecrawl credentials.
---

# Web Fetch / Literature Retrieval Skill

Use this skill for source-page and full-text recovery after candidate discovery. Use the
`retrieval` skill for literature searches, candidate collection, query expansion, pagination, and
deduplication.

## Goal

Retrieve candidate literature for:

```txt
What is the effect of <substance> on human health when orally ingested?
```

Live web fetching should support the staged research workflow. It should not become a shortcut for unsupported summaries.

## Inputs

- substance name
- aliases, spelling variants, and parent class
- ingredient form or exposure form when applicable
- oral-ingestion framing
- target candidate paper count
- depth: `light`, `standard`, or `deep`

## Discovery Sources

Use structured literature sources first:

- PubMed / NCBI E-utilities
- Europe PMC
- OpenAlex
- Semantic Scholar
- Crossref

Use broader web retrieval only as fallback or for full-text recovery:

- DOI landing pages
- publisher pages
- PubMed Central
- Europe PMC full-text XML
- OpenAlex open-access locations
- Crossref full-text links
- MDPI direct URLs when DOI pattern supports them
- Elsevier Article API when configured and the paper is Elsevier
- Firecrawl when configured

## Query Strategy

Generate a small set of auditable queries:

- direct substance name with oral-ingestion human-health terms
- aliases and spelling variants
- parent class terms when direct evidence is sparse
- study-type terms such as randomized, clinical trial, systematic review, meta-analysis, observational, cohort
- form terms such as food, juice, beverage, pulp, puree, supplement, extract, or dietary intake when relevant

Avoid query drift toward:

- topical, injected, inhaled, or pharmaceutical use
- animal-only or in-vitro-only evidence
- agriculture, extraction, processing, chemistry-only, or composition-only papers
- marketing pages and unverifiable supplement claims

## Candidate Record Schema

Each candidate should preserve:

- `source_id`
- `title`
- `authors` when available
- `year`
- `doi`
- `pmid`
- `abstract`
- `source_url`
- `retrieval_query`
- `retrieval_source`
- `retrieval_tier` when available
- `selection_reason`

## Retrieval Loop

1. Search targeted human-health queries in PubMed and Europe PMC.
2. Broaden to OpenAlex, Semantic Scholar, and Crossref if target count is not reached.
3. Deduplicate by DOI, PMID, then normalized title.
4. Enrich missing abstracts from DOI/PMID metadata where possible.
5. Store metadata under `workspace/sources/<run-id>/`.
6. Record failed collectors and blocked sources in `workspace/notes/<run-id>/`.

## Stop Criteria

Stop when one of these is true:

- target retained candidate count is reached
- maximum retrieval pages are exhausted
- no new unique candidates are found after query expansion
- the user explicitly asks to stop

Default target retained papers:

- standard: 25
- deep: 40

## Output

Write source metadata, retrieval queries, source failures, and deduplication notes under `workspace/sources/<run-id>/`. Do not write final health claims from this skill.
