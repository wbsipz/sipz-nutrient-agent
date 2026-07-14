---
name: retrieval
description: Plans and executes literature retrieval for nutrients, bioactives, botanicals, additives, and ingredients. Use when the user asks to find papers, run a literature search, gather candidate sources, diagnose sparse retrieval, expand synonyms, or prepare source metadata for oral-ingestion human-health research.
compatibility: Requires network/search tools or local source files; can operate in planning mode without live retrieval.
metadata:
  version: 1.0.0
  domain: nutrient-health-research
---

# Retrieval Skill

Use this skill to build an auditable candidate-paper set for oral-ingestion human-health research. This is a workflow skill, not a PubMed wrapper. PubMed, Europe PMC, OpenAlex, Semantic Scholar, Crossref, DOI landing pages, and full-text endpoints are tools or source-specific tactics used inside the retrieval workflow.

The canonical retrieval question is:

```txt
What is the effect of the substance on human health when orally ingested?
```

## When To Use

Use this skill when the task involves any of these requests:

- find candidate papers for a nutrient, bioactive, botanical, additive, or ingredient
- run or plan a literature search
- improve sparse retrieval results
- expand synonyms, aliases, spelling variants, parent classes, or ingredient forms
- diagnose why a run found too few or too many papers
- prepare source metadata for screening, claim extraction, validation, or export

Do not use this skill to decide whether a health claim is true. Retrieval only proposes candidate sources.

## Skill Boundary

This skill owns:

- entity and alias-aware retrieval planning
- query generation
- source priority and fallback order
- candidate metadata collection
- deduplication rules
- retrieval provenance
- sparse-result and failure handling

This skill does not own:

- final paper inclusion decisions
- claim extraction
- claim validation
- quote grounding
- evidence grading
- report export
- database writes

## Core Workflow

1. Define the retrieval target.
   - Normalize the substance name.
   - Identify aliases, spelling variants, parent classes, botanical names, ingredient forms, and supplement or food forms.
   - Preserve whether the run is for a nutrient, bioactive, phytochemical class, botanical/extract, additive, whole ingredient, or ingredient form.

2. Frame the oral-ingestion query.
   - Include oral consumption, dietary intake, beverage/food form, supplement form, or ingredient form as applicable.
   - Exclude topical, inhaled, injected, pharmaceutical, agriculture, extraction, processing, chemistry-only, and composition-only drift.

3. Generate a small query set.
   - Start with direct substance queries.
   - Add alias and spelling queries.
   - Add parent-class queries only when direct evidence is sparse or the user wants a broader review.
   - Include targeted human-study queries before broad scholarly search.

4. Search high-value literature sources first.
   - PubMed and Europe PMC first for human-health biomedical literature.
   - OpenAlex and Crossref next for broader DOI/metadata coverage.
   - Use Semantic Scholar and Firecrawl only when the primary sources finish below the retained-paper target.
   - Use DOI landing pages, publisher pages, open-access locations, and configured full-text APIs after candidate papers exist.

5. Deduplicate and enrich.
   - Deduplicate by DOI first, then PMID, then normalized title.
   - Enrich missing abstracts, DOI, PMID, year, and source URL where possible.
   - Preserve every retrieval query and source for audit.

6. Stop deliberately.
   - Screening-only requests target retained papers; complete workflows target usable full texts.
   - Defaults are 10 usable full texts for light, 25 for standard, and 40 for deep runs.
   - Expand automatically when below target. Pi chooses 2-4 new queries from the coverage checkpoint;
     deterministic guardrails enforce maximum rounds, candidate caps, and zero-yield stopping.
   - Use at most 3 rounds for light, 5 for standard, and 10 for deep, with 200 unique candidates total.
   - Stop after two consecutive rounds add no new retained or usable papers.
   - If sparse, record what expansions were attempted and why retrieval stopped.

7. Write retrieval artifacts.
   - Save candidate metadata under `workspace/sources`.
   - Save failed collectors, sparse-result notes, and manual lookup queues under `workspace/notes`.

## Available Tools

### Concurrency

- Screen independent candidates concurrently, defaulting to 5 LLM workers and allowing 1-10.
- Retrieve independent full texts concurrently, defaulting to 10 network workers and allowing 1-10.
- Keep artifact writes and count reconciliation on the coordinating thread.
- Resume only unfinished screening records and unattempted full texts.
- Retry transient `408`, `429`, `502`, `503`, and `504` model responses with bounded backoff.
- Use `RESEARCH_MAX_LLM_WORKERS` and `RESEARCH_MAX_RETRIEVAL_WORKERS` to change defaults.

Use the tools at workflow boundaries rather than manually coordinating source APIs:

- `retrieve_candidates`: primary discovery tool. Supply the resolved substance, aliases, optional queries, depth, target count, and maximum pages.
- `screen_candidates`: classify every retrieved candidate before full-text retrieval; require reconciled counts and persisted decisions.
- `enrich_candidate`: recover missing abstract, DOI, PMID, or canonical metadata for one known candidate.
- `retrieve_full_text`: retrieve body text for one retained candidate and save it when an output directory is supplied.
- `retrieve_full_text_batch`: retrieve all papers from `retained_sources.json`, resume prior work,
  and skip successful existing texts. Prefer this for subject-level full-text requests.
- `advance_research_pipeline`: run one adaptive coverage round. When it returns
  `expansion_recommended=true`, propose new queries not present in `queries_attempted` and call it
  again with `expansion_queries` before proceeding downstream.
- `inspect_retrieval_run`: diagnose an existing run from its persisted artifacts.

Do not call one source repeatedly by hand when `retrieve_candidates` can manage source priority, pagination, deduplication, and partial failures. Do not interpret successful retrieval as evidence that a claim is true.

## Output Contract

Each candidate source should preserve:

```json
{
  "source_id": "string",
  "title": "string",
  "authors": ["string"],
  "year": 2026,
  "doi": "string or null",
  "pmid": "string or null",
  "abstract": "string or null",
  "source_url": "string or null",
  "retrieval_query": "string",
  "retrieval_source": "pubmed|europe_pmc|openalex|semantic_scholar|crossref|firecrawl|manual",
  "retrieval_tier": "targeted_human|review|dietary_exposure|broad_scholarly|fallback",
  "selection_reason": "string"
}
```

For every retrieval run, also record:

- normalized target name
- aliases used
- parent classes used
- query list
- source order
- page counts
- raw candidate count
- unique candidate count
- source failures
- stop reason

## Source Priority

Use this default order:

1. Targeted PubMed and Europe PMC human-study searches.
2. Targeted PubMed and Europe PMC review searches.
3. Dietary or oral-exposure PubMed and Europe PMC searches.
4. Broad OpenAlex and Crossref searches.
5. Semantic Scholar fallback when the retained-paper target is still unmet.
6. Firecrawl or generic web search fallback when configured.
7. DOI/open-access/full-text recovery for retained candidates.

## Query Rules

A good query includes three ideas:

```txt
entity terms AND human/oral terms AND study/outcome terms
```

Prefer narrow, auditable query batches over one very broad query.

Examples:

```txt
("acai" OR "açaí" OR "Euterpe oleracea") AND (randomized OR placebo OR "clinical trial" OR crossover)
("acai" OR "açaí" OR "Euterpe oleracea") AND ("systematic review" OR "meta-analysis" OR review)
("acai" OR "açaí" OR "Euterpe oleracea") AND (human OR adults) AND (consumption OR dietary OR juice OR pulp)
```

For source-specific query syntax and paging details, read:

- `references/query-design.md`
- `references/source-specific-usage.md`
- `references/candidate-schema.md`
- `references/troubleshooting.md`

## Handoff To Screening

After retrieval, hand off candidates to paper screening. Do not claim a source supports a health effect just because it was retrieved. The next stage must screen title/abstract and later validate claims against body text.

Pass the `candidates_path` returned by `retrieve_candidates` directly to `screen_candidates`; do not
load or echo the full candidate array into the conversation. For a full-text request, pass the
resulting `retained_sources.json` directly to `retrieve_full_text_batch` and stop after that stage.

## Troubleshooting Shortcuts

- Too few papers: expand aliases, add botanical/scientific names, use parent class, then broaden sources.
- Too many irrelevant papers: add oral/human/study-type terms and exclude topical, animal, in-vitro, extraction, agriculture, and composition terms.
- PubMed sparse but OpenAlex broad: keep OpenAlex candidates but require stricter screening.
- Missing abstract: enrich by DOI/PMID from Crossref, OpenAlex, Europe PMC, or PubMed.
- Publisher blocked: record manual retrieval need; do not infer full-text claims from a blocked page.
