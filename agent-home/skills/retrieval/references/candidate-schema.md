# Candidate Metadata Schema

Use this schema for retrieval outputs. The exact storage format can vary, but these fields should be preserved whenever available.

## Candidate Source

```json
{
  "source_id": "string",
  "title": "string",
  "authors": ["string"],
  "year": 2026,
  "doi": "string or null",
  "pmid": "string or null",
  "pmcid": "string or null",
  "abstract": "string or null",
  "source_url": "string or null",
  "publisher_url": "string or null",
  "retrieval_query": "string",
  "retrieval_source": "pubmed|europe_pmc|openalex|semantic_scholar|crossref|firecrawl|manual",
  "retrieval_tier": "targeted_human|review|dietary_exposure|broad_scholarly|fallback",
  "selection_reason": "string",
  "metadata_sources": ["string"]
}
```

## Retrieval Run Summary

```json
{
  "run_id": "string",
  "target_name": "string",
  "normalized_target_name": "string",
  "target_type": "nutrient|bioactive|class|botanical|extract|additive|ingredient|ingredient_form",
  "aliases_used": ["string"],
  "parent_classes_used": ["string"],
  "queries": ["string"],
  "source_order": ["pubmed", "europe_pmc", "openalex", "semantic_scholar", "crossref"],
  "raw_candidate_count": 0,
  "unique_candidate_count": 0,
  "retained_candidate_target": 25,
  "retrieval_pages_attempted": 0,
  "source_failures": ["string"],
  "stop_reason": "target_reached|max_pages|no_new_unique_candidates|user_stopped|retrieval_error"
}
```

## Deduplication Key Order

1. normalized DOI
2. PMID
3. PMCID
4. normalized title

When merging duplicates, preserve all useful metadata and append a note to `selection_reason` or `metadata_sources`.

## Audit Requirement

Every retained candidate must be traceable:

```txt
query -> retrieval source -> candidate source -> screening decision -> claim extraction input
```

Retrieval does not prove any health claim.
