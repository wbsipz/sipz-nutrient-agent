# Example Session

This is a sanitized illustration of the public workflow. Counts and identifiers are representative;
no provider credentials, downloaded papers, or session history are included.

## Request

```text
Run a complete standard-depth research workflow for Example Nutrient with a target of six usable
full-text papers. Extract and validate human oral-health claims, then export the report.
```

## Progress

```text
[entity_resolution] Resolving Example Nutrient and aliases...
[retrieval] Round 1: searching PubMed, Europe PMC, OpenAlex, Semantic Scholar, and Crossref...
[screening] Screening 24 unique candidates with 5 workers...
[full_text] Retrieving 7 retained papers with 10 workers...
[coverage] 4/6 usable full texts; expansion recommended.
[retrieval] Round 2: running 3 untried outcome-specific queries...
[coverage] 6/6 usable full texts; target met.
[claim_extraction] Processing paper 1/6...
[claim_validation] Validating claim 1/9 against sanitized body text...
[report_export] Writing Markdown, JSON, CSV, and manifest...
```

## Terminal Summary

```text
## Example Nutrient Human-Health Evidence

- Sources retained after screening: 9
- Papers reviewed from usable full text: 6
- Validated claims: 4
- Rejected claims: 5
- Held for better retrieval: 0
- Validation failures: 0
- Requested usable full texts: 6
- Usable full texts retrieved: 6
- Retrieval target met: yes
```

```text
Saved report folder: /absolute/path/to/workspace/reports/example-nutrient
Main report file: /absolute/path/to/workspace/reports/example-nutrient/claims_report.md
```

The Markdown report presents the validated claims as a table. JSON retains source metadata,
retrieval coverage, model provenance, rejected claims, limitations, and grounded quote records. CSV
contains one structured row per validated claim for downstream analysis.
