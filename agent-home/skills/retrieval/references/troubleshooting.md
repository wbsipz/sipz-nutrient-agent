# Retrieval Troubleshooting

Use this reference when retrieval quality is poor, sparse, too broad, or blocked.

## Too Few Papers

Likely causes:

- target is niche
- synonym expansion is weak
- source is more commonly studied under a parent class
- spelling variants or botanical names are missing
- query is too strict

Actions:

1. Add aliases, spelling variants, and scientific names.
2. Add parent class terms.
3. Add form terms such as juice, extract, supplement, or dietary intake.
4. Run review queries.
5. Broaden from PubMed/Europe PMC to OpenAlex, Semantic Scholar, and Crossref.
6. Record sparse-evidence status if still thin.

## Too Many Irrelevant Papers

Likely causes:

- entity name is ambiguous
- query lacks oral/human terms
- query lacks study-type terms
- source returns chemistry, agriculture, or extraction literature

Actions:

1. Add human and oral-ingestion terms.
2. Add study-type terms such as clinical trial, randomized, systematic review, or cohort.
3. Add form-specific terms.
4. Down-rank extraction, processing, agriculture, animal, in-vitro, and composition-only papers.
5. Keep broad candidates but require stricter screening.

## Ambiguous Names

Examples:

- acronyms
- brand-like names
- common words
- plant names with multiple species
- ingredient forms that are really blends or flavors

Actions:

1. Prefer scientific name or CAS-like specificity when available.
2. Add known false-positive exclusions.
3. Ask for clarification only if ambiguity cannot be resolved from context.
4. Record ambiguity in retrieval notes.

## API Or Source Failures

Common failures:

- Semantic Scholar rate limit
- Europe PMC temporary 5xx
- publisher 403 or access denied
- DOI landing page redirects to abstract only
- missing full text

Actions:

1. Continue with other sources.
2. Record failed collector, query, and error.
3. Do not retry indefinitely.
4. Add manual retrieval queue for important blocked papers.
5. Never infer claims from unavailable body text.

## Abstract Available But Body Text Missing

Actions:

1. Try DOI landing page.
2. Try PubMed Central or Europe PMC XML.
3. Try OpenAlex OA locations.
4. Try Crossref full-text links.
5. Try publisher page.
6. Try configured Elsevier API only for Elsevier candidates.
7. Try configured Firecrawl scrape.
8. If still missing, mark full text unavailable and do not validate claims from the abstract.
