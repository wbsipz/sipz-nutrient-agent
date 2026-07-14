# Source-Specific Usage

This reference describes how to use literature sources inside the retrieval workflow. These are source tactics, not separate top-level skills.

## PubMed

Use for biomedical literature, human trials, reviews, and PMID-backed records.

Useful metadata:

- PMID
- title
- publication year
- DOI from article IDs
- abstract via PubMed abstract fetch
- PubMed URL

Pagination pattern:

- `retmax`: page size
- `retstart`: page index times page size
- sort by relevance for discovery

Use targeted human-study and review queries first. PubMed should be high priority for nutrients, bioactives, supplements, and clinical endpoints.

Configure `NCBI_EMAIL` for application identification and `NCBI_API_KEY` for authenticated
E-utilities limits. The retrieval tools attach both internally and must not persist them in source
URLs or failure messages.

## Europe PMC

Use for PubMed-adjacent coverage, PMC records, abstracts, full-text availability, and Europe PMC XML lookup.

Useful metadata:

- PMID
- PMCID when available
- DOI
- title
- abstract
- source URL
- full-text URL or XML endpoint

Pagination pattern:

- `pageSize`: page size
- `page`: one-based page index

Europe PMC is especially useful when PubMed has a PMID but body text may be available through PMC or Europe PMC XML.

## OpenAlex

Use to broaden scholarly discovery, recover DOI records, and locate open-access locations.
Authenticate with `OPENALEX_API_KEY` when configured. The retrieval tools attach the key
internally; never include it in an agent query, provenance record, error message, or saved source
URL. Anonymous fallback remains available when no key is configured, but should not be relied on
for batch retrieval.

Useful metadata:

- DOI
- publication year
- title
- landing page URL
- abstract inverted index when available
- open-access status and OA URL
- best OA location and PDF URL

OpenAlex can be broad and noisy. Keep its candidates, but require stricter paper screening.

## Unpaywall

Use Unpaywall during full-text recovery when a retained candidate has a DOI. It is an
open-access location resolver, not a literature discovery or relevance source.

Prefer locations in this order:

1. published version
2. accepted manuscript
3. submitted manuscript

Within a version, prefer direct PDF links before landing pages. Preserve the manuscript
version, host type, license, and OA status in retrieval records. Configure `UNPAYWALL_EMAIL`;
`NCBI_EMAIL` is used as a fallback when it is already configured.

## Semantic Scholar

Use only as a fallback to broaden discovery and recover DOI/PMID links when PubMed, Europe PMC,
OpenAlex, and Crossref finish below the retained-paper target. An API key is optional; anonymous
requests remain deliberately rate-limited and must tolerate `429` responses without failing the
overall retrieval run.

Useful metadata:

- paper ID
- title
- year
- abstract
- DOI
- PubMed ID
- source URL

Semantic Scholar can return rate-limit errors. Record failures and continue with other sources.

## Crossref

Use for DOI-centric discovery and full-text link metadata.

Useful metadata:

- DOI
- title
- publication date
- URL
- abstract when available
- license and link records

Crossref is often better for DOI/full-text enrichment than for deciding health relevance. Keep provenance and let screening decide.

Configure `CROSSREF_MAILTO` so requests identify the application to Crossref's polite pool. The
email is attached internally and must not appear in saved source URLs or audit failures.

## DOI And Publisher Pages

Use only after candidate papers exist. DOI landing pages and publisher pages can help recover article body text or full-text links. Do not use citation pages, abstract pages, references, or headings as proof for health claims.

## Firecrawl Or Generic Web Fetch

Use only when configured and needed:

- fallback search over literature sites
- publisher page scraping after a candidate DOI or landing URL exists
- visible-page summary when APIs do not expose an abstract

Do not treat a scraped page as proof unless body text is present and quote grounding succeeds.
