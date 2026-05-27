# Nutrient Research Skill

Use this skill when implementing candidate retrieval, claim extraction, study orchestration, or nutrient/bioactive research behavior.

## Goal

Turn a nutrient, bioactive, or class name into candidate research artifacts without overstating health effects.

## Rules

- Candidate discovery only finds potential sources.
- Candidate discovery does not prove claims.
- Paper-reader logic may propose short candidate claims.
- Candidate claims must be validated before export.
- Prefer human evidence over animal, in vitro, or mechanistic evidence.
- Preserve claim scope when evidence is indirect or preclinical.

## MVP Behavior

For demo mode:

- Load bundled fixtures from `examples/demo-corpus`.
- Use deterministic claims from `src/sipz_agent/core/extraction.py`.
- Avoid network calls.

For live candidate retrieval:

- Query PubMed with NCBI E-utilities.
- Query Europe PMC and fetch full-text XML body text when available.
- Query OpenAlex works.
- Query Semantic Scholar Graph API.
- Query Crossref works.
- If `FIRECRAWL_API_KEY` is set, use Firecrawl search as a supplemental site-search fallback.
- Keep live retrieval output compatible with `CandidateCitation`.
- Store retrieval query, source metadata, and a `selection_reason` for auditability.
- Write both structured `sources.json` and human-readable `sources.md` candidate lists.
- Candidate retrieval is not claim proof; extraction and validation remain separate stages.

## Files

- `src/sipz_agent/core/retrieval.py`
- `src/sipz_agent/core/extraction.py`
- `src/sipz_agent/core/orchestrator.py`
- `src/sipz_agent/schemas/citations.py`
- `src/sipz_agent/schemas/claims.py`
