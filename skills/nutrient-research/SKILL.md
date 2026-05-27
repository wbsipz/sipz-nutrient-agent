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

For future live mode:

- Add Firecrawl or literature APIs behind `src/sipz_agent/core/retrieval.py`.
- Keep live retrieval output compatible with `CandidateCitation`.
- Store retrieval query and source metadata for auditability.

## Files

- `src/sipz_agent/core/retrieval.py`
- `src/sipz_agent/core/extraction.py`
- `src/sipz_agent/core/orchestrator.py`
- `src/sipz_agent/schemas/citations.py`
- `src/sipz_agent/schemas/claims.py`
