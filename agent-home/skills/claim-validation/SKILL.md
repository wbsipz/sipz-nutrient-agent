---
name: claim-validation
description: Validates proposed nutrient or ingredient health claims one claim at a time against cleaned paper body text and requires deterministically grounded supporting quotes. Use when the user asks to verify claims, audit paper support, check quote grounding, continue from proposed claims, resume validation failures, or advance a subject through claim validation.
compatibility: Requires proposed claims and body text with title, abstract, conclusion, and references excluded where possible.
---

# Claim Validation Skill

Test each proposed claim independently against paper body text. Validation is a terminal stage when
requested on its own; do not continue to export.

Before validating claims, assess body adequacy once per paper. This is an LLM assessment of whether
the sanitized body contains enough human population, oral exposure, study-design, substantive-result,
and uncertainty detail. Do not classify adequacy from character count or keywords alone.

## Tool Routing

- When exact artifact paths are supplied, call `validate_claims` directly.
- Do not pre-read full claim, source, manifest, or paper-text artifacts merely to count or inspect
  them. The tool validates these inputs and returns authoritative reconciled counts.
- For a subject-only request, call `inspect_research_state`, then call
  `advance_research_pipeline` with `target_stage=claim_validation`.
- Let the controller create only missing prerequisites. Reuse completed full text and proposed
  claims instead of repeating retrieval or extraction.
- Default to resume mode. Set `retry_failed=true` only when the user requests another attempt or a
  prior failure has been fixed.
- Use 5 validation workers by default and never exceed 10.
- Retry a claim at most three times, and only for malformed/truncated JSON, schema errors, timeouts,
  rate limits, or transient provider/connection failures. Keep the proposed claim immutable.
- Never retry a completed `unsupported`, `over_scoped`, non-oral, non-human, wrong-substance, or
  `quote_not_found` decision. A retry must not become an attempt to persuade the model to pass.
- Validate claims only for papers assessed `adequate`. Hold `limited` and `inadequate` papers from
  accepted and rejected outputs and write them to the body retrieval queue.
- Treat held claims as missing validation capability, not evidence against the claim. Do not
  automatically rerun retrieval unless the user explicitly requests recovery.

## Required Checks

For each claim, verify from body text:

1. The substance was orally consumed by humans.
2. Population, exposure form, dose, duration, comparator, outcome, and direction are preserved.
3. Certainty does not exceed the study design or reported result.
4. Food-level evidence is not inferred from supplement-level evidence, or vice versa.
5. At least one short supporting quote is copied from Methods, Results, or Discussion.
6. The supporting entity is the requested species or an explicit alias, not a related species.
7. At least one quote directly reports the result or a review synthesis. Study counts, methods,
   measured outcomes, and background statements are context rather than proof.

The validator context must exclude title, abstract, conclusion, references, navigation text, and
citation metadata where possible.

## Decisions

- `supported`: the claim is supported at its proposed scope.
- `supported_with_limitations`: body evidence supports a narrower or less certain statement.
- `unsupported`: body evidence does not support the claim.
- `over_scoped`: the claim cannot be narrowed into a defensible human oral claim.

Reject animal-only, in-vitro-only, topical, injected, inhaled, pharmaceutical, abstract-only, or
materially overgeneralized support. Do not use deterministic keyword filters as a substitute for
reading the evidence.

## Grounding and Output

Deterministic quote matching is authoritative but exact presence alone is insufficient. If a
supported decision has no grounded, result-bearing quote, the tool may make one focused quote-repair
attempt; otherwise it rejects the claim. Do not validate in
conversational prose or manually construct accepted claims.

Use the tool's accepted, rejected, held, failed, and pending counts and canonical artifact paths. Preserve
the status, context hash, model, attempt count, and audit records. Do not write final reports.
Never infer counts from a partial or truncated file read.
