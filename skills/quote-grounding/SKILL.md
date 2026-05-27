# Quote Grounding Skill

Use this skill when implementing validator quote checks, text normalization, or audit behavior.

## Goal

Ensure accepted health claims are supported by quotes that can be found in the paper body text.

## Required Matching Tiers

```txt
exact
normalized_whitespace
dehyphenated_ligature_normalized
not_found
```

## Rules

- Accepted claims must have at least one supporting quote.
- Supporting quotes must be searched in body text, not abstract text.
- If a quote cannot be found, the claim must be rejected or fail audit.
- Do not silently accept paraphrases as quotes.
- Normalization exists only to handle PDF extraction artifacts.

## Files

- `src/sipz_agent/core/quote_grounding.py`
- `src/sipz_agent/core/validation.py`
- `src/sipz_agent/core/audit.py`
- `tests/test_quote_grounding.py`
