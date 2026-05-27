# Sipz Export Skill

Use this skill when implementing CSV generation, artifact export, or future Supabase import workflows.

## Goal

Produce a Supabase-ready `effects.csv` that can be reviewed and imported.

## CSV Columns

```txt
id,nutrient_id,effect_slug,effect_label,description,score,evidence_level,tags,sources,created_at,updated_at,nutrient_name,match_status,match_confidence,match_notes
```

## Rules

- Only accepted, quote-grounded claims can become CSV rows.
- Quote-level evidence does not belong in `effects.csv`.
- Quote-level evidence belongs in `validated_claims.json`.
- Link validation evidence to CSV rows with `effect_row_id`.
- Do not export `insufficient` evidence rows.
- Do not write directly to Supabase in the MVP.

## Files

- `src/sipz_agent/core/synthesis.py`
- `src/sipz_agent/core/artifacts.py`
- `src/sipz_agent/cli.py`
- `src/sipz_agent/schemas/effects.py`
