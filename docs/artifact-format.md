# Artifact Format

Each run writes:

```txt
effects.csv
validated_claims.json
rejected_claims.json
sources.json
packet.json
summary.md
audit_log.jsonl
```

`effects.csv` is the primary output for Supabase import.

`validated_claims.json` contains quote-level validation evidence linked by `effect_row_id`.

`rejected_claims.json` contains unsupported, over-scoped, or ungrounded claims.

## CSV Columns

```txt
id,nutrient_id,effect_slug,effect_label,description,score,evidence_level,tags,sources,created_at,updated_at,nutrient_name,match_status,match_confidence,match_notes
```

`tags` and `sources` are JSON strings inside CSV cells.
