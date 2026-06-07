# Nutrient Health Effects Bulk Import Format

This report defines the exact output shape an LLM should produce for bulk nutrient health-effect ingestion, and how that output maps into the current Supabase workflow.

## Current Workflow Summary

The health workflow uses two nutrient-effect data surfaces:

1. `beverage.nutrient_health_effects`
   - Used by dietary-goal matching.
   - Exposed through `public.nutrient_health_effects_public_v1`.
   - Drives `public.nutrient_tag_lookup_v1` and `public.nutrient_tag_lookup_mv1`.
   - This is where detailed effect rows belong.

2. `beverage.nutrient_effects_compressed`
   - Used by generated recipe health reports.
   - Exposed through `public.nutrient_effects_compressed_v1`.
   - This is where positive and negative summary text belongs.

The existing `/admin/nutrients` UI writes one detailed effect at a time to `beverage.nutrient_health_effects` and refreshes lookup materialized views after each write. It is not suitable for large batch imports unless extended.

## Recommended LLM Output Format

Use JSONL, one nutrient object per line. JSONL is preferred over one large JSON array because it is easier to validate, retry, split, and resume.

Each line must follow this shape:

```json
{
  "nutrient": {
    "id": "uuid-if-known",
    "name": "Vitamin C",
    "code": "vitamin_c"
  },
  "effects": [
    {
      "effect_slug": "immune_support",
      "effect_label": "Supports immune function",
      "description": "Vitamin C contributes to normal immune system function and supports antioxidant defense.",
      "score": 0.82,
      "evidence_level": "strong",
      "tags": ["immune_support", "antioxidant_support"],
      "sources": [
        {
          "title": "Vitamin C Fact Sheet",
          "url": "https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/",
          "publisher": "NIH Office of Dietary Supplements",
          "accessed_at": "2026-06-07"
        }
      ]
    }
  ],
  "compressed_effects": [
    {
      "evidence_level": "strong",
      "health_effect_positive": "Vitamin C may support immune function and antioxidant protection.",
      "health_effect_negative": null,
      "tags": ["immune_support", "antioxidant_support"],
      "source_effect_slugs": ["immune_support"]
    }
  ]
}
```

## Field Requirements

### `nutrient`

At least one stable identifier is required.

Preferred:

```json
{
  "id": "beverage.nutrients.id UUID",
  "name": "Vitamin C",
  "code": "vitamin_c"
}
```

Acceptable if IDs are not available:

```json
{
  "name": "Vitamin C",
  "code": "vitamin_c"
}
```

Import logic must resolve this to:

```text
beverage.nutrients.id
```

Do not insert effects unless the nutrient resolves to exactly one nutrient row.

### `effects[]`

Each object maps to one row in `beverage.nutrient_health_effects`.

Required fields:

```text
effect_slug
effect_label
description
score
evidence_level
tags
sources
```

Rules:

- `effect_slug` must already exist in `beverage.health_effects.effect_slug`.
- `effect_label` must be a short human-readable label.
- `description` must be a specific evidence-grounded effect statement.
- `score` must be a number from `0` to `1`.
- `evidence_level` must be one of `strong`, `moderate`, or `limited`.
- `tags` must be an array of canonical health tags when possible.
- `sources` must be a JSON array. Use `[]` if no source data is available, but sourced data is strongly preferred.

Recommended `sources[]` object:

```json
{
  "title": "Source title",
  "url": "https://example.com/source",
  "publisher": "Publisher name",
  "doi": "optional DOI",
  "pmid": "optional PubMed ID",
  "accessed_at": "2026-06-07"
}
```

### `compressed_effects[]`

Each object maps to one row in `beverage.nutrient_effects_compressed`, keyed by:

```text
nutrient_id
evidence_level
```

Required fields:

```text
evidence_level
health_effect_positive
health_effect_negative
tags
```

Rules:

- `evidence_level` must be one of `strong`, `moderate`, or `limited`.
- `health_effect_positive` can be `null` or a concise summary.
- `health_effect_negative` can be `null` or a concise summary.
- At least one of `health_effect_positive` or `health_effect_negative` should be non-empty.
- `tags` must be a JSON array.
- `source_effect_slugs` is not stored by the current table but is useful for audit logs and importer validation.

## Database Mapping

### Detailed Effect Row

For each `effects[]` item, insert or upsert into:

```text
beverage.nutrient_health_effects
```

Database row shape:

```json
{
  "nutrient_id": "resolved nutrient UUID",
  "nutrient_name": "Vitamin C",
  "effect_slug": "immune_support",
  "effect_label": "Supports immune function",
  "description": "Vitamin C contributes to normal immune system function and supports antioxidant defense.",
  "score": 0.82,
  "evidence_level": "strong",
  "tags": ["immune_support", "antioxidant_support"],
  "sources": [
    {
      "title": "Vitamin C Fact Sheet",
      "url": "https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/",
      "publisher": "NIH Office of Dietary Supplements",
      "accessed_at": "2026-06-07"
    }
  ]
}
```

Suggested upsert key for a bulk importer:

```text
nutrient_id + effect_slug
```

The current schema does not define this unique constraint, so either add one or have the importer check for an existing row before insert.

### Compressed Summary Row

For each `compressed_effects[]` item, upsert into:

```text
beverage.nutrient_effects_compressed
```

Database row shape:

```json
{
  "nutrient_id": "resolved nutrient UUID",
  "nutrient_name": "Vitamin C",
  "evidence_level": "strong",
  "health_effect_positive": "Vitamin C may support immune function and antioxidant protection.",
  "health_effect_negative": null,
  "tags": ["immune_support", "antioxidant_support"]
}
```

Existing primary key:

```text
nutrient_id + evidence_level
```

Use:

```sql
on conflict (nutrient_id, evidence_level)
do update
```

## Exact Import Sequence

1. Load all LLM JSONL rows.
2. Validate JSON syntax line by line.
3. Resolve each nutrient to `beverage.nutrients.id`.
4. Reject or quarantine rows with ambiguous or missing nutrient matches.
5. Validate every `effect_slug` against `beverage.health_effects.effect_slug`.
6. Validate `score`, `evidence_level`, `tags`, and `sources`.
7. Optionally map/filter tags against `beverage.health_tag_vocab`.
8. Insert or upsert detailed rows into `beverage.nutrient_health_effects`.
9. Upsert compressed rows into `beverage.nutrient_effects_compressed`.
10. Refresh health lookup materialized views once:

```sql
select public.refresh_health_tag_lookup_mvs_v1();
```

## Minimal SQL Shape

Detailed effects:

```sql
insert into beverage.nutrient_health_effects (
  nutrient_id,
  nutrient_name,
  effect_slug,
  effect_label,
  description,
  score,
  evidence_level,
  tags,
  sources
)
values (
  :nutrient_id,
  :nutrient_name,
  :effect_slug,
  :effect_label,
  :description,
  :score,
  :evidence_level,
  :tags::text[],
  :sources::jsonb
);
```

Compressed summaries:

```sql
insert into beverage.nutrient_effects_compressed (
  nutrient_id,
  nutrient_name,
  evidence_level,
  health_effect_positive,
  health_effect_negative,
  tags
)
values (
  :nutrient_id,
  :nutrient_name,
  :evidence_level,
  :health_effect_positive,
  :health_effect_negative,
  :tags::jsonb
)
on conflict (nutrient_id, evidence_level)
do update set
  nutrient_name = excluded.nutrient_name,
  health_effect_positive = excluded.health_effect_positive,
  health_effect_negative = excluded.health_effect_negative,
  tags = excluded.tags,
  updated_at = now();
```

Then:

```sql
select public.refresh_health_tag_lookup_mvs_v1();
```

## LLM Prompt Contract

Give the LLM this instruction:

```text
Return JSONL only. Each line must be a complete JSON object with keys:
nutrient, effects, compressed_effects.

Do not include markdown.
Do not include comments.
Do not invent nutrient IDs unless provided.
Use only effect_slug values from the allowed list.
Use evidence_level only as strong, moderate, or limited.
Use score as a number between 0 and 1.
Use tags as snake_case strings.
Use null for unavailable positive or negative compressed summaries.
Every health claim must be tied to a source when possible.
```

Provide the LLM:

1. A nutrient lookup table containing `id`, `name`, and `code`.
2. The allowed `effect_slug` list from `beverage.health_effects`.
3. The canonical tag vocabulary from `beverage.health_tag_vocab`, or a reduced list of allowed tags.

## Validation Checklist

A row is importable only if:

- It is valid JSON.
- `nutrient.id` or `nutrient.name/code` resolves to exactly one nutrient.
- Every `effect_slug` exists.
- Every `score` is between `0` and `1`.
- Every `evidence_level` is valid.
- `effects[].description` is non-empty.
- `effects[].tags` is an array.
- `effects[].sources` is an array.
- Every compressed row has at least one non-empty positive or negative summary.

## Practical Recommendation

Do not bulk-call the existing single-row admin endpoint. It refreshes lookup materialized views after each create/update. For large batches, add a dedicated importer that:

- validates the whole batch,
- performs inserts/upserts in chunks,
- writes rejects to a review file or staging table,
- refreshes `refresh_health_tag_lookup_mvs_v1()` once at the end.

