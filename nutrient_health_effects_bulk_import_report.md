# Unified Nutrient and Bioactive Health Evidence Bulk Import

This report defines the exact LLM output and import procedure for loading nutrient and bioactive health effects into the current Supabase workflow.

## Temporary Source-of-Truth Decision

Use this table as the common detailed health-evidence source:

```text
beverage.bioactive_health_evidence
```

Despite its name, the table supports both:

```text
bioactive_type = nutrient
bioactive_type = polyphenol
```

For this import, do not duplicate detailed nutrient effects into:

```text
beverage.nutrient_health_effects
```

This is an intentional temporary architecture decision. The table should eventually be renamed or migrated to a properly named unified health-evidence table.

## How the Common Source Enters Recipe Generation

`beverage.bioactive_health_evidence` feeds:

```text
public.bioactive_health_evidence_lookup_v1
public.bioactive_tag_lookup_v1
public.bioactive_tag_lookup_mv1
```

The dietary-goal workflow uses those objects through:

```text
retrieve_health_candidates_from_tags_v1
retrieve_health_effects_for_candidate_v1
```

Both nutrients and polyphenols stored in this table are exposed to the application as `candidate_type = bioactive`. The underlying `bioactive_type` distinction is retained in the table but is not preserved in the current candidate response.

## Required LLM Output

Use JSONL, with one complete evidence object per line.

JSONL is preferred because each line can be independently validated, retried, rejected, or imported. The output can be converted directly into the JSON array accepted by the existing bulk RPC.

### Nutrient Example

```json
{"bioactive_type":"nutrient","bioactive_id":"4db79318-c3d9-4d43-9e27-b4ce00b6db39","bioactive_name":"Vitamin C","effect_slug":"immune_support","effect_label":"Supports immune function","description":"Vitamin C contributes to normal immune system function and supports antioxidant defense.","score":0.82,"evidence_level":"strong","tags":["immune_support","antioxidant_support"],"sources":[{"title":"Vitamin C Fact Sheet for Health Professionals","url":"https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/","publisher":"NIH Office of Dietary Supplements","accessed_at":"2026-06-07"}],"review_status":"generated","review_notes":null}
```

### Polyphenol Example

```json
{"bioactive_type":"polyphenol","bioactive_id":"5a2c6ec8-bf27-4d71-aa22-fc85241df908","bioactive_name":"Quercetin","effect_slug":"antioxidant_support","effect_label":"Supports antioxidant defense","description":"Quercetin has antioxidant activity and may contribute to cellular protection from oxidative stress.","score":0.68,"evidence_level":"moderate","tags":["antioxidant_support","oxidative_stress"],"sources":[{"title":"Source title","url":"https://example.org/source","publisher":"Publisher","doi":"10.xxxx/example","accessed_at":"2026-06-07"}],"review_status":"generated","review_notes":null}
```

## Exact Record Contract

Every JSONL line must contain:

```json
{
  "bioactive_type": "nutrient or polyphenol",
  "bioactive_id": "UUID",
  "bioactive_name": "Canonical display name",
  "effect_slug": "allowed_effect_slug",
  "effect_label": "Short human-readable effect label",
  "description": "Evidence-grounded effect description",
  "score": 0.75,
  "evidence_level": "strong, moderate, or limited",
  "tags": ["canonical_health_tag"],
  "sources": [],
  "review_status": "generated",
  "review_notes": null
}
```

### Field Rules

`bioactive_type`

- Required.
- Must be exactly `nutrient` or `polyphenol`.
- Use `nutrient` for vitamins, minerals, macronutrients, amino acids, fatty acids, and other entities represented by the nutrient identity system.
- Use `polyphenol` for polyphenols and other entities represented by the current bioactive/compound identity system.

`bioactive_id`

- Required UUID.
- For `nutrient`, use the canonical ID from `beverage.nutrients.id`.
- For `polyphenol`, use the UUID used by the existing polyphenol/bioactive lookup and measurement mapping.
- Do not let the LLM invent this value.
- Reject the record if the supplied name does not match the referenced entity.

`bioactive_name`

- Required canonical display name.
- Must match the entity represented by `bioactive_id`.
- Do not use marketing names or dosage descriptions.

`effect_slug`

- Required.
- Must already exist in `beverage.health_effects.effect_slug`.
- Do not allow the LLM to invent slugs outside the supplied allowed list.

`effect_label`

- Required.
- Short human-readable label for the effect.
- Recommended maximum: 200 characters.

`description`

- Required.
- Must describe the supported or adverse health effect without dosage instructions or unsupported medical certainty.
- Each distinct effect should be a separate record.

`score`

- Required number from `0` to `1`.
- Must represent evidence confidence or support strength consistently across the dataset.
- The scoring rubric should be supplied to the LLM rather than inferred separately for each record.

`evidence_level`

- Required for this import.
- Must be exactly `strong`, `moderate`, or `limited`.

`tags`

- Required JSON array of strings.
- Tags should come from `beverage.health_tag_vocab`.
- Use canonical snake-case tags.
- Tags drive dietary-goal candidate retrieval, so an evidence row without useful canonical tags will not participate effectively in goal matching.

`sources`

- Required JSON array.
- Use `[]` only when evidence provenance is genuinely unavailable.
- Every claim should include at least one source when possible.

Recommended source object:

```json
{
  "title": "Source title",
  "url": "https://example.org/source",
  "publisher": "Publisher",
  "doi": "optional DOI",
  "pmid": "optional PubMed ID",
  "publication_date": "optional ISO date",
  "accessed_at": "2026-06-07"
}
```

`review_status`

- Required for bulk LLM output.
- Use `generated` for newly generated, unreviewed records.
- Allowed database values are `generated`, `reviewed`, and `rejected`.
- Do not let the LLM mark its own output as `reviewed`.

`review_notes`

- Optional string or `null`.
- Use for import warnings, ambiguity notes, or reviewer comments.

## Existing Bulk RPC

The database already provides:

```text
public.upsert_bioactive_health_evidence(p_rows jsonb)
```

It accepts a JSON array of the records defined above.

It upserts on:

```text
bioactive_type + bioactive_id + effect_slug
```

This corresponds to the existing unique index:

```text
bhe_unique_bioactive_effect
```

### RPC Payload

Convert the JSONL records into one or more JSON arrays:

```json
[
  {
    "bioactive_type": "nutrient",
    "bioactive_id": "4db79318-c3d9-4d43-9e27-b4ce00b6db39",
    "bioactive_name": "Vitamin C",
    "effect_slug": "immune_support",
    "effect_label": "Supports immune function",
    "description": "Vitamin C contributes to normal immune system function and supports antioxidant defense.",
    "score": 0.82,
    "evidence_level": "strong",
    "tags": ["immune_support", "antioxidant_support"],
    "sources": [
      {
        "title": "Vitamin C Fact Sheet for Health Professionals",
        "url": "https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/",
        "publisher": "NIH Office of Dietary Supplements",
        "accessed_at": "2026-06-07"
      }
    ],
    "review_status": "generated",
    "review_notes": null
  }
]
```

Call:

```sql
select public.upsert_bioactive_health_evidence(
  '<JSON array>'::jsonb
);
```

For a large dataset, import in chunks rather than sending one extremely large RPC payload.

Recommended initial chunk size:

```text
100 to 500 evidence records per RPC call
```

## Required Post-Import Refresh

After all chunks succeed, refresh the tag lookup materialized views once:

```sql
select public.refresh_health_tag_lookup_mvs_v1();
```

This refreshes:

```text
public.nutrient_tag_lookup_mv1
public.bioactive_tag_lookup_mv1
```

For this unified import, `public.bioactive_tag_lookup_mv1` is the relevant candidate lookup.

Do not refresh after every row.

## Import Sequence

1. Export canonical nutrient and polyphenol identity records with their UUIDs and names.
2. Export allowed effect slugs from `beverage.health_effects`.
3. Export the canonical tag vocabulary from `beverage.health_tag_vocab`.
4. Give those controlled lists and the evidence source material to the LLM.
5. Require JSONL output using the contract in this report.
6. Parse and validate every JSONL line before database writes.
7. Confirm `bioactive_id` and `bioactive_name` resolve to the same entity.
8. Confirm every `effect_slug` exists.
9. Filter or reject tags that are not in the canonical health-tag vocabulary.
10. Reject invalid scores, evidence levels, review statuses, URLs, or malformed source arrays.
11. Convert validated JSONL records into JSON arrays of 100 to 500 records.
12. Call `public.upsert_bioactive_health_evidence` for each chunk.
13. Record accepted, rejected, and failed rows in an import audit file or table.
14. Call `public.refresh_health_tag_lookup_mvs_v1()` once after all writes.
15. Run test dietary goals and confirm the imported entities appear as bioactive candidates.

## Validation Checklist

A record is importable only if:

- It is valid JSON.
- `bioactive_type` is `nutrient` or `polyphenol`.
- `bioactive_id` is a valid UUID supplied by the identity lookup.
- `bioactive_id` and `bioactive_name` identify the same entity.
- `effect_slug` exists in `beverage.health_effects`.
- `effect_label` is non-empty.
- `description` is non-empty.
- `score` is between `0` and `1`.
- `evidence_level` is `strong`, `moderate`, or `limited`.
- `tags` is a JSON array containing canonical tags.
- `sources` is a JSON array.
- `review_status` is `generated`, `reviewed`, or `rejected`.
- Newly generated LLM records use `review_status = generated`.

## LLM Prompt Contract

Use the following output instruction:

```text
Return JSONL only, with exactly one complete health-evidence object per line.

Do not include markdown, code fences, headings, comments, or explanatory text.
Do not invent entity UUIDs.
Use only entity IDs and names from the supplied identity list.
Set bioactive_type to exactly "nutrient" or "polyphenol".
Use only effect_slug values from the supplied allowed list.
Use evidence_level only as "strong", "moderate", or "limited".
Use score as a JSON number between 0 and 1.
Use only canonical tags from the supplied tag vocabulary.
Set review_status to "generated".
Set review_notes to null unless an ambiguity must be recorded.
Use a JSON array for sources and include evidence provenance whenever available.
Create a separate JSONL record for every unique entity and effect_slug combination.
```

The LLM must receive:

1. Canonical nutrient IDs and names.
2. Canonical polyphenol/bioactive IDs and names.
3. Allowed effect slugs and labels.
4. Canonical health tags.
5. A fixed evidence scoring rubric.
6. The source material from which claims and citations should be derived.

## Existing Final Health-Report Dependency

The post-generation recipe health report currently retrieves nutrient summaries through:

```text
fetch_nutrient_effects_for_nutrients_v1
→ public.nutrient_effects_compressed_v1
→ beverage.nutrient_effects_compressed
```

Changing the detailed common source to `beverage.bioactive_health_evidence` does not automatically change that separate report path.

Therefore:

- `beverage.bioactive_health_evidence` is the temporary authoritative detailed evidence source for dietary-goal matching.
- `beverage.nutrient_effects_compressed` remains a temporary legacy dependency for post-generation nutrient summaries.
- Polyphenol/bioactive rows are not currently included in that compressed report path.

Until the report workflow is refactored, either:

1. Continue producing compressed summary rows for nutrient entities only, or
2. Accept that newly imported evidence affects dietary-goal matching but not the final compressed nutrient report.

Do not duplicate detailed nutrient evidence into `beverage.nutrient_health_effects` merely to support the compressed report. That report reads `beverage.nutrient_effects_compressed`, not `beverage.nutrient_health_effects`.

## Optional Legacy Nutrient Summary Output

If the import must also update the current final health report, generate a second JSONL file for nutrient entities only:

```json
{"nutrient_id":"4db79318-c3d9-4d43-9e27-b4ce00b6db39","nutrient_name":"Vitamin C","evidence_level":"strong","health_effect_positive":"Vitamin C may support immune function and antioxidant protection.","health_effect_negative":null,"tags":["immune_support","antioxidant_support"]}
```

Upsert these rows into:

```text
beverage.nutrient_effects_compressed
```

using its primary key:

```text
nutrient_id + evidence_level
```

This optional file is a compatibility output, not the authoritative detailed evidence dataset.

## Recommended Import Deliverables

The bulk generation process should produce:

```text
unified_health_evidence.jsonl
legacy_nutrient_effects_compressed.jsonl
rejected_health_evidence.jsonl
import_summary.json
```

`legacy_nutrient_effects_compressed.jsonl` is required only when the current post-generation report must also reflect the new nutrient evidence.

