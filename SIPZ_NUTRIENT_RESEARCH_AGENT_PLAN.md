# Sipz Nutrient Research Agent Plan

## Purpose

Build a turnkey, publicly hostable agent that supports the Sipz platform by researching nutrients, bioactives, and phytochemical classes. The agent should improve health-effect summaries for existing Sipz nutrients, suggest new nutrients or bioactives to add, map specific compounds into broader categories, and identify ingredients that contain those nutrients or classes.

The project should also be designed as a portfolio-quality demonstration of agentic engineering: clear architecture, transparent evidence handling, reproducible outputs, local/free model support, and safe human-reviewed database integration.

## Current Sipz Context

The current Sipz codebase already has an admin nutrient workflow:

- `beverage.nutrients`
- `beverage.health_effects`
- `beverage.nutrient_health_effects`
- admin endpoints for creating nutrients and nutrient effects
- `sources` support on nutrient health effects
- tag generation for health-effect lookup

Because of that, the agent should be built as a separate public GitHub project that produces reviewed research packets and database-ready patches, rather than being embedded directly into the Nuxt app.

## Core Product

Suggested future repository name:

```txt
sipz-nutrient-research-agent
```

Primary goal:

```txt
Given a nutrient, bioactive, or category, perform literature-backed research, summarize likely human-health effects, grade evidence quality, identify food/beverage ingredient sources, and produce structured records ready for Sipz ingestion.
```

The agent should support two main modes:

```bash
sipz-agent study "3,5-Diferuloylquinic acid"
sipz-agent discover --exclude existing_nutrients.csv --domain beverages
```

The output should be a versioned research artifact, not just a database write:

```txt
research_runs/
  2026-05-26_3_5_diferuloylquinic_acid/
    packet.json
    summary.md
    citations.bib
    proposed_nutrient_patch.json
    effects.csv
    proposed_ingredient_sources_patch.json
    audit_log.jsonl
```

## High-Level Architecture

Use a staged, deterministic Python pipeline with LLMs used only where they add value.

```txt
CLI / Web UI
   |
Research Orchestrator
   |
   |-- Entity Resolver
   |-- Literature Retriever
   |-- Paper Screener
   |-- Evidence Extractor
   |-- Claim Validator
   |-- Quote Grounding Checker
   |-- Effect Synthesizer
   |-- Category Mapper
   |-- Ingredient Source Finder
   |-- Quality Auditor
   |-- Human Review Exporter
   |
Sipz Import JSON / SQL / Admin API
```

The workflow should feel agentic, but internally it should be structured and auditable. That makes it cheaper, more reliable, easier to test, and easier to explain publicly.

## Model Strategy

Do not hard-code the system to one model provider. Use a provider abstraction:

```py
class LlmProvider(Protocol):
    def complete_json(self, prompt: str, adapter: TypeAdapter[T]) -> T: ...
```

Support:

- local Ollama models on the mini-PC
- OpenAI-compatible endpoints
- DeepSeek-compatible endpoints
- optional paid fallback model for difficult audit passes

Good LLM use cases:

- finding candidate papers for a nutrient, bioactive, or class
- synonym expansion
- paper relevance classification
- structured evidence extraction
- body-text claim validation
- summary drafting
- contradiction checks

Avoid relying on LLMs for:

- citation acceptance without quote verification
- quote acceptance without locating the quote in source text
- final evidence grading without rules
- production database writes without review

The key principle is:

```txt
LLM can propose.
LLM cannot prove.
Proof must be grounded in retrievable paper text.
```

## Research Pipeline

### 1. Entity Resolution

For an input like `3,5-Diferuloylquinic acid`, resolve:

- preferred name
- synonyms
- spelling variants
- parent class
- related compounds
- identifiers where available

Example hierarchy:

```txt
3,5-Diferuloylquinic acid
  -> diferuloylquinic acids
  -> chlorogenic acids
  -> hydroxycinnamic acids
  -> polyphenols
  -> bioactives
```

This matters because very specific compounds may have weak direct human evidence, while the parent class may have stronger evidence.

### 2. Literature Retrieval

Use retrieval APIs and scrape-free sources where possible:

- PubMed / NCBI E-utilities
- Europe PMC
- Semantic Scholar
- Crossref
- OpenAlex
- clinical trial registries where useful

Store every candidate paper with metadata:

```json
{
  "title": "...",
  "doi": "...",
  "pmid": "...",
  "year": 2021,
  "study_type": "human_rct",
  "abstract": "...",
  "url": "...",
  "retrieval_query": "..."
}
```

The first LLM loop may use search tooling such as Firecrawl to find candidate papers that appear to discuss the nutrient or bioactive and health outcomes. This agent should only produce candidate citations and metadata. It should not decide that a health claim is true.

### 3. Paper Screening

Screen papers into categories:

```txt
include_direct_human
include_indirect_human
include_animal
include_in_vitro
include_food_composition
exclude_irrelevant
exclude_supplement_marketing
exclude_unverifiable
```

For health-effect summaries, prioritize:

1. systematic reviews / meta-analyses
2. human randomized trials
3. human observational studies
4. mechanistic human studies
5. animal studies
6. in vitro studies

### 4. Evidence Extraction

Each candidate paper is sent to a paper-reader agent. The reader proposes short health statements or conclusions, usually 2-3 sentences long. These are candidate statements only.

Extract structured claims from each included paper:

```json
{
  "compound": "chlorogenic acids",
  "effect": "postprandial glucose response",
  "direction": "beneficial",
  "population": "adults with impaired glucose metabolism",
  "dose_or_exposure": "...",
  "outcome": "...",
  "study_type": "human_rct",
  "confidence": 0.72,
  "limitations": [
    "short duration",
    "small sample size"
  ],
  "citation_ids": ["pmid:..."]
}
```

### 5. Body-Only Claim Validation

Each proposed statement is passed to a validator agent.

The validator receives:

- the proposed statement
- citation metadata
- the paper body text
- no abstract text

The abstract should be excluded from the validator context because abstracts often contain broad summary claims that are supported elsewhere in the paper. The goal is to prevent the validator from simply echoing the abstract. The validator must construct an argument using snippets from the body of the paper.

Example input:

```txt
Nutrient: quercetin
Candidate paper: Recent Advances in Potential Health Benefits of Quercetin
Proposed statement: Quercetin exhibits anticancer effects by inhibiting cancer cell proliferation and inducing apoptosis.
Validator context: full paper text excluding abstract
```

The validator should produce an accepted output row shaped like the current Sipz nutrient health-effect table so it can be dropped into the database or import preview with minimal transformation.

Target output fields:

```txt
id
nutrient_id
effect_slug
effect_label
description
score
evidence_level
tags
sources
created_at
updated_at
nutrient_name
match_status
match_confidence
match_notes
```

Example accepted validator output:

```json
{
  "id": "45929237-0046-4769-bd55-bd0c8de2199b",
  "nutrient_id": "0291c7ef-13e0-4740-a669-a1aca1b9655c",
  "effect_slug": "dental_caries_prevention",
  "effect_label": "Dental caries prevention",
  "description": "Human evidence supports that oral fluoride exposure from fluoridated drinking water reduces dental caries risk, especially in children. Recent systematic-review evidence still suggests benefit, although contemporary effect estimates are smaller and more uncertain than older studies.",
  "score": 0.84,
  "evidence_level": "moderate",
  "tags": [
    "cavity_prevention",
    "tooth_decay",
    "dental_caries",
    "oral_health",
    "strong_teeth",
    "tooth_protection",
    "kids_dental_health",
    "adult_dental_health",
    "preventive_dental_care",
    "enamel_support",
    "tooth_enamel",
    "tooth_strength",
    "decay_reduction",
    "tooth_decay_prevention",
    "caries_risk_reduction",
    "dental_health",
    "tooth_health",
    "oral_hygiene",
    "fluoride_exposure",
    "fluoridated_water",
    "community_water_fluoridation",
    "children_dental_health",
    "long_term_tooth_protection"
  ],
  "sources": [
    {
      "url": "https://pubmed.ncbi.nlm.nih.gov/39362658/",
      "type": "web_source",
      "notes": ""
    },
    {
      "url": "https://ods.od.nih.gov/factsheets/Fluoride-HealthProfessional/?stream=top",
      "type": "web_source",
      "notes": ""
    },
    {
      "url": "https://www.cdc.gov/fluoridation/about/statement-on-the-evidence-supporting-the-safety-and-effectiveness-of-community-water-fluoridation.html",
      "type": "web_source",
      "notes": ""
    },
    {
      "url": "https://www.who.int/teams/environment-climate-change-and-health/chemical-safety-and-health/health-impacts/chemicals/inadequate-or-excess-fluoride",
      "type": "web_source",
      "notes": ""
    }
  ],
  "created_at": "2026-03-27T01:01:27.305242Z",
  "updated_at": "2026-03-27T01:01:27.305242Z",
  "nutrient_name": "Fluoride",
  "match_status": "auto_llm",
  "match_confidence": 1.0,
  "match_notes": "Generated via LLM from nutrient list."
}
```

The claim-validation evidence should be stored beside the table row as audit metadata, not mixed into the table-compatible row:

```json
{
  "effect_row_id": "45929237-0046-4769-bd55-bd0c8de2199b",
  "verdict": "supported_with_limitations",
  "support_level": "human_systematic_review",
  "claim_scope": "human evidence for dental caries prevention from oral fluoride exposure; benefit estimate varies by population and exposure context",
  "supporting_quotes": [
    {
      "quote": "exact quote from paper body",
      "section": "Results",
      "reason": "Supports reduced caries risk."
    },
    {
      "quote": "exact quote from paper body",
      "section": "Discussion",
      "reason": "Supports uncertainty or limitations in contemporary evidence."
    }
  ],
  "limitations": [
    "Effect size may differ between older and contemporary studies.",
    "Evidence depends on exposure route, population, and baseline fluoride availability."
  ],
  "accepted": true
}
```

This creates two useful artifacts:

```txt
effects.csv
  Supabase-ready rows ready for import or preview.

validated_claims.json
  Audit trail with body-text quotes, quote-match status, claim scope, and validator reasoning.
```

The validator must check two things separately:

```txt
1. Text support:
   Does the paper body actually support this statement?

2. Evidence scope:
   Is the statement phrased at the right strength?
```

This prevents claim inflation.

Example:

```txt
Bad:
Quercetin has anticancer effects.

Better:
Quercetin has shown anticancer mechanisms in preclinical research, including effects on cancer cell proliferation and apoptosis. Human clinical evidence is not sufficient to treat this as a proven cancer-prevention effect.
```

### 6. Quote Grounding Check

After validation, every validator quote must be searched in the paper text programmatically.

If a quote cannot be located, assume the validator hallucinated it and reject the statement or send it to manual review.

Use matching tiers:

```txt
Tier 1: exact substring match
Tier 2: normalized whitespace match
Tier 3: dehyphenated / ligature-normalized match
Tier 4: reject unless manually reviewed
```

This is necessary because PDF extraction often introduces line breaks, ligatures, hyphenation, and whitespace changes.

A claim can be accepted only if:

- the validator marks it supported
- the supporting quotes are found in the paper text
- the claim scope is not stronger than the evidence
- the citation metadata is valid

### 7. Claim Scope Classification

For each accepted claim, classify the underlying evidence type:

```txt
human_clinical
human_observational
human_mechanistic
animal
in_vitro
mechanistic_theory
review_author_interpretation
composition_data
```

This distinction is especially important for review papers. A review may accurately state a mechanism, but the Sipz health summary should not treat that mechanism as a proven human outcome.

### 8. Health Effect Synthesis

Generate Sipz-ready summaries only after extraction.

Example:

```json
{
  "effect_slug": "glucose_metabolism",
  "effect_label": "Glucose metabolism",
  "description": "Chlorogenic-acid-rich compounds may support healthier post-meal glucose handling, though evidence varies by dose, source ingredient, and study population.",
  "score": 0.64,
  "evidence_level": "moderate",
  "sources": []
}
```

Important rule:

If evidence exists only for a parent category, the agent should say so.

Example:

```txt
Direct evidence for 3,5-Diferuloylquinic acid is limited. Most human evidence applies to chlorogenic acids as a broader class, commonly studied through coffee, yerba mate, or plant extracts.
```

That distinction is crucial for data quality.

## Proposed Data Model Additions

Keep the existing `nutrients` and `nutrient_health_effects` tables, but add supporting tables for hierarchy, provenance, and ingredient source mapping.

### `nutrient_categories`

```sql
id
slug
name
description
parent_id
category_type -- nutrient_class, bioactive_class, phytochemical_class
```

Examples:

```txt
bioactives
polyphenols
hydroxycinnamic_acids
chlorogenic_acids
diferuloylquinic_acids
```

### `nutrient_category_memberships`

```sql
nutrient_id
category_id
relationship_type -- exact_member, subclass_member, related_compound
confidence
source
```

### `nutrient_research_runs`

```sql
id
nutrient_id nullable
input_name
status
model_provider
started_at
completed_at
artifact_path
quality_score
review_status -- draft, needs_review, approved, rejected
```

### `nutrient_evidence_claims`

```sql
id
nutrient_id nullable
category_id nullable
effect_slug
claim_text
direction
study_type
population
evidence_strength
citation_ids
validated_quotes
claim_scope
limitations
created_by_run_id
```

Each stored evidence claim should preserve the validated quotes and the claim-scope classification that justified it. This makes it possible to inspect why a health effect was accepted later.

### `nutrient_ingredient_sources`

```sql
id
nutrient_id nullable
category_id nullable
ingredient_name
canonical_ingredient_id nullable
evidence_type -- composition_database, paper, inferred_category
concentration_text nullable
confidence
sources
```

This allows Sipz to represent ingredient relationships accurately:

```txt
Coffee contains chlorogenic acids.
Yerba mate contains caffeoylquinic acids.
Artichoke contains chlorogenic-acid derivatives.
```

without pretending the system knows the exact amount of every specific molecule in every ingredient.

## Agent Commands

Design the public repo around clean commands:

```bash
sipz-agent study "chlorogenic acids"
sipz-agent study "3,5-Diferuloylquinic acid" --depth deep
sipz-agent improve --input current_nutrients.csv
sipz-agent discover --exclude current_nutrients.csv --domain beverages
sipz-agent export --run runs/abc123 --format csv
sipz-agent audit --run runs/abc123
```

## Human Review Workflow

Do not auto-update production.

Recommended flow:

```txt
Agent researches nutrient
Agent validates claims against paper body text
Agent verifies validator quotes with string matching
Agent creates research packet
Human reviews summary.md
Human approves or edits
Agent exports Supabase-ready CSV
Human imports CSV into Supabase
Database updates
```

Later, add an authenticated admin endpoint:

```txt
POST /api/admin/nutrient-research/import
```

Keep review mandatory.

## Evidence Scoring

Use a transparent score, not just an LLM confidence number.

Evidence grading should be based only on accepted claims that passed quote grounding. Rejected, ungrounded, or over-scoped claims should not contribute to the final evidence score.

Example rubric:

```txt
+0.35 systematic review
+0.25 human RCT
+0.15 observational human evidence
+0.10 plausible mechanism
-0.20 conflicting results
-0.15 only animal/in-vitro evidence
-0.10 small sample sizes
-0.10 unclear dosage
```

Map the final score:

```txt
strong: >= 0.75
moderate: 0.50-0.74
limited: 0.25-0.49
insufficient: < 0.25
```

The current Sipz app supports `strong`, `moderate`, and `limited`. The agent can use `insufficient` internally and avoid importing those as user-facing health effects.

## MVP Roadmap

### Phase 1: Research Packet Generator

Build the standalone CLI.

Inputs:

- nutrient name
- optional existing nutrient CSV
- optional target health effects

Outputs:

- `summary.md`
- `packet.json`
- `sources.json`
- `validated_claims.json`
- `rejected_claims.json`
- `effects.csv`

No database writes yet.

### Phase 2: CSV Export Contract

Define a stable CSV that maps to the target Supabase table:

```txt
effects.csv
```

Use rows with these columns:

```txt
id,nutrient_id,effect_slug,effect_label,description,score,evidence_level,tags,sources,created_at,updated_at,nutrient_name,match_status,match_confidence,match_notes
```

Keep quote-level validator evidence in `validated_claims.json`, linked by `effect_row_id`, so the CSV remains clean while the research packet remains auditable.

### Phase 3: Admin Import Tool

Add an importer to Sipz that lets an admin preview:

- new nutrient
- changed summaries
- new health effects
- evidence level changes
- sources
- ingredient-source suggestions

Then approve manually.

### Phase 4: Discovery Mode

Agent scans common beverage ingredients and suggests missing bioactives.

Examples:

```txt
green tea -> catechins, EGCG, L-theanine
coffee -> chlorogenic acids, trigonelline
hibiscus -> anthocyanins, hibiscus acid
cocoa -> flavanols, theobromine
turmeric -> curcuminoids
ginger -> gingerols, shogaols
```

### Phase 5: Public GitHub Polish

To market the project as agentic engineering work, the repo should include:

- clear architecture diagram
- Docker Compose setup
- local model support
- sample completed research packets
- eval suite
- fake/demo mode with bundled sample papers
- safety section
- hallucinated-citation prevention section showing body-only validation and quote grounding
- screenshots or terminal recordings
- Sipz integration adapter

## Recommended Repo Structure

```txt
sipz-nutrient-research-agent/
  apps/
    cli/
    web/
  packages/
    core/
      orchestrator/
      models/
      retrieval/
      extraction/
      validation/
      quote-grounding/
      synthesis/
      audit/
    sipz-adapter/
    schemas/
  examples/
    chlorogenic-acids/
    diferuloylquinic-acid/
  docs/
    architecture.md
    evidence-grading.md
    sipz-integration.md
  docker-compose.yml
  README.md
```

## Key Design Principle

Separate these three things:

```txt
Evidence claim:
  "A study found X under Y conditions."

Health-effect summary:
  "This may support Z, with moderate evidence."

Sipz product behavior:
  "Use this nutrient when matching recipes to a health goal."
```

That separation will make the data cleaner and the engineering story much stronger.

## Recommended First Demo

Start with `chlorogenic acids` as the first end-to-end demo, not `3,5-Diferuloylquinic acid`.

Reason: it gives enough literature to demonstrate deep research, category hierarchy, specific-compound inheritance, ingredient-source mapping, and evidence grading.

Then use `3,5-Diferuloylquinic acid` as the second demo to show how the agent handles sparse evidence responsibly.
