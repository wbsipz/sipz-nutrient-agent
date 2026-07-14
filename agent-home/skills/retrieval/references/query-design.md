# Retrieval Query Design

Use this reference when generating literature-search queries for oral-ingestion human-health research.

## Query Shape

Each retrieval query should combine:

```txt
entity terms AND oral/human terms AND study/outcome terms
```

Entity terms can include:

- exact input name
- common aliases
- spelling variants
- scientific names
- parent class
- ingredient form
- supplement or extract form

Oral/human terms can include:

- human
- adults
- clinical
- oral
- consumption
- ingestion
- dietary
- diet
- food
- beverage
- juice
- supplement

Study/outcome terms can include:

- randomized
- placebo
- crossover
- clinical trial
- systematic review
- meta-analysis
- cohort
- observational
- biomarker
- cardiovascular
- glucose
- inflammation
- sleep
- cognition
- antioxidant

## Tiered Queries

Use tiers instead of one broad search.

### Targeted Human Intervention

```txt
(entity terms) AND (randomized OR placebo OR "clinical trial" OR crossover)
```

### Review

```txt
(entity terms) AND ("systematic review" OR "meta-analysis" OR review)
```

### Dietary Exposure

```txt
(entity terms) AND (human OR adults) AND (consumption OR dietary OR oral OR food OR beverage OR supplement)
```

### Broader Parent Class

Use only when direct evidence is sparse or explicitly requested:

```txt
(parent class terms) AND (human OR clinical OR dietary) AND (health OR outcome terms)
```

## Ingredient-Specific Queries

For ingredients, include form terms:

- whole food
- juice
- beverage
- pulp
- puree
- powder
- concentrate
- extract
- supplement

Do not assume every ingredient row should be researched directly. Processed blends, juice blends, vague flavor names, or nutritionally redundant forms may be skipped or mapped to a canonical target.

## Exclusion Terms

Use exclusions carefully. They can improve precision but may remove useful records if overused.

Common terms to exclude or down-rank:

- topical
- inhalation
- injection
- mouse
- mice
- rat
- in vitro
- cell line
- animal model
- extraction
- processing
- agriculture
- pesticide
- chemical composition
- phytochemical profile

Prefer prompt-level screening over aggressive deterministic exclusion when user recall is more important than precision.
