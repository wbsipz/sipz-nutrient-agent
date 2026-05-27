# Evidence Grading

The MVP uses deterministic support-level scoring.

```txt
human_systematic_review: 0.80+
human_rct: 0.75
human_observational: 0.60
human_mechanistic: 0.50
animal: 0.35
in_vitro: 0.30
mechanistic_theory: 0.25
review_author_interpretation: 0.45
composition_data: 0.40
```

Only accepted, quote-grounded claims can become exported rows in `effects.csv`.
