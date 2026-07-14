#!/usr/bin/env bash
set -euo pipefail

RERUN_EXISTING=1 \
SEARCH_AS_INGREDIENT_NAME=1 \
TARGETS_CSV=ingredient_preparation_added/ingredient_research_run_targets.rerun_harmful_search_targets.csv \
LOOKUP_CSV=ingredient_preparation_added/added_ingredients_lookup.csv \
LOG_DIR=batch_logs/retrieval_added_harmful_rerun \
bash scripts/run_ingredient_retrieval_batch.sh
