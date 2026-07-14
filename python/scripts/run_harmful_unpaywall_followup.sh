#!/usr/bin/env bash
set -euo pipefail

UNPAYWALL_EMAIL="${UNPAYWALL_EMAIL:-w.benn@live.ca}" \
uv run python scripts/try_unpaywall_retrieval.py \
  --ingredients-file ingredient_preparation_added/rerun_harmful_search_targets.txt
