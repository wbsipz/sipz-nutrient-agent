#!/usr/bin/env bash
set -uo pipefail

INGREDIENT_WORKERS="${INGREDIENT_WORKERS:-5}"
PAPER_WORKERS="${PAPER_WORKERS:-10}"
PROVIDER="${PROVIDER:-deepseek}"
LOG_DIR="${LOG_DIR:-batch_logs/claims_all_parallel}"
RUN_ROOT="${RUN_ROOT:-ingredient_runs}"
FORCE_VALIDATE="${FORCE_VALIDATE:-0}"

mkdir -p "$LOG_DIR"

find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -I {} -P "$INGREDIENT_WORKERS" bash -c '
set -uo pipefail

RUN="$1"
NAME="$(basename "$RUN")"
LOG_DIR="$2"
PROVIDER="$3"
PAPER_WORKERS="$4"
FORCE_VALIDATE="$5"

if [ ! -f "$RUN/ingredient_packet.json" ] || \
   [ ! -f "$RUN/sources.json" ] || \
   [ ! -f "$RUN/raw_texts.json" ] || \
   [ ! -d "$RUN/raw_texts" ]; then
  echo "=== Skipping non-ready run: $NAME ==="
  exit 0
fi

FULL_TEXT_COUNT="$(
  uv run python - "$RUN/raw_texts.json" <<'"'"'PY'"'"'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    records = json.load(handle)
print(sum(1 for record in records if record.get("status") == "full_text_found"))
PY
)"

if [ "$FULL_TEXT_COUNT" -eq 0 ]; then
  echo "=== Skipping $NAME: no full text ==="
  exit 0
fi

echo "=== $NAME: $FULL_TEXT_COUNT full-text papers ==="

if [ ! -f "$RUN/proposed_ingredient_claims.json" ]; then
  echo "--- Proposing claims: $NAME"
  uv run sipz-ingredients propose-claims \
    --run "$RUN" \
    --provider "$PROVIDER" \
    --workers "$PAPER_WORKERS" \
    > "$LOG_DIR/${NAME}_propose.log" 2>&1

  if [ $? -ne 0 ]; then
    echo "FAILED propose: $NAME"
    exit 1
  fi
else
  echo "--- Proposed claims already exist; reusing: $NAME"
fi

if [ "$FORCE_VALIDATE" != "1" ] && \
   [ -f "$RUN/validated_ingredient_claims.json" ] && \
   [ -f "$RUN/rejected_ingredient_claims.json" ] && \
   [ -f "$RUN/ingredient_validation_failures.json" ] && \
   [ -f "$RUN/ingredient_validation_summary.md" ]; then
  echo "--- Validation already complete; skipping: $NAME"
  exit 0
fi

echo "--- Validating claims: $NAME"
uv run sipz-ingredients validate-claims \
  --run "$RUN" \
  --provider "$PROVIDER" \
  --workers "$PAPER_WORKERS" \
  > "$LOG_DIR/${NAME}_validate.log" 2>&1

if [ $? -ne 0 ]; then
  echo "FAILED validate: $NAME"
  exit 1
fi

echo "--- Done: $NAME"
' _ {} "$LOG_DIR" "$PROVIDER" "$PAPER_WORKERS" "$FORCE_VALIDATE"
