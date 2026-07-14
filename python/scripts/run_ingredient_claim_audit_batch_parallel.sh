#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="${RUN_ROOT:-ingredient_runs}"
LOG_DIR="${LOG_DIR:-batch_logs/ingredient_claim_audit}"
PROVIDER="${PROVIDER:-deepseek}"
INGREDIENT_WORKERS="${INGREDIENT_WORKERS:-5}"
PAPER_WORKERS="${PAPER_WORKERS:-10}"
FORCE_AUDIT="${FORCE_AUDIT:-0}"

mkdir -p "$LOG_DIR"

find "$RUN_ROOT" -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -I {} -P "$INGREDIENT_WORKERS" bash -c '
RUN="$1"
LOG_DIR="$2"
PROVIDER="$3"
PAPER_WORKERS="$4"
FORCE_AUDIT="$5"
NAME="$(basename "$RUN")"

if [ ! -f "$RUN/ingredient_packet.json" ] || \
   [ ! -f "$RUN/proposed_ingredient_claims.json" ] || \
   [ ! -f "$RUN/validated_ingredient_claims.json" ] || \
   [ ! -f "$RUN/sources.json" ]; then
  echo "=== Skipping non-ready run: $NAME ==="
  exit 0
fi

ACCEPTED_COUNT="$(
  uv run python - "$RUN/validated_ingredient_claims.json" <<'"'"'PY'"'"'
import json
import sys

claims = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(1 for claim in claims if claim.get("accepted")))
PY
)"

if [ "$ACCEPTED_COUNT" -eq 0 ]; then
  echo "=== Skipping $NAME: no accepted claims ==="
  exit 0
fi

if [ "$FORCE_AUDIT" != "1" ] && [ -f "$RUN/audited_ingredient_claims.json" ]; then
  echo "=== Audit already complete; skipping: $NAME ==="
  exit 0
fi

echo "=== Auditing $NAME: $ACCEPTED_COUNT accepted validation claims ==="
uv run sipz-ingredients audit-claims \
  --run "$RUN" \
  --provider "$PROVIDER" \
  --workers "$PAPER_WORKERS" \
  > "$LOG_DIR/${NAME}_audit.log" 2>&1

if [ $? -ne 0 ]; then
  echo "FAILED audit: $NAME"
  exit 1
fi

echo "--- Done audit: $NAME"
' _ {} "$LOG_DIR" "$PROVIDER" "$PAPER_WORKERS" "$FORCE_AUDIT"
