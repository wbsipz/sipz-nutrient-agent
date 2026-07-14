#!/usr/bin/env bash
set -uo pipefail

INGREDIENTS_FILE="${INGREDIENTS_FILE:-ingredient_preparation_added/latest_retrieval_batch_ingredients.txt}"
RUN_ROOT="${RUN_ROOT:-ingredient_runs}"
LOG_DIR="${LOG_DIR:-batch_logs/claims_added_latest_parallel}"
PROVIDER="${PROVIDER:-deepseek}"
INGREDIENT_WORKERS="${INGREDIENT_WORKERS:-5}"
PAPER_WORKERS="${PAPER_WORKERS:-10}"
FORCE_VALIDATE="${FORCE_VALIDATE:-0}"
RUNS_TSV="${RUNS_TSV:-/tmp/sipz_added_latest_claim_runs.tsv}"

mkdir -p "$LOG_DIR"

if [ ! -f "$INGREDIENTS_FILE" ]; then
  echo "ERROR: INGREDIENTS_FILE does not exist: $INGREDIENTS_FILE" >&2
  exit 1
fi

uv run python - "$INGREDIENTS_FILE" "$RUN_ROOT" "$RUNS_TSV" <<'PY'
import re
import sys
from pathlib import Path

ingredients_file = Path(sys.argv[1])
runs_root = Path(sys.argv[2])
runs_tsv = Path(sys.argv[3])


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "ingredient"


rows = []
missing = []
for ingredient in [
    line.strip() for line in ingredients_file.read_text(encoding="utf-8").splitlines() if line.strip()
]:
    matches = sorted(runs_root.glob(f"*_{slugify(ingredient)}"), key=lambda path: path.name, reverse=True)
    if not matches:
        missing.append(ingredient)
        continue
    rows.append((ingredient, str(matches[0])))

if missing:
    raise SystemExit("Missing run directories for: " + ", ".join(missing))

runs_tsv.write_text(
    "".join(f"{ingredient}\t{run_dir}\n" for ingredient, run_dir in rows),
    encoding="utf-8",
)
print(f"Ingredients: {len(rows)}")
print(f"Wrote: {runs_tsv}")
PY

while IFS=$'\t' read -r ingredient run_dir; do
  printf "%s\0" "$run_dir"
done < "$RUNS_TSV" \
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
