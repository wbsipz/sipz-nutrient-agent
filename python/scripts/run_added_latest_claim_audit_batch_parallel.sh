#!/usr/bin/env bash
set -uo pipefail

INGREDIENTS_FILE="${INGREDIENTS_FILE:-ingredient_preparation_added/latest_retrieval_batch_ingredients.txt}"
RUN_ROOT="${RUN_ROOT:-ingredient_runs}"
LOG_DIR="${LOG_DIR:-batch_logs/ingredient_claim_audit_added_latest}"
PROVIDER="${PROVIDER:-deepseek}"
INGREDIENT_WORKERS="${INGREDIENT_WORKERS:-5}"
PAPER_WORKERS="${PAPER_WORKERS:-10}"
FORCE_AUDIT="${FORCE_AUDIT:-0}"
RUNS_TSV="${RUNS_TSV:-/tmp/sipz_added_latest_audit_runs.tsv}"

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
