#!/usr/bin/env bash
set -euo pipefail

TARGETS_CSV="${TARGETS_CSV:-ingredient_preparation_full/ingredient_research_run_targets.final.csv}"
LOOKUP_CSV="${LOOKUP_CSV:-health_reports_ingredients_v1_rows.csv}"
PENDING_TSV="${PENDING_TSV:-/tmp/sipz_pending_ingredients_resume.tsv}"
LOG_DIR="${LOG_DIR:-batch_logs}"
PROVIDER="${PROVIDER:-deepseek}"
DEPTH="${DEPTH:-standard}"
FULL_TEXT_WORKERS="${FULL_TEXT_WORKERS:-4}"
LIMIT="${LIMIT:-}"
PER_INGREDIENT_TIMEOUT="${PER_INGREDIENT_TIMEOUT:-}"
SEARCH_AS_INGREDIENT_NAME="${SEARCH_AS_INGREDIENT_NAME:-0}"
RERUN_EXISTING="${RERUN_EXISTING:-0}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$TARGETS_CSV" ]]; then
  printf "ERROR: TARGETS_CSV does not exist: %q\n" "$TARGETS_CSV" >&2
  exit 1
fi
if [[ ! -f "$LOOKUP_CSV" ]]; then
  printf "ERROR: LOOKUP_CSV does not exist: %q\n" "$LOOKUP_CSV" >&2
  exit 1
fi

rm -f "$PENDING_TSV"

uv run python - "$TARGETS_CSV" "$PENDING_TSV" "$RERUN_EXISTING" <<'PY'
import csv
import re
import sys
from pathlib import Path

targets_csv = Path(sys.argv[1])
pending_tsv = Path(sys.argv[2])
rerun_existing = sys.argv[3] == "1"
runs_dir = Path("ingredient_runs")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "ingredient"


completed_slugs = set()
for run_dir in runs_dir.glob("*"):
    if not run_dir.is_dir():
        continue
    match = re.match(r".*?_([^/]+)$", run_dir.name)
    if match:
        completed_slugs.add(match.group(1))

rows = list(csv.DictReader(targets_csv.open(newline="", encoding="utf-8")))
pending = rows if rerun_existing else [
    row for row in rows if slugify(row["run_target_name"]) not in completed_slugs
]

with pending_tsv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    for row in pending:
        writer.writerow(
            [
                row["run_target_name"],
                row.get("representative_canonical_beverage_id") or "",
            ]
        )

print(f"Final targets: {len(rows)}")
print(f"Completed matching final targets: {len(rows) - len(pending)}")
print(f"Pending: {len(pending)}")
print(f"Wrote: {pending_tsv}")
PY

status_file="$LOG_DIR/retrieval_batch_status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf "timestamp\tingredient\tstatus\texit_code\tlog\n" > "$status_file"
fi

processed=0
while IFS=$'\t' read -r ingredient canonical_id; do
  ingredient="${ingredient%$'\r'}"
  canonical_id="${canonical_id%$'\r'}"
  if [[ -n "$LIMIT" && "$processed" -ge "$LIMIT" ]]; then
    echo "Reached LIMIT=$LIMIT"
    break
  fi
  processed=$((processed + 1))

  safe_name="$(printf '%s' "$ingredient" | tr ' /' '--' | tr -cd '[:alnum:]_.-')"
  log="$LOG_DIR/retrieval_${safe_name}.log"

  echo "=== Running retrieval: $ingredient ($canonical_id) ==="
  cmd=(
    uv run sipz-ingredients study "$ingredient"
    --lookup "$LOOKUP_CSV"
    --provider "$PROVIDER"
    --depth "$DEPTH"
    --retrieve-full-text
    --full-text-workers "$FULL_TEXT_WORKERS"
  )
  if [[ -n "$canonical_id" ]]; then
    cmd+=(--canonical-beverage-id "$canonical_id")
  fi
  if [[ "$SEARCH_AS_INGREDIENT_NAME" == "1" ]]; then
    cmd+=(--search-as-ingredient-name)
  fi

  exit_code=0
  if [[ -n "$PER_INGREDIENT_TIMEOUT" && "$(command -v timeout)" ]]; then
    timeout "$PER_INGREDIENT_TIMEOUT" "${cmd[@]}" > "$log" 2>&1 || exit_code=$?
  else
    "${cmd[@]}" > "$log" 2>&1 || exit_code=$?
  fi
  timestamp="$(date -Is)"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "OK: $ingredient"
    printf "%s\t%s\tok\t%s\t%s\n" "$timestamp" "$ingredient" "$exit_code" "$log" >> "$status_file"
  else
    echo "FAILED: $ingredient, exit=$exit_code, log=$log"
    printf "%s\t%s\tfailed\t%s\t%s\n" "$timestamp" "$ingredient" "$exit_code" "$log" >> "$status_file"
  fi
done < "$PENDING_TSV"

echo "Processed this invocation: $processed"
echo "Status log: $status_file"
