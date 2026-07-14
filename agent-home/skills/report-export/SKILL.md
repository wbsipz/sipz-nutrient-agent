---
name: report-export
description: Creates deterministic public Markdown, JSON, and CSV reports from completed claim validation and returns a terminal-ready claims table and report link. Use when the user asks for results, a report, a claims table, structured research output, downloadable artifacts, or the final stage of a nutrient, bioactive, botanical, or ingredient review.
compatibility: Requires finalized source, proposed-claim, validated-claim, and rejected-claim artifacts; held, adequacy, and failure artifacts are optional.
---

# Public Report Export

Call `export_research_report`; never draft, merge, summarize, or recalculate evidence manually. The
tool deterministically writes `claims_report.md`, `claims_report.json`, `claims_report.csv`, and
`report_manifest.json`. It does not call an LLM and does not create database-import records.

After the tool returns, print `terminal_response_markdown` verbatim as the entire response. It is a
preassembled deterministic block containing the summary, complete validated-claims table, absolute
save location, report and folder links, and JSON/CSV paths. Do not preface, summarize, shorten, or
rewrite it.

Do not omit harmful or mixed findings. Do not put rejected or held claims in the validated table.
Report their counts and leave their details in the full Markdown and JSON artifacts. Do not expose
Supabase fields, unresolved database IDs, effect scores, or internal import formats.

For a complete new run, use the adaptive `advance_research_pipeline` loop through claim validation,
then pass its validation paths and `expansion_state_path` to this tool. For a report-only request,
reuse existing validation artifacts and run only this stage.
