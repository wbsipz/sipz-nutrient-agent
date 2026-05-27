from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import orjson

from sipz_agent.schemas.artifacts import Packet
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ValidatedClaim
from sipz_agent.schemas.effects import EFFECT_CSV_COLUMNS, EffectRow


class StudyArtifacts:
    def __init__(
        self,
        *,
        packet: Packet,
        sources: list[CandidateCitation],
        effect_rows: list[EffectRow],
        validated_claims: list[ValidatedClaim],
        rejected_claims: list[ValidatedClaim],
        summary_markdown: str,
        audit_events: list[dict[str, Any]],
    ) -> None:
        self.packet = packet
        self.sources = sources
        self.effect_rows = effect_rows
        self.validated_claims = validated_claims
        self.rejected_claims = rejected_claims
        self.summary_markdown = summary_markdown
        self.audit_events = audit_events


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE))


def model_dump_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [model_dump_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def write_effects_csv(path: Path, rows: list[EffectRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EFFECT_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def write_sources_markdown(path: Path, sources: list[CandidateCitation]) -> None:
    lines = ["# Candidate Sources", ""]
    if not sources:
        lines.extend(["No candidate sources were collected.", ""])
    for index, source in enumerate(sources, start=1):
        lines.append(f"## {index}. {source.title}")
        lines.append("")
        if source.url:
            lines.append(f"- URL: {source.url}")
        if source.doi:
            lines.append(f"- DOI: {source.doi}")
        if source.pmid:
            lines.append(f"- PMID: {source.pmid}")
        if source.year:
            lines.append(f"- Year: {source.year}")
        lines.append(f"- Source: {source.source}")
        lines.append(f"- Retrieval query: {source.retrieval_query}")
        if source.selection_reason:
            lines.append(f"- Selection reason: {source.selection_reason}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_study_artifacts(out_dir: str | Path, run_id: str, artifacts: StudyArtifacts) -> Path:
    run_dir = Path(out_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "packet.json", artifacts.packet.model_dump(mode="json"))
    write_json(run_dir / "sources.json", model_dump_jsonable(artifacts.sources))
    write_sources_markdown(run_dir / "sources.md", artifacts.sources)
    write_json(run_dir / "validated_claims.json", model_dump_jsonable(artifacts.validated_claims))
    write_json(run_dir / "rejected_claims.json", model_dump_jsonable(artifacts.rejected_claims))
    write_effects_csv(run_dir / "effects.csv", artifacts.effect_rows)
    (run_dir / "summary.md").write_text(artifacts.summary_markdown, encoding="utf-8")
    (run_dir / "audit_log.jsonl").write_text(
        "".join(orjson.dumps(event).decode("utf-8") + "\n" for event in artifacts.audit_events),
        encoding="utf-8",
    )

    return run_dir
