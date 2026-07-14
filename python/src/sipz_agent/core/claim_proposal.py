from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

import orjson
from pydantic import BaseModel, TypeAdapter

from sipz_agent.core.artifacts import model_dump_jsonable, write_json
from sipz_agent.core.models import LlmProvider
from sipz_agent.core.retrieval import truncate_text
from sipz_agent.schemas.artifacts import Packet
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim
from sipz_agent.schemas.raw_texts import RawTextRecord

SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])
PROPOSED_CLAIMS_ADAPTER = TypeAdapter(list[ProposedClaim])
PACKET_ADAPTER = TypeAdapter(Packet)

ALLOWED_EXPOSURE_CATEGORIES = {"natural_food_level", "supplement_level", "unclear"}
ClaimProposalProgress = Callable[[int, int, CandidateCitation], None]


class ClaimProposalResponse(BaseModel):
    claims: list[ProposedClaim]
    skipped_reason: str | None = None


CLAIM_PROPOSAL_ADAPTER = TypeAdapter(ClaimProposalResponse)
CLAIM_CONTEXT_MAX_CHARS = 45_000


def build_claim_extraction_context(body_text: str) -> tuple[str, dict[str, Any]]:
    original_length = len(body_text)
    without_references = re.split(
        r"(?im)^\s*(?:\d+[.)]?\s*)?(?:references|bibliography)\s*$",
        body_text,
        maxsplit=1,
    )[0]
    without_abstract = re.sub(
        r"(?is)\babstract\s+.*?(?=\b(?:keywords?|introduction|background|methods?|materials\s+and\s+methods)\b)",
        "",
        without_references,
        count=1,
    ).strip()
    if len(without_abstract) <= CLAIM_CONTEXT_MAX_CHARS:
        return without_abstract, {
            "original_char_count": original_length,
            "supplied_char_count": len(without_abstract),
            "strategy": "complete_body_without_abstract_or_references",
            "truncated": False,
            "selected_sections": ["complete_body"],
        }

    heading_pattern = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?"
        r"(introduction|background|methods?|materials\s+and\s+methods|results?|discussion|conclusions?)\s*$"
    )
    matches = list(heading_pattern.finditer(without_abstract))
    if not matches:
        inline_heading_pattern = re.compile(
            r"\b(Methods|Results|Discussion|Conclusions|Authors' conclusions)\b"
            r"(?=\s+(?:Criteria|Search|Types|Data|Description|Effects|Summary|Overall|In\b))"
        )
        matches = list(inline_heading_pattern.finditer(without_abstract))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(without_abstract)
        name = match.group(1).lower()
        if name == "authors' conclusions":
            name = "conclusions"
        sections.append((name, without_abstract[match.start():end].strip()))

    priorities = ("methods", "method", "materials and methods", "results", "result", "discussion", "conclusion", "conclusions", "introduction", "background")
    selected: list[tuple[int, str, str]] = []
    remaining = CLAIM_CONTEXT_MAX_CHARS
    for priority in priorities:
        for index, (name, text) in enumerate(sections):
            if name != priority or any(item[0] == index for item in selected):
                continue
            excerpt = text[:remaining]
            if excerpt:
                selected.append((index, name, excerpt))
                remaining -= len(excerpt) + 2
            if remaining <= 0:
                break
        if remaining <= 0:
            break
    if selected:
        selected.sort(key=lambda item: item[0])
        context = "\n\n".join(item[2] for item in selected)
        section_names = [item[1] for item in selected]
        strategy = "section_priority"
    else:
        context = truncate_text(without_abstract, max_chars=CLAIM_CONTEXT_MAX_CHARS)
        section_names = ["body_prefix"]
        strategy = "body_prefix_fallback"
    return context, {
        "original_char_count": original_length,
        "supplied_char_count": len(context),
        "strategy": strategy,
        "truncated": len(context) < len(without_abstract),
        "selected_sections": section_names,
    }


def body_text_for_record(raw_texts_dir: Path, record: RawTextRecord) -> str | None:
    if record.status != "full_text_found" or not record.text_path:
        return None
    path = raw_texts_dir / Path(record.text_path).name
    if not path.exists():
        path = raw_texts_dir.parent / record.text_path
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def claim_proposal_prompt(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    body_text: str,
) -> str:
    context, _ = build_claim_extraction_context(body_text)
    return f"""You are the paper-reader stage for the Sipz nutrient research agent.

Task:
- Propose candidate health-effect claims only. Do not prove them.
- Focus on human health effects from oral consumption of the requested nutrient/bioactive.
- Treat biological species as distinct research entities. Do not create a claim from a related
  species unless the requested target explicitly names the broader genus/class. Record the exact
  species or compound supporting each claim in compound.
- Reject animal-only and in-vitro-only evidence. Do not turn preclinical findings into human
  candidate claims, even when a review author describes them as relevant or promising.
- Do not propose a human claim whose population, outcome, or dose is extrapolated from animals,
  isolated cells, or other preclinical models.
- For reviews, classify evidence_type from the underlying evidence supporting the specific claim.
  Use review_author_interpretation only when the reviewed evidence includes human oral evidence.
- If the paper contains no human oral evidence suitable for a claim, return an empty claims array
  and explain that in skipped_reason.
- Separate effects possible at natural ingredient/food concentrations from supplement-level concentrations.
- Do not propose claims centered on pharmaceutical use, injected delivery, topical use, isolated cell dosing, or non-oral formulations.
- Do not propose a claim for the requested nutrient/bioactive from a multi-ingredient intervention
  unless its independent contribution is isolatable. A claim explicitly about the complete
  formulation is allowed only when the requested research target is that formulation.
- Include dose/concentration text if present; otherwise mark it unclear.
- If the text is a review, prefer claims that summarize human-relevant oral evidence and mark evidence_type as review_author_interpretation unless direct RCT/meta-analysis evidence is clear.
- Return at most 3 candidate claims.
- Make proposed_effect_slug lowercase snake_case and describe only the health effect/outcome. Do not
  include the nutrient/bioactive name in the slug.
- Keep each string field under 500 characters.
- Return at most 3 short limitations per claim.
- Return JSON only matching this shape: {{"claims":[...] , "skipped_reason": null}}.

Each claim must include:
- id: any non-empty temporary id
- nutrient_name
- citation_id
- statement
- proposed_effect_slug
- proposed_effect_label
- compound
- effect
- direction: exactly one of beneficial, harmful, neutral, mixed, or unclear; do not add qualifiers
- population
- dose_or_exposure
- outcome
- study_type
- limitations: always a JSON array of strings, even when there is only one limitation
- evidence_type: human_clinical, human_observational, human_mechanistic, animal, in_vitro, mechanistic_theory, review_author_interpretation, composition_data, or unclear
- intake_route: oral, topical, injection, in_vitro, or unclear
- exposure_category: natural_food_level, supplement_level, pharmaceutical_level, or unclear
- natural_concentration_relevance: a JSON string explaining relevance, or null; never a boolean
- supplement_level_relevance: a JSON string explaining relevance, or null; never a boolean
- Write relevance fields as direct prose. Do not prefix them with True, False, Yes, or No.
- pharmaceutical_centered
- concentration_notes

Requested nutrient/bioactive: {nutrient_name}
Citation ID: {citation.id}
Title: {citation.title}
DOI: {citation.doi or "unknown"}
PMID: {citation.pmid or "unknown"}
Year: {citation.year or "unknown"}

Paper body text, not abstract:
{context}
"""


def normalize_claim_ids(
    *,
    claims: list[ProposedClaim],
    nutrient_name: str,
    citation_id: str,
) -> list[ProposedClaim]:
    normalized = []
    for index, claim in enumerate(claims, start=1):
        stable_id = str(
            uuid5(
                NAMESPACE_URL,
                f"sipz-claim:{nutrient_name.casefold()}:{citation_id}:{index}:{claim.statement}",
            )
        )
        normalized.append(
            claim.model_copy(
                update={
                    "id": stable_id,
                    "nutrient_name": claim.nutrient_name or nutrient_name,
                    "citation_id": citation_id,
                }
            )
        )
    return normalized


def accepted_proposed_claim(claim: ProposedClaim) -> bool:
    if claim.intake_route != "oral":
        return False
    if claim.pharmaceutical_centered:
        return False
    if claim.exposure_category not in ALLOWED_EXPOSURE_CATEGORIES:
        return False
    if claim.evidence_type in {"animal", "in_vitro", "mechanistic_theory"}:
        return False
    return True


def propose_claims_for_source(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    body_text: str,
    provider: LlmProvider,
) -> list[ProposedClaim]:
    response = provider.complete_json(
        claim_proposal_prompt(nutrient_name=nutrient_name, citation=citation, body_text=body_text),
        CLAIM_PROPOSAL_ADAPTER,
    )
    claims = normalize_claim_ids(
        claims=response.claims,
        nutrient_name=nutrient_name,
        citation_id=citation.id,
    )
    return [claim for claim in claims if accepted_proposed_claim(claim)]


def write_proposed_claims_markdown(path: Path, claims: list[ProposedClaim]) -> None:
    lines = ["# Proposed Claims", ""]
    if not claims:
        lines.extend(["No candidate claims were proposed.", ""])
    for index, claim in enumerate(claims, start=1):
        lines.append(f"## {index}. {claim.proposed_effect_label}")
        lines.append("")
        lines.append(f"- Citation ID: {claim.citation_id}")
        lines.append(f"- Statement: {claim.statement}")
        lines.append(f"- Effect: {claim.effect or claim.proposed_effect_slug}")
        lines.append(f"- Direction: {claim.direction}")
        lines.append(f"- Evidence type: {claim.evidence_type}")
        lines.append(f"- Intake route: {claim.intake_route}")
        lines.append(f"- Exposure category: {claim.exposure_category}")
        if claim.dose_or_exposure:
            lines.append(f"- Dose/exposure: {claim.dose_or_exposure}")
        if claim.concentration_notes:
            lines.append(f"- Concentration notes: {claim.concentration_notes}")
        if claim.limitations:
            lines.append(f"- Limitations: {'; '.join(claim.limitations)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_audit_event(run_dir: Path, event: dict[str, Any]) -> None:
    audit_path = run_dir / "audit_log.jsonl"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(orjson.dumps(event).decode("utf-8") + "\n")


def update_packet_proposed_claims(run_dir: Path, count: int) -> None:
    packet_path = run_dir / "packet.json"
    if not packet_path.exists():
        return
    packet = PACKET_ADAPTER.validate_python(orjson.loads(packet_path.read_bytes()))
    packet_data = packet.model_dump(mode="json")
    packet_data["counts"]["proposed_claims"] = count
    packet_data["completed_at"] = datetime.now(UTC).isoformat()
    write_json(packet_path, packet_data)


def propose_claims_from_raw_texts(
    *,
    nutrient_name: str,
    sources_path: Path,
    raw_texts_manifest_path: Path,
    raw_texts_dir: Path,
    out_dir: Path,
    provider: LlmProvider,
    update_run_packet: bool = False,
    progress: ClaimProposalProgress | None = None,
) -> list[ProposedClaim]:
    sources = SOURCES_ADAPTER.validate_python(orjson.loads(sources_path.read_bytes()))
    raw_records = RAW_TEXTS_ADAPTER.validate_python(orjson.loads(raw_texts_manifest_path.read_bytes()))
    citations_by_id = {citation.id: citation for citation in sources}

    if update_run_packet:
        append_audit_event(
            out_dir,
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "claim_proposal_started",
                "raw_text_records": len(raw_records),
            },
        )

    papers: list[tuple[CandidateCitation, str]] = []
    for record in raw_records:
        citation = citations_by_id.get(record.source_id)
        if citation is None:
            continue
        body_text = body_text_for_record(raw_texts_dir, record)
        if not body_text:
            continue
        papers.append((citation, body_text))

    claims: list[ProposedClaim] = []
    total_papers = len(papers)
    for index, (citation, body_text) in enumerate(papers, start=1):
        if progress is not None:
            progress(index, total_papers, citation)
        claims.extend(
            propose_claims_for_source(
                nutrient_name=nutrient_name,
                citation=citation,
                body_text=body_text,
                provider=provider,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "proposed_claims.json", model_dump_jsonable(claims))
    write_proposed_claims_markdown(out_dir / "proposed_claims.md", claims)

    if update_run_packet:
        update_packet_proposed_claims(out_dir, len(claims))
        append_audit_event(
            out_dir,
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "claim_proposal_completed",
                "proposed_claims": len(claims),
            },
        )

    return claims
