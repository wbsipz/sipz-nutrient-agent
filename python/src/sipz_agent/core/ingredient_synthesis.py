from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import orjson
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.models import HeuristicProvider, LlmProvider
from sipz_agent.schemas.artifacts import Packet
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, ValidatedClaim
from sipz_agent.schemas.ingredients import (
    IngredientPacket,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)


VALIDATED_ADAPTER = TypeAdapter(list[ValidatedClaim])
PROPOSED_ADAPTER = TypeAdapter(list[ProposedClaim])
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
INGREDIENT_PACKET_ADAPTER = TypeAdapter(IngredientPacket)
PROPOSED_INGREDIENT_ADAPTER = TypeAdapter(list[ProposedIngredientClaim])
VALIDATED_INGREDIENT_ADAPTER = TypeAdapter(list[ValidatedIngredientClaim])


INGREDIENT_HEALTH_REPORT_COLUMNS = [
    "canonical_beverage_id",
    "canonical_beverage_name",
    "health_effect_positive",
    "health_effect_negative",
    "health_effect_positive_embedding",
    "health_effect_negative_embedding",
    "health_effect_positive_tags",
    "health_effect_negative_tags",
    "payload_nutrients_count",
    "matched_nutrients_count",
    "skipped_keys_count",
    "missing_summary_count",
    "missing_amount_count",
    "source",
    "embedding_model",
    "embedded_at",
    "created_at",
    "updated_at",
]

SOURCE_MARKER = "literature_agent_v1"


class IngredientReportTag(BaseModel):
    tag: str = Field(min_length=1, max_length=80)
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, value: object) -> str:
        return slug_text(str(value)) or "health_effect"


class PolishedIngredientReport(BaseModel):
    health_effect_positive: str = Field(default="", max_length=4000)
    health_effect_negative: str = Field(default="", max_length=4000)
    health_effect_positive_tags: list[IngredientReportTag] = Field(default_factory=list)
    health_effect_negative_tags: list[IngredientReportTag] = Field(default_factory=list)
    synthesis_notes: str = Field(default="", max_length=1000)


POLISHED_REPORT_ADAPTER = TypeAdapter(PolishedIngredientReport)


class IngredientSynthesisResult:
    def __init__(
        self,
        *,
        updated_row: dict[str, str],
        claim_sources: list[dict[str, Any]],
        notes: dict[str, Any],
        rejected_items: list[dict[str, Any]],
    ) -> None:
        self.updated_row = updated_row
        self.claim_sources = claim_sources
        self.notes = notes
        self.rejected_items = rejected_items


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


def slug_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.casefold())
    return re.sub(r"_+", "_", slug).strip("_")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def claim_key(citation_id: str, proposed_claim_id: str) -> tuple[str, str]:
    return (citation_id, proposed_claim_id)


def ingredient_exposure_to_legacy(value: str) -> str:
    if value in {"whole_food", "juice_or_beverage"}:
        return "natural_food_level"
    if value in {"powder_or_concentrate", "extract_or_supplement"}:
        return "supplement_level"
    return "unclear"


def normalized_claim_exposure_category(claim: ProposedIngredientClaim) -> str:
    text = " ".join(
        [
            claim.ingredient_form or "",
            claim.oral_exposure or "",
            claim.dose_or_serving or "",
            claim.food_matrix or "",
            claim.concentration_notes or "",
        ]
    ).lower()
    if re.search(r"\b(juice|beverage|smoothie)\b", text):
        return "juice_or_beverage"
    if re.search(r"\b(pulp|puree|purée|whole fruit|fruit|berry|berries)\b", text) and not re.search(
        r"\b(powder|capsule|extract|supplement)\b",
        text,
    ):
        return "whole_food"
    return claim.exposure_category


def legacy_from_ingredient_claim(claim: ProposedIngredientClaim) -> ProposedClaim:
    exposure_category = normalized_claim_exposure_category(claim)
    return ProposedClaim(
        id=claim.id,
        nutrient_name=claim.ingredient_name,
        citation_id=claim.citation_id,
        statement=claim.statement,
        proposed_effect_slug=claim.proposed_effect_slug,
        proposed_effect_label=claim.proposed_effect_label,
        compound=claim.ingredient_name,
        effect=claim.effect,
        direction=claim.claim_direction,
        population=claim.population,
        dose_or_exposure=claim.dose_or_serving or claim.oral_exposure or None,
        outcome=claim.outcome,
        study_type=claim.study_type,
        limitations=claim.limitations,
        evidence_type=claim.evidence_type,
        intake_route="oral" if claim.oral_exposure else "unclear",
        exposure_category=ingredient_exposure_to_legacy(exposure_category),
        natural_concentration_relevance=(
            "Ingredient-level evidence."
            if exposure_category in {"whole_food", "juice_or_beverage"}
            else None
        ),
        supplement_level_relevance=(
            "Concentrated, powder, or extract-like ingredient evidence."
            if exposure_category in {"powder_or_concentrate", "extract_or_supplement"}
            else None
        ),
        pharmaceutical_centered=False,
        concentration_notes=claim.concentration_notes,
    )


def legacy_from_validated_ingredient_claim(claim: ValidatedIngredientClaim) -> ValidatedClaim:
    return ValidatedClaim(
        effect_row_id=claim.effect_row_id,
        proposed_claim_id=claim.proposed_ingredient_claim_id,
        citation_id=claim.citation_id,
        verdict=claim.verdict,
        support_level=claim.support_level,
        claim_scope=claim.claim_scope,
        validated_statement=claim.validated_statement,
        validator_reasoning=claim.validator_reasoning,
        supporting_quotes=claim.supporting_quotes,
        limitations=claim.limitations,
        accepted=claim.accepted,
    )


def packet_input_name(run_dir: Path) -> str:
    ingredient_packet_path = run_dir / "ingredient_packet.json"
    if ingredient_packet_path.exists():
        ingredient_packet = INGREDIENT_PACKET_ADAPTER.validate_python(
            orjson.loads(ingredient_packet_path.read_bytes())
        )
        return ingredient_packet.input.ingredient_name
    packet = Packet.model_validate_json((run_dir / "packet.json").read_text(encoding="utf-8"))
    return packet.input.nutrient_name


def packet_canonical_beverage_id(run_dir: Path) -> str | None:
    ingredient_packet_path = run_dir / "ingredient_packet.json"
    if not ingredient_packet_path.exists():
        return None
    ingredient_packet = INGREDIENT_PACKET_ADAPTER.validate_python(
        orjson.loads(ingredient_packet_path.read_bytes())
    )
    return ingredient_packet.input.canonical_beverage_id


def run_id_for_synthesis(run_dir: Path) -> str:
    ingredient_packet_path = run_dir / "ingredient_packet.json"
    if ingredient_packet_path.exists():
        ingredient_packet = INGREDIENT_PACKET_ADAPTER.validate_python(
            orjson.loads(ingredient_packet_path.read_bytes())
        )
        return ingredient_packet.run_id
    packet = Packet.model_validate_json((run_dir / "packet.json").read_text(encoding="utf-8"))
    return packet.run_id


def read_proposed_for_synthesis(run_dir: Path) -> list[ProposedClaim]:
    ingredient_path = run_dir / "proposed_ingredient_claims.json"
    if ingredient_path.exists():
        claims = PROPOSED_INGREDIENT_ADAPTER.validate_python(
            orjson.loads(ingredient_path.read_bytes())
        )
        return [legacy_from_ingredient_claim(claim) for claim in claims]
    return PROPOSED_ADAPTER.validate_python(
        orjson.loads((run_dir / "proposed_claims.json").read_bytes())
    )


def read_validated_for_synthesis(run_dir: Path) -> list[ValidatedClaim]:
    audited_ingredient_path = run_dir / "audited_ingredient_claims.json"
    if audited_ingredient_path.exists():
        claims = VALIDATED_INGREDIENT_ADAPTER.validate_python(
            orjson.loads(audited_ingredient_path.read_bytes())
        )
        return [legacy_from_validated_ingredient_claim(claim) for claim in claims]
    ingredient_path = run_dir / "validated_ingredient_claims.json"
    if ingredient_path.exists():
        claims = VALIDATED_INGREDIENT_ADAPTER.validate_python(
            orjson.loads(ingredient_path.read_bytes())
        )
        return [legacy_from_validated_ingredient_claim(claim) for claim in claims]
    return VALIDATED_ADAPTER.validate_python(
        orjson.loads((run_dir / "validated_claims.json").read_bytes())
    )


def close_lookup_matches(
    rows: list[dict[str, str]],
    target_name: str,
    *,
    limit: int = 8,
) -> list[dict[str, str]]:
    target = normalize_text(target_name)
    target_tokens = set(target.split())
    matches = []
    for row in rows:
        name = row.get("canonical_beverage_name", "")
        normalized = normalize_text(name)
        tokens = set(normalized.split())
        score = 0
        if target and target in normalized:
            score += 4
        if normalized and normalized in target:
            score += 3
        score += len(target_tokens & tokens)
        if score:
            matches.append((score, row))
    matches.sort(key=lambda item: (-item[0], item[1].get("canonical_beverage_name", "")))
    return [
        {
            "canonical_beverage_id": row.get("canonical_beverage_id", ""),
            "canonical_beverage_name": row.get("canonical_beverage_name", ""),
        }
        for _, row in matches[:limit]
    ]


def resolve_target_row(
    *,
    rows: list[dict[str, str]],
    packet_name: str,
    ingredient_name: str | None = None,
    canonical_beverage_id: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if canonical_beverage_id:
        matches = [
            row for row in rows if row.get("canonical_beverage_id") == canonical_beverage_id
        ]
        if len(matches) == 1:
            return matches[0], {
                "method": "canonical_beverage_id",
                "value": canonical_beverage_id,
            }
        raise ValueError(f"canonical_beverage_id_not_found:{canonical_beverage_id}")

    target_name = ingredient_name or packet_name
    normalized_target = normalize_text(target_name)
    exact = [
        row
        for row in rows
        if normalize_text(row.get("canonical_beverage_name", "")) == normalized_target
    ]
    if len(exact) == 1:
        return exact[0], {"method": "exact_name", "value": target_name}
    if len(exact) > 1:
        ids = ", ".join(row.get("canonical_beverage_id", "") for row in exact)
        raise ValueError(f"ambiguous_ingredient_name:{target_name}; ids={ids}")

    suggestions = close_lookup_matches(rows, target_name)
    raise ValueError(
        "ingredient_lookup_row_not_found:"
        + target_name
        + "; suggestions="
        + json.dumps(suggestions, ensure_ascii=False)
    )


def support_confidence(support_level: str) -> float:
    return {
        "human_systematic_review": 0.90,
        "human_rct": 0.86,
        "human_observational": 0.74,
        "human_mechanistic": 0.68,
        "review_author_interpretation": 0.60,
    }.get(support_level, 0.55)


def tag_from_claim(proposed: ProposedClaim, claim: ValidatedClaim) -> IngredientReportTag:
    raw_tag = proposed.proposed_effect_slug or proposed.effect or proposed.proposed_effect_label
    return IngredientReportTag(
        tag=raw_tag,
        score=round(min(0.95, max(0.35, support_confidence(claim.support_level))), 2),
        confidence=round(min(0.95, max(0.5, support_confidence(claim.support_level))), 2),
    )


def unique_tags(tags: list[IngredientReportTag]) -> list[IngredientReportTag]:
    by_tag: dict[str, IngredientReportTag] = {}
    for tag in tags:
        existing = by_tag.get(tag.tag)
        if existing is None or tag.confidence > existing.confidence:
            by_tag[tag.tag] = tag
    return list(by_tag.values())


def compact_sentences(sentences: list[str], *, max_chars: int = 1800) -> str:
    output: list[str] = []
    size = 0
    for sentence in sentences:
        clean = " ".join(sentence.strip().split())
        if not clean:
            continue
        if clean[-1] not in ".!?":
            clean += "."
        next_size = size + len(clean) + (1 if output else 0)
        if next_size > max_chars:
            break
        output.append(clean)
        size = next_size
    return " ".join(output)


def statement_for_claim(claim: ValidatedClaim, proposed: ProposedClaim) -> str:
    statement = claim.validated_statement or claim.claim_scope
    details = []
    if proposed.dose_or_exposure:
        details.append(f"dose/exposure: {proposed.dose_or_exposure}")
    if proposed.population:
        details.append(f"population: {proposed.population}")
    if proposed.exposure_category and proposed.exposure_category != "unclear":
        details.append(f"evidence form: {proposed.exposure_category.replace('_', ' ')}")
    if claim.limitations:
        details.append(f"caveat: {claim.limitations[0]}")
    if details:
        return f"{statement} ({'; '.join(details)})."
    return statement


def deterministic_report(
    *,
    row: dict[str, str],
    used: list[tuple[ValidatedClaim, ProposedClaim, CandidateCitation]],
) -> tuple[PolishedIngredientReport, list[dict[str, Any]]]:
    positive_sentences: list[str] = []
    negative_sentences: list[str] = []
    positive_tags: list[IngredientReportTag] = []
    negative_tags: list[IngredientReportTag] = []
    rejected: list[dict[str, Any]] = []

    for claim, proposed, _ in used:
        direction = proposed.direction
        sentence = statement_for_claim(claim, proposed)
        if direction == "beneficial":
            positive_sentences.append(sentence)
            positive_tags.append(tag_from_claim(proposed, claim))
        elif direction == "harmful":
            negative_sentences.append(sentence)
            negative_tags.append(tag_from_claim(proposed, claim))
        else:
            rejected.append(
                {
                    "type": "unsupported_direction_for_report_side",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                    "direction": direction,
                }
            )

    positive = compact_sentences(positive_sentences) or row.get("health_effect_positive", "")
    negative = compact_sentences(negative_sentences) or row.get("health_effect_negative", "")
    if not positive_sentences and row.get("health_effect_positive"):
        positive_tags.extend(parse_existing_tags(row.get("health_effect_positive_tags", "")))
    if not negative_sentences and row.get("health_effect_negative"):
        negative_tags.extend(parse_existing_tags(row.get("health_effect_negative_tags", "")))

    return (
        PolishedIngredientReport(
            health_effect_positive=positive,
            health_effect_negative=negative,
            health_effect_positive_tags=unique_tags(positive_tags),
            health_effect_negative_tags=unique_tags(negative_tags),
            synthesis_notes="Deterministic synthesis from accepted quote-grounded claims.",
        ),
        rejected,
    )


def parse_existing_tags(value: str) -> list[IngredientReportTag]:
    if not value.strip():
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    tags = []
    if not isinstance(raw, list):
        return []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            tags.append(IngredientReportTag.model_validate(item))
        except ValidationError:
            continue
    return tags


def polish_prompt(
    *,
    ingredient_name: str,
    draft: PolishedIngredientReport,
    used: list[tuple[ValidatedClaim, ProposedClaim, CandidateCitation]],
) -> str:
    claims_payload = []
    for claim, proposed, source in used:
        claims_payload.append(
            {
                "citation_id": claim.citation_id,
                "proposed_claim_id": claim.proposed_claim_id,
                "direction": proposed.direction,
                "validated_statement": claim.validated_statement,
                "claim_scope": claim.claim_scope,
                "support_level": claim.support_level,
                "limitations": claim.limitations,
                "dose_or_exposure": proposed.dose_or_exposure,
                "population": proposed.population,
                "exposure_category": proposed.exposure_category,
                "source_title": source.title,
            }
        )
    return f"""Polish an ingredient health report row for Sipz.

Rules:
- Use only the accepted validated claims supplied below.
- Do not add new benefits, harms, mechanisms, populations, or doses.
- Preserve form, dose, population, and evidence limitations.
- Keep each health_effect field under 1000 characters.
- Return JSON only in this shape:
{{
  "health_effect_positive": "...",
  "health_effect_negative": "...",
  "health_effect_positive_tags": [{{"tag":"snake_case","score":0.7,"confidence":0.8}}],
  "health_effect_negative_tags": [{{"tag":"snake_case","score":0.7,"confidence":0.8}}],
  "synthesis_notes": "..."
}}

Ingredient row name: {ingredient_name}

Deterministic draft:
{orjson.dumps(draft.model_dump(mode="json"), option=orjson.OPT_INDENT_2).decode("utf-8")}

Accepted validated claims:
{orjson.dumps(claims_payload, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


def polished_report_is_usable(
    *,
    report: PolishedIngredientReport,
    draft: PolishedIngredientReport,
) -> tuple[bool, str | None]:
    if draft.health_effect_positive and not report.health_effect_positive.strip():
        return False, "polish_removed_positive_text"
    if draft.health_effect_negative and not report.health_effect_negative.strip():
        return False, "polish_removed_negative_text"
    return True, None


def maybe_polish_report(
    *,
    ingredient_name: str,
    draft: PolishedIngredientReport,
    used: list[tuple[ValidatedClaim, ProposedClaim, CandidateCitation]],
    provider: LlmProvider,
) -> tuple[PolishedIngredientReport, dict[str, Any]]:
    if isinstance(provider, HeuristicProvider):
        return draft, {"used": False, "reason": "heuristic_provider"}
    try:
        report = provider.complete_json(
            polish_prompt(ingredient_name=ingredient_name, draft=draft, used=used),
            POLISHED_REPORT_ADAPTER,
        )
        usable, reason = polished_report_is_usable(report=report, draft=draft)
        if not usable:
            return draft, {"used": False, "reason": reason}
        return report, {"used": True, "reason": None}
    except Exception as exc:
        return draft, {
            "used": False,
            "reason": f"{type(exc).__name__}: {str(exc) or type(exc).__name__}",
        }


def build_claim_sources(
    used: list[tuple[ValidatedClaim, ProposedClaim, CandidateCitation]],
) -> list[dict[str, Any]]:
    rows = []
    for claim, proposed, source in used:
        rows.append(
            {
                "citation_id": claim.citation_id,
                "proposed_claim_id": claim.proposed_claim_id,
                "direction": proposed.direction,
                "validated_statement": claim.validated_statement,
                "claim_scope": claim.claim_scope,
                "support_level": claim.support_level,
                "limitations": claim.limitations,
                "source": {
                    "title": source.title,
                    "url": str(source.url) if source.url else None,
                    "doi": source.doi,
                    "pmid": source.pmid,
                    "year": source.year,
                },
                "supporting_quotes": [
                    quote.model_dump(mode="json") for quote in claim.supporting_quotes
                ],
            }
        )
    return rows


def collect_usable_claims(
    *,
    accepted_claims: list[ValidatedClaim],
    proposed_claims: list[ProposedClaim],
    sources: list[CandidateCitation],
) -> tuple[
    list[tuple[ValidatedClaim, ProposedClaim, CandidateCitation]],
    list[dict[str, Any]],
]:
    proposed_by_key = {claim_key(item.citation_id, item.id): item for item in proposed_claims}
    sources_by_id = {source.id: source for source in sources}
    used = []
    rejected = []
    for claim in accepted_claims:
        key = claim_key(claim.citation_id, claim.proposed_claim_id)
        proposed = proposed_by_key.get(key)
        source = sources_by_id.get(claim.citation_id)
        if proposed is None:
            rejected.append(
                {
                    "type": "missing_proposed_claim",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                }
            )
            continue
        if source is None:
            rejected.append(
                {
                    "type": "missing_source_metadata",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                }
            )
            continue
        used.append((claim, proposed, source))
    return used, rejected


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def updated_report_row(
    *,
    row: dict[str, str],
    report: PolishedIngredientReport,
    now: str,
    replace_existing: bool,
) -> dict[str, str]:
    updated = dict(row)
    if replace_existing or report.health_effect_positive:
        updated["health_effect_positive"] = report.health_effect_positive
    if replace_existing or report.health_effect_negative:
        updated["health_effect_negative"] = report.health_effect_negative
    updated["health_effect_positive_tags"] = json_cell(
        [tag.model_dump(mode="json") for tag in report.health_effect_positive_tags]
    )
    updated["health_effect_negative_tags"] = json_cell(
        [tag.model_dump(mode="json") for tag in report.health_effect_negative_tags]
    )
    updated["source"] = SOURCE_MARKER
    updated["updated_at"] = now
    return updated


def synthesize_ingredient_report(
    *,
    run_dir: Path,
    lookup_path: Path,
    out_dir: Path,
    provider: LlmProvider,
    ingredient_name: str | None = None,
    canonical_beverage_id: str | None = None,
    replace_existing: bool = True,
) -> IngredientSynthesisResult:
    input_name = packet_input_name(run_dir)
    run_id = run_id_for_synthesis(run_dir)
    claim_source_artifact = (
        "audited_ingredient_claims.json"
        if (run_dir / "audited_ingredient_claims.json").exists()
        else "validated_ingredient_claims.json"
        if (run_dir / "validated_ingredient_claims.json").exists()
        else "validated_claims.json"
    )
    accepted_claims = [claim for claim in read_validated_for_synthesis(run_dir) if claim.accepted]
    proposed_claims = read_proposed_for_synthesis(run_dir)
    sources = SOURCES_ADAPTER.validate_python(orjson.loads((run_dir / "sources.json").read_bytes()))
    columns, lookup_rows = read_csv_rows(lookup_path)
    if columns != INGREDIENT_HEALTH_REPORT_COLUMNS:
        raise ValueError("ingredient_lookup_columns_do_not_match_expected_contract")

    packet_row_id = None
    if canonical_beverage_id is None and ingredient_name is None:
        packet_row_id = packet_canonical_beverage_id(run_dir)

    target_row, resolution = resolve_target_row(
        rows=lookup_rows,
        packet_name=input_name,
        ingredient_name=ingredient_name,
        canonical_beverage_id=canonical_beverage_id or packet_row_id,
    )
    used, rejected = collect_usable_claims(
        accepted_claims=accepted_claims,
        proposed_claims=proposed_claims,
        sources=sources,
    )
    draft, deterministic_rejections = deterministic_report(row=target_row, used=used)
    rejected.extend(deterministic_rejections)
    ingredient_label = target_row.get("canonical_beverage_name") or ingredient_name or input_name
    report, polish = maybe_polish_report(
        ingredient_name=ingredient_label,
        draft=draft,
        used=used,
        provider=provider,
    )
    now = datetime.now(UTC).isoformat()
    updated = updated_report_row(
        row=target_row,
        report=report,
        now=now,
        replace_existing=replace_existing,
    )
    claim_sources = build_claim_sources(used)
    notes = {
        "run_id": run_id,
        "packet_input_name": input_name,
        "target_resolution": resolution,
        "target_row": {
            "canonical_beverage_id": target_row.get("canonical_beverage_id", ""),
            "canonical_beverage_name": target_row.get("canonical_beverage_name", ""),
        },
        "accepted_claims": len(accepted_claims),
        "claim_source_artifact": claim_source_artifact,
        "used_audited_claims": claim_source_artifact == "audited_ingredient_claims.json",
        "used_claims": len(used),
        "rejected_synthesis_items": len(rejected),
        "replace_existing": replace_existing,
        "deterministic_draft": draft.model_dump(mode="json"),
        "llm_polish": polish,
        "synthesis_notes": report.synthesis_notes,
        "created_at": now,
    }
    result = IngredientSynthesisResult(
        updated_row=updated,
        claim_sources=claim_sources,
        notes=notes,
        rejected_items=rejected,
    )
    write_ingredient_synthesis_artifacts(out_dir=out_dir, result=result)
    return result


def write_ingredient_synthesis_artifacts(
    *,
    out_dir: Path,
    result: IngredientSynthesisResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        out_dir / "ingredient_health_report_rows.updated.csv",
        INGREDIENT_HEALTH_REPORT_COLUMNS,
        [result.updated_row],
    )
    write_json(out_dir / "ingredient_claim_sources.json", result.claim_sources)
    write_json(out_dir / "ingredient_synthesis_notes.json", result.notes)
    write_json(out_dir / "ingredient_rejected_synthesis_items.json", result.rejected_items)
