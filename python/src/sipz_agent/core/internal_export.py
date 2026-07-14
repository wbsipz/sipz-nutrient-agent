from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal
from uuid import UUID, uuid5

import orjson
from pydantic import TypeAdapter, ValidationError

from sipz_agent.core.models import LlmProvider
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, ValidatedClaim
from sipz_agent.schemas.internal_export import (
    BioactiveHealthEvidenceExportRow,
    BioactiveType,
    ClaimFormattingResponse,
    ExposureContextRow,
    ExportClaimReference,
    ExportSource,
    InternalSynthesisResponse,
    NewBioactiveEntityRow,
    NewHealthEffectRow,
)

VALIDATED_ADAPTER = TypeAdapter(list[ValidatedClaim])
PROPOSED_ADAPTER = TypeAdapter(list[ProposedClaim])
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
CLAIM_FORMATTING_ADAPTER = TypeAdapter(ClaimFormattingResponse)

EXPORT_NAMESPACE = UUID("c6044f5f-770b-40c5-a87d-4c9fc18d922b")
EVIDENCE_COLUMNS = [
    "id",
    "bioactive_type",
    "bioactive_id",
    "bioactive_name",
    "effect_slug",
    "effect_label",
    "description",
    "score",
    "evidence_level",
    "tags",
    "sources",
    "review_status",
    "review_notes",
    "created_at",
    "updated_at",
]
NEW_EFFECT_COLUMNS = [
    "id",
    "effect_slug",
    "effect_label",
    "description",
    "tags",
    "created_at",
    "updated_at",
]
NEW_ENTITY_COLUMNS = [
    "bioactive_type",
    "bioactive_id",
    "bioactive_name",
    "created_at",
    "resolution_note",
]
EXPOSURE_COLUMNS = [
    "id",
    "evidence_row_id",
    "bioactive_type",
    "bioactive_id",
    "bioactive_name",
    "effect_slug",
    "exposure_category",
    "dose_or_exposure",
    "concentration_notes",
    "source_claims",
    "created_at",
]


class InternalExportResult:
    def __init__(
        self,
        *,
        evidence_rows: list[BioactiveHealthEvidenceExportRow],
        new_effects: list[NewHealthEffectRow],
        new_entities: list[NewBioactiveEntityRow],
        exposure_rows: list[ExposureContextRow],
        rejected: list[dict[str, Any]],
    ) -> None:
        self.evidence_rows = evidence_rows
        self.new_effects = new_effects
        self.new_entities = new_entities
        self.exposure_rows = exposure_rows
        self.rejected = rejected


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def stable_uuid(kind: str, *parts: str) -> str:
    key = "|".join([kind, *(normalize_name(part) for part in parts)])
    return str(uuid5(EXPORT_NAMESPACE, key))


def read_lookup_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def identity_index(rows: list[dict[str, str]]) -> dict[str, list[tuple[BioactiveType, str, str]]]:
    index: dict[str, set[tuple[BioactiveType, str, str]]] = {}
    for row in rows:
        raw_type = row.get("bioactive_type", "").strip()
        if raw_type not in {"nutrient", "polyphenol"}:
            continue
        name = row.get("bioactive_name", "").strip()
        bioactive_id = row.get("bioactive_id", "").strip()
        if not name or not bioactive_id:
            continue
        index.setdefault(normalize_name(name), set()).add(
            (raw_type, bioactive_id, name)  # type: ignore[arg-type]
        )
    return {key: sorted(value) for key, value in index.items()}


def effect_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    effects: dict[str, dict[str, str]] = {}
    for row in rows:
        slug = row.get("effect_slug", "").strip()
        if not slug:
            continue
        effects.setdefault(
            slug,
            {
                "effect_slug": slug,
                "effect_label": row.get("effect_label", "").strip(),
                "description": row.get("description", "").strip(),
                "tags": row.get("tags", "").strip(),
            },
        )
    return effects


def existing_evidence_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (
            row.get("bioactive_type", "").strip(),
            row.get("bioactive_id", "").strip(),
            row.get("effect_slug", "").strip(),
        ): row
        for row in rows
        if row.get("id") and row.get("bioactive_id") and row.get("effect_slug")
    }


def support_score(support_level: str) -> float:
    return {
        "human_systematic_review": 0.84,
        "human_rct": 0.75,
        "human_observational": 0.60,
        "human_mechanistic": 0.50,
        "review_author_interpretation": 0.45,
    }.get(support_level, 0.40)


def output_evidence_level(score: float) -> Literal["strong", "moderate", "limited"]:
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    return "limited"


def claim_key(citation_id: str, proposed_claim_id: str) -> tuple[str, str]:
    return (citation_id, proposed_claim_id)


def resolve_claim_reference(
    ref: ExportClaimReference,
    accepted_claims: list[ValidatedClaim],
) -> tuple[str, str] | None:
    if ref.citation_id:
        return claim_key(ref.citation_id, ref.proposed_claim_id)
    matches = [
        claim
        for claim in accepted_claims
        if claim.proposed_claim_id == ref.proposed_claim_id
    ]
    if len(matches) != 1:
        return None
    return claim_key(matches[0].citation_id, ref.proposed_claim_id)


def claim_formatting_prompt(
    *,
    entity_name: str,
    accepted_claim: ValidatedClaim,
    proposed_by_key: dict[tuple[str, str], ProposedClaim],
) -> str:
    proposed = proposed_by_key.get(
        claim_key(accepted_claim.citation_id, accepted_claim.proposed_claim_id)
    )
    claim_payload = {
        "citation_id": accepted_claim.citation_id,
        "proposed_claim_id": accepted_claim.proposed_claim_id,
        "validated_statement": accepted_claim.validated_statement,
        "claim_scope": accepted_claim.claim_scope,
        "support_level": accepted_claim.support_level,
        "limitations": accepted_claim.limitations,
        "effect": proposed.effect if proposed else None,
        "proposed_effect_slug": proposed.proposed_effect_slug if proposed else None,
        "proposed_effect_label": proposed.proposed_effect_label if proposed else None,
        "population": proposed.population if proposed else None,
        "dose_or_exposure": proposed.dose_or_exposure if proposed else None,
        "exposure_category": proposed.exposure_category if proposed else "unclear",
        "concentration_notes": proposed.concentration_notes if proposed else None,
    }
    return f"""You format one already accepted and validated health claim for Sipz database import.

Rules:
- This claim has already passed validation. Do not reject, exclude, merge, or reassess it.
- Return exactly one effect object for exactly this claim.
- Create a concise snake_case effect_slug from the claim.
- Semantic similarity to other possible effects is irrelevant; do not broaden the claim.
- Preserve population, route, dose, exposure category, concentration, and limitations.
- Use the exact citation_id and proposed_claim_id supplied below in source_claims.
- Suggested bioactive type is routing metadata. Use polyphenol only when the entity is clearly a
  polyphenol with confidence at least 0.9; otherwise use nutrient.
- Return only JSON in this exact shape:
{{
  "suggested_bioactive_type": "nutrient",
  "type_confidence": 0.0,
  "effect": {{
    "effect_slug": "snake_case_slug",
    "effect_label": "Short human-readable label",
    "effect_description": "Generic description of this health effect",
    "description": "Claim-specific database description with limitations",
    "tags": ["snake_case_tag"],
    "source_claims": [
      {{
        "citation_id": "{accepted_claim.citation_id}",
        "proposed_claim_id": "{accepted_claim.proposed_claim_id}"
      }}
    ],
    "exposure_category": "natural_food_level|supplement_level|mixed|unclear",
    "dose_or_exposure": ["relevant dose or exposure"],
    "concentration_notes": null,
    "review_notes": null
  }}
}}

Entity: {entity_name}

Accepted validated claim:
{orjson.dumps(claim_payload, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


def resolve_identity(
    *,
    entity_name: str,
    identities: dict[str, list[tuple[BioactiveType, str, str]]],
    response: InternalSynthesisResponse,
    type_override: Literal["auto", "nutrient", "polyphenol"],
    now: str,
) -> tuple[BioactiveType, str, str, list[NewBioactiveEntityRow]]:
    matches = identities.get(normalize_name(entity_name), [])
    if len(matches) > 1:
        raise ValueError("ambiguous_bioactive_identity")
    if matches:
        bioactive_type, bioactive_id, canonical_name = matches[0]
        return bioactive_type, bioactive_id, canonical_name, []

    if type_override == "auto":
        bioactive_type: BioactiveType = (
            "polyphenol"
            if response.suggested_bioactive_type == "polyphenol"
            and response.type_confidence >= 0.9
            else "nutrient"
        )
    else:
        bioactive_type = type_override
    bioactive_id = stable_uuid("bioactive", bioactive_type, entity_name)
    entity = NewBioactiveEntityRow(
        bioactive_type=bioactive_type,
        bioactive_id=bioactive_id,
        bioactive_name=entity_name,
        created_at=now,
        resolution_note="Generated because no normalized-name match existed in the lookup CSV.",
    )
    return bioactive_type, bioactive_id, entity_name, [entity]


def source_for_citation(citation: CandidateCitation) -> ExportSource:
    return ExportSource(
        title=citation.title,
        url=str(citation.url) if citation.url else None,
        doi=citation.doi,
        pmid=citation.pmid,
        year=citation.year,
    )


def evidence_description(synthesized_description: str, used_claims: list[ValidatedClaim]) -> str:
    if len(synthesized_description.split()) >= 8:
        return synthesized_description
    return " ".join(claim.validated_statement for claim in used_claims)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_internal_export(out_dir: Path, result: InternalExportResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dicts = []
    for row in result.evidence_rows:
        data = row.model_dump(mode="json")
        data["tags"] = _json_cell(data["tags"])
        data["sources"] = _json_cell(data["sources"])
        evidence_dicts.append(data)
    _write_csv(
        out_dir / "bioactive_health_evidence_rows.generated.csv",
        EVIDENCE_COLUMNS,
        evidence_dicts,
    )
    with (out_dir / "bioactive_health_evidence_rows.generated.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in result.evidence_rows:
            handle.write(orjson.dumps(row.model_dump(mode="json")).decode("utf-8") + "\n")

    new_effect_dicts = []
    for row in result.new_effects:
        data = row.model_dump(mode="json")
        data["tags"] = _json_cell(data["tags"])
        new_effect_dicts.append(data)
    _write_csv(out_dir / "health_effects_new.csv", NEW_EFFECT_COLUMNS, new_effect_dicts)
    _write_csv(
        out_dir / "new_bioactive_entities.csv",
        NEW_ENTITY_COLUMNS,
        [row.model_dump(mode="json") for row in result.new_entities],
    )
    exposure_dicts = []
    for row in result.exposure_rows:
        data = row.model_dump(mode="json")
        data["dose_or_exposure"] = _json_cell(data["dose_or_exposure"])
        data["source_claims"] = _json_cell(data["source_claims"])
        exposure_dicts.append(data)
    _write_csv(
        out_dir / "bioactive_health_evidence_exposure_context.csv",
        EXPOSURE_COLUMNS,
        exposure_dicts,
    )
    with (out_dir / "rejected_health_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in result.rejected:
            handle.write(orjson.dumps(row).decode("utf-8") + "\n")
    summary = {
        "evidence_rows": len(result.evidence_rows),
        "new_health_effects": len(result.new_effects),
        "new_entities": len(result.new_entities),
        "exposure_rows": len(result.exposure_rows),
        "rejected": len(result.rejected),
    }
    (out_dir / "internal_export_summary.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    (out_dir / "IMPORT_INSTRUCTIONS.md").write_text(
        """# Internal Export Import Order

Import these files into different tables, in this order:

1. `health_effects_new.csv` -> `beverage.health_effects`
2. `bioactive_health_evidence_rows.generated.csv` -> `beverage.bioactive_health_evidence`

Do not import `health_effects_new.csv` into `beverage.bioactive_health_evidence`. It contains only
health-effect taxonomy columns and therefore does not contain `bioactive_type`, `bioactive_id`, or
`bioactive_name`.

Review-only files:

- `new_bioactive_entities.csv`: entities missing from the supplied lookup.
- `bioactive_health_evidence_exposure_context.csv`: exposure context not yet represented by the
  production evidence table.
- `rejected_health_evidence.jsonl`: rows skipped because of exact duplicates or invalid formatting.
""",
        encoding="utf-8",
    )


def export_health_evidence(
    *,
    validated_claims_path: Path,
    proposed_claims_path: Path,
    sources_path: Path,
    lookup_path: Path,
    entity_name: str,
    lookup_entity_name: str | None = None,
    out_dir: Path,
    provider: LlmProvider,
    bioactive_type: Literal["auto", "nutrient", "polyphenol"] = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> InternalExportResult:
    accepted_claims = [
        claim
        for claim in VALIDATED_ADAPTER.validate_python(
            orjson.loads(validated_claims_path.read_bytes())
        )
        if claim.accepted
    ]
    proposed = PROPOSED_ADAPTER.validate_python(orjson.loads(proposed_claims_path.read_bytes()))
    citations = SOURCES_ADAPTER.validate_python(orjson.loads(sources_path.read_bytes()))
    proposed_by_key = {
        claim_key(claim.citation_id, claim.id): claim for claim in proposed
    }
    citations_by_id = {citation.id: citation for citation in citations}
    lookup_rows = read_lookup_rows(lookup_path)
    effects = effect_index(lookup_rows)
    identities = identity_index(lookup_rows)
    identity_name = lookup_entity_name or entity_name
    identity_matches = identities.get(normalize_name(identity_name), [])
    if len(identity_matches) > 1:
        raise ValueError("ambiguous_bioactive_identity")
    if lookup_entity_name and not identity_matches:
        raise ValueError(f"bioactive_lookup_name_not_found:{lookup_entity_name}")
    existing_rows = existing_evidence_index(lookup_rows)
    formatted_claims: list[tuple[ValidatedClaim, ClaimFormattingResponse]] = []
    formatting_rejections: list[dict[str, Any]] = []
    for claim_number, claim in enumerate(accepted_claims, start=1):
        if progress:
            progress(claim_number, len(accepted_claims))
        try:
            formatted = provider.complete_json(
                claim_formatting_prompt(
                    entity_name=entity_name,
                    accepted_claim=claim,
                    proposed_by_key=proposed_by_key,
                ),
                CLAIM_FORMATTING_ADAPTER,
            )
        except ValidationError as exc:
            formatting_rejections.append(
                {
                    "type": "invalid_csv_format",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                    "message": str(exc),
                }
            )
            continue
        except RuntimeError as exc:
            if str(exc) not in {
                "llm_provider_invalid_json",
                "llm_provider_invalid_response_json",
            }:
                raise
            formatting_rejections.append(
                {
                    "type": "invalid_csv_format",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                    "message": str(exc),
                }
            )
            continue
        formatted.effect.source_claims = [
            ExportClaimReference(
                citation_id=claim.citation_id,
                proposed_claim_id=claim.proposed_claim_id,
            )
        ]
        formatted_claims.append((claim, formatted))

    if formatted_claims:
        type_response = max(
            (formatted for _, formatted in formatted_claims),
            key=lambda item: item.type_confidence,
        )
    else:
        type_response = InternalSynthesisResponse()

    now = datetime.now(UTC).isoformat()
    resolved_type, bioactive_id, canonical_name, new_entities = resolve_identity(
        entity_name=identity_name,
        identities=identities,
        response=type_response,
        type_override=bioactive_type,
        now=now,
    )
    evidence_rows: list[BioactiveHealthEvidenceExportRow] = []
    new_effects: list[NewHealthEffectRow] = []
    exposure_rows: list[ExposureContextRow] = []
    rejected = formatting_rejections
    seen_slugs: set[str] = set()

    for claim, formatted in formatted_claims:
        synthesized = formatted.effect
        if (
            synthesized.effect_slug in seen_slugs
            or (resolved_type, bioactive_id, synthesized.effect_slug) in existing_rows
        ):
            rejected.append(
                {
                    "type": "existing_entity_effect_skipped",
                    "effect_slug": synthesized.effect_slug,
                    "bioactive_type": resolved_type,
                    "bioactive_id": bioactive_id,
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                }
            )
            continue
        seen_slugs.add(synthesized.effect_slug)
        score = support_score(claim.support_level)
        level = output_evidence_level(score)
        citation = citations_by_id.get(claim.citation_id)
        if citation is None:
            rejected.append(
                {
                    "type": "invalid_csv_format",
                    "effect_slug": synthesized.effect_slug,
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.proposed_claim_id,
                    "message": "Missing source metadata for the validated claim.",
                }
            )
            continue
        sources = [source_for_citation(citation)]
        existing_effect = effects.get(synthesized.effect_slug)
        effect_label = (
            existing_effect["effect_label"]
            if existing_effect and existing_effect["effect_label"]
            else synthesized.effect_label
        )
        evidence_id = stable_uuid(
            "evidence", resolved_type, bioactive_id, synthesized.effect_slug
        )
        evidence_rows.append(
            BioactiveHealthEvidenceExportRow(
                id=evidence_id,
                bioactive_type=resolved_type,
                bioactive_id=bioactive_id,
                bioactive_name=canonical_name,
                effect_slug=synthesized.effect_slug,
                effect_label=effect_label,
                description=evidence_description(synthesized.description, [claim]),
                score=score,
                evidence_level=level,
                tags=synthesized.tags,
                sources=sources,
                review_status="generated",
                review_notes=synthesized.review_notes,
                created_at=now,
                updated_at=now,
            )
        )
        if existing_effect is None:
            new_effects.append(
                NewHealthEffectRow(
                    id=stable_uuid("health-effect", synthesized.effect_slug),
                    effect_slug=synthesized.effect_slug,
                    effect_label=synthesized.effect_label,
                    description=synthesized.effect_description,
                    tags=synthesized.tags,
                    created_at=now,
                    updated_at=now,
                )
            )
        exposure_rows.append(
            ExposureContextRow(
                id=stable_uuid("exposure", evidence_id),
                evidence_row_id=evidence_id,
                bioactive_type=resolved_type,
                bioactive_id=bioactive_id,
                bioactive_name=canonical_name,
                effect_slug=synthesized.effect_slug,
                exposure_category=synthesized.exposure_category,
                dose_or_exposure=synthesized.dose_or_exposure,
                concentration_notes=synthesized.concentration_notes,
                source_claims=[
                    ExportClaimReference(
                        citation_id=claim.citation_id,
                        proposed_claim_id=claim.proposed_claim_id,
                    )
                ],
                created_at=now,
            )
        )

    result = InternalExportResult(
        evidence_rows=evidence_rows,
        new_effects=new_effects,
        new_entities=new_entities,
        exposure_rows=exposure_rows,
        rejected=rejected,
    )
    write_internal_export(out_dir, result)
    return result
