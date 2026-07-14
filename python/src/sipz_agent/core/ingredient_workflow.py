from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import unicodedata
from typing import Any, Callable, cast
from uuid import uuid4

import orjson
from pydantic import TypeAdapter

from sipz_agent.core.artifacts import (
    model_dump_jsonable,
    write_effects_csv,
    write_json,
    write_rejected_sources_markdown,
    write_sources_markdown,
)
from sipz_agent.core.claim_proposal import body_text_for_record, write_proposed_claims_markdown
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.full_text import FULL_TEXT_STATUS_FIELDS, retrieve_full_text_for_run
from sipz_agent.core.ingredient_preparation import canonical_search_name, relationship_for_name
from sipz_agent.core.ingredient_synthesis import (
    INGREDIENT_HEALTH_REPORT_COLUMNS,
    read_csv_rows,
    resolve_target_row,
)
from sipz_agent.core.models import LlmProvider, create_llm_provider
from sipz_agent.core.retrieval import (
    CandidatePageOutput,
    citation_key,
    deduplicate_citations,
    find_live_candidate_page,
)
from sipz_agent.core.screening import SourceScreeningUnavailable, screen_sources
from sipz_agent.core.validation import (
    PAPER_VALIDATION_ADAPTER,
    ValidatorDecision,
    acceptable_grounded_quotes,
    grounded_supporting_quotes,
    repair_supporting_quotes_once,
    sanitize_paper_body,
    substantive_grounded_quote_count,
    validation_failure_code,
    validation_prompt,
    write_sanitized_body_preview,
)
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel, StudyDepth
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, ValidatedClaim
from sipz_agent.schemas.ingredients import (
    IngredientClaimProposalResponse,
    IngredientEntityPlan,
    IngredientEntityPlanEnrichmentResponse,
    IngredientExposureCategory,
    IngredientPacket,
    IngredientPacketInput,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)
from sipz_agent.schemas.raw_texts import RawTextRecord


INGREDIENT_PACKET_ADAPTER = TypeAdapter(IngredientPacket)
INGREDIENT_ENTITY_PLAN_ADAPTER = TypeAdapter(IngredientEntityPlan)
INGREDIENT_ENTITY_PLAN_ENRICHMENT_ADAPTER = TypeAdapter(
    IngredientEntityPlanEnrichmentResponse
)
PROPOSED_INGREDIENT_CLAIMS_ADAPTER = TypeAdapter(list[ProposedIngredientClaim])
VALIDATED_INGREDIENT_CLAIMS_ADAPTER = TypeAdapter(list[ValidatedIngredientClaim])
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])
INGREDIENT_CLAIM_PROPOSAL_ADAPTER = TypeAdapter(IngredientClaimProposalResponse)

LIVE_RETRIEVAL_PAGE_SIZE = 10
LIVE_RETRIEVAL_MAX_PAGES: dict[StudyDepth, int] = {
    "light": 5,
    "standard": 5,
    "deep": 10,
}
RETAINED_TARGETS: dict[StudyDepth, int] = {
    "light": 10,
    "standard": 25,
    "deep": 40,
}
MAX_ENTITY_ALIASES = 12
MAX_ENTITY_FORMS = 8
MAX_ENTITY_QUERY_TERMS = 16
MAX_ENTITY_WARNINGS = 8
MAX_INGREDIENT_RETRIEVAL_QUERIES = 6
HUMAN_ORAL_QUERY_TERMS = [
    "human",
    "clinical",
    "trial",
    "randomized",
    "consumption",
    "oral",
    "dietary",
]
PREFERRED_SOURCE_TYPE_TERMS = [
    "randomized",
    "trial",
    "systematic review",
    "meta-analysis",
]
REVIEW_SOURCE_TYPE_TERMS = [
    "review",
    "critical review",
    "systematic review",
    "meta-analysis",
    "clinical trials",
]
DEFAULT_EXCLUDED_QUERY_TERMS = [
    "in vitro",
    "mouse",
    "mice",
    "rat",
    "cell line",
    "animal model",
    "extraction",
    "processing",
    "agriculture",
    "pesticide",
    "phytochemical",
    "chemical composition",
]
GENERIC_ENTITY_TERMS = {
    "beverage",
    "drink",
    "fresh",
    "fruit",
    "health",
    "human",
    "juice",
    "organic",
    "powder",
    "raw",
    "supplement",
    "whole",
}


@dataclass(frozen=True)
class IngredientStudyResult:
    run_dir: Path
    run_id: str


IngredientClaimProgress = Callable[[int, int, CandidateCitation], None]
IngredientValidationProgress = Callable[[int, int, CandidateCitation, int], None]


def slugify(value: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", value.strip().lower()))


def normalize_entity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).split())


def timestamp_for_run_id() -> str:
    return datetime.now(UTC).isoformat().replace(":", "-").replace(".", "-")


def ingredient_form_from_relationship(relationship: str) -> str:
    if relationship == "juice_or_beverage_form":
        return "juice_or_beverage"
    if relationship == "powder_or_concentrate_form":
        return "powder_or_concentrate"
    if relationship == "extract_or_supplement_form":
        return "extract_or_supplement"
    if relationship in {"same_ingredient", "minimally_processed_form"}:
        return "whole_food"
    return "mixed_or_unclear"


def ingredient_exposure_to_legacy(value: IngredientExposureCategory) -> str:
    if value in {"whole_food", "juice_or_beverage"}:
        return "natural_food_level"
    if value in {"powder_or_concentrate", "extract_or_supplement"}:
        return "supplement_level"
    return "unclear"


def normalized_claim_exposure_category(claim: ProposedIngredientClaim) -> IngredientExposureCategory:
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


def infer_group_applicability(value: IngredientExposureCategory) -> str:
    if value == "whole_food":
        return "same_ingredient"
    if value == "juice_or_beverage":
        return "similar_forms"
    if value in {"powder_or_concentrate", "extract_or_supplement"}:
        return "form_specific"
    return "direct_only"


def clean_entity_term(value: str) -> str:
    return " ".join(str(value).strip().split())


def dedupe_entity_terms(values: list[str], *, limit: int, drop_generic: bool = False) -> list[str]:
    terms = []
    seen: set[str] = set()
    for value in values:
        term = clean_entity_term(value)
        key = normalize_entity_key(term)
        if not term or not key:
            continue
        if drop_generic and key in GENERIC_ENTITY_TERMS:
            continue
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def curated_alias_profile(canonical_search_name_value: str) -> dict[str, list[str]]:
    key = normalize_entity_key(canonical_search_name_value)
    if key == "tart cherry":
        return {
            "aliases": ["sour cherry", "Montmorency cherry", "Prunus cerasus"],
            "included_forms": ["whole fruit", "juice", "concentrate"],
            "excluded_forms": ["capsule extract unless explicitly ingredient-relevant"],
            "excluded_query_terms": [],
            "query_terms": [
                "tart cherry",
                "sour cherry",
                "Montmorency cherry",
                "Prunus cerasus",
                "tart cherry juice",
                "tart cherry concentrate",
            ],
            "search_warnings": [
                (
                    "Do not generalize capsule or extract evidence to juice or whole fruit "
                    "unless the paper is explicitly ingredient-relevant."
                )
            ],
        }
    if key == "acai":
        return {
            "aliases": ["açaí", "Euterpe oleracea"],
            "included_forms": ["pulp", "juice", "whole fruit"],
            "excluded_forms": ["capsule extract unless explicitly ingredient-relevant"],
            "excluded_query_terms": [
                "accelerated cost adjustment",
                "automated content access interface",
            ],
            "query_terms": [
                "acai",
                "açaí",
                "Euterpe oleracea",
                "acai pulp",
                "acai juice",
            ],
            "search_warnings": [
                (
                    "Do not generalize capsule or extract evidence to pulp, juice, or whole "
                    "fruit unless the paper is explicitly ingredient-relevant."
                )
            ],
        }
    return {
        "aliases": [],
        "included_forms": [],
        "excluded_forms": [],
        "excluded_query_terms": [],
        "query_terms": [],
        "search_warnings": [],
    }


def included_forms_from_relationship(
    *,
    canonical_search_name_value: str,
    row_name: str | None,
    ingredient_form: str,
) -> list[str]:
    forms = []
    if ingredient_form == "whole_food":
        forms.append("whole food")
    elif ingredient_form == "juice_or_beverage":
        forms.append("juice")
    elif ingredient_form == "powder_or_concentrate":
        forms.append("powder or concentrate")
    elif ingredient_form == "extract_or_supplement":
        forms.append("extract or supplement")

    lowered_row = (row_name or "").casefold()
    if "concentrate" in lowered_row:
        forms.append("concentrate")
    if "pulp" in lowered_row or "puree" in lowered_row:
        forms.append("pulp")
    if not forms and canonical_search_name_value:
        forms.append("mixed or unclear")
    return forms


def default_excluded_forms(ingredient_form: str) -> list[str]:
    if ingredient_form == "extract_or_supplement":
        return ["topical or non-oral uses", "isolated pharmaceutical preparations"]
    return ["capsule extract unless explicitly ingredient-relevant"]


def quoted_or_grouped_terms(terms: list[str]) -> str:
    return " OR ".join(f'"{term}"' for term in terms)


def boolean_group(terms: list[str]) -> str:
    return f"({quoted_or_grouped_terms(terms)})"


def filtered_ingredient_query(
    *,
    positive_terms: list[str],
    excluded_terms: list[str],
    include_source_type_boost: bool = False,
) -> str:
    groups = [
        boolean_group(positive_terms),
        boolean_group(HUMAN_ORAL_QUERY_TERMS),
    ]
    if include_source_type_boost:
        groups.append(boolean_group(PREFERRED_SOURCE_TYPE_TERMS))
    query = " AND ".join(groups)
    if excluded_terms:
        query += f" NOT {boolean_group(excluded_terms)}"
    return query


def review_focused_ingredient_query(*, positive_terms: list[str]) -> str:
    return " AND ".join(
        [
            boolean_group(positive_terms),
            boolean_group(["human", "health", "consumption", "dietary", "clinical"]),
            boolean_group(REVIEW_SOURCE_TYPE_TERMS),
        ]
    )


def build_ingredient_retrieval_queries(plan: IngredientEntityPlan) -> list[str]:
    canonical = clean_entity_term(plan.canonical_search_name)
    excluded_terms = dedupe_entity_terms(
        [*DEFAULT_EXCLUDED_QUERY_TERMS, *plan.excluded_query_terms],
        limit=MAX_ENTITY_QUERY_TERMS,
        drop_generic=False,
    )
    primary_terms = dedupe_entity_terms(
        [canonical, *plan.aliases],
        limit=MAX_ENTITY_ALIASES,
        drop_generic=False,
    )
    queries = [
        filtered_ingredient_query(
            positive_terms=primary_terms or [canonical],
            excluded_terms=excluded_terms,
        )
    ]
    queries.append(
        filtered_ingredient_query(
            positive_terms=primary_terms or [canonical],
            excluded_terms=excluded_terms,
            include_source_type_boost=True,
        )
    )
    queries.append(
        review_focused_ingredient_query(
            positive_terms=primary_terms or [canonical],
        )
    )

    alias_terms = [
        term
        for term in plan.aliases
        if normalize_entity_key(term) != normalize_entity_key(canonical)
    ]
    for index in range(0, len(alias_terms), 4):
        chunk = alias_terms[index : index + 4]
        if chunk:
            queries.append(
                filtered_ingredient_query(
                    positive_terms=chunk,
                    excluded_terms=excluded_terms,
                )
            )

    form_terms = []
    for term in plan.query_terms:
        key = normalize_entity_key(term)
        if key == normalize_entity_key(canonical):
            continue
        if any(key == normalize_entity_key(alias) for alias in plan.aliases):
            continue
        form_terms.append(term)
    for form in plan.included_forms:
        form_key = normalize_entity_key(form)
        if form_key in {"whole food", "mixed or unclear"}:
            continue
        form_terms.append(f"{canonical} {form}")

    form_terms = dedupe_entity_terms(
        form_terms,
        limit=MAX_ENTITY_QUERY_TERMS,
        drop_generic=True,
    )
    for index in range(0, len(form_terms), 4):
        chunk = form_terms[index : index + 4]
        if chunk:
            queries.append(
                filtered_ingredient_query(
                    positive_terms=chunk,
                    excluded_terms=excluded_terms,
                )
            )

    return dedupe_entity_terms(
        queries,
        limit=MAX_INGREDIENT_RETRIEVAL_QUERIES,
        drop_generic=False,
    )


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


def ingredient_from_legacy_validated_claim(claim: ValidatedClaim) -> ValidatedIngredientClaim:
    return ValidatedIngredientClaim(
        effect_row_id=claim.effect_row_id,
        proposed_ingredient_claim_id=claim.proposed_claim_id,
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


def compatibility_packet(packet: IngredientPacket) -> Packet:
    return Packet(
        run_id=packet.run_id,
        input=PacketInput(
            nutrient_name=packet.input.ingredient_name,
            depth=packet.input.depth,
            demo=packet.input.demo,
            retrieval_queries=packet.input.retrieval_queries,
        ),
        model=packet.model,
        status=packet.status,
        created_at=packet.created_at,
        completed_at=packet.completed_at,
        counts=packet.counts,
    )


def write_ingredient_packet(run_dir: Path, packet: IngredientPacket) -> None:
    write_json(run_dir / "ingredient_packet.json", packet.model_dump(mode="json"))
    write_json(run_dir / "packet.json", compatibility_packet(packet).model_dump(mode="json"))


def read_ingredient_packet(run_dir: Path) -> IngredientPacket:
    return INGREDIENT_PACKET_ADAPTER.validate_python(
        orjson.loads((run_dir / "ingredient_packet.json").read_bytes())
    )


def update_ingredient_packet_counts(
    run_dir: Path,
    *,
    proposed_claims: int | None = None,
    validated_claims: int | None = None,
    rejected_claims: int | None = None,
) -> None:
    packet = read_ingredient_packet(run_dir)
    counts = packet.counts.model_dump(mode="json")
    if proposed_claims is not None:
        counts["proposed_claims"] = proposed_claims
    if validated_claims is not None:
        counts["validated_claims"] = validated_claims
    if rejected_claims is not None:
        counts["rejected_claims"] = rejected_claims
    updated = packet.model_copy(
        update={
            "counts": PacketCounts(**counts),
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    write_ingredient_packet(run_dir, updated)


def append_audit_event(run_dir: Path, event: dict[str, Any]) -> None:
    with (run_dir / "audit_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(orjson.dumps(event).decode("utf-8") + "\n")


def resolve_ingredient_target(
    *,
    lookup_path: Path,
    ingredient_name: str,
    canonical_beverage_id: str | None = None,
    search_as_ingredient_name: bool = False,
) -> tuple[dict[str, str] | None, str, str, str]:
    columns, rows = read_csv_rows(lookup_path)
    if columns != INGREDIENT_HEALTH_REPORT_COLUMNS:
        raise ValueError("ingredient_lookup_columns_do_not_match_expected_contract")
    if canonical_beverage_id:
        row, _ = resolve_target_row(
            rows=rows,
            packet_name=ingredient_name,
            canonical_beverage_id=canonical_beverage_id,
        )
        search_name = (
            literal_search_name(ingredient_name)
            if search_as_ingredient_name
            else canonical_search_name(row["canonical_beverage_name"])
        )
        relationship = relationship_for_name(row["canonical_beverage_name"])
        return row, search_name, relationship, ingredient_form_from_relationship(relationship)
    row, _ = resolve_target_row(rows=rows, packet_name=ingredient_name)
    search_name = (
        literal_search_name(ingredient_name)
        if search_as_ingredient_name
        else canonical_search_name(row["canonical_beverage_name"])
    )
    relationship = relationship_for_name(row["canonical_beverage_name"])
    return row, search_name, relationship, ingredient_form_from_relationship(relationship)


def literal_search_name(name: str) -> str:
    return normalize_entity_key(name) or " ".join(name.strip().split())


def build_entity_plan(
    *,
    ingredient_name: str,
    canonical_search_name_value: str,
    row: dict[str, str] | None,
    relationship: str,
    ingredient_form: str,
    provider: LlmProvider,
    use_llm_enrichment: bool,
) -> IngredientEntityPlan:
    row_name = row.get("canonical_beverage_name") if row else None
    curated = curated_alias_profile(canonical_search_name_value)
    aliases = list(curated["aliases"])
    included_forms = [
        *included_forms_from_relationship(
            canonical_search_name_value=canonical_search_name_value,
            row_name=row_name,
            ingredient_form=ingredient_form,
        ),
        *curated["included_forms"],
    ]
    excluded_forms = [*default_excluded_forms(ingredient_form), *curated["excluded_forms"]]
    excluded_query_terms = [*DEFAULT_EXCLUDED_QUERY_TERMS, *curated["excluded_query_terms"]]
    query_terms = [canonical_search_name_value]
    if row_name:
        query_terms.append(row_name)
    query_terms.extend(curated["query_terms"])
    search_warnings = list(curated["search_warnings"])
    rationale_parts = [
        "Deterministic ingredient entity plan from lookup row and relationship.",
    ]

    if use_llm_enrichment:
        try:
            enrichment = provider.complete_json(
                ingredient_entity_plan_prompt(
                    ingredient_name=ingredient_name,
                    canonical_search_name_value=canonical_search_name_value,
                    canonical_beverage_name=row_name,
                    ingredient_form=ingredient_form,
                    relationship=relationship,
                    seeded_aliases=aliases,
                    seeded_included_forms=included_forms,
                    seeded_excluded_forms=excluded_forms,
                    seeded_excluded_query_terms=excluded_query_terms,
                    seeded_query_terms=query_terms,
                ),
                INGREDIENT_ENTITY_PLAN_ENRICHMENT_ADAPTER,
            )
            aliases.extend(enrichment.aliases)
            included_forms.extend(enrichment.included_forms)
            excluded_forms.extend(enrichment.excluded_forms)
            excluded_query_terms.extend(enrichment.excluded_query_terms)
            query_terms.extend(enrichment.query_terms)
            search_warnings.extend(enrichment.search_warnings)
            if enrichment.rationale.strip():
                rationale_parts.append(enrichment.rationale.strip())
        except Exception as exc:
            search_warnings.append(
                f"LLM entity enrichment failed; used deterministic plan. Reason: {type(exc).__name__}."
            )

    normalized_aliases = dedupe_entity_terms(
        aliases,
        limit=MAX_ENTITY_ALIASES,
        drop_generic=True,
    )
    normalized_included_forms = dedupe_entity_terms(
        included_forms,
        limit=MAX_ENTITY_FORMS,
        drop_generic=False,
    )
    normalized_excluded_forms = dedupe_entity_terms(
        excluded_forms,
        limit=MAX_ENTITY_FORMS,
        drop_generic=False,
    )
    normalized_query_terms = dedupe_entity_terms(
        query_terms,
        limit=MAX_ENTITY_QUERY_TERMS,
        drop_generic=True,
    )
    normalized_excluded_query_terms = dedupe_entity_terms(
        excluded_query_terms,
        limit=MAX_ENTITY_QUERY_TERMS,
        drop_generic=False,
    )
    normalized_warnings = dedupe_entity_terms(
        search_warnings,
        limit=MAX_ENTITY_WARNINGS,
        drop_generic=False,
    )
    plan = IngredientEntityPlan(
        ingredient_name=ingredient_name,
        canonical_search_name=canonical_search_name_value,
        canonical_beverage_id=row.get("canonical_beverage_id") if row else None,
        canonical_beverage_name=row.get("canonical_beverage_name") if row else None,
        ingredient_form=ingredient_form,
        relationship=relationship,  # type: ignore[arg-type]
        aliases=normalized_aliases,
        included_forms=normalized_included_forms,
        excluded_forms=normalized_excluded_forms,
        query_terms=normalized_query_terms,
        excluded_query_terms=normalized_excluded_query_terms,
        search_warnings=normalized_warnings,
        retrieval_queries=[],
        rationale=" ".join(rationale_parts),
    )
    return plan.model_copy(
        update={"retrieval_queries": build_ingredient_retrieval_queries(plan)}
    )


def ingredient_entity_plan_prompt(
    *,
    ingredient_name: str,
    canonical_search_name_value: str,
    canonical_beverage_name: str | None,
    ingredient_form: str,
    relationship: str,
    seeded_aliases: list[str],
    seeded_included_forms: list[str],
    seeded_excluded_forms: list[str],
    seeded_excluded_query_terms: list[str],
    seeded_query_terms: list[str],
) -> str:
    return f"""You are the entity-resolution planner for a food ingredient literature-search agent.

Task:
- Improve search coverage for human health papers about oral consumption of the ingredient.
- Return only precise aliases, botanical/scientific names, spelling variants, food forms, and query terms.
- Do not invent claims, doses, identifiers, DOI fragments, or PubMed IDs.
- Do not broaden to unrelated parent classes unless papers commonly index the ingredient that way.
- Exclude pharmaceutical, topical, animal-feed, and extract-only terms unless they are
  explicitly relevant to oral ingredient consumption.
- Keep lists short and high precision.
- Return JSON only matching this shape:
{{
  "aliases": ["exact synonym or scientific name"],
  "included_forms": ["juice"],
  "excluded_forms": ["capsule extract unless explicitly ingredient-relevant"],
  "query_terms": ["specific search phrase"],
  "excluded_query_terms": ["unrelated acronym or irrelevant domain"],
  "search_warnings": ["scope warning"],
  "rationale": "short reason"
}}

Requested ingredient: {ingredient_name}
Canonical search name: {canonical_search_name_value}
Canonical beverage row: {canonical_beverage_name or "unknown"}
Ingredient form: {ingredient_form}
Relationship: {relationship}

Seeded aliases:
{orjson.dumps(seeded_aliases, option=orjson.OPT_INDENT_2).decode("utf-8")}

Seeded included forms:
{orjson.dumps(seeded_included_forms, option=orjson.OPT_INDENT_2).decode("utf-8")}

Seeded excluded forms:
{orjson.dumps(seeded_excluded_forms, option=orjson.OPT_INDENT_2).decode("utf-8")}

Seeded excluded query terms:
{orjson.dumps(seeded_excluded_query_terms, option=orjson.OPT_INDENT_2).decode("utf-8")}

Seeded query terms:
{orjson.dumps(seeded_query_terms, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


SOURCE_TYPE_BOOST_PATTERNS: list[tuple[str, str, int]] = [
    ("systematic_review", r"\bsystematic review\b", 7),
    ("meta_analysis", r"\bmeta[- ]analysis\b", 7),
    ("randomized_trial", r"\brandomi[sz]ed\b|\brct\b", 6),
    ("clinical_trial", r"\bclinical trial\b|\btrial\b", 5),
    ("human_observational", r"\bcohort\b|\bcross[- ]sectional\b|\bobservational\b", 3),
    ("human_evidence", r"\bhuman\b|\badult[s]?\b|\bparticipant[s]?\b|\bvolunteer[s]?\b", 2),
    ("oral_dietary", r"\boral\b|\bdietary\b|\bconsum(?:e|ed|ption)\b|\bintake\b", 2),
]
SOURCE_TYPE_PENALTY_PATTERNS: list[tuple[str, str, int]] = [
    ("in_vitro", r"\bin vitro\b|\bcell line\b|\bcells\b", -8),
    ("animal", r"\bmouse\b|\bmice\b|\brat[s]?\b|\banimal model\b|\bmurine\b", -7),
    ("agriculture", r"\bagricultur(?:e|al)\b|\bcultivar\b|\bharvest\b|\bsoil\b", -5),
    ("processing", r"\bextraction\b|\bprocessing\b|\bdrying\b|\bencapsulation\b", -5),
    ("chemistry_only", r"\bphytochemical\b|\bchemical composition\b|\banthocyanin profile\b", -4),
    ("residue", r"\bpesticide\b|\bresidue\b|\bheavy metal\b", -5),
]


def citation_rank_text(citation: CandidateCitation) -> str:
    return " ".join(
        part
        for part in [
            citation.title,
            citation.abstract,
            citation.page_summary,
            citation.selection_reason,
        ]
        if part
    ).casefold()


def rank_ingredient_candidate(citation: CandidateCitation) -> tuple[int, list[str], list[str]]:
    text = citation_rank_text(citation)
    score = 0
    boosts = []
    penalties = []
    for label, pattern, value in SOURCE_TYPE_BOOST_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += value
            boosts.append(label)
    for label, pattern, value in SOURCE_TYPE_PENALTY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += value
            penalties.append(label)
    if citation.source in {"pubmed", "europe_pmc"}:
        score += 1
        boosts.append("literature_database")
    if citation.pmid:
        score += 1
        boosts.append("pmid")
    return score, boosts, penalties


def ranking_note(score: int, boosts: list[str], penalties: list[str]) -> str:
    parts = [f"score={score}"]
    if boosts:
        parts.append("boosts=" + ",".join(boosts[:4]))
    if penalties:
        parts.append("penalties=" + ",".join(penalties[:4]))
    return "Ingredient retrieval ranking: " + "; ".join(parts) + "."


def rank_ingredient_candidates(citations: list[CandidateCitation]) -> list[CandidateCitation]:
    ranked_items = []
    for index, citation in enumerate(citations):
        score, boosts, penalties = rank_ingredient_candidate(citation)
        reason = citation.selection_reason or "Selected by retrieval."
        if "Ingredient retrieval ranking:" not in reason:
            reason = f"{reason} {ranking_note(score, boosts, penalties)}"
        ranked_items.append(
            (
                -score,
                index,
                citation.model_copy(update={"selection_reason": reason}),
            )
        )
    ranked_items.sort(key=lambda item: (item[0], item[1]))
    return [citation for _, _, citation in ranked_items]


def paginated_retrieve_sources(
    *,
    ingredient_name: str,
    depth: StudyDepth,
    queries: list[str],
    aliases: list[str],
    provider: LlmProvider,
    screening_available: bool,
    audit_events: list[dict[str, Any]],
) -> tuple[list[CandidateCitation], list[Any], dict[str, Any]]:
    target = RETAINED_TARGETS[depth]
    max_pages = LIVE_RETRIEVAL_MAX_PAGES[depth]
    unique_candidates: list[CandidateCitation] = []
    accepted_keys: set[str] = set()
    rejected_sources = []
    disabled_collectors: set[str] = set()
    disabled_metadata_sources: set[str] = set()
    raw_candidate_count = 0
    retrieval_pages_attempted = 0
    retrieval_stop_reason = "max_pages"

    retrieval_schedule = [
        *[("primary", page_index) for page_index in range(max_pages)],
        *[("fallback", page_index) for page_index in range(2)],
    ]
    for collector_mode, page_index in retrieval_schedule:
        if collector_mode == "fallback" and page_index == 0:
            audit_events.append(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "event": "ingredient_fallback_source_retrieval_started",
                    "sources": ["semantic_scholar", "firecrawl"],
                    "accepted_total": len(accepted_keys),
                    "retained_target": target,
                }
            )
        page: CandidatePageOutput = find_live_candidate_page(
            ingredient_name,
            depth,
            page_index=page_index,
            page_size=LIVE_RETRIEVAL_PAGE_SIZE,
            queries=queries,
            disabled_collectors=disabled_collectors,
            disabled_metadata_sources=disabled_metadata_sources,
            collector_mode=collector_mode,
        )
        retrieval_pages_attempted += 1
        raw_candidate_count += page.raw_candidate_count
        previous_keys = {citation_key(item) for item in unique_candidates}
        unique_candidates = deduplicate_citations(unique_candidates + page.citations)
        new_candidates = [
            item for item in unique_candidates if citation_key(item) not in previous_keys
        ]
        if new_candidates:
            new_candidates = rank_ingredient_candidates(new_candidates)

        for failure in page.source_failures:
            audit_events.append(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "event": "ingredient_candidate_source_page_failed",
                    "retrieval_tier": collector_mode,
                    "page": page_index + 1,
                    "failure": failure,
                }
            )

        accepted_this_page = 0
        rejected_this_page = 0
        if new_candidates and screening_available:
            try:
                screening = screen_sources(
                    nutrient_name=ingredient_name,
                    citations=new_candidates,
                    provider=provider,
                    aliases=aliases,
                )
                accepted_keys.update(citation_key(item) for item in screening.accepted)
                rejected_sources.extend(screening.rejected)
                accepted_this_page = len(screening.accepted)
                rejected_this_page = len(screening.rejected)
            except SourceScreeningUnavailable as error:
                screening_available = False
                accepted_keys.update(citation_key(item) for item in new_candidates)
                accepted_this_page = len(new_candidates)
                audit_events.append(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "ingredient_source_screening_skipped",
                        "reason": str(error),
                    }
                )
        elif new_candidates:
            accepted_keys.update(citation_key(item) for item in new_candidates)
            accepted_this_page = len(new_candidates)

        audit_events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "ingredient_candidate_page_processed",
                "retrieval_tier": collector_mode,
                "page": page_index + 1,
                "raw_candidates": page.raw_candidate_count,
                "new_unique_candidates": len(new_candidates),
                "accepted_this_page": accepted_this_page,
                "rejected_this_page": rejected_this_page,
                "accepted_total": len(accepted_keys),
                "unique_total": len(unique_candidates),
            }
        )

        if len(accepted_keys) >= target:
            retrieval_stop_reason = "retained_target_reached"
            break
        if not new_candidates and collector_mode == "fallback":
            retrieval_stop_reason = "no_new_unique_candidates"
            break

    accepted = rank_ingredient_candidates(
        [item for item in unique_candidates if citation_key(item) in accepted_keys]
    )
    counts = {
        "raw_candidates_retrieved": raw_candidate_count,
        "unique_candidates": len(unique_candidates),
        "retrieval_pages_attempted": retrieval_pages_attempted,
        "retrieval_stop_reason": retrieval_stop_reason,
        "screened_sources": len(accepted),
        "rejected_sources": len(rejected_sources),
    }
    return accepted, rejected_sources, counts


def write_ingredient_study_artifacts(
    *,
    run_dir: Path,
    packet: IngredientPacket,
    entity_plan: IngredientEntityPlan,
    sources: list[CandidateCitation],
    rejected_sources: list[Any],
    audit_events: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_ingredient_packet(run_dir, packet)
    write_json(run_dir / "ingredient_entity_plan.json", entity_plan.model_dump(mode="json"))
    write_json(run_dir / "sources.json", model_dump_jsonable(sources))
    write_sources_markdown(run_dir / "sources.md", sources)
    write_json(run_dir / "rejected_sources.json", model_dump_jsonable(rejected_sources))
    write_rejected_sources_markdown(run_dir / "rejected_sources.md", rejected_sources)
    write_effects_csv(run_dir / "effects.csv", [])
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {packet.input.ingredient_name} Ingredient Research Summary",
                "",
                f"Screened sources: {len(sources)}",
                "",
                "This is research support for data curation, not medical advice.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "audit_log.jsonl").write_text(
        "".join(orjson.dumps(event).decode("utf-8") + "\n" for event in audit_events),
        encoding="utf-8",
    )


def run_ingredient_study(
    *,
    ingredient_name: str,
    lookup_path: Path,
    depth: StudyDepth,
    out_dir: Path,
    provider: str | None = None,
    model: str | None = None,
    canonical_beverage_id: str | None = None,
    search_as_ingredient_name: bool = False,
    retrieve_full_text: bool = False,
    full_text_workers: int = 4,
) -> IngredientStudyResult:
    model_config = resolve_model_config(provider=provider, model=model)
    llm_provider = create_llm_provider(model_config)
    created_at = datetime.now(UTC).isoformat()
    row, search_name, relationship, ingredient_form = resolve_ingredient_target(
        lookup_path=lookup_path,
        ingredient_name=ingredient_name,
        canonical_beverage_id=canonical_beverage_id,
        search_as_ingredient_name=search_as_ingredient_name,
    )
    entity_plan = build_entity_plan(
        ingredient_name=ingredient_name.strip(),
        canonical_search_name_value=search_name,
        row=row,
        relationship=relationship,
        ingredient_form=ingredient_form,
        provider=llm_provider,
        use_llm_enrichment=model_config.provider != "heuristic",
    )
    audit_events = [
        {
            "ts": created_at,
            "event": "ingredient_study_started",
            "ingredient": ingredient_name,
            "canonical_search_name": search_name,
            "model_provider": model_config.provider,
            "model_name": model_config.model_name,
        }
    ]
    audit_events.append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "ingredient_entity_plan_created",
            "aliases": len(entity_plan.aliases),
            "included_forms": len(entity_plan.included_forms),
            "excluded_forms": len(entity_plan.excluded_forms),
            "query_terms": len(entity_plan.query_terms),
            "search_warnings": len(entity_plan.search_warnings),
            "retrieval_queries": len(entity_plan.retrieval_queries),
        }
    )
    sources, rejected_sources, retrieval_counts = paginated_retrieve_sources(
        ingredient_name=search_name,
        depth=depth,
        queries=entity_plan.retrieval_queries,
        aliases=[entity_plan.ingredient_name, *entity_plan.aliases, *entity_plan.query_terms],
        provider=llm_provider,
        screening_available=model_config.provider != "heuristic",
        audit_events=audit_events,
    )
    completed_at = datetime.now(UTC).isoformat()
    counts = PacketCounts(
        candidate_citations=retrieval_counts["unique_candidates"],
        raw_candidates_retrieved=retrieval_counts["raw_candidates_retrieved"],
        unique_candidates=retrieval_counts["unique_candidates"],
        retrieval_pages_attempted=retrieval_counts["retrieval_pages_attempted"],
        retrieval_stop_reason=retrieval_counts["retrieval_stop_reason"],
        screened_sources=retrieval_counts["screened_sources"],
        rejected_sources=retrieval_counts["rejected_sources"],
        proposed_claims=0,
        validated_claims=0,
        rejected_claims=0,
        effect_rows=0,
    )
    run_id = f"{timestamp_for_run_id()}_{slugify(ingredient_name)}"
    packet = IngredientPacket(
        run_id=run_id,
        input=IngredientPacketInput(
            ingredient_name=ingredient_name.strip(),
            canonical_search_name=search_name,
            ingredient_form=ingredient_form,
            canonical_beverage_id=row.get("canonical_beverage_id") if row else None,
            canonical_beverage_name=row.get("canonical_beverage_name") if row else None,
            depth=depth,
            demo=False,
            retrieval_queries=entity_plan.retrieval_queries,
        ),
        model=PacketModel(provider=model_config.provider, model_name=model_config.model_name),
        status="completed",
        created_at=created_at,
        completed_at=completed_at,
        counts=counts,
    )
    audit_events.append(
        {"ts": completed_at, "event": "ingredient_study_completed", "counts": counts.model_dump()}
    )
    run_dir = out_dir.resolve() / run_id
    write_ingredient_study_artifacts(
        run_dir=run_dir,
        packet=packet,
        entity_plan=entity_plan,
        sources=sources,
        rejected_sources=rejected_sources,
        audit_events=audit_events,
    )
    if retrieve_full_text:
        records = retrieve_full_text_for_run(run_dir, max_workers=full_text_workers)
        raw_counts = packet.counts.model_dump(mode="json")
        for status in FULL_TEXT_STATUS_FIELDS:
            raw_counts[status] = sum(1 for record in records if record.status == status)
        packet = packet.model_copy(update={"counts": PacketCounts(**raw_counts)})
        write_ingredient_packet(run_dir, packet)
    return IngredientStudyResult(run_dir=run_dir, run_id=run_id)


def ingredient_claim_proposal_prompt(
    *,
    packet: IngredientPacket,
    citation: CandidateCitation,
    body_text: str,
) -> str:
    from sipz_agent.core.retrieval import truncate_text

    context = truncate_text(body_text, max_chars=45_000)
    return f"""You are the paper-reader stage for the Sipz ingredient health report agent.

Task:
- Propose candidate health-effect claims only. Do not prove them.
- Focus on human health effects when this ingredient is consumed orally.
- A claim is eligible only when the specific supporting passage describes human oral intake,
  human dietary consumption, or a review/meta-analysis of human oral intake studies.
- Preserve ingredient form, serving/dose, food matrix, population, and outcome.
- Reject animal-only and in-vitro-only evidence. Do not turn preclinical findings into human
  candidate claims, even when a review author describes them as relevant, promising, or mechanistic.
- Do not propose a human claim whose population, dose, or outcome is extrapolated from isolated
  cells, tissue assays, animal models, buffers, cell-culture concentrations, or injected/topical use.
- Do not convert extract or supplement evidence into a whole-food or beverage claim.
- For reviews, classify evidence_type from the underlying evidence supporting the specific claim.
  Use review_author_interpretation only when the reviewed evidence includes human oral evidence.
- If a review only supports a statement with in vitro, animal, or chemistry-only evidence, skip that
  statement entirely instead of proposing it with limitations.
- If the paper contains no human oral evidence suitable for a claim, return an empty claims array
  and explain that in skipped_reason.
- Include positive, negative, neutral, or mixed findings when supported.
- Return at most 3 candidate claims.
- Return JSON only matching this shape: {{"claims":[...], "skipped_reason": null}}.

Each claim must include:
- id
- ingredient_name
- ingredient_form
- citation_id
- statement
- proposed_effect_slug
- proposed_effect_label
- effect
- claim_direction: beneficial, harmful, neutral, mixed, or unclear
- population
- oral_exposure
- dose_or_serving
- food_matrix
- outcome
- study_type
- limitations: JSON array
- evidence_type: human_clinical, human_observational, human_mechanistic, animal, in_vitro, mechanistic_theory, review_author_interpretation, composition_data, or unclear
- exposure_category: whole_food, juice_or_beverage, powder_or_concentrate, extract_or_supplement, or mixed_or_unclear
- concentration_notes
- claim_applies_to_group_members: direct_only, same_ingredient, similar_forms, form_specific, or do_not_propagate

Requested ingredient: {packet.input.ingredient_name}
Canonical search name: {packet.input.canonical_search_name}
Target ingredient form: {packet.input.ingredient_form}
Canonical beverage row: {packet.input.canonical_beverage_name or "unknown"}

Citation ID: {citation.id}
Title: {citation.title}
DOI: {citation.doi or "unknown"}
PMID: {citation.pmid or "unknown"}
Year: {citation.year or "unknown"}

Paper body text, not abstract:
{context}
"""


PRECLINICAL_CLAIM_PATTERNS = [
    r"\bin[\s-]?vitro\b",
    r"\bcell line(s)?\b",
    r"\bcultured cells?\b",
    r"\badipocytes?\b",
    r"\bastrocytes?\b",
    r"\bneurons?\b",
    r"\bcolon cancer cells?\b",
    r"\b3t3-l1\b",
    r"\bcaco-?2\b",
    r"\bhank'?s buffer\b",
    r"\bmouse cell\b",
    r"\brat model\b",
    r"\bmouse model\b",
    r"\bmice\b",
    r"\brats?\b",
    r"\banimal model\b",
]


def claim_text_for_preclinical_filter(claim: ProposedIngredientClaim) -> str:
    values = [
        claim.statement,
        claim.population or "",
        claim.oral_exposure,
        claim.dose_or_serving or "",
        claim.outcome or "",
        claim.study_type or "",
        " ".join(claim.limitations),
        claim.concentration_notes or "",
    ]
    return " ".join(values).lower()


def appears_preclinical_only_ingredient_claim(claim: ProposedIngredientClaim) -> bool:
    if claim.evidence_type in {"animal", "in_vitro", "mechanistic_theory"}:
        return True
    text = claim_text_for_preclinical_filter(claim)
    if any(re.search(pattern, text) for pattern in PRECLINICAL_CLAIM_PATTERNS):
        return True
    if re.search(r"\b\d+(\.\d+)?\s*(ug|µg|mcg|mg)\s*/\s*(ml|l)\b", text):
        return True
    return False


def accepted_proposed_ingredient_claim(claim: ProposedIngredientClaim) -> bool:
    if not claim.oral_exposure.strip():
        return False
    if appears_preclinical_only_ingredient_claim(claim):
        return False
    if claim.exposure_category == "extract_or_supplement":
        # Keep supplement/extract evidence, but prevent automatic propagation.
        claim.claim_applies_to_group_members = "form_specific"
    return True


def normalize_ingredient_claims(
    *,
    packet: IngredientPacket,
    citation_id: str,
    claims: list[ProposedIngredientClaim],
) -> list[ProposedIngredientClaim]:
    normalized = []
    for claim in claims:
        exposure_category = claim.exposure_category
        group_applicability = claim.claim_applies_to_group_members
        if group_applicability == "direct_only":
            group_applicability = infer_group_applicability(exposure_category)
        normalized.append(
            claim.model_copy(
                update={
                    "id": claim.id or str(uuid4()),
                    "ingredient_name": claim.ingredient_name or packet.input.ingredient_name,
                    "ingredient_form": claim.ingredient_form or packet.input.ingredient_form,
                    "citation_id": citation_id,
                    "claim_applies_to_group_members": group_applicability,
                }
            )
        )
    return normalized


def propose_ingredient_claims_for_source(
    *,
    packet: IngredientPacket,
    citation: CandidateCitation,
    body_text: str,
    provider: LlmProvider,
) -> list[ProposedIngredientClaim]:
    response = provider.complete_json(
        ingredient_claim_proposal_prompt(packet=packet, citation=citation, body_text=body_text),
        INGREDIENT_CLAIM_PROPOSAL_ADAPTER,
    )
    claims = normalize_ingredient_claims(
        packet=packet,
        citation_id=citation.id,
        claims=response.claims,
    )
    return [claim for claim in claims if accepted_proposed_ingredient_claim(claim)]


def write_proposed_ingredient_claims_markdown(
    path: Path,
    claims: list[ProposedIngredientClaim],
) -> None:
    lines = ["# Proposed Ingredient Claims", ""]
    if not claims:
        lines.extend(["No candidate ingredient claims were proposed.", ""])
    for index, claim in enumerate(claims, start=1):
        lines.append(f"## {index}. {claim.proposed_effect_label}")
        lines.append("")
        lines.append(f"- Citation ID: {claim.citation_id}")
        lines.append(f"- Statement: {claim.statement}")
        lines.append(f"- Direction: {claim.claim_direction}")
        lines.append(f"- Ingredient form: {claim.ingredient_form}")
        lines.append(f"- Exposure category: {claim.exposure_category}")
        lines.append(f"- Group applicability: {claim.claim_applies_to_group_members}")
        if claim.dose_or_serving:
            lines.append(f"- Dose/serving: {claim.dose_or_serving}")
        if claim.food_matrix:
            lines.append(f"- Food matrix: {claim.food_matrix}")
        if claim.limitations:
            lines.append(f"- Limitations: {'; '.join(claim.limitations)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def propose_ingredient_claims_from_raw_texts(
    *,
    run_dir: Path,
    provider: LlmProvider,
    progress: IngredientClaimProgress | None = None,
    workers: int = 1,
) -> list[ProposedIngredientClaim]:
    if workers < 1:
        raise ValueError("claim_workers_must_be_positive")
    packet = read_ingredient_packet(run_dir)
    sources = SOURCES_ADAPTER.validate_python(orjson.loads((run_dir / "sources.json").read_bytes()))
    raw_records = RAW_TEXTS_ADAPTER.validate_python(
        orjson.loads((run_dir / "raw_texts.json").read_bytes())
    )
    citations_by_id = {citation.id: citation for citation in sources}
    papers: list[tuple[CandidateCitation, str]] = []
    for record in raw_records:
        citation = citations_by_id.get(record.source_id)
        if citation is None:
            continue
        body_text = body_text_for_record(run_dir / "raw_texts", record)
        if body_text:
            papers.append((citation, body_text))

    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "ingredient_claim_proposal_started",
            "raw_text_records": len(raw_records),
        },
    )
    claims: list[ProposedIngredientClaim] = []
    if workers == 1:
        for index, (citation, body_text) in enumerate(papers, start=1):
            if progress:
                progress(index, len(papers), citation)
            claims.extend(
                propose_ingredient_claims_for_source(
                    packet=packet,
                    citation=citation,
                    body_text=body_text,
                    provider=provider,
                )
            )
    else:
        indexed_results: list[tuple[int, list[ProposedIngredientClaim]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, (citation, body_text) in enumerate(papers, start=1):
                if progress:
                    progress(index, len(papers), citation)
                future = executor.submit(
                    propose_ingredient_claims_for_source,
                    packet=packet,
                    citation=citation,
                    body_text=body_text,
                    provider=provider,
                )
                futures[future] = index
            for future in as_completed(futures):
                indexed_results.append((futures[future], future.result()))
        for _, paper_claims in sorted(indexed_results, key=lambda item: item[0]):
            claims.extend(paper_claims)

    write_json(run_dir / "proposed_ingredient_claims.json", model_dump_jsonable(claims))
    write_proposed_ingredient_claims_markdown(run_dir / "proposed_ingredient_claims.md", claims)
    legacy_claims = [legacy_from_ingredient_claim(claim) for claim in claims]
    write_json(run_dir / "proposed_claims.json", model_dump_jsonable(legacy_claims))
    write_proposed_claims_markdown(run_dir / "proposed_claims.md", legacy_claims)
    update_ingredient_packet_counts(run_dir, proposed_claims=len(claims))
    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "ingredient_claim_proposal_completed",
            "proposed_claims": len(claims),
        },
    )
    return claims


def ingredient_validation_prompt(
    *,
    packet: IngredientPacket,
    citation: CandidateCitation,
    claims: list[ProposedIngredientClaim],
    sanitized_body: str,
) -> str:
    legacy_claims = [legacy_from_ingredient_claim(claim) for claim in claims]
    base_prompt = validation_prompt(
        nutrient_name=packet.input.ingredient_name,
        citation=citation,
        claims=legacy_claims,
        sanitized_body=sanitized_body,
    )
    ingredient_payload = [
        {
            "proposed_claim_id": claim.id,
            "ingredient_form": claim.ingredient_form,
            "oral_exposure": claim.oral_exposure,
            "dose_or_serving": claim.dose_or_serving,
            "food_matrix": claim.food_matrix,
            "exposure_category": claim.exposure_category,
            "group_applicability": claim.claim_applies_to_group_members,
        }
        for claim in claims
    ]
    return (
        base_prompt
        + "\n\nIngredient-specific validation rules:\n"
        + "- Reject non-oral claims.\n"
        + "- Reject animal-only and in vitro-only evidence for final ingredient reports.\n"
        + "- Reject claims that overgeneralize ingredient form, dose, serving, or food matrix.\n"
        + "- Do not validate extract/supplement evidence as a whole-food or beverage claim.\n"
        + "- Preserve whether support is specific to powder, concentrate, juice, extract, or whole food.\n\n"
        + "Ingredient-specific claim context:\n"
        + orjson.dumps(ingredient_payload, option=orjson.OPT_INDENT_2).decode("utf-8")
    )


def grounded_ingredient_validation(
    *,
    claim: ProposedIngredientClaim,
    decision: ValidatorDecision,
    citation: CandidateCitation,
    sanitized_body: str,
    provider: LlmProvider | None = None,
) -> ValidatedIngredientClaim:
    quotes = grounded_supporting_quotes(
        validator_quotes=decision.supporting_quotes,
        citation_title=citation.title,
        sanitized_body=sanitized_body,
    )
    model_supported = decision.verdict in {"supported", "supported_with_limitations"}
    grounded_quote_count = sum(1 for quote in quotes if quote.match_status != "not_found")
    has_substantive_grounded_quote = (
        substantive_grounded_quote_count(quotes=quotes, citation_title=citation.title) > 0
    )
    if model_supported and not has_substantive_grounded_quote and provider is not None:
        repaired_quotes = repair_supporting_quotes_once(
            claim=claim,
            decision=decision,
            citation_title=citation.title,
            sanitized_body=sanitized_body,
            provider=provider,
        )
        if repaired_quotes is not None:
            quotes = repaired_quotes
            grounded_quote_count = sum(1 for quote in quotes if quote.match_status != "not_found")
            has_substantive_grounded_quote = True
    has_validated_statement = bool(decision.validated_statement.strip())
    has_claim_scope = bool(decision.claim_scope.strip())
    accepted = (
        model_supported
        and has_substantive_grounded_quote
        and has_validated_statement
        and has_claim_scope
    )
    verdict = (
        cast(Any, decision.verdict)
        if accepted
        else ("quote_not_found" if model_supported else decision.verdict)
    )
    limitations = list(decision.limitations)
    if model_supported and grounded_quote_count == 0:
        limitations.append("No validator quote could be grounded in the sanitized paper body.")
    elif model_supported and not has_substantive_grounded_quote:
        limitations.append("The only grounded validator quote matched title/header-like text.")
    validated_statement = decision.validated_statement.strip() or (
        "Rejected because the paper body did not support this proposed ingredient claim."
    )
    claim_scope = decision.claim_scope.strip() or (
        "Rejected because the paper body did not support a validated ingredient claim scope."
    )
    output_quotes = (
        acceptable_grounded_quotes(quotes=quotes, citation_title=citation.title)
        if accepted
        else quotes
    )
    return ValidatedIngredientClaim(
        effect_row_id=str(uuid4()),
        proposed_ingredient_claim_id=claim.id,
        citation_id=citation.id,
        verdict=verdict,
        support_level=decision.support_level,
        claim_scope=claim_scope,
        validated_statement=validated_statement,
        validator_reasoning=decision.reasoning,
        supporting_quotes=output_quotes,
        limitations=limitations,
        accepted=accepted,
    )


def validate_ingredient_claims_for_paper(
    *,
    packet: IngredientPacket,
    citation: CandidateCitation,
    claims: list[ProposedIngredientClaim],
    sanitized_body: str,
    provider: LlmProvider,
) -> list[ValidatedIngredientClaim]:
    response = provider.complete_json(
        ingredient_validation_prompt(
            packet=packet,
            citation=citation,
            claims=claims,
            sanitized_body=sanitized_body,
        ),
        PAPER_VALIDATION_ADAPTER,
    )
    decisions = {decision.proposed_claim_id: decision for decision in response.decisions}
    results = []
    for claim in claims:
        decision = decisions.get(claim.id)
        if decision is None:
            raise ValueError(f"validator_missing_decision:{claim.id}")
        results.append(
            grounded_ingredient_validation(
                claim=claim,
                decision=decision,
                citation=citation,
                sanitized_body=sanitized_body,
                provider=provider,
            )
        )
    return results


def validate_ingredient_claims(
    *,
    run_dir: Path,
    provider: LlmProvider,
    progress: IngredientValidationProgress | None = None,
    max_body_chars: int = 750_000,
    workers: int = 1,
) -> tuple[list[ValidatedIngredientClaim], list[ValidatedIngredientClaim], list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("claim_workers_must_be_positive")
    packet = read_ingredient_packet(run_dir)
    proposed = PROPOSED_INGREDIENT_CLAIMS_ADAPTER.validate_python(
        orjson.loads((run_dir / "proposed_ingredient_claims.json").read_bytes())
    )
    sources = SOURCES_ADAPTER.validate_python(orjson.loads((run_dir / "sources.json").read_bytes()))
    raw_records = RAW_TEXTS_ADAPTER.validate_python(
        orjson.loads((run_dir / "raw_texts.json").read_bytes())
    )
    citations = {citation.id: citation for citation in sources}
    records = {record.source_id: record for record in raw_records}
    claims_by_source: dict[str, list[ProposedIngredientClaim]] = {}
    for claim in proposed:
        claims_by_source.setdefault(claim.citation_id, []).append(claim)

    accepted: list[ValidatedIngredientClaim] = []
    rejected: list[ValidatedIngredientClaim] = []
    failures: list[dict[str, Any]] = []
    papers = list(claims_by_source.items())
    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "ingredient_claim_validation_started",
            "papers": len(papers),
            "claims": sum(len(items) for _, items in papers),
        },
    )
    def validate_one_paper(
        index: int,
        citation_id: str,
        paper_claims: list[ProposedIngredientClaim],
    ) -> tuple[int, list[ValidatedIngredientClaim], list[ValidatedIngredientClaim], dict[str, Any] | None]:
        citation = citations.get(citation_id)
        record = records.get(citation_id)
        body_text: str | None = None
        try:
            if citation is None:
                raise ValueError("citation_not_found")
            if record is None:
                raise ValueError("raw_text_record_not_found")
            body_text = body_text_for_record(run_dir / "raw_texts", record)
            if not body_text:
                raise ValueError("full_text_not_found")
            sanitized = sanitize_paper_body(body_text=body_text, citation=citation)
            if len(sanitized) > max_body_chars:
                raise ValueError(
                    f"paper_exceeds_validation_context:{len(sanitized)}>{max_body_chars}"
                )
            paper_results = validate_ingredient_claims_for_paper(
                packet=packet,
                citation=citation,
                claims=paper_claims,
                sanitized_body=sanitized,
                provider=provider,
            )
            return (
                index,
                [result for result in paper_results if result.accepted],
                [result for result in paper_results if not result.accepted],
                None,
            )
        except Exception as exc:
            failure_code = validation_failure_code(exc)
            preview_path = write_sanitized_body_preview(
                out_dir=run_dir,
                citation_id=citation_id,
                body_text=body_text,
                exc=exc,
            )
            failure = {
                "citation_id": citation_id,
                "proposed_ingredient_claim_ids": [claim.id for claim in paper_claims],
                "failure_code": failure_code,
                "error": str(exc) or type(exc).__name__,
            }
            if preview_path:
                failure["sanitized_body_preview_path"] = preview_path
            return (index, [], [], failure)

    indexed_results: list[
        tuple[int, list[ValidatedIngredientClaim], list[ValidatedIngredientClaim], dict[str, Any] | None]
    ] = []
    if workers == 1:
        for index, (citation_id, paper_claims) in enumerate(papers, start=1):
            citation = citations.get(citation_id)
            if citation is not None and progress:
                progress(index, len(papers), citation, len(paper_claims))
            indexed_results.append(validate_one_paper(index, citation_id, paper_claims))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, (citation_id, paper_claims) in enumerate(papers, start=1):
                citation = citations.get(citation_id)
                if citation is not None and progress:
                    progress(index, len(papers), citation, len(paper_claims))
                future = executor.submit(validate_one_paper, index, citation_id, paper_claims)
                futures[future] = index
            for future in as_completed(futures):
                indexed_results.append(future.result())

    for _, paper_accepted, paper_rejected, failure in sorted(
        indexed_results,
        key=lambda item: item[0],
    ):
        accepted.extend(paper_accepted)
        rejected.extend(paper_rejected)
        if failure is not None:
            failures.append(failure)

    write_json(run_dir / "validated_ingredient_claims.json", model_dump_jsonable(accepted))
    write_json(run_dir / "rejected_ingredient_claims.json", model_dump_jsonable(rejected))
    write_json(run_dir / "ingredient_validation_failures.json", failures)
    (run_dir / "ingredient_validation_summary.md").write_text(
        "\n".join(
            [
                "# Ingredient Claim Validation Summary",
                "",
                f"Accepted claims: {len(accepted)}",
                f"Rejected claims: {len(rejected)}",
                f"Paper failures: {len(failures)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        run_dir / "validated_claims.json",
        model_dump_jsonable([legacy_from_validated_ingredient_claim(claim) for claim in accepted]),
    )
    write_json(
        run_dir / "rejected_claims.json",
        model_dump_jsonable([legacy_from_validated_ingredient_claim(claim) for claim in rejected]),
    )
    write_json(run_dir / "validation_failures.json", failures)
    (run_dir / "validation_summary.md").write_text(
        (run_dir / "ingredient_validation_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    update_ingredient_packet_counts(
        run_dir,
        validated_claims=len(accepted),
        rejected_claims=len(rejected),
    )
    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "ingredient_claim_validation_completed",
            "accepted": len(accepted),
            "rejected": len(rejected),
            "failures": len(failures),
        },
    )
    return accepted, rejected, failures
