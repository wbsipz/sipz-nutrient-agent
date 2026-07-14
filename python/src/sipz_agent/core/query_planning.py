from __future__ import annotations

from pydantic import TypeAdapter

from sipz_agent.core.models import LlmProvider
from sipz_agent.core.retrieval import candidate_query
from sipz_agent.schemas.citations import RetrievalQueryPlan

QUERY_PLAN_ADAPTER = TypeAdapter(RetrievalQueryPlan)
MAX_RETRIEVAL_QUERIES = 6
MAX_SPECIFIC_SYNONYMS = 8


def fallback_query_plan(nutrient_name: str, rationale: str = "Used default query.") -> RetrievalQueryPlan:
    normalized = nutrient_name.strip()
    return RetrievalQueryPlan(
        canonical_name=normalized,
        is_niche=False,
        specific_synonyms=[],
        source_terms=[],
        recommended_queries=[candidate_query(normalized)],
        rationale=rationale,
    )


def normalize_query_plan(nutrient_name: str, plan: RetrievalQueryPlan) -> RetrievalQueryPlan:
    normalized_name = nutrient_name.strip()
    canonical_name = plan.canonical_name.strip() or normalized_name
    synonyms = list(
        dict.fromkeys(
            item.strip()
            for item in plan.specific_synonyms
            if item and item.strip().lower() != normalized_name.lower()
        )
    )[:MAX_SPECIFIC_SYNONYMS]
    queries: list[str] = []
    for query in [candidate_query(normalized_name), *plan.recommended_queries]:
        clean = " ".join(query.split())
        if clean and clean.lower() not in {item.lower() for item in queries}:
            queries.append(clean)
        if len(queries) >= MAX_RETRIEVAL_QUERIES - 1:
            break

    searchable_text = " ".join(queries).lower()
    coverage_terms = [
        item
        for item in [canonical_name, *synonyms]
        if item.lower() != normalized_name.lower()
    ]
    missing_synonyms = [item for item in coverage_terms if item.lower() not in searchable_text]
    if missing_synonyms:
        alternatives = " OR ".join(f'"{item}"' for item in missing_synonyms)
        synonym_query = (
            f"({alternatives}) AND (human OR clinical OR dietary OR oral OR consumption)"
        )
        if len(queries) >= MAX_RETRIEVAL_QUERIES:
            queries[-1] = synonym_query
        else:
            queries.append(synonym_query)

    if not queries:
        queries = [candidate_query(normalized_name)]

    return plan.model_copy(
        update={
            "canonical_name": canonical_name,
            "specific_synonyms": synonyms,
            "source_terms": [item.strip() for item in plan.source_terms[:8] if item and item.strip()],
            "recommended_queries": queries,
            "rationale": plan.rationale.strip() or "Generated search-query expansion plan.",
        }
    )


def query_planning_prompt(nutrient_name: str) -> str:
    return f"""
You are the retrieval query planner for a nutrient and bioactive literature-search agent.

Requested nutrient/bioactive: {nutrient_name}

Create a concise search plan for finding human-health papers about oral consumption of this
nutrient or bioactive.

Rules:
- If the requested item is broad or common, keep the plan conservative with one or two queries.
- If the requested item is niche, identify exact scientific synonyms, spelling variants, and
  source terms that are likely to appear in titles or abstracts.
- Recommended queries must still target human health and oral exposure, not pharmaceutical use.
- Do not include broad parent classes unless the requested item is often indexed under that class.
- Do not invent registry identifiers, DOI fragments, or unsupported claims.
- Return only JSON matching this shape:
{{
  "canonical_name": "preferred name",
  "is_niche": true,
  "specific_synonyms": ["exact synonym"],
  "source_terms": ["major natural source term"],
  "recommended_queries": ["query string"],
  "rationale": "short reason"
}}
""".strip()


def plan_retrieval_queries(nutrient_name: str, provider: LlmProvider) -> RetrievalQueryPlan:
    try:
        plan = provider.complete_json(query_planning_prompt(nutrient_name), QUERY_PLAN_ADAPTER)
    except Exception as exc:
        return fallback_query_plan(
            nutrient_name,
            rationale=f"Query planning failed; used default query. Reason: {type(exc).__name__}: {exc}",
        )
    return normalize_query_plan(nutrient_name, plan)
