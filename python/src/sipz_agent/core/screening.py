from __future__ import annotations

import re
import unicodedata

from pydantic import TypeAdapter

from sipz_agent.core.models import LlmProvider
from sipz_agent.schemas.citations import (
    CandidateCitation,
    RejectedCitation,
    SourceRejectionCode,
    SourceScreeningDecision,
)

SCREENING_ADAPTER = TypeAdapter(SourceScreeningDecision)


class SourceScreeningUnavailable(RuntimeError):
    pass


class SourceScreeningOutput:
    def __init__(
        self,
        *,
        accepted: list[CandidateCitation],
        rejected: list[RejectedCitation],
        decisions: dict[str, SourceScreeningDecision] | None = None,
    ) -> None:
        self.accepted = accepted
        self.rejected = rejected
        self.decisions = decisions or {}


def missing_abstract_decision() -> SourceScreeningDecision:
    return SourceScreeningDecision(
        accepted=False,
        human_health_relevance=False,
        mentions_nutrient_or_bioactive=False,
        conclusiveness="not_applicable",
        rationale=(
            "Rejected because no abstract was available, so the title and metadata were "
            "insufficient for light human-health screening."
        ),
    )


def screening_context(citation: CandidateCitation) -> tuple[str | None, str]:
    if citation.abstract:
        return citation.abstract, "API abstract"
    if citation.page_summary:
        return citation.page_summary, "visible landing-page summary"
    return None, "missing"


def screening_error_decision(error: Exception) -> SourceScreeningDecision:
    return SourceScreeningDecision(
        accepted=False,
        human_health_relevance=False,
        mentions_nutrient_or_bioactive=False,
        conclusiveness="not_applicable",
        rationale=f"screening_error: {type(error).__name__}: {error}",
    )


def is_provider_unavailable_error(error: Exception) -> bool:
    return "llm_provider_payment_required" in str(error)


def rejected_inconsistent_acceptance_decision(
    decision: SourceScreeningDecision,
) -> SourceScreeningDecision:
    return decision.model_copy(
        update={
            "accepted": False,
            "rationale": (
                "Rejected because the screening response was internally inconsistent: "
                f"{decision.rationale}"
            ),
        }
    )


def is_accepted_decision(decision: SourceScreeningDecision) -> bool:
    if decision.entity_match in {"different_species", "different_entity"}:
        return False
    if decision.relevance_class in {"indirect_background", "reject"}:
        return False
    if decision.intervention_specificity in {"mixed_confounded", "mention_only"}:
        return False
    if decision.publication_type in {"case_report", "news_editorial", "broad_review", "preclinical"}:
        return False
    return (
        decision.accepted
        and decision.human_health_relevance
        and decision.mentions_nutrient_or_bioactive
    )


def normalized_screening_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).split())


def citation_screening_text(citation: CandidateCitation, decision: SourceScreeningDecision) -> str:
    return " ".join(
        part
        for part in [
            citation.title,
            citation.abstract,
            citation.page_summary,
            citation.retrieval_query,
            decision.rationale,
        ]
        if part
    )


def matched_source_alias(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    aliases: list[str] | None = None,
) -> str | None:
    text = normalized_screening_text(
        " ".join(
            part
            for part in [
                citation.title,
                citation.abstract,
                citation.page_summary,
                citation.retrieval_query,
            ]
            if part
        )
    )
    padded_text = f" {text} "
    for alias in [*(aliases or []), nutrient_name]:
        clean = " ".join(str(alias).split())
        key = normalized_screening_text(clean)
        if clean and key and f" {key} " in padded_text:
            return clean
    return None


def rejection_code_from_decision(
    *,
    decision: SourceScreeningDecision,
    citation: CandidateCitation,
) -> tuple[SourceRejectionCode, str, float]:
    text = normalized_screening_text(citation_screening_text(citation, decision))
    rationale = normalized_screening_text(decision.rationale)
    if "screening error" in rationale:
        return "screening_error", "screening_execution", 0.95
    if "internally inconsistent" in rationale:
        return "inconsistent_screening", "screening_consistency", 0.95
    if "no abstract was available" in rationale or "insufficient" in rationale:
        return "insufficient_metadata", "screening_context", 0.9
    if decision.entity_match in {"different_species", "different_entity"}:
        return "wrong_entity", "entity_match", 0.95
    if not decision.mentions_nutrient_or_bioactive:
        return "wrong_entity", "entity_match", 0.85
    if re.search(r"\bin vitro\b|\bcell line\b|\bcells\b|\bmouse\b|\bmice\b|\brat\b|\banimal", text):
        return "preclinical_only", "human_evidence", 0.9
    if re.search(r"\btopical\b|\bdermal\b|\bskin\b|\binjection\b|\bintravenous\b", text):
        return "not_oral", "oral_consumption", 0.85
    if re.search(
        r"\bextraction\b|\bprocessing\b|\bagriculture\b|\bpesticide\b|\bresidue\b|"
        r"\bchemical composition\b|\bphytochemical\b|\bcultivar\b|\bharvest\b",
        text,
    ):
        return "food_processing_only", "source_scope", 0.85
    if re.search(r"\bbackground\b|\bnarrative review\b|\boverview\b", text):
        return "background_review_only", "health_effect", 0.7
    if not decision.human_health_relevance:
        return "not_health_effect", "health_effect", 0.75
    if "human" not in text and "adult" not in text and "participant" not in text:
        return "not_human", "human_evidence", 0.65
    return "unclear", "source_screening", 0.5


def rejected_citation(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    decision: SourceScreeningDecision,
    aliases: list[str] | None = None,
) -> RejectedCitation:
    code, failed_requirement, confidence = rejection_code_from_decision(
        decision=decision,
        citation=citation,
    )
    return RejectedCitation(
        citation=citation,
        screening=decision,
        rejection_code=code,
        screening_confidence=confidence,
        matched_alias=matched_source_alias(
            nutrient_name=nutrient_name,
            citation=citation,
            aliases=aliases,
        ),
        failed_requirement=failed_requirement,
    )


def source_screening_prompt(nutrient_name: str, citation: CandidateCitation) -> str:
    context, context_label = screening_context(citation)
    return "\n".join(
        [
            "Screen this candidate paper for a nutrient research bibliography.",
            "",
            "Acceptance criteria:",
            "- Accept direct human studies only when the requested substance is an isolated "
            "or separately interpretable oral exposure.",
            "- Accept reviews only when the requested substance is a central subject and the "
            "review evaluates human-health evidence for oral consumption.",
            "- Treat biological species as distinct entities. A related species is not an alias. "
            "Reject a paper centered on a different species unless the requested target is the "
            "broader genus/class and that broader scope is explicit.",
            "- Reject animal-only, in-vitro-only, unrelated disease/compound papers, broad "
            "nutrition papers without the requested nutrient-health relationship, and "
            "records with insufficient evidence in the abstract.",
            "- Reject mixed interventions when the requested substance's contribution cannot "
            "be separated from herbs, other vitamins, drugs, or co-interventions.",
            "- Reject papers that merely mention the substance, broad multi-nutrient reviews, "
            "news briefs, editorials, table fragments, protocols without results, and case reports.",
            "- If accepted, classify whether the abstract presents results as conclusive, "
            "says further research is required, or is unclear.",
            "",
            "Return JSON exactly matching this schema:",
            "{"
            '"accepted": boolean, '
            '"human_health_relevance": boolean, '
            '"mentions_nutrient_or_bioactive": boolean, '
            '"conclusiveness": "conclusive" | "needs_more_research" | "unclear" | '
            '"not_applicable", '
            '"relevance_class": "direct_human" | "focused_review" | "indirect_background" | "reject", '
            '"intervention_specificity": "isolated" | "separable" | "mixed_confounded" | "mention_only" | "not_applicable", '
            '"publication_type": "primary_human_study" | "systematic_review_meta_analysis" | "focused_narrative_review" | "case_report" | "news_editorial" | "broad_review" | "preclinical" | "other", '
            '"entity_match": "exact_or_alias" | "broader_class" | "different_species" | "different_entity" | "unclear", '
            '"rationale": string'
            "}",
            "",
            f"Requested nutrient/bioactive: {nutrient_name}",
            f"Title: {citation.title}",
            f"Source: {citation.source}",
            f"Year: {citation.year or 'unknown'}",
            f"DOI: {citation.doi or 'unknown'}",
            f"PMID: {citation.pmid or 'unknown'}",
            "",
            f"Screening context type: {context_label}",
            f"Screening context: {context}",
        ]
    )


def screen_source(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    provider: LlmProvider,
) -> SourceScreeningDecision:
    context, _ = screening_context(citation)
    if not context:
        return missing_abstract_decision()
    return provider.complete_json(
        source_screening_prompt(nutrient_name, citation),
        SCREENING_ADAPTER,
    )


def screen_sources(
    *,
    nutrient_name: str,
    citations: list[CandidateCitation],
    provider: LlmProvider,
    aliases: list[str] | None = None,
) -> SourceScreeningOutput:
    accepted: list[CandidateCitation] = []
    rejected: list[RejectedCitation] = []
    decisions: dict[str, SourceScreeningDecision] = {}

    for citation in citations:
        try:
            decision = screen_source(
                nutrient_name=nutrient_name,
                citation=citation,
                provider=provider,
            )
        except Exception as error:
            if is_provider_unavailable_error(error):
                raise SourceScreeningUnavailable(str(error)) from error
            decision = screening_error_decision(error)

        decisions[citation.id] = decision

        if is_accepted_decision(decision):
            accepted.append(
                citation.model_copy(
                    update={
                        "selection_reason": (
                            f"{citation.selection_reason or 'Selected by retrieval.'} "
                            f"LLM source screening accepted this paper: {decision.rationale}"
                        )
                    }
                )
            )
        else:
            if decision.accepted and decision.entity_match in {
                "different_species",
                "different_entity",
            }:
                decision = decision.model_copy(
                    update={
                        "accepted": False,
                        "rationale": f"Rejected for entity mismatch: {decision.rationale}",
                    }
                )
                decisions[citation.id] = decision
            elif decision.accepted:
                decision = rejected_inconsistent_acceptance_decision(decision)
                decisions[citation.id] = decision
            rejected.append(
                rejected_citation(
                    nutrient_name=nutrient_name,
                    citation=citation,
                    decision=decision,
                    aliases=aliases,
                )
            )

    return SourceScreeningOutput(accepted=accepted, rejected=rejected, decisions=decisions)
