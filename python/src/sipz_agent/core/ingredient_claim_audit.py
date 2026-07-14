from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable, Literal

import orjson
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from sipz_agent.core.artifacts import write_json
from sipz_agent.core.models import HeuristicProvider, LlmProvider
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.ingredients import (
    IngredientPacket,
    ProposedIngredientClaim,
    ValidatedIngredientClaim,
)


PROPOSED_INGREDIENT_ADAPTER = TypeAdapter(list[ProposedIngredientClaim])
VALIDATED_INGREDIENT_ADAPTER = TypeAdapter(list[ValidatedIngredientClaim])
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
INGREDIENT_PACKET_ADAPTER = TypeAdapter(IngredientPacket)

AuditVerdict = Literal["pass", "needs_review", "reject_for_synthesis"]
PaperSupportCheck = Literal[
    "validated_and_quote_grounded",
    "grounding_or_artifact_problem",
    "scope_needs_review",
]
ReportFitCheck = Literal[
    "suitable",
    "suitable_with_caveats",
    "needs_review",
    "not_suitable",
]
ModernEvidenceCheck = Literal[
    "generally_consistent",
    "limited_or_uncertain",
    "possibly_contradicted",
    "implausible_or_outdated",
    "not_assessed",
]


class IngredientClaimAuditDecision(BaseModel):
    proposed_ingredient_claim_id: str = Field(min_length=1)
    citation_id: str = Field(min_length=1)
    verdict: AuditVerdict
    paper_support_check: PaperSupportCheck = "validated_and_quote_grounded"
    report_fit_check: ReportFitCheck = "suitable_with_caveats"
    modern_evidence_check: ModernEvidenceCheck = "limited_or_uncertain"
    issue_categories: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=1200)
    paper_support_reasoning: str | None = Field(default=None, max_length=800)
    report_fit_reasoning: str | None = Field(default=None, max_length=800)
    modern_evidence_reasoning: str | None = Field(default=None, max_length=800)
    suggested_scope: str | None = Field(default=None, max_length=1200)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("issue_categories", mode="before")
    @classmethod
    def normalize_issue_categories(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value


class IngredientClaimAuditResponse(BaseModel):
    decisions: list[IngredientClaimAuditDecision]


INGREDIENT_CLAIM_AUDIT_RESPONSE_ADAPTER = TypeAdapter(IngredientClaimAuditResponse)


@dataclass
class IngredientClaimAuditResult:
    audited_claims: list[ValidatedIngredientClaim]
    rejected_claims: list[ValidatedIngredientClaim]
    findings: list[dict[str, Any]]
    skipped: bool = False
    skip_reason: str | None = None


def claim_key(citation_id: str, proposed_claim_id: str) -> tuple[str, str]:
    return (citation_id, proposed_claim_id)


def grounded_quote_count(claim: ValidatedIngredientClaim) -> int:
    return sum(1 for quote in claim.supporting_quotes if quote.match_status != "not_found")


def _normalized_text(*values: object) -> str:
    return " ".join(str(value or "").casefold() for value in values)


def deterministic_issue_categories(
    *,
    claim: ValidatedIngredientClaim,
    proposed: ProposedIngredientClaim | None,
    source: CandidateCitation | None,
) -> list[str]:
    issues: list[str] = []
    if grounded_quote_count(claim) == 0:
        issues.append("no_grounded_quote")
    if proposed is None:
        issues.append("missing_proposed_claim")
        return issues
    if source is None:
        issues.append("missing_source_metadata")

    claim_text = _normalized_text(
        proposed.statement,
        proposed.ingredient_form,
        proposed.oral_exposure,
        proposed.dose_or_serving,
        proposed.food_matrix,
        proposed.population,
        proposed.outcome,
        proposed.study_type,
        proposed.limitations,
        proposed.concentration_notes,
        claim.claim_scope,
        claim.validated_statement,
        claim.validator_reasoning,
        claim.limitations,
    )
    if proposed.evidence_type in {"animal", "in_vitro", "mechanistic_theory"}:
        issues.append("animal_or_in_vitro_used_as_human")

    non_oral_patterns = [
        r"\btopical\b",
        r"\bmouth\s?wash\b",
        r"\bmouth\s?rinse\b",
        r"\boil pulling\b",
        r"\bgargl",
        r"\boral hygiene\b",
        r"\bdental plaque\b",
        r"\bperiodontal\b",
        r"\bsalivar",
        r"\bstrep(?:tococcus)? mutans\b",
        r"\bmucositis\b",
        r"\baromatherapy\b",
        r"\binhal",
        r"\bnasal\b",
        r"\bodou?r\b",
        r"\bessential oil\b",
    ]
    if any(re.search(pattern, claim_text) for pattern in non_oral_patterns):
        issues.append("topical_or_oral_hygiene_evidence")

    if proposed.oral_exposure.strip() == "" and not re.search(
        r"\b(consum|oral|ingest|diet|intake|supplement|capsule|drink|eat|ate|food|beverage)\b",
        claim_text,
    ):
        issues.append("missing_oral_exposure")

    return list(dict.fromkeys(issues))


def audit_prompt(
    *,
    packet: IngredientPacket,
    source: CandidateCitation,
    items: list[tuple[ValidatedIngredientClaim, ProposedIngredientClaim]],
) -> str:
    payload = []
    for claim, proposed in items:
        payload.append(
            {
                "citation_id": claim.citation_id,
                "proposed_ingredient_claim_id": claim.proposed_ingredient_claim_id,
                "proposed": proposed.model_dump(mode="json"),
                "validated": claim.model_dump(mode="json"),
            }
        )
    return f"""Audit ingredient health claims before synthesis.

Important:
- This audit has three responsibilities:
  1. Paper support sanity: confirm the prior validator artifact says the paper support was accepted and quote-grounded. Do not re-litigate the full paper.
  2. Report fit: decide whether the claim is suitable and cautious enough for a consumer ingredient health report.
  3. Modern model-knowledge check: based only on your internal scientific/medical knowledge, decide whether broader modern evidence makes the claim plausible, uncertain, contradicted, implausible, or outdated. Do not browse or invent citations.
- This is NOT paper validation.
- A prior validator already checked that the paper body supports the claim and that quotes are grounded.
- Do not re-judge whether the supplied paper actually found the effect unless the validator artifact itself has a scope/grounding problem.
- Your main job is to ask whether the health claim still makes logical sense for a modern consumer ingredient health report.
- A cautious "may" claim is acceptable when the evidence is relevant but limited.

Goal:
- Keep claims that are logically plausible, quote-grounded, and useful for an ingredient health report.
- Prefer passing cautiously scoped claims over suppressing them.
- The ingredient should be consumed orally as food, beverage, powder/concentrate, supplement, extract, or as part of a studied food group/formulation.
- Do not reject a claim solely because the ingredient was studied as part of a combination, food group, dietary pattern, or formulation. If the validated wording already makes that caveat clear, pass it and use suggested_scope for careful wording.
- Reject only obvious bad fits: topical, mouthwash, oil pulling, dental plaque/oral hygiene, aromatherapy, inhalation, animal-only, in-vitro-only, chemistry-only, wrong entity, or claims that are implausible/obsolete/contradicted by broad modern evidence.

Verdicts:
- pass: claim is suitable for synthesis as a cautious "may" or "was associated with" statement, even if caveats are needed.
- needs_review: claim might be useful, but modern plausibility or report fit is uncertain enough that a human/audit-repair pass should inspect it.
- reject_for_synthesis: claim is clearly misleading for the ingredient row, non-oral, preclinical-only, obsolete, biologically implausible, contradicted by broad modern evidence, or wrong-entity.

Required check fields:
- paper_support_check:
  - validated_and_quote_grounded: accepted validator output includes grounded quote support.
  - grounding_or_artifact_problem: missing/ungrounded quote, missing proposed claim, missing source, or internally inconsistent artifact.
  - scope_needs_review: validator support exists, but the validated statement appears to drift beyond what the artifact says.
- report_fit_check:
  - suitable: ready for report.
  - suitable_with_caveats: useful if caveats/form/group/population are preserved.
  - needs_review: report wording/scope needs human or repair review.
  - not_suitable: should not appear in a consumer ingredient report.
- modern_evidence_check:
  - generally_consistent: makes sense based on broad model knowledge.
  - limited_or_uncertain: plausible but evidence is likely limited, mixed, or indirect.
  - possibly_contradicted: may conflict with broader modern evidence; needs review or web audit.
  - implausible_or_outdated: claim conflicts with current broad understanding.
  - not_assessed: only use if no model-knowledge assessment is possible.

Issue categories to use when relevant:
- overgeneralized_claim
- wrong_ingredient_form
- extract_or_supplement_overgeneralized
- formulated_product_misrepresented_as_whole_food
- combination_or_food_group_evidence
- dose_or_serving_missing
- positive_effect_too_strong
- negative_effect_missing
- too_specific_for_final_report
- duplicate_or_overlapping_claim
- small_sample_size
- no_control_group
- composition_data_not_human_trial
- implausible_or_outdated_claim
- contradicted_by_modern_evidence
- non_oral_evidence
- animal_or_in_vitro_used_as_human
- topical_or_oral_hygiene_evidence

Guidance:
- Use combination_or_food_group_evidence as a caveat, not an automatic rejection reason.
- Do not mark overgeneralized_claim merely because the paper did not isolate the ingredient, as long as the validated statement preserves that limitation.
- Dose or serving gaps should usually pass with caveats unless the claim would become unsafe or misleading.

Return JSON only:
{{
  "decisions": [
    {{
      "proposed_ingredient_claim_id": "...",
      "citation_id": "...",
      "verdict": "pass|needs_review|reject_for_synthesis",
      "paper_support_check": "validated_and_quote_grounded|grounding_or_artifact_problem|scope_needs_review",
      "report_fit_check": "suitable|suitable_with_caveats|needs_review|not_suitable",
      "modern_evidence_check": "generally_consistent|limited_or_uncertain|possibly_contradicted|implausible_or_outdated|not_assessed",
      "issue_categories": ["..."],
      "reasoning": "Short reason grounded in the supplied claim/source metadata.",
      "paper_support_reasoning": "Short note on the validator/quote artifact.",
      "report_fit_reasoning": "Short note on consumer report suitability and caveats.",
      "modern_evidence_reasoning": "Short note based on model knowledge only; do not cite searched evidence.",
      "suggested_scope": "Optional narrower wording if useful.",
      "confidence": 0.0
    }}
  ]
}}

Ingredient packet:
{orjson.dumps(packet.model_dump(mode="json"), option=orjson.OPT_INDENT_2).decode("utf-8")}

Source:
{orjson.dumps(source.model_dump(mode="json"), option=orjson.OPT_INDENT_2).decode("utf-8")}

Claims to audit:
{orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")}
"""


def finding_from_decision(
    *,
    decision: IngredientClaimAuditDecision,
    stage: str,
    source: CandidateCitation | None,
) -> dict[str, Any]:
    return {
        "audit_stage": stage,
        "citation_id": decision.citation_id,
        "proposed_ingredient_claim_id": decision.proposed_ingredient_claim_id,
        "verdict": decision.verdict,
        "paper_support_check": decision.paper_support_check,
        "report_fit_check": decision.report_fit_check,
        "modern_evidence_check": decision.modern_evidence_check,
        "issue_categories": decision.issue_categories,
        "reasoning": decision.reasoning,
        "paper_support_reasoning": decision.paper_support_reasoning,
        "report_fit_reasoning": decision.report_fit_reasoning,
        "modern_evidence_reasoning": decision.modern_evidence_reasoning,
        "suggested_scope": decision.suggested_scope,
        "confidence": decision.confidence,
        "source": (
            {
                "title": source.title,
                "url": str(source.url) if source.url else None,
                "doi": source.doi,
                "pmid": source.pmid,
                "year": source.year,
            }
            if source is not None
            else None
        ),
    }


def deterministic_rejection_decision(
    *,
    claim: ValidatedIngredientClaim,
    issues: list[str],
) -> IngredientClaimAuditDecision:
    artifact_issues = {
        "no_grounded_quote",
        "missing_proposed_claim",
        "missing_source_metadata",
    }
    report_fit_issues = {
        "animal_or_in_vitro_used_as_human",
        "topical_or_oral_hygiene_evidence",
        "missing_oral_exposure",
    }
    return IngredientClaimAuditDecision(
        proposed_ingredient_claim_id=claim.proposed_ingredient_claim_id,
        citation_id=claim.citation_id,
        verdict="reject_for_synthesis",
        paper_support_check=(
            "grounding_or_artifact_problem"
            if any(issue in artifact_issues for issue in issues)
            else "validated_and_quote_grounded"
        ),
        report_fit_check=(
            "not_suitable"
            if any(issue in report_fit_issues for issue in issues)
            else "needs_review"
        ),
        modern_evidence_check="not_assessed",
        issue_categories=issues,
        reasoning="Rejected by deterministic ingredient audit checks: " + ", ".join(issues),
        paper_support_reasoning=(
            "The validator artifact is missing required quote/proposed/source support."
            if any(issue in artifact_issues for issue in issues)
            else "The validator artifact reports accepted quote-grounded support."
        ),
        report_fit_reasoning="Deterministic checks found a route, evidence-type, or artifact mismatch.",
        modern_evidence_reasoning="No model-knowledge assessment was performed for this deterministic rejection.",
        confidence=0.95,
    )


def pass_decision(
    *,
    claim: ValidatedIngredientClaim,
    stage: str,
) -> IngredientClaimAuditDecision:
    return IngredientClaimAuditDecision(
        proposed_ingredient_claim_id=claim.proposed_ingredient_claim_id,
        citation_id=claim.citation_id,
        verdict="pass",
        paper_support_check="validated_and_quote_grounded",
        report_fit_check="suitable_with_caveats",
        modern_evidence_check="not_assessed",
        issue_categories=[],
        reasoning=f"Passed {stage} ingredient audit checks.",
        paper_support_reasoning="The validator artifact reports accepted quote-grounded support.",
        report_fit_reasoning="No deterministic report-fit problems were found.",
        modern_evidence_reasoning=(
            "The heuristic provider cannot assess broader modern evidence from model knowledge."
        ),
        confidence=0.8,
    )


def audit_source_claims(
    *,
    packet: IngredientPacket,
    source: CandidateCitation,
    items: list[tuple[ValidatedIngredientClaim, ProposedIngredientClaim]],
    provider: LlmProvider,
) -> list[IngredientClaimAuditDecision]:
    if isinstance(provider, HeuristicProvider):
        return [pass_decision(claim=claim, stage="heuristic") for claim, _ in items]
    response = provider.complete_json(
        audit_prompt(packet=packet, source=source, items=items),
        INGREDIENT_CLAIM_AUDIT_RESPONSE_ADAPTER,
    )
    return response.decisions


def _load_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def write_audit_summary(
    path: Path,
    *,
    accepted_count: int,
    audited_count: int,
    rejected_count: int,
    findings: list[dict[str, Any]],
) -> None:
    verdict_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    paper_support_counts: dict[str, int] = {}
    report_fit_counts: dict[str, int] = {}
    modern_evidence_counts: dict[str, int] = {}
    for finding in findings:
        verdict = str(finding.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        paper_support = str(finding.get("paper_support_check") or "unknown")
        paper_support_counts[paper_support] = paper_support_counts.get(paper_support, 0) + 1
        report_fit = str(finding.get("report_fit_check") or "unknown")
        report_fit_counts[report_fit] = report_fit_counts.get(report_fit, 0) + 1
        modern_evidence = str(finding.get("modern_evidence_check") or "unknown")
        modern_evidence_counts[modern_evidence] = (
            modern_evidence_counts.get(modern_evidence, 0) + 1
        )
        for issue in finding.get("issue_categories") or []:
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1

    lines = [
        "# Ingredient Claim Audit Summary",
        "",
        f"- Accepted validation claims reviewed: {accepted_count}",
        f"- Passed for synthesis: {audited_count}",
        f"- Excluded from synthesis: {rejected_count}",
        "",
        "## Verdict Counts",
        "",
    ]
    if verdict_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(verdict_counts.items()))
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Paper Support Checks", ""])
    if paper_support_counts:
        lines.extend(
            f"- {key}: {value}" for key, value in sorted(paper_support_counts.items())
        )
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Report Fit Checks", ""])
    if report_fit_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(report_fit_counts.items()))
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Modern Evidence Checks", ""])
    if modern_evidence_counts:
        lines.extend(
            f"- {key}: {value}" for key, value in sorted(modern_evidence_counts.items())
        )
    else:
        lines.append("- none: 0")
    lines.extend(["", "## Issue Counts", ""])
    if issue_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(issue_counts.items()))
    else:
        lines.append("- none: 0")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def append_audit_event(run_dir: Path, event: dict[str, Any]) -> None:
    path = run_dir / "audit_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(orjson.dumps(event).decode("utf-8") + "\n")


def audit_ingredient_claims(
    *,
    run_dir: Path,
    provider: LlmProvider,
    workers: int = 1,
    progress: Callable[[int, int, CandidateCitation, int], None] | None = None,
) -> IngredientClaimAuditResult:
    packet = INGREDIENT_PACKET_ADAPTER.validate_python(_load_json(run_dir / "ingredient_packet.json"))
    proposed_claims = PROPOSED_INGREDIENT_ADAPTER.validate_python(
        _load_json(run_dir / "proposed_ingredient_claims.json")
    )
    validated_claims = VALIDATED_INGREDIENT_ADAPTER.validate_python(
        _load_json(run_dir / "validated_ingredient_claims.json")
    )
    sources = SOURCES_ADAPTER.validate_python(_load_json(run_dir / "sources.json"))

    accepted_claims = [claim for claim in validated_claims if claim.accepted]
    if not accepted_claims:
        return IngredientClaimAuditResult(
            audited_claims=[],
            rejected_claims=[],
            findings=[],
            skipped=True,
            skip_reason="no_accepted_claims",
        )

    proposed_by_key = {claim_key(claim.citation_id, claim.id): claim for claim in proposed_claims}
    sources_by_id = {source.id: source for source in sources}
    audited: list[ValidatedIngredientClaim] = []
    rejected: list[ValidatedIngredientClaim] = []
    findings: list[dict[str, Any]] = []
    pending_by_source: dict[str, list[tuple[ValidatedIngredientClaim, ProposedIngredientClaim]]] = {}

    for claim in accepted_claims:
        proposed = proposed_by_key.get(
            claim_key(claim.citation_id, claim.proposed_ingredient_claim_id)
        )
        source = sources_by_id.get(claim.citation_id)
        issues = deterministic_issue_categories(claim=claim, proposed=proposed, source=source)
        if issues:
            decision = deterministic_rejection_decision(claim=claim, issues=issues)
            rejected.append(claim)
            findings.append(finding_from_decision(decision=decision, stage="deterministic", source=source))
            continue
        if proposed is None or source is None:
            continue
        pending_by_source.setdefault(claim.citation_id, []).append((claim, proposed))

    source_items = [
        (sources_by_id[citation_id], items)
        for citation_id, items in pending_by_source.items()
        if citation_id in sources_by_id
    ]
    decisions_by_key: dict[tuple[str, str], IngredientClaimAuditDecision] = {}

    if workers == 1 or len(source_items) <= 1:
        for index, (source, items) in enumerate(source_items, start=1):
            if progress:
                progress(index, len(source_items), source, len(items))
            try:
                decisions = audit_source_claims(
                    packet=packet,
                    source=source,
                    items=items,
                    provider=provider,
                )
            except Exception as exc:
                decisions = [
                    IngredientClaimAuditDecision(
                        proposed_ingredient_claim_id=claim.proposed_ingredient_claim_id,
                        citation_id=claim.citation_id,
                        verdict="needs_review",
                        paper_support_check="validated_and_quote_grounded",
                        report_fit_check="needs_review",
                        modern_evidence_check="not_assessed",
                        issue_categories=["audit_provider_error"],
                        reasoning=f"{type(exc).__name__}: {str(exc) or type(exc).__name__}",
                        paper_support_reasoning=(
                            "The validator artifact reports accepted quote-grounded support."
                        ),
                        report_fit_reasoning=(
                            "Report suitability could not be assessed because the audit provider failed."
                        ),
                        modern_evidence_reasoning=(
                            "Model-knowledge assessment could not be completed because the audit provider failed."
                        ),
                        confidence=0.0,
                    )
                    for claim, _ in items
                ]
            for decision in decisions:
                decisions_by_key[claim_key(decision.citation_id, decision.proposed_ingredient_claim_id)] = decision
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    audit_source_claims,
                    packet=packet,
                    source=source,
                    items=items,
                    provider=provider,
                ): (source, items)
                for source, items in source_items
            }
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                source, items = futures[future]
                completed += 1
                if progress:
                    progress(completed, total, source, len(items))
                try:
                    decisions = future.result()
                except Exception as exc:
                    decisions = [
                        IngredientClaimAuditDecision(
                            proposed_ingredient_claim_id=claim.proposed_ingredient_claim_id,
                            citation_id=claim.citation_id,
                            verdict="needs_review",
                            paper_support_check="validated_and_quote_grounded",
                            report_fit_check="needs_review",
                            modern_evidence_check="not_assessed",
                            issue_categories=["audit_provider_error"],
                            reasoning=f"{type(exc).__name__}: {str(exc) or type(exc).__name__}",
                            paper_support_reasoning=(
                                "The validator artifact reports accepted quote-grounded support."
                            ),
                            report_fit_reasoning=(
                                "Report suitability could not be assessed because the audit provider failed."
                            ),
                            modern_evidence_reasoning=(
                                "Model-knowledge assessment could not be completed because the audit provider failed."
                            ),
                            confidence=0.0,
                        )
                        for claim, _ in items
                    ]
                for decision in decisions:
                    decisions_by_key[
                        claim_key(decision.citation_id, decision.proposed_ingredient_claim_id)
                    ] = decision

    for source, items in source_items:
        for claim, _ in items:
            key = claim_key(claim.citation_id, claim.proposed_ingredient_claim_id)
            decision = decisions_by_key.get(key)
            if decision is None:
                decision = IngredientClaimAuditDecision(
                    proposed_ingredient_claim_id=claim.proposed_ingredient_claim_id,
                    citation_id=claim.citation_id,
                    verdict="needs_review",
                    paper_support_check="validated_and_quote_grounded",
                    report_fit_check="needs_review",
                    modern_evidence_check="not_assessed",
                    issue_categories=["missing_audit_decision"],
                    reasoning="Audit provider did not return a decision for this claim.",
                    paper_support_reasoning=(
                        "The validator artifact reports accepted quote-grounded support."
                    ),
                    report_fit_reasoning="Report suitability was not assessed.",
                    modern_evidence_reasoning="Model-knowledge assessment was not completed.",
                    confidence=0.0,
                )
            findings.append(
                finding_from_decision(decision=decision, stage="llm", source=source)
            )
            if decision.verdict == "pass":
                audited.append(claim)
            else:
                rejected.append(claim)

    write_json(
        run_dir / "audited_ingredient_claims.json",
        [claim.model_dump(mode="json") for claim in audited],
    )
    write_json(
        run_dir / "rejected_audited_ingredient_claims.json",
        [claim.model_dump(mode="json") for claim in rejected],
    )
    write_json(run_dir / "ingredient_claim_audit_findings.json", findings)
    write_audit_summary(
        run_dir / "ingredient_claim_audit_summary.md",
        accepted_count=len(accepted_claims),
        audited_count=len(audited),
        rejected_count=len(rejected),
        findings=findings,
    )
    append_audit_event(
        run_dir,
        {
            "event": "ingredient_claim_audit_completed",
            "timestamp": datetime.now(UTC).isoformat(),
            "accepted_validation_claims": len(accepted_claims),
            "audited_claims": len(audited),
            "rejected_claims": len(rejected),
        },
    )
    return IngredientClaimAuditResult(
        audited_claims=audited,
        rejected_claims=rejected,
        findings=findings,
    )
