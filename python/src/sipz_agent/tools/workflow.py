from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import orjson
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from sipz_agent.core.claim_proposal import (
    body_text_for_record,
    build_claim_extraction_context,
    propose_claims_for_source,
    write_proposed_claims_markdown,
)
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.models import LlmProvider, create_llm_provider
from sipz_agent.core.validation import (
    BodySanitizationError,
    sanitize_paper_body,
    validate_claims_for_paper,
    validation_failure_code,
    write_sanitized_body_preview,
)
from sipz_agent.core.quote_grounding import ground_quote
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, ValidatedClaim
from sipz_agent.schemas.raw_texts import FullTextRetrievalAttempt, RawTextRecord
from sipz_agent.tools.progress import emit_progress


PROPOSED_ADAPTER = TypeAdapter(list[ProposedClaim])
VALIDATED_ADAPTER = TypeAdapter(list[ValidatedClaim])
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])


def _provider(provider: str | None, model: str | None) -> tuple[LlmProvider, str, str]:
    config = resolve_model_config(
        provider=(
            provider
            or os.getenv("WORKER_MODEL_PROVIDER")
            or os.getenv("RESEARCH_MODEL_PROVIDER")
        ),
        model=model or os.getenv("WORKER_MODEL_ID") or os.getenv("RESEARCH_MODEL_ID"),
    )
    if config.provider == "heuristic":
        raise RuntimeError("workflow_tool_requires_llm_provider")
    return create_llm_provider(config), config.provider, config.model_name


def _require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError("missing_required_artifacts:" + ",".join(missing))


def _workspace_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "agent-home").is_dir() and (root / "workspace").is_dir():
        return root / "workspace"
    return root


class ExtractClaimsInput(BaseModel):
    substance: str = Field(min_length=1)
    sources_path: Path
    raw_texts_manifest_path: Path
    raw_texts_dir: Path
    output_dir: Path
    provider: str | None = None
    model: str | None = None
    resume: bool = True
    max_workers: int = Field(default=5, ge=1, le=10)


class ExtractClaimsOutput(BaseModel):
    substance: str
    provider: str
    model: str
    source_count: int
    usable_paper_count: int
    proposed_claim_count: int
    failure_count: int
    completed_paper_count: int
    zero_claim_paper_count: int
    pending_paper_count: int
    proposed_claims_path: str
    failures_path: str
    status_path: str
    contexts_path: str


def extract_claims(payload: ExtractClaimsInput) -> ExtractClaimsOutput:
    _require(payload.sources_path, payload.raw_texts_manifest_path, payload.raw_texts_dir)
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    claims_path = payload.output_dir / "proposed_claims.json"
    failures_path = payload.output_dir / "extraction_failures.json"
    status_path = payload.output_dir / "extraction_status.json"
    contexts_path = payload.output_dir / "extraction_contexts.json"
    provider, provider_name, model_name = _provider(payload.provider, payload.model)
    sources = SOURCES_ADAPTER.validate_json(payload.sources_path.read_bytes())
    raw_records = RAW_TEXTS_ADAPTER.validate_json(payload.raw_texts_manifest_path.read_bytes())
    usable = sum(record.status == "full_text_found" for record in raw_records)
    claims = (
        PROPOSED_ADAPTER.validate_json(claims_path.read_bytes())
        if payload.resume and claims_path.exists()
        else []
    )
    statuses: dict[str, dict[str, Any]] = {}
    if payload.resume and status_path.exists():
        statuses = json.loads(status_path.read_text(encoding="utf-8"))
    contexts: dict[str, dict[str, Any]] = {}
    if payload.resume and contexts_path.exists():
        contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    completed_ids = {
        source_id for source_id, status in statuses.items() if status.get("status") == "completed"
    }
    citations = {source.id: source for source in sources}
    eligible = [record for record in raw_records if record.status == "full_text_found"]
    pending = [record for record in eligible if record.source_id not in completed_ids]

    def extract_one(record: RawTextRecord):
        citation = citations.get(record.source_id)
        if citation is None:
            raise ValueError("citation_not_found")
        body_text = body_text_for_record(payload.raw_texts_dir, record)
        if not body_text:
            raise ValueError("full_text_not_found")
        context, context_metadata = build_claim_extraction_context(body_text)
        context_metadata.update(
            {
                "citation_id": record.source_id,
                "text_path": record.text_path,
                "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            }
        )
        paper_claims = propose_claims_for_source(
            nutrient_name=payload.substance.strip(),
            citation=citation,
            body_text=body_text,
            provider=provider,
        )
        return record, paper_claims, context_metadata

    def persist() -> None:
        claims.sort(key=lambda claim: (claim.citation_id, claim.id))
        failures = [status for status in statuses.values() if status.get("status") == "failed"]
        claims_path.write_bytes(PROPOSED_ADAPTER.dump_json(claims, indent=2))
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        contexts_path.write_text(json.dumps(contexts, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=min(payload.max_workers, len(pending) or 1)) as executor:
        futures = {executor.submit(extract_one, record): record for record in pending}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                _, paper_claims, context_metadata = future.result()
                claims = [claim for claim in claims if claim.citation_id != record.source_id]
                claims.extend(paper_claims)
                contexts[record.source_id] = context_metadata
                statuses[record.source_id] = {
                    "status": "completed",
                    "claim_count": len(paper_claims),
                    "zero_claims": len(paper_claims) == 0,
                }
            except Exception as exc:
                failure = {
                    "citation_id": record.source_id,
                    "failure_code": "claim_extraction_failed",
                    "error": str(exc) or type(exc).__name__,
                }
                statuses[record.source_id] = {"status": "failed", **failure}
            persist()
            citation = citations.get(record.source_id)
            status = statuses[record.source_id]
            emit_progress(
                f"Claims paper {completed_index}/{len(pending)}: {status['status']} - "
                f"{citation.title if citation else record.source_id}",
                stage="claim_extraction", current=completed_index, total=len(pending),
                title=citation.title if citation else record.source_id, status=status["status"],
                claim_count=status.get("claim_count", 0),
            )
    persist()
    failures = [status for status in statuses.values() if status.get("status") == "failed"]
    write_proposed_claims_markdown(payload.output_dir / "proposed_claims.md", claims)
    completed = [status for status in statuses.values() if status.get("status") == "completed"]

    return ExtractClaimsOutput(
        substance=payload.substance.strip(),
        provider=provider_name,
        model=model_name,
        source_count=len(sources),
        usable_paper_count=usable,
        proposed_claim_count=len(claims),
        failure_count=len(failures),
        completed_paper_count=len(completed),
        zero_claim_paper_count=sum(bool(status.get("zero_claims")) for status in completed),
        pending_paper_count=max(0, len(eligible) - len(completed) - len(failures)),
        proposed_claims_path=str(claims_path.resolve()),
        failures_path=str(failures_path.resolve()),
        status_path=str(status_path.resolve()),
        contexts_path=str(contexts_path.resolve()),
    )


class ValidateClaimsInput(BaseModel):
    proposed_claims_path: Path
    sources_path: Path
    raw_texts_manifest_path: Path
    raw_texts_dir: Path
    output_dir: Path
    provider: str | None = None
    model: str | None = None
    max_body_chars: int = Field(default=750_000, ge=1)
    resume: bool = True
    retry_failed: bool = False
    max_workers: int = Field(default=5, ge=1, le=10)
    max_attempts: int = Field(default=3, ge=1, le=3)


class ValidateClaimsOutput(BaseModel):
    provider: str
    model: str
    proposed_claim_count: int
    accepted_count: int
    rejected_count: int
    failure_count: int
    completed_claim_count: int
    pending_claim_count: int
    held_claim_count: int
    adequate_paper_count: int
    limited_paper_count: int
    inadequate_paper_count: int
    assessment_failed_paper_count: int
    validated_claims_path: str
    rejected_claims_path: str
    failures_path: str
    status_path: str
    contexts_path: str
    audit_path: str
    adequacy_path: str
    held_claims_path: str
    retrieval_queue_path: str


BodyAdequacyStatus = Literal["adequate", "limited", "inadequate", "assessment_failed"]


class BodyAdequacyExcerpt(BaseModel):
    quote: str = Field(min_length=1, max_length=800)
    section: str | None = None
    reason: str = Field(min_length=1, max_length=500)


class BodyAdequacyCoverage(BaseModel):
    human_population: bool = False
    oral_exposure: bool = False
    intervention_or_exposure: bool = False
    study_design: bool = False
    comparator_when_applicable: bool = False
    substantive_results: bool = False
    dose_or_duration: bool = False
    limitations_or_uncertainty: bool = False


class BodyAdequacyResponse(BaseModel):
    status: Literal["adequate", "limited", "inadequate"]
    coverage: BodyAdequacyCoverage
    reason_codes: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=1200)
    diagnostic_excerpts: list[BodyAdequacyExcerpt] = Field(default_factory=list)


BODY_ADEQUACY_ADAPTER = TypeAdapter(BodyAdequacyResponse)


def _body_adequacy_prompt(*, citation_id: str, sanitized_body: str) -> str:
    return f"""You assess whether sanitized paper body text is sufficient for reliable claim validation.

Judge text sufficiency only. Do not decide whether any health claim is true. The title, abstract,
conclusion, and references have been excluded where possible.

Statuses:
- adequate: enough substantive body detail to assess human population, oral exposure,
  intervention/form, study design or evidence source, relevant results, and uncertainty.
- limited: substantive results are present, but missing methods, population, exposure, comparator,
  or uncertainty prevents confident validation.
- inadequate: abstract/page-summary/citation/navigation-like content or no substantive study result.

Reviews may be adequate without an RCT-style Methods section when they provide enough detail about
the underlying human oral evidence, results, and limitations. Do not use character count alone.
For adequate, include at least two short exact excerpts copied from the body that demonstrate study
context and substantive results. For limited or inadequate, quote available diagnostic text when useful.

Return JSON only:
{{"status":"adequate|limited|inadequate","coverage":{{"human_population":true,"oral_exposure":true,"intervention_or_exposure":true,"study_design":true,"comparator_when_applicable":true,"substantive_results":true,"dose_or_duration":true,"limitations_or_uncertainty":true}},"reason_codes":["..."],"reasoning":"...","diagnostic_excerpts":[{{"quote":"exact body text","section":"Methods|Results|Discussion|other","reason":"..."}}]}}

Citation ID: {citation_id}
Sanitized body:
{sanitized_body}
"""


def _validation_key(citation_id: str, claim_id: str) -> str:
    return f"{citation_id}::{claim_id}"


class ValidationAttemptsExhausted(RuntimeError):
    def __init__(self, cause: Exception, attempts: int) -> None:
        super().__init__(str(cause) or type(cause).__name__)
        self.cause = cause
        self.attempts = attempts


def _retryable_validation_exception(exc: Exception) -> bool:
    if isinstance(exc, (ValidationError, json.JSONDecodeError, TimeoutError, ConnectionError)):
        return True
    status_code = getattr(exc, "code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in [
            "invalid json",
            "invalid_json",
            "invalid response json",
            "invalid_response_json",
            "eof while parsing",
            "truncated",
            "timed out",
            "timeout",
            "rate limit",
            "temporarily unavailable",
            "connection reset",
            "body_adequacy_quotes_not_grounded",
        ]
    )


def _validation_markdown(path: Path, title: str, claims: list[ValidatedClaim]) -> None:
    lines = [f"# {title}", ""]
    for claim in sorted(claims, key=lambda item: (item.citation_id, item.proposed_claim_id)):
        lines.extend(
            [
                f"## {claim.proposed_claim_id}",
                "",
                claim.validated_statement or "No validated statement.",
                "",
                f"- Source: {claim.citation_id}",
                f"- Verdict: {claim.verdict}",
                f"- Support: {claim.support_level}",
                f"- Grounded quotes: {sum(q.match_status != 'not_found' for q in claim.supporting_quotes)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_claims(payload: ValidateClaimsInput) -> ValidateClaimsOutput:
    _require(
        payload.proposed_claims_path,
        payload.sources_path,
        payload.raw_texts_manifest_path,
        payload.raw_texts_dir,
    )
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    provider, provider_name, model_name = _provider(payload.provider, payload.model)
    proposed = PROPOSED_ADAPTER.validate_json(payload.proposed_claims_path.read_bytes())
    sources = SOURCES_ADAPTER.validate_json(payload.sources_path.read_bytes())
    records = RAW_TEXTS_ADAPTER.validate_json(payload.raw_texts_manifest_path.read_bytes())
    accepted_path = payload.output_dir / "validated_claims.json"
    rejected_path = payload.output_dir / "rejected_claims.json"
    failures_path = payload.output_dir / "validation_failures.json"
    status_path = payload.output_dir / "validation_status.json"
    contexts_path = payload.output_dir / "validation_contexts.json"
    audit_path = payload.output_dir / "validation_audit.jsonl"
    adequacy_path = payload.output_dir / "body_adequacy.json"
    held_claims_path = payload.output_dir / "held_claims.json"
    retrieval_queue_path = payload.output_dir / "body_retrieval_queue.json"
    accepted = VALIDATED_ADAPTER.validate_json(accepted_path.read_bytes()) if payload.resume and accepted_path.exists() else []
    rejected = VALIDATED_ADAPTER.validate_json(rejected_path.read_bytes()) if payload.resume and rejected_path.exists() else []
    statuses: dict[str, dict[str, Any]] = json.loads(status_path.read_text()) if payload.resume and status_path.exists() else {}
    contexts: dict[str, dict[str, Any]] = json.loads(contexts_path.read_text()) if payload.resume and contexts_path.exists() else {}
    adequacy: dict[str, dict[str, Any]] = json.loads(adequacy_path.read_text()) if payload.resume and adequacy_path.exists() else {}
    failures: list[dict[str, Any]] = json.loads(failures_path.read_text()) if payload.resume and failures_path.exists() else []
    proposed_keys = {_validation_key(claim.citation_id, claim.id) for claim in proposed}
    accepted = [item for item in accepted if _validation_key(item.citation_id, item.proposed_claim_id) in proposed_keys]
    rejected = [item for item in rejected if _validation_key(item.citation_id, item.proposed_claim_id) in proposed_keys]
    for item in accepted + rejected:
        key = _validation_key(item.citation_id, item.proposed_claim_id)
        statuses.setdefault(
            key,
            {
                "status": "accepted" if item.accepted else "rejected",
                "citation_id": item.citation_id,
                "proposed_claim_id": item.proposed_claim_id,
                "verdict": item.verdict,
                "provider": "legacy_artifact",
                "model": "unknown",
            },
        )
    citations = {source.id: source for source in sources}
    records_by_id = {record.source_id: record for record in records}
    sanitized_by_source: dict[str, str] = {}

    def persist() -> None:
        accepted.sort(key=lambda item: (item.citation_id, item.proposed_claim_id))
        rejected.sort(key=lambda item: (item.citation_id, item.proposed_claim_id))
        active_failures = [value for value in statuses.values() if value.get("status") == "failed"]
        accepted_path.write_bytes(VALIDATED_ADAPTER.dump_json(accepted, indent=2))
        rejected_path.write_bytes(VALIDATED_ADAPTER.dump_json(rejected, indent=2))
        failures_path.write_text(json.dumps(active_failures, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        contexts_path.write_text(json.dumps(contexts, indent=2), encoding="utf-8")
        held_claims = []
        for claim in proposed:
            state = statuses.get(_validation_key(claim.citation_id, claim.id), {})
            if str(state.get("status", "")).startswith("held_"):
                held_claims.append({"claim": claim.model_dump(mode="json"), **state})
        queue = []
        for citation_id, assessment in adequacy.items():
            if assessment.get("status") not in {"limited", "inadequate", "assessment_failed"}:
                continue
            citation = citations.get(citation_id)
            record = records_by_id.get(citation_id)
            queue.append(
                {
                    "citation_id": citation_id,
                    "title": citation.title if citation else None,
                    "doi": citation.doi if citation else None,
                    "pmid": citation.pmid if citation else None,
                    "url": str(citation.url) if citation and citation.url else None,
                    "current_status": assessment.get("status"),
                    "reason_codes": assessment.get("reason_codes", []),
                    "retrieval_method": record.retrieval_method if record else None,
                    "text_char_count": record.text_char_count if record else 0,
                    "attempted_urls": record.attempted_urls if record else [],
                }
            )
        adequacy_path.write_text(json.dumps(adequacy, indent=2), encoding="utf-8")
        held_claims_path.write_text(json.dumps(held_claims, indent=2), encoding="utf-8")
        retrieval_queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
        counts = {status: sum(item.get("status") == status for item in adequacy.values()) for status in ["adequate", "limited", "inadequate", "assessment_failed"]}
        (payload.output_dir / "body_adequacy_summary.md").write_text(
            "# Body Adequacy Summary\n\n"
            + "\n".join(f"- {status.replace('_', ' ').title()}: {count}" for status, count in counts.items())
            + f"\n- Held claims: {len(held_claims)}\n",
            encoding="utf-8",
        )
        _validation_markdown(payload.output_dir / "validated_claims.md", "Validated Claims", accepted)
        _validation_markdown(payload.output_dir / "rejected_claims.md", "Rejected Claims", rejected)

    def audit(event: str, **details: Any) -> None:
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "event": event, **details}) + "\n")

    def sanitized_context(claim: ProposedClaim) -> str:
        if claim.citation_id in sanitized_by_source:
            return sanitized_by_source[claim.citation_id]
        citation = citations.get(claim.citation_id)
        record = records_by_id.get(claim.citation_id)
        if citation is None:
            raise ValueError("citation_not_found")
        if record is None:
            raise ValueError("raw_text_record_not_found")
        body = body_text_for_record(payload.raw_texts_dir, record)
        if not body:
            raise ValueError("full_text_not_found")
        sanitized = sanitize_paper_body(body_text=body, citation=citation)
        if len(sanitized) > payload.max_body_chars:
            raise ValueError(f"paper_exceeds_validation_context:{len(sanitized)}>{payload.max_body_chars}")
        sanitized_by_source[claim.citation_id] = sanitized
        contexts[claim.citation_id] = {
            "citation_id": claim.citation_id,
            "text_path": record.text_path,
            "original_char_count": len(body),
            "supplied_char_count": len(sanitized),
            "excluded_sections": ["title", "abstract", "conclusion", "references"],
            "context_sha256": hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
        }
        return sanitized

    def assess_source(citation_id: str, claims: list[ProposedClaim]) -> tuple[str, dict[str, Any]]:
        try:
            sanitized = sanitized_context(claims[0])
        except Exception as exc:
            return citation_id, {
                "status": "inadequate",
                "coverage": BodyAdequacyCoverage().model_dump(mode="json"),
                "reason_codes": [validation_failure_code(exc)],
                "reasoning": str(exc) or type(exc).__name__,
                "diagnostic_excerpts": [],
                "attempts": 0,
                "provider": provider_name,
                "model": model_name,
                "context_sha256": None,
                "assessed_at": datetime.now(UTC).isoformat(),
            }
        last_error: Exception | None = None
        for attempt in range(1, payload.max_attempts + 1):
            try:
                response = provider.complete_json(
                    _body_adequacy_prompt(citation_id=citation_id, sanitized_body=sanitized),
                    BODY_ADEQUACY_ADAPTER,
                )
                grounded_excerpts = []
                for excerpt in response.diagnostic_excerpts:
                    match = ground_quote(excerpt.quote, sanitized)
                    grounded_excerpts.append(
                        {
                            **excerpt.model_dump(mode="json"),
                            "match_status": match.match_status,
                        }
                    )
                grounded_count = sum(item["match_status"] != "not_found" for item in grounded_excerpts)
                reason_codes = list(response.reason_codes)
                if response.diagnostic_excerpts and grounded_count < len(response.diagnostic_excerpts):
                    reason_codes.append("diagnostic_excerpt_not_grounded")
                return citation_id, {
                    **response.model_dump(mode="json", exclude={"diagnostic_excerpts", "reason_codes"}),
                    "reason_codes": reason_codes,
                    "diagnostic_excerpts": grounded_excerpts,
                    "attempts": attempt,
                    "provider": provider_name,
                    "model": model_name,
                    "context_sha256": contexts[citation_id]["context_sha256"],
                    "assessed_at": datetime.now(UTC).isoformat(),
                }
            except Exception as exc:
                last_error = exc
                if attempt == payload.max_attempts or not _retryable_validation_exception(exc):
                    break
        assert last_error is not None
        return citation_id, {
            "status": "assessment_failed",
            "coverage": BodyAdequacyCoverage().model_dump(mode="json"),
            "reason_codes": [validation_failure_code(last_error)],
            "reasoning": str(last_error) or type(last_error).__name__,
            "diagnostic_excerpts": [],
            "attempts": attempt,
            "provider": provider_name,
            "model": model_name,
            "context_sha256": contexts.get(citation_id, {}).get("context_sha256"),
            "assessed_at": datetime.now(UTC).isoformat(),
        }

    completed = {
        key for key, value in statuses.items() if value.get("status") in {"accepted", "rejected"}
    }
    pending = []
    for claim in proposed:
        key = _validation_key(claim.citation_id, claim.id)
        status = statuses.get(key, {}).get("status")
        if key in completed:
            continue
        if status == "failed" and not payload.retry_failed:
            continue
        pending.append(claim)

    pending_by_source: dict[str, list[ProposedClaim]] = {}
    for claim in pending:
        pending_by_source.setdefault(claim.citation_id, []).append(claim)
    sources_to_assess = {
        citation_id: claims
        for citation_id, claims in pending_by_source.items()
        if citation_id not in adequacy
        or (adequacy[citation_id].get("status") == "assessment_failed" and payload.retry_failed)
    }
    audit("body_adequacy_started", papers=len(sources_to_assess), max_workers=payload.max_workers)
    with ThreadPoolExecutor(max_workers=min(payload.max_workers, len(sources_to_assess) or 1)) as executor:
        futures = {
            executor.submit(assess_source, citation_id, claims): citation_id
            for citation_id, claims in sources_to_assess.items()
        }
        for completed_index, future in enumerate(as_completed(futures), start=1):
            citation_id, assessment = future.result()
            adequacy[citation_id] = assessment
            contexts.setdefault(citation_id, {})["body_adequacy_status"] = assessment["status"]
            audit("body_adequacy_completed", citation_id=citation_id, status=assessment["status"], attempts=assessment["attempts"])
            persist()
            citation = citations.get(citation_id)
            emit_progress(
                f"Body adequacy {completed_index}/{len(sources_to_assess)}: {assessment['status']} - "
                f"{citation.title if citation else citation_id}",
                stage="body_adequacy", current=completed_index, total=len(sources_to_assess),
                title=citation.title if citation else citation_id, status=assessment["status"],
            )

    eligible_pending: list[ProposedClaim] = []
    for claim in pending:
        assessment_status = adequacy.get(claim.citation_id, {}).get("status", "assessment_failed")
        if assessment_status == "adequate":
            eligible_pending.append(claim)
            continue
        key = _validation_key(claim.citation_id, claim.id)
        statuses[key] = {
            "status": "held_assessment_failed" if assessment_status == "assessment_failed" else "held_insufficient_body",
            "citation_id": claim.citation_id,
            "proposed_claim_id": claim.id,
            "body_adequacy_status": assessment_status,
            "reason_codes": adequacy.get(claim.citation_id, {}).get("reason_codes", []),
            "completed_at": datetime.now(UTC).isoformat(),
        }
    pending = eligible_pending
    persist()

    def validate_one(claim: ProposedClaim) -> tuple[ProposedClaim, ValidatedClaim, int]:
        citation = citations.get(claim.citation_id)
        if citation is None:
            raise ValueError("citation_not_found")
        sanitized = sanitized_context(claim)
        for attempt in range(1, payload.max_attempts + 1):
            try:
                result = validate_claims_for_paper(
                    nutrient_name=claim.nutrient_name,
                    citation=citation,
                    claims=[claim],
                    sanitized_body=sanitized,
                    provider=provider,
                )[0]
                return claim, result.model_copy(
                    update={"effect_row_id": str(uuid5(NAMESPACE_URL, key_for_claim(claim)))}
                ), attempt
            except Exception as exc:
                if attempt == payload.max_attempts or not _retryable_validation_exception(exc):
                    raise ValidationAttemptsExhausted(exc, attempt) from exc
        raise AssertionError("unreachable_validation_attempt_loop")

    def key_for_claim(claim: ProposedClaim) -> str:
        return f"sipz-validation:{claim.citation_id}:{claim.id}"

    audit("claim_validation_started", claims=len(pending), max_workers=payload.max_workers)
    with ThreadPoolExecutor(max_workers=min(payload.max_workers, len(pending) or 1)) as executor:
        futures = {executor.submit(validate_one, claim): claim for claim in pending}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            claim = futures[future]
            key = _validation_key(claim.citation_id, claim.id)
            try:
                _, result, attempts = future.result()
                accepted = [item for item in accepted if _validation_key(item.citation_id, item.proposed_claim_id) != key]
                rejected = [item for item in rejected if _validation_key(item.citation_id, item.proposed_claim_id) != key]
                (accepted if result.accepted else rejected).append(result)
                statuses[key] = {
                    "status": "accepted" if result.accepted else "rejected",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.id,
                    "verdict": result.verdict,
                    "provider": provider_name,
                    "model": model_name,
                    "attempts": attempts,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
                audit("claim_validation_completed", key=key, verdict=result.verdict, accepted=result.accepted)
            except Exception as exc:
                original_exc = exc.cause if isinstance(exc, ValidationAttemptsExhausted) else exc
                attempts = exc.attempts if isinstance(exc, ValidationAttemptsExhausted) else 1
                failure = {
                    "status": "failed",
                    "citation_id": claim.citation_id,
                    "proposed_claim_id": claim.id,
                    "failure_code": validation_failure_code(original_exc),
                    "error": str(original_exc) or type(original_exc).__name__,
                    "attempts": attempts,
                    "retryable": _retryable_validation_exception(original_exc),
                    "failed_at": datetime.now(UTC).isoformat(),
                }
                if isinstance(original_exc, BodySanitizationError):
                    preview = write_sanitized_body_preview(
                        out_dir=payload.output_dir,
                        citation_id=claim.citation_id,
                        body_text=None,
                        exc=original_exc,
                    )
                    if preview:
                        failure["sanitized_body_preview_path"] = preview
                statuses[key] = failure
                audit("claim_validation_failed", key=key, failure_code=failure["failure_code"])
            persist()
            state = statuses[key]
            emit_progress(
                f"Validated claim {completed_index}/{len(pending)}: {state['status']} - {claim.statement[:100]}",
                stage="claim_validation", current=completed_index, total=len(pending),
                citation_id=claim.citation_id, claim_id=claim.id, status=state["status"],
            )
    persist()
    failures = [value for value in statuses.values() if value.get("status") == "failed"]
    completed_count = sum(value.get("status") in {"accepted", "rejected"} for value in statuses.values())
    held_count = sum(str(value.get("status", "")).startswith("held_") for value in statuses.values())
    adequacy_counts = {
        status: sum(item.get("status") == status for item in adequacy.values())
        for status in ["adequate", "limited", "inadequate", "assessment_failed"]
    }
    return ValidateClaimsOutput(
        provider=provider_name,
        model=model_name,
        proposed_claim_count=len(proposed),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        failure_count=len(failures),
        completed_claim_count=completed_count,
        pending_claim_count=max(0, len(proposed) - completed_count - held_count - len(failures)),
        held_claim_count=held_count,
        adequate_paper_count=adequacy_counts["adequate"],
        limited_paper_count=adequacy_counts["limited"],
        inadequate_paper_count=adequacy_counts["inadequate"],
        assessment_failed_paper_count=adequacy_counts["assessment_failed"],
        validated_claims_path=str(accepted_path.resolve()),
        rejected_claims_path=str(rejected_path.resolve()),
        failures_path=str(failures_path.resolve()),
        status_path=str(status_path.resolve()),
        contexts_path=str(contexts_path.resolve()),
        audit_path=str(audit_path.resolve()),
        adequacy_path=str(adequacy_path.resolve()),
        held_claims_path=str(held_claims_path.resolve()),
        retrieval_queue_path=str(retrieval_queue_path.resolve()),
    )


RefereeVerdict = Literal["pass", "needs_review", "fail"]


class RefereeDecision(BaseModel):
    proposed_claim_id: str
    citation_id: str
    decision: RefereeVerdict
    reason_codes: list[str] = Field(default_factory=list)
    feedback: str
    validation_attempt: int = Field(ge=1, le=3)


class RefereeResponse(BaseModel):
    decision: RefereeVerdict
    reason_codes: list[str] = Field(default_factory=list)
    feedback: str = Field(min_length=1)


REFEREE_ADAPTER = TypeAdapter(RefereeResponse)


class RefereeClaimsInput(BaseModel):
    substance: str = Field(min_length=1)
    proposed_claims_path: Path
    validated_claims_path: Path
    rejected_claims_path: Path
    sources_path: Path
    output_dir: Path
    provider: str | None = None
    model: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=3)
    resume: bool = True


class RefereeClaimsOutput(BaseModel):
    provider: str
    model: str
    input_count: int
    pass_count: int
    needs_review_count: int
    fail_count: int
    decisions_path: str


def _referee_prompt(
    substance: str,
    proposed: ProposedClaim,
    validated: ValidatedClaim,
    citation: CandidateCitation | None,
    previous_feedback: str | None,
) -> str:
    return f"""You are the final referee for an oral-consumption human-health evidence pipeline.
Return JSON exactly as {{"decision":"pass|needs_review|fail","reason_codes":[],"feedback":"..."}}.

Pass only when the validated claim answers the effect of {substance} on human health when orally
consumed, preserves population, dose/form, outcome, certainty and limitations, and has at least one
grounded body-text quote. Fail source mismatch, non-oral exposure, animal/in-vitro-only evidence,
missing grounded quotes, or material overstatement. Use needs_review only for a genuinely resolvable
ambiguity.

Source: {citation.model_dump_json() if citation else 'missing'}
Proposed claim: {proposed.model_dump_json()}
Validator result: {validated.model_dump_json()}
Previous referee feedback: {previous_feedback or 'none'}
"""


def referee_claims(payload: RefereeClaimsInput) -> RefereeClaimsOutput:
    _require(
        payload.proposed_claims_path,
        payload.validated_claims_path,
        payload.rejected_claims_path,
        payload.sources_path,
    )
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    path = payload.output_dir / "referee_decisions.json"
    provider, provider_name, model_name = _provider(payload.provider, payload.model)
    proposed = PROPOSED_ADAPTER.validate_json(payload.proposed_claims_path.read_bytes())
    validated = VALIDATED_ADAPTER.validate_json(payload.validated_claims_path.read_bytes())
    rejected = VALIDATED_ADAPTER.validate_json(payload.rejected_claims_path.read_bytes())
    sources = SOURCES_ADAPTER.validate_json(payload.sources_path.read_bytes())
    proposed_by_id = {claim.id: claim for claim in proposed}
    sources_by_id = {source.id: source for source in sources}

    if payload.resume and path.exists():
        decisions = TypeAdapter(list[RefereeDecision]).validate_json(path.read_bytes())
    else:
        decisions: list[RefereeDecision] = []
        for result in rejected:
            decisions.append(
                RefereeDecision(
                    proposed_claim_id=result.proposed_claim_id,
                    citation_id=result.citation_id,
                    decision="fail",
                    reason_codes=[result.verdict],
                    feedback=result.validator_reasoning or "Validator rejected the claim.",
                    validation_attempt=1,
                )
            )
        for result in validated:
            claim = proposed_by_id.get(result.proposed_claim_id)
            grounded = any(q.match_status != "not_found" for q in result.supporting_quotes)
            if claim is None or not grounded or claim.citation_id != result.citation_id:
                decisions.append(
                    RefereeDecision(
                        proposed_claim_id=result.proposed_claim_id,
                        citation_id=result.citation_id,
                        decision="fail",
                        reason_codes=["missing_claim" if claim is None else "grounding_or_source_mismatch"],
                        feedback="Deterministic referee precondition failed.",
                        validation_attempt=1,
                    )
                )
                continue
            previous: str | None = None
            response: RefereeResponse | None = None
            for attempt in range(1, payload.max_attempts + 1):
                response = provider.complete_json(
                    _referee_prompt(
                        payload.substance, claim, result, sources_by_id.get(result.citation_id), previous
                    ),
                    REFEREE_ADAPTER,
                )
                if response.decision != "needs_review" or attempt == payload.max_attempts:
                    break
                previous = response.feedback
            assert response is not None
            decision: RefereeVerdict = response.decision
            if decision == "needs_review" and attempt >= payload.max_attempts:
                decision = "needs_review"
            decisions.append(
                RefereeDecision(
                    proposed_claim_id=result.proposed_claim_id,
                    citation_id=result.citation_id,
                    decision=decision,
                    reason_codes=response.reason_codes,
                    feedback=response.feedback,
                    validation_attempt=attempt,
                )
            )
        path.write_text(
            TypeAdapter(list[RefereeDecision]).dump_json(decisions, indent=2).decode(),
            encoding="utf-8",
        )

    return RefereeClaimsOutput(
        provider=provider_name,
        model=model_name,
        input_count=len(decisions),
        pass_count=sum(item.decision == "pass" for item in decisions),
        needs_review_count=sum(item.decision == "needs_review" for item in decisions),
        fail_count=sum(item.decision == "fail" for item in decisions),
        decisions_path=str(path.resolve()),
    )


class ExportResearchReportInput(BaseModel):
    substance: str = Field(min_length=1)
    proposed_claims_path: Path
    validated_claims_path: Path
    rejected_claims_path: Path
    sources_path: Path
    output_dir: Path
    held_claims_path: Path | None = None
    body_adequacy_path: Path | None = None
    validation_failures_path: Path | None = None
    retrieval_expansion_path: Path | None = None


class ExportResearchReportOutput(BaseModel):
    accepted_count: int
    rejected_count: int
    held_count: int
    failure_count: int
    source_count: int
    report_dir: str
    markdown_path: str
    json_path: str
    csv_path: str
    manifest_path: str
    terminal_summary_markdown: str
    terminal_table_markdown: str
    report_link_markdown: str
    report_directory_link_markdown: str
    report_location_markdown: str
    terminal_response_markdown: str


def _public_evidence_label(value: str) -> str:
    return {
        "human_systematic_review": "Human systematic review",
        "human_rct": "Human RCT",
        "human_observational": "Human observational study",
        "human_mechanistic": "Human mechanistic study",
        "review_author_interpretation": "Review of human evidence",
        "unsupported": "Unsupported",
    }.get(value, value.replace("_", " ").title())


def _public_source_url(source: CandidateCitation | None) -> str | None:
    if source is None:
        return None
    if source.doi:
        return f"https://doi.org/{source.doi}"
    if source.pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{source.pmid}/"
    return str(source.url) if source.url else None


def _markdown_cell(value: Any) -> str:
    if value is None or value == "":
        return "Not reported"
    return " ".join(str(value).split()).replace("|", "\\|")


PUBLIC_TABLE_HEADERS = [
    "Health effect", "Direction", "Population", "Oral exposure", "Evidence", "Finding", "Status", "Source"
]


def _claims_table_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(PUBLIC_TABLE_HEADERS) + " |",
        "|" + "|".join(["---"] * len(PUBLIC_TABLE_HEADERS)) + "|",
    ]
    for row in rows:
        source = _markdown_cell(row["source_title"])
        if row.get("source_url"):
            source = f"[{source}]({row['source_url']})"
        values = [
            row["health_effect"], row["direction"], row["population"], row["oral_exposure"],
            row["evidence"], row["finding"], row["status"], source,
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    if not rows:
        lines.append("| No validated claims | - | - | - | - | - | - | - |")
    return "\n".join(lines)


def export_research_report(payload: ExportResearchReportInput) -> ExportResearchReportOutput:
    emit_progress("Building public claims report...", stage="report_export")
    _require(
        payload.proposed_claims_path,
        payload.validated_claims_path,
        payload.rejected_claims_path,
        payload.sources_path,
    )
    payload.output_dir.mkdir(parents=True, exist_ok=True)
    proposed = PROPOSED_ADAPTER.validate_json(payload.proposed_claims_path.read_bytes())
    validated = VALIDATED_ADAPTER.validate_json(payload.validated_claims_path.read_bytes())
    validator_rejected = VALIDATED_ADAPTER.validate_json(payload.rejected_claims_path.read_bytes())
    sources = SOURCES_ADAPTER.validate_json(payload.sources_path.read_bytes())
    proposed_by_id = {claim.id: claim for claim in proposed}
    sources_by_id = {source.id: source for source in sources}
    now = datetime.now(UTC).isoformat()
    held = json.loads(payload.held_claims_path.read_text()) if payload.held_claims_path and payload.held_claims_path.exists() else []
    adequacy = json.loads(payload.body_adequacy_path.read_text()) if payload.body_adequacy_path and payload.body_adequacy_path.exists() else {}
    failures = json.loads(payload.validation_failures_path.read_text()) if payload.validation_failures_path and payload.validation_failures_path.exists() else []
    retrieval_coverage = (
        json.loads(payload.retrieval_expansion_path.read_text(encoding="utf-8"))
        if payload.retrieval_expansion_path and payload.retrieval_expansion_path.exists()
        else None
    )
    rows: list[dict[str, Any]] = []
    for result in validated:
        claim = proposed_by_id[result.proposed_claim_id]
        source = sources_by_id.get(result.citation_id)
        rows.append({
            "claim_id": result.proposed_claim_id,
            "citation_id": result.citation_id,
            "health_effect": claim.proposed_effect_label,
            "effect_slug": claim.proposed_effect_slug,
            "direction": claim.direction.title(),
            "population": claim.population or "Not reported",
            "oral_exposure": claim.dose_or_exposure or claim.exposure_category.replace("_", " "),
            "dose_or_exposure": claim.dose_or_exposure,
            "exposure_category": claim.exposure_category,
            "study_type": claim.study_type,
            "evidence": _public_evidence_label(result.support_level),
            "finding": result.validated_statement or claim.statement,
            "status": "Supported with limitations" if result.verdict == "supported_with_limitations" else "Supported",
            "limitations": result.limitations,
            "supporting_quotes": [quote.model_dump(mode="json") for quote in result.supporting_quotes],
            "source_title": source.title if source else result.citation_id,
            "doi": source.doi if source else None,
            "pmid": source.pmid if source else None,
            "source_url": _public_source_url(source),
        })
    rows.sort(key=lambda row: (row["health_effect"].casefold(), row["citation_id"], row["claim_id"]))
    table = _claims_table_markdown(rows)
    reviewed_citation_ids = {claim.citation_id for claim in proposed}
    counts = {
        "retained_sources": len(sources),
        "papers_reviewed_from_full_text": len(reviewed_citation_ids),
        "validated_claims": len(rows),
        "rejected_claims": len(validator_rejected),
        "held_claims": len(held),
        "validation_failures": len(failures),
    }
    summary_counts = (
        f"- Sources retained after screening: {counts['retained_sources']}\n"
        f"- Papers reviewed from usable full text: {counts['papers_reviewed_from_full_text']}\n"
        f"- Validated claims: {counts['validated_claims']}\n"
        f"- Rejected claims: {counts['rejected_claims']}\n"
        f"- Held for better retrieval: {counts['held_claims']}\n"
        f"- Validation failures: {counts['validation_failures']}"
    )
    if retrieval_coverage:
        totals = retrieval_coverage.get("totals", {})
        summary_counts += (
            f"\n- Requested usable full texts: {retrieval_coverage.get('requested_count', 0)}"
            f"\n- Usable full texts retrieved: {totals.get('usable_full_texts', 0)}"
            f"\n- Retrieval target met: {'yes' if retrieval_coverage.get('target_met') else 'no'}"
            f"\n- Retrieval stop reason: {retrieval_coverage.get('stop_reason', 'unknown')}"
        )
    summary = f"## {payload.substance} Human-Health Evidence\n\n{summary_counts}"
    csv_path = payload.output_dir / "claims_report.csv"
    csv_columns = [
        "claim_id", "citation_id", "effect_slug", "health_effect", "direction", "population",
        "oral_exposure", "dose_or_exposure", "exposure_category", "study_type", "evidence",
        "finding", "status", "limitations", "supporting_quotes", "source_title", "doi", "pmid", "source_url",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in csv_columns}
            csv_row["limitations"] = json.dumps(row["limitations"], ensure_ascii=True)
            csv_row["supporting_quotes"] = json.dumps(row["supporting_quotes"], ensure_ascii=True)
            writer.writerow(csv_row)
    json_path = payload.output_dir / "claims_report.json"
    report_payload = {
        "schema_version": "1.0",
        "substance": payload.substance,
        "canonical_question": f"What is the effect of {payload.substance} on human health when orally ingested?",
        "generated_at": now,
        "counts": counts,
        "validated_claims": rows,
        "rejected_claims": [item.model_dump(mode="json") for item in validator_rejected],
        "held_claims": held,
        "body_adequacy": adequacy,
        "validation_failures": failures,
        "retrieval_coverage": retrieval_coverage,
        "sources": [source.model_dump(mode="json") for source in sources],
        "model_provenance": {
            "orchestrator": {
                "provider": os.getenv("ORCHESTRATOR_MODEL_PROVIDER"),
                "model": os.getenv("ORCHESTRATOR_MODEL_ID"),
                "thinking": os.getenv("ORCHESTRATOR_THINKING"),
            },
            "worker": {
                "provider": os.getenv("WORKER_MODEL_PROVIDER") or os.getenv("RESEARCH_MODEL_PROVIDER"),
                "model": os.getenv("WORKER_MODEL_ID") or os.getenv("RESEARCH_MODEL_ID"),
            },
        },
    }
    json_path.write_text(json.dumps(report_payload, indent=2, default=str), encoding="utf-8")
    markdown_path = payload.output_dir / "claims_report.md"
    details = []
    for row in rows:
        limitations = "\n".join(f"- {item}" for item in row["limitations"]) or "- None reported."
        quotes = "\n\n".join(f"> {quote['quote']}" for quote in row["supporting_quotes"]) or "> No quote available."
        source_link = f"[{row['source_title']}]({row['source_url']})" if row.get("source_url") else row["source_title"]
        details.append(
            f"### {row['health_effect']}\n\n{row['finding']}\n\n"
            f"- Direction: {row['direction']}\n- Population: {row['population']}\n"
            f"- Oral exposure: {row['oral_exposure']}\n- Evidence: {row['evidence']}\n"
            f"- Status: {row['status']}\n- Source: {source_link}\n\n"
            f"**Limitations**\n\n{limitations}\n\n**Grounded evidence**\n\n{quotes}"
        )
    rejected_lines = [f"- `{item.proposed_claim_id}` ({item.citation_id}): {item.validated_statement}" for item in validator_rejected] or ["- None."]
    held_lines = [f"- `{item.get('proposed_claim_id')}` ({item.get('citation_id')}): {item.get('body_adequacy_status')}" for item in held] or ["- None."]
    source_lines = [f"- [{source.title}]({_public_source_url(source)})" if _public_source_url(source) else f"- {source.title}" for source in sources]
    details_text = "\n\n".join(details) if details else "No claims were validated."
    rejected_text = "\n".join(rejected_lines)
    held_text = "\n".join(held_lines)
    sources_text = "\n".join(source_lines)
    markdown_path.write_text(
        f"# {payload.substance} Human-Health Evidence\n\n"
        "This report summarizes research evidence and is not medical advice.\n\n"
        f"{summary_counts}\n\n## Validated Claims\n\n{table}\n\n"
        f"## Claim Details\n\n{details_text}\n\n"
        f"## Rejected Claims\n\n{rejected_text}\n\n"
        f"## Held for Better Retrieval\n\n{held_text}\n\n"
        f"## Sources\n\n{sources_text}\n",
        encoding="utf-8",
    )
    manifest_path = payload.output_dir / "report_manifest.json"
    report_link = f"[Open the full claims report]({markdown_path.resolve()})"
    report_directory_link = f"[Open the report folder]({payload.output_dir.resolve()})"
    report_location = (
        f"**Saved report folder:** `{payload.output_dir.resolve()}`\n\n"
        f"**Main report file:** `{markdown_path.resolve()}`"
    )
    terminal_response = (
        f"{summary}\n\n{table}\n\n{report_location}\n\n"
        f"{report_link}\n\n{report_directory_link}\n\n"
        f"**JSON:** `{json_path.resolve()}`\n\n"
        f"**CSV:** `{csv_path.resolve()}`"
    )
    manifest = {
        "schema_version": "1.0", "substance": payload.substance, "generated_at": now,
        "counts": counts,
        "model_provenance": report_payload["model_provenance"],
        "artifacts": {"markdown": str(markdown_path.resolve()), "json": str(json_path.resolve()), "csv": str(csv_path.resolve())},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ExportResearchReportOutput(
        accepted_count=len(rows),
        rejected_count=len(validator_rejected),
        held_count=len(held),
        failure_count=len(failures),
        source_count=len(sources),
        report_dir=str(payload.output_dir.resolve()),
        markdown_path=str(markdown_path.resolve()),
        json_path=str(json_path.resolve()),
        csv_path=str(csv_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        terminal_summary_markdown=summary,
        terminal_table_markdown=table,
        report_link_markdown=report_link,
        report_directory_link_markdown=report_directory_link,
        report_location_markdown=report_location,
        terminal_response_markdown=terminal_response,
    )


class RunResearchPipelineInput(BaseModel):
    substance: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    depth: Literal["light", "standard", "deep"] = "standard"
    provider: str | None = None
    model: str | None = None
    output_root: Path = Path("workspace/runs")
    run_id: str | None = None
    target_count: int | None = Field(default=None, ge=1, le=200)
    max_pages: int = Field(default=5, ge=1, le=20)
    page_size: int = Field(default=10, ge=1, le=50)
    resume: bool = True


class RunResearchPipelineOutput(BaseModel):
    run_id: str
    run_dir: str
    status: Literal["completed", "failed"]
    stage_counts: dict[str, int]
    report_dir: str | None = None
    markdown_path: str | None = None
    json_path: str | None = None
    csv_path: str | None = None
    terminal_summary_markdown: str | None = None
    terminal_table_markdown: str | None = None
    report_link_markdown: str | None = None
    report_directory_link_markdown: str | None = None
    report_location_markdown: str | None = None
    terminal_response_markdown: str | None = None


class InspectResearchStateInput(BaseModel):
    substance: str = Field(min_length=1)
    workspace_root: Path = Path("workspace")


class ResearchStateMatch(BaseModel):
    path: str
    artifact_type: str
    modified_at: str


class InspectResearchStateOutput(BaseModel):
    substance: str
    matches: list[ResearchStateMatch]
    retained_sources_path: str | None = None
    full_text_manifest_path: str | None = None
    full_text_status_counts: dict[str, int] = Field(default_factory=dict)
    verified_full_text_paths: list[str] = Field(default_factory=list)
    proposed_claims_path: str | None = None
    extraction_status_path: str | None = None
    extraction_status_counts: dict[str, int] = Field(default_factory=dict)
    validated_claims_path: str | None = None
    rejected_claims_path: str | None = None
    validation_status_path: str | None = None
    validation_status_counts: dict[str, int] = Field(default_factory=dict)
    validation_failure_count: int = 0
    held_claim_count: int = 0
    limited_body_count: int = 0
    inadequate_body_count: int = 0
    held_claim_count: int = 0
    body_adequacy_status_counts: dict[str, int] = Field(default_factory=dict)
    claims_report_path: str | None = None
    claims_report_json_path: str | None = None
    claims_report_csv_path: str | None = None
    suggested_action: Literal[
        "retrieve_and_screen", "retrieve_full_text", "resume_full_text", "full_text_complete",
        "extract_claims", "resume_claim_extraction", "claim_extraction_complete",
        "validate_claims", "resume_claim_validation", "claim_validation_complete",
        "retrieve_better_body_text", "export_report", "report_complete"
    ]


class AdvanceResearchPipelineInput(BaseModel):
    substance: str = Field(min_length=1)
    target_stage: Literal["screening", "full_text", "claim_extraction", "claim_validation"]
    aliases: list[str] = Field(default_factory=list)
    expansion_queries: list[str] = Field(default_factory=list)
    requested_count: int = Field(default=10, ge=1, le=200)
    depth: Literal["light", "standard", "deep"] = "standard"
    provider: str | None = None
    model: str | None = None
    workspace_root: Path = Path("workspace")
    run_dir_override: Path | None = None
    resume: bool = True
    screening_workers: int = Field(default=5, ge=1, le=10)
    full_text_workers: int = Field(default=10, ge=1, le=10)
    extraction_workers: int = Field(default=5, ge=1, le=10)
    validation_workers: int = Field(default=5, ge=1, le=10)
    retry_failed_validation: bool = False
    max_expansion_rounds: int | None = Field(default=None, ge=1, le=10)
    max_candidates: int = Field(default=200, ge=10, le=200)


class AdvanceResearchPipelineOutput(BaseModel):
    substance: str
    target_stage: str
    run_dir: str
    candidates_count: int
    retained_count: int
    rejected_count: int
    full_text_retrieved_count: int = 0
    full_text_unavailable_count: int = 0
    skipped_existing_count: int = 0
    retained_sources_path: str
    full_text_manifest_path: str | None = None
    proposed_claim_count: int = 0
    extraction_failure_count: int = 0
    proposed_claims_path: str | None = None
    validated_claim_count: int = 0
    rejected_claim_count: int = 0
    validation_failure_count: int = 0
    held_claim_count: int = 0
    limited_body_count: int = 0
    inadequate_body_count: int = 0
    validated_claims_path: str | None = None
    rejected_claims_path: str | None = None
    held_claims_path: str | None = None
    body_adequacy_path: str | None = None
    validation_failures_path: str | None = None
    requested_count: int = 0
    target_metric: Literal["retained_papers", "usable_full_texts"] = "usable_full_texts"
    target_met: bool = False
    expansion_recommended: bool = False
    expansion_round: int = 0
    max_expansion_rounds: int = 0
    new_candidate_count: int = 0
    new_retained_count: int = 0
    new_usable_full_text_count: int = 0
    stop_reason: Literal[
        "target_met",
        "expansion_required",
        "max_rounds",
        "candidate_cap",
        "no_new_usable_papers",
    ] = "expansion_required"
    queries_attempted: list[str] = Field(default_factory=list)
    retained_titles: list[str] = Field(default_factory=list)
    publication_type_counts: dict[str, int] = Field(default_factory=dict)
    rejection_code_counts: dict[str, int] = Field(default_factory=dict)
    full_text_status_counts: dict[str, int] = Field(default_factory=dict)
    expansion_state_path: str | None = None


def inspect_research_state(payload: InspectResearchStateInput) -> InspectResearchStateOutput:
    root = _workspace_root(payload.workspace_root)
    slug_variants = {
        _slug(payload.substance),
        _slug(payload.substance).replace("-", "_"),
        re.sub(r"[^a-z0-9]", "", payload.substance.lower()),
    }
    matches: list[ResearchStateMatch] = []
    retained_paths: list[Path] = []
    manifest_paths: list[Path] = []
    proposed_paths: list[Path] = []
    extraction_status_paths: list[Path] = []
    validated_paths: list[Path] = []
    rejected_claim_paths: list[Path] = []
    validation_status_paths: list[Path] = []
    adequacy_paths: list[Path] = []
    report_markdown_paths: list[Path] = []
    report_json_paths: list[Path] = []
    report_csv_paths: list[Path] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            normalized = re.sub(r"[^a-z0-9]", "", str(path.relative_to(root)).lower())
            if not any(variant.replace("-", "").replace("_", "") in normalized for variant in slug_variants):
                continue
            artifact_type = "other"
            if path.name == "retained_sources.json":
                artifact_type = "retained_sources"
                retained_paths.append(path)
            elif path.name in {"manifest.json", "raw_texts.json"}:
                artifact_type = "full_text_manifest"
                manifest_paths.append(path)
            elif "candidate" in path.name or "retrieval" in path.name:
                artifact_type = "retrieval_candidates"
            elif path.name == "proposed_claims.json":
                artifact_type = "proposed_claims"
                proposed_paths.append(path)
            elif path.name == "extraction_status.json":
                artifact_type = "extraction_status"
                extraction_status_paths.append(path)
            elif path.name == "validated_claims.json":
                artifact_type = "validated_claims"
                validated_paths.append(path)
            elif path.name == "rejected_claims.json":
                artifact_type = "rejected_claims"
                rejected_claim_paths.append(path)
            elif path.name == "validation_status.json":
                artifact_type = "validation_status"
                validation_status_paths.append(path)
            elif path.name == "body_adequacy.json":
                artifact_type = "body_adequacy"
                adequacy_paths.append(path)
            elif path.name == "claims_report.md":
                artifact_type = "claims_report"
                report_markdown_paths.append(path)
            elif path.name == "claims_report.json":
                artifact_type = "claims_report_json"
                report_json_paths.append(path)
            elif path.name == "claims_report.csv":
                artifact_type = "claims_report_csv"
                report_csv_paths.append(path)
            elif path.suffix == ".txt":
                artifact_type = "full_text"
            matches.append(
                ResearchStateMatch(
                    path=str(path),
                    artifact_type=artifact_type,
                    modified_at=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                )
            )
    retained = max(retained_paths, key=lambda item: item.stat().st_mtime, default=None)
    manifest = max(manifest_paths, key=lambda item: item.stat().st_mtime, default=None)
    proposed_path = max(proposed_paths, key=lambda item: item.stat().st_mtime, default=None)
    extraction_status_path = max(
        extraction_status_paths, key=lambda item: item.stat().st_mtime, default=None
    )
    validated_path = max(validated_paths, key=lambda item: item.stat().st_mtime, default=None)
    rejected_path = max(rejected_claim_paths, key=lambda item: item.stat().st_mtime, default=None)
    validation_status_path = max(
        validation_status_paths, key=lambda item: item.stat().st_mtime, default=None
    )
    adequacy_path = max(adequacy_paths, key=lambda item: item.stat().st_mtime, default=None)
    report_markdown = max(report_markdown_paths, key=lambda item: item.stat().st_mtime, default=None)
    report_json = max(report_json_paths, key=lambda item: item.stat().st_mtime, default=None)
    report_csv = max(report_csv_paths, key=lambda item: item.stat().st_mtime, default=None)
    suggested: Literal[
        "retrieve_and_screen", "retrieve_full_text", "resume_full_text", "full_text_complete",
        "extract_claims", "resume_claim_extraction", "claim_extraction_complete",
        "validate_claims", "resume_claim_validation", "claim_validation_complete",
        "retrieve_better_body_text", "export_report", "report_complete"
    ] = "retrieve_and_screen"
    status_counts: dict[str, int] = {}
    verified_paths: list[str] = []
    extraction_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    adequacy_counts: dict[str, int] = {}
    proposed_count: int | None = None
    if retained:
        suggested = "retrieve_full_text"
    if manifest:
        try:
            records = RAW_TEXTS_ADAPTER.validate_json(manifest.read_bytes())
            for record in records:
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
                if record.status == "full_text_found" and record.text_path:
                    text_path = Path(record.text_path)
                    if not text_path.is_absolute():
                        text_path = manifest.parent / "papers" / text_path.name
                    if text_path.exists() and text_path.stat().st_size > 0:
                        verified_paths.append(str(text_path.resolve()))
            suggested = (
                "full_text_complete"
                if records and len(verified_paths) == len(records)
                else "resume_full_text"
            )
        except Exception:
            suggested = "resume_full_text"
    if verified_paths and not proposed_path:
        suggested = "extract_claims"
    if proposed_path:
        try:
            proposed_count = len(PROPOSED_ADAPTER.validate_json(proposed_path.read_bytes()))
        except Exception:
            proposed_count = None
    if extraction_status_path:
        try:
            extraction_statuses = json.loads(extraction_status_path.read_text(encoding="utf-8"))
            for status in extraction_statuses.values():
                key = str(status.get("status", "unknown"))
                extraction_counts[key] = extraction_counts.get(key, 0) + 1
            suggested = (
                "claim_extraction_complete"
                if extraction_counts.get("completed", 0) and not extraction_counts.get("failed", 0)
                else "resume_claim_extraction"
            )
        except Exception:
            suggested = "resume_claim_extraction"
    if proposed_path and not validation_status_path:
        suggested = "validate_claims"
    if validation_status_path:
        try:
            validation_statuses = json.loads(validation_status_path.read_text(encoding="utf-8"))
            for status in validation_statuses.values():
                key = str(status.get("status", "unknown"))
                validation_counts[key] = validation_counts.get(key, 0) + 1
            completed_validations = (
                validation_counts.get("accepted", 0) + validation_counts.get("rejected", 0)
            )
            held_validations = sum(
                count for key, count in validation_counts.items() if key.startswith("held_")
            )
            suggested = (
                "claim_validation_complete"
                if proposed_count is not None
                and completed_validations + held_validations == proposed_count
                and not validation_counts.get("failed", 0)
                else "resume_claim_validation"
            )
        except Exception:
            suggested = "resume_claim_validation"
    if adequacy_path:
        try:
            assessments = json.loads(adequacy_path.read_text(encoding="utf-8"))
            for assessment in assessments.values():
                key = str(assessment.get("status", "unknown"))
                adequacy_counts[key] = adequacy_counts.get(key, 0) + 1
            if adequacy_counts.get("limited", 0) or adequacy_counts.get("inadequate", 0):
                suggested = "retrieve_better_body_text"
            elif adequacy_counts.get("assessment_failed", 0):
                suggested = "resume_claim_validation"
        except Exception:
            suggested = "resume_claim_validation"
    if validation_status_path and suggested == "claim_validation_complete":
        suggested = "export_report"
    if report_markdown and report_json and report_csv:
        suggested = "report_complete"
    return InspectResearchStateOutput(
        substance=payload.substance.strip(),
        matches=sorted(matches, key=lambda item: item.modified_at, reverse=True)[:50],
        retained_sources_path=str(retained) if retained else None,
        full_text_manifest_path=str(manifest) if manifest else None,
        full_text_status_counts=status_counts,
        verified_full_text_paths=verified_paths,
        proposed_claims_path=str(proposed_path) if proposed_path else None,
        extraction_status_path=str(extraction_status_path) if extraction_status_path else None,
        extraction_status_counts=extraction_counts,
        validated_claims_path=str(validated_path) if validated_path else None,
        rejected_claims_path=str(rejected_path) if rejected_path else None,
        validation_status_path=str(validation_status_path) if validation_status_path else None,
        validation_status_counts=validation_counts,
        validation_failure_count=validation_counts.get("failed", 0),
        held_claim_count=sum(
            count for key, count in validation_counts.items() if key.startswith("held_")
        ),
        body_adequacy_status_counts=adequacy_counts,
        claims_report_path=str(report_markdown) if report_markdown else None,
        claims_report_json_path=str(report_json) if report_json else None,
        claims_report_csv_path=str(report_csv) if report_csv else None,
        suggested_action=suggested,
    )


def advance_research_pipeline(payload: AdvanceResearchPipelineInput) -> AdvanceResearchPipelineOutput:
    from sipz_agent.tools.literature import (
        RetrieveCandidatesInput,
        RetrieveCandidatesOutput,
        RetrieveFullTextBatchInput,
        RetrievalSourceCounts,
        ScreenCandidatesInput,
        retrieve_candidates,
        retrieve_full_text_batch,
        screen_candidates,
    )
    from sipz_agent.core.retrieval import deduplicate_citations

    root = _workspace_root(payload.workspace_root)
    run_dir = (
        payload.run_dir_override.expanduser().resolve()
        if payload.run_dir_override
        else root / "runs" / _slug(payload.substance)
    )
    retrieval_dir = run_dir / "retrieval"
    screening_dir = run_dir / "screening"
    full_text_dir = run_dir / "full_text"
    claims_dir = run_dir / "claims"
    validation_dir = run_dir / "validation"
    candidates_path = retrieval_dir / "candidates.json"
    retained_path = screening_dir / "retained_sources.json"
    selected_path = screening_dir / "selected_sources.json"
    canonical_manifest = full_text_dir / "manifest.json"
    expansion_path = retrieval_dir / "retrieval_expansion.json"
    max_rounds = payload.max_expansion_rounds or {"light": 3, "standard": 5, "deep": 10}[payload.depth]
    target_metric: Literal["retained_papers", "usable_full_texts"] = (
        "retained_papers" if payload.target_stage == "screening" else "usable_full_texts"
    )

    state: dict[str, Any] = {
        "schema_version": "1.0",
        "substance": payload.substance,
        "requested_count": payload.requested_count,
        "target_metric": target_metric,
        "max_expansion_rounds": max_rounds,
        "max_candidates": payload.max_candidates,
        "aliases": [],
        "queries_attempted": [],
        "zero_yield_rounds": 0,
        "rounds": [],
    }
    if payload.resume and expansion_path.exists():
        state.update(json.loads(expansion_path.read_text(encoding="utf-8")))
    previous_target_metric = state.get("target_metric")
    state.update({
        "requested_count": payload.requested_count,
        "target_metric": target_metric,
        "max_expansion_rounds": max_rounds,
        "max_candidates": payload.max_candidates,
    })
    if previous_target_metric and previous_target_metric != target_metric:
        state["zero_yield_rounds"] = 0
    state["aliases"] = list(dict.fromkeys([*state.get("aliases", []), *payload.aliases]))

    retrieval: RetrieveCandidatesOutput | None = None
    if candidates_path.exists():
        raw = json.loads(candidates_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "candidates" in raw:
            retrieval = RetrieveCandidatesOutput.model_validate(raw)
        else:
            existing = SOURCES_ADAPTER.validate_python(raw)
            retrieval = RetrieveCandidatesOutput(
                substance=payload.substance,
                aliases=state["aliases"],
                queries=[],
                candidates=existing,
                raw_candidate_count=len(existing),
                unique_candidate_count=len(existing),
                pages_attempted=0,
                source_counts=RetrievalSourceCounts(),
                failures=[],
                stop_reason="max_pages",
                output_path=str(candidates_path),
            )

    used_queries = set(state.get("queries_attempted", []))
    proposed_queries = list(
        dict.fromkeys(query.strip() for query in payload.expansion_queries if query.strip())
    )
    new_queries = [query for query in proposed_queries if query not in used_queries]
    retrieval_performed = retrieval is None or bool(new_queries)
    prior_candidates = len(retrieval.candidates) if retrieval else 0
    prior_totals = state.get("rounds", [{}])[-1].get("totals", {}) if state.get("rounds") else {}
    prior_retained = int(prior_totals.get("retained", 0))
    prior_usable = int(prior_totals.get("usable_full_texts", 0))
    round_retrieval = None

    if retrieval_performed:
        remaining_capacity = payload.max_candidates - prior_candidates
        existing_usable = prior_usable
        batch_target = min(
            remaining_capacity,
            max(10, max(1, payload.requested_count - existing_usable) * 3),
        )
        if batch_target > 0:
            round_retrieval = retrieve_candidates(
                RetrieveCandidatesInput(
                    substance=payload.substance,
                    aliases=state["aliases"],
                    queries=new_queries,
                    depth=payload.depth,
                    target_count=batch_target,
                    max_pages={"light": 3, "standard": 5, "deep": 10}[payload.depth],
                    page_size=min(10, batch_target),
                )
            )
            merged = deduplicate_citations(
                [*(retrieval.candidates if retrieval else []), *round_retrieval.candidates]
            )[: payload.max_candidates]
            all_queries = list(
                dict.fromkeys([
                    *(retrieval.queries if retrieval else []),
                    *round_retrieval.queries,
                ])
            )
            source_unique: dict[str, int] = {}
            for citation in merged:
                source_unique[citation.source] = source_unique.get(citation.source, 0) + 1
            raw_counts = dict(retrieval.source_counts.raw) if retrieval else {}
            for source, count in round_retrieval.source_counts.raw.items():
                raw_counts[source] = raw_counts.get(source, 0) + count
            retrieval = RetrieveCandidatesOutput(
                substance=payload.substance,
                aliases=state["aliases"],
                queries=all_queries,
                candidates=merged,
                raw_candidate_count=(retrieval.raw_candidate_count if retrieval else 0)
                + round_retrieval.raw_candidate_count,
                unique_candidate_count=len(merged),
                pages_attempted=(retrieval.pages_attempted if retrieval else 0)
                + round_retrieval.pages_attempted,
                source_counts=RetrievalSourceCounts(raw=raw_counts, unique=source_unique),
                failures=[*(retrieval.failures if retrieval else []), *round_retrieval.failures],
                stop_reason=round_retrieval.stop_reason,
                output_path=str(candidates_path),
            )
            candidates_path.parent.mkdir(parents=True, exist_ok=True)
            candidates_path.write_text(retrieval.model_dump_json(indent=2), encoding="utf-8")

    candidates = retrieval.candidates if retrieval else []
    candidate_count = len(candidates)
    if candidates:
        screened = screen_candidates(
            ScreenCandidatesInput(
                substance=payload.substance,
                aliases=state["aliases"],
                candidates=candidates,
                provider=payload.provider,
                model=payload.model,
                output_dir=screening_dir,
                resume=payload.resume,
                max_workers=payload.screening_workers,
            )
        )
        retained = screened.retained
        rejected_count = screened.counts.rejected
        publication_type_counts: dict[str, int] = {}
        rejection_code_counts: dict[str, int] = {}
        for record in screened.records:
            if record.publication_type:
                publication_type_counts[record.publication_type] = (
                    publication_type_counts.get(record.publication_type, 0) + 1
                )
            if record.rejection_code:
                rejection_code_counts[record.rejection_code] = (
                    rejection_code_counts.get(record.rejection_code, 0) + 1
                )
    else:
        retained = []
        rejected_count = 0
        publication_type_counts = {}
        rejection_code_counts = {}
        screening_dir.mkdir(parents=True, exist_ok=True)
        retained_path.write_bytes(SOURCES_ADAPTER.dump_json([], indent=2))

    selected = retained[: payload.requested_count] if target_metric == "retained_papers" else retained
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_bytes(SOURCES_ADAPTER.dump_json(selected, indent=2))

    full_text = None
    records: list[RawTextRecord] = []
    status_counts: dict[str, int] = {}
    if target_metric == "usable_full_texts":
        full_text = retrieve_full_text_batch(
            RetrieveFullTextBatchInput(
                retained_sources_path=selected_path,
                output_dir=full_text_dir,
                resume=payload.resume,
                max_workers=payload.full_text_workers,
            )
        )
        records = RAW_TEXTS_ADAPTER.validate_json(Path(full_text.manifest_path).read_bytes())
        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1

    retained_count = len(retained)
    usable_count = sum(record.status == "full_text_found" for record in records)
    current_metric = retained_count if target_metric == "retained_papers" else usable_count
    new_candidate_count = max(0, candidate_count - prior_candidates)
    new_retained_count = max(0, retained_count - prior_retained)
    new_usable_count = max(0, usable_count - prior_usable)
    metric_delta = new_retained_count if target_metric == "retained_papers" else new_usable_count

    record_round = retrieval_performed or not state.get("rounds")
    if record_round:
        state["zero_yield_rounds"] = 0 if metric_delta > 0 else int(state.get("zero_yield_rounds", 0)) + 1
        round_queries = round_retrieval.queries if round_retrieval else (retrieval.queries if retrieval else [])
        state["queries_attempted"] = list(
            dict.fromkeys([*state.get("queries_attempted", []), *round_queries])
        )
        state.setdefault("rounds", []).append({
            "round": len(state.get("rounds", [])) + 1,
            "queries": round_queries,
            "retrieval_stop_reason": round_retrieval.stop_reason if round_retrieval else None,
            "new": {
                "candidates": new_candidate_count,
                "retained": new_retained_count,
                "usable_full_texts": new_usable_count,
            },
            "totals": {
                "candidates": candidate_count,
                "retained": retained_count,
                "usable_full_texts": usable_count,
            },
            "coverage": {
                "retained_titles": [citation.title for citation in retained],
                "publication_type_counts": publication_type_counts,
                "rejection_code_counts": rejection_code_counts,
                "full_text_status_counts": status_counts,
            },
        })

    target_met = current_metric >= payload.requested_count
    round_count = len(state.get("rounds", []))
    if target_met:
        stop_reason = "target_met"
    elif candidate_count >= payload.max_candidates:
        stop_reason = "candidate_cap"
    elif round_count >= max_rounds:
        stop_reason = "max_rounds"
    elif int(state.get("zero_yield_rounds", 0)) >= 2:
        stop_reason = "no_new_usable_papers"
    else:
        stop_reason = "expansion_required"
    expansion_recommended = stop_reason == "expansion_required"
    state.update({
        "target_met": target_met,
        "expansion_recommended": expansion_recommended,
        "stop_reason": stop_reason,
        "totals": {
            "candidates": candidate_count,
            "retained": retained_count,
            "usable_full_texts": usable_count,
        },
        "coverage": {
            "retained_titles": [citation.title for citation in retained],
            "publication_type_counts": publication_type_counts,
            "rejection_code_counts": rejection_code_counts,
            "full_text_status_counts": status_counts,
        },
        "updated_at": datetime.now(UTC).isoformat(),
    })
    _write_atomic_json(expansion_path, state)
    emit_progress(
        f"Coverage checkpoint: {current_metric}/{payload.requested_count} {target_metric.replace('_', ' ')}; "
        f"{stop_reason.replace('_', ' ')}.",
        stage="retrieval_coverage",
        current=current_metric,
        total=payload.requested_count,
        target_metric=target_metric,
        target_met=target_met,
        expansion_recommended=expansion_recommended,
        stop_reason=stop_reason,
    )

    result = AdvanceResearchPipelineOutput(
        substance=payload.substance,
        target_stage=payload.target_stage,
        run_dir=str(run_dir),
        candidates_count=candidate_count,
        retained_count=len(selected) if target_metric == "retained_papers" else retained_count,
        rejected_count=rejected_count,
        full_text_retrieved_count=usable_count,
        full_text_unavailable_count=sum(record.status != "full_text_found" for record in records),
        skipped_existing_count=full_text.skipped_existing_count if full_text else 0,
        retained_sources_path=str(selected_path),
        full_text_manifest_path=full_text.manifest_path if full_text else None,
        requested_count=payload.requested_count,
        target_metric=target_metric,
        target_met=target_met,
        expansion_recommended=expansion_recommended,
        expansion_round=round_count,
        max_expansion_rounds=max_rounds,
        new_candidate_count=new_candidate_count,
        new_retained_count=new_retained_count,
        new_usable_full_text_count=new_usable_count,
        stop_reason=stop_reason,
        queries_attempted=state.get("queries_attempted", []),
        retained_titles=[citation.title for citation in retained],
        publication_type_counts=publication_type_counts,
        rejection_code_counts=rejection_code_counts,
        full_text_status_counts=status_counts,
        expansion_state_path=str(expansion_path),
    )
    if payload.target_stage in {"screening", "full_text"} or expansion_recommended:
        return result

    extraction = extract_claims(
        ExtractClaimsInput(
            substance=payload.substance,
            sources_path=selected_path,
            raw_texts_manifest_path=Path(result.full_text_manifest_path),
            raw_texts_dir=full_text_dir / "papers",
            output_dir=claims_dir,
            provider=payload.provider,
            model=payload.model,
            resume=payload.resume,
            max_workers=payload.extraction_workers,
        )
    )
    extracted_result = result.model_copy(
        update={
            "proposed_claim_count": extraction.proposed_claim_count,
            "extraction_failure_count": extraction.failure_count,
            "proposed_claims_path": extraction.proposed_claims_path,
        }
    )
    if payload.target_stage == "claim_extraction":
        return extracted_result
    validation = validate_claims(
        ValidateClaimsInput(
            proposed_claims_path=Path(extraction.proposed_claims_path),
            sources_path=selected_path,
            raw_texts_manifest_path=Path(result.full_text_manifest_path),
            raw_texts_dir=full_text_dir / "papers",
            output_dir=validation_dir,
            provider=payload.provider,
            model=payload.model,
            resume=payload.resume,
            retry_failed=payload.retry_failed_validation,
            max_workers=payload.validation_workers,
        )
    )
    return extracted_result.model_copy(update={
        "validated_claim_count": validation.accepted_count,
        "rejected_claim_count": validation.rejected_count,
        "validation_failure_count": validation.failure_count,
        "held_claim_count": validation.held_claim_count,
        "limited_body_count": validation.limited_paper_count,
        "inadequate_body_count": validation.inadequate_paper_count,
        "validated_claims_path": validation.validated_claims_path,
        "rejected_claims_path": validation.rejected_claims_path,
        "held_claims_path": validation.held_claims_path,
        "body_adequacy_path": validation.adequacy_path,
        "validation_failures_path": validation.failures_path,
    })


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "research"


def _write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _run_research_pipeline_legacy(payload: RunResearchPipelineInput) -> RunResearchPipelineOutput:
    from sipz_agent.tools.literature import (
        RetrieveCandidatesInput,
        RetrieveFullTextInput,
        ScreenCandidatesInput,
        retrieve_candidates,
        retrieve_full_text,
        screen_candidates,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = payload.run_id or f"{timestamp}_{_slug(payload.substance)}"
    run_dir = (payload.output_root / run_id).expanduser().resolve()
    retrieval_dir = run_dir / "retrieval"
    screening_dir = run_dir / "screening"
    full_text_dir = run_dir / "full_text"
    papers_dir = full_text_dir / "papers"
    claims_dir = run_dir / "claims"
    validation_dir = run_dir / "validation"
    report_dir = run_dir / "report"
    run_path = run_dir / "run.json"
    audit_path = run_dir / "audit_log.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_id,
        "substance": payload.substance,
        "aliases": payload.aliases,
        "depth": payload.depth,
        "models": {
            "orchestrator": {
                "provider": os.getenv("ORCHESTRATOR_MODEL_PROVIDER"),
                "model": os.getenv("ORCHESTRATOR_MODEL_ID"),
                "thinking": os.getenv("ORCHESTRATOR_THINKING"),
            },
            "worker": {
                "provider": payload.provider or os.getenv("WORKER_MODEL_PROVIDER") or os.getenv("RESEARCH_MODEL_PROVIDER"),
                "model": payload.model or os.getenv("WORKER_MODEL_ID") or os.getenv("RESEARCH_MODEL_ID"),
            },
        },
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "stages": {},
    }
    if payload.resume and run_path.exists():
        state = json.loads(run_path.read_text(encoding="utf-8"))
        state["status"] = "running"
        state["models"] = {
            "orchestrator": {
                "provider": os.getenv("ORCHESTRATOR_MODEL_PROVIDER"),
                "model": os.getenv("ORCHESTRATOR_MODEL_ID"),
                "thinking": os.getenv("ORCHESTRATOR_THINKING"),
            },
            "worker": {
                "provider": payload.provider or os.getenv("WORKER_MODEL_PROVIDER") or os.getenv("RESEARCH_MODEL_PROVIDER"),
                "model": payload.model or os.getenv("WORKER_MODEL_ID") or os.getenv("RESEARCH_MODEL_ID"),
            },
        }
    _write_atomic_json(run_path, state)

    def complete(stage: str, counts: dict[str, int], artifacts: dict[str, str]) -> None:
        state["stages"][stage] = {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "artifacts": artifacts,
        }
        _write_atomic_json(run_path, state)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "event": f"{stage}_completed", "counts": counts}) + "\n")
        emit_progress(
            f"Completed {stage.replace('_', ' ')}: "
            + ", ".join(f"{key}={value}" for key, value in counts.items()),
            stage=stage, status="completed", counts=counts,
        )

    try:
        emit_progress(f"Starting research pipeline for {payload.substance}...", stage="pipeline")
        retrieval_path = retrieval_dir / "candidates.json"
        if not (payload.resume and retrieval_path.exists()):
            retrieval = retrieve_candidates(
                RetrieveCandidatesInput(
                    substance=payload.substance,
                    aliases=payload.aliases,
                    depth=payload.depth,
                    target_count=payload.target_count,
                    max_pages=payload.max_pages,
                    page_size=payload.page_size,
                    output_path=retrieval_path,
                )
            )
        else:
            from sipz_agent.tools.literature import RetrieveCandidatesOutput
            retrieval = RetrieveCandidatesOutput.model_validate_json(retrieval_path.read_text())
        complete(
            "retrieval",
            {"raw": retrieval.raw_candidate_count, "unique": retrieval.unique_candidate_count},
            {"candidates": str(retrieval_path)},
        )

        screening_summary = screening_dir / "screening_summary.json"
        retained_path = screening_dir / "retained_sources.json"
        if not (payload.resume and screening_summary.exists() and retained_path.exists()):
            screened = screen_candidates(
                ScreenCandidatesInput(
                    substance=payload.substance,
                    aliases=payload.aliases,
                    candidates=retrieval.candidates,
                    provider=payload.provider,
                    model=payload.model,
                    output_dir=screening_dir,
                )
            )
        else:
            retained = SOURCES_ADAPTER.validate_json(retained_path.read_bytes())
            counts = json.loads(screening_summary.read_text(encoding="utf-8"))
            screened = None
        retained = screened.retained if screened else retained
        screen_counts = screened.counts.model_dump() if screened else counts
        if screen_counts["input"] != screen_counts["retained"] + screen_counts["rejected"]:
            raise RuntimeError("screening_count_mismatch")
        complete("screening", screen_counts, {"retained_sources": str(retained_path)})

        manifest_path = full_text_dir / "manifest.json"
        attempts_path = full_text_dir / "attempts.json"
        if not (payload.resume and manifest_path.exists() and attempts_path.exists()):
            records: list[RawTextRecord] = []
            attempts: list[FullTextRetrievalAttempt] = []
            for citation_index, citation in enumerate(retained, start=1):
                emit_progress(
                    f"Retrieving full text {citation_index}/{len(retained)}: {citation.title}",
                    stage="full_text_retrieval", current=citation_index, total=len(retained),
                    title=citation.title,
                )
                result = retrieve_full_text(
                    RetrieveFullTextInput(
                        id=citation.id,
                        title=citation.title,
                        doi=citation.doi,
                        pmid=citation.pmid,
                        url=citation.url,
                        source=citation.source,
                        retrieval_query=citation.retrieval_query,
                        abstract=citation.abstract,
                        body_text=citation.body_text,
                        output_dir=papers_dir,
                    )
                )
                records.append(result.record)
                attempts.extend(result.attempts)
                emit_progress(
                    f"Full text {citation_index}/{len(retained)}: {result.record.status} - {citation.title}",
                    stage="full_text_retrieval", current=citation_index, total=len(retained),
                    title=citation.title, status=result.record.status,
                )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(TypeAdapter(list[RawTextRecord]).dump_json(records, indent=2))
            attempts_path.write_bytes(
                TypeAdapter(list[FullTextRetrievalAttempt]).dump_json(attempts, indent=2)
            )
        else:
            records = TypeAdapter(list[RawTextRecord]).validate_json(manifest_path.read_bytes())
        complete(
            "full_text",
            {
                "input": len(retained),
                "retrieved": sum(item.status == "full_text_found" for item in records),
                "unavailable": sum(item.status != "full_text_found" for item in records),
            },
            {"manifest": str(manifest_path), "attempts": str(attempts_path)},
        )

        extraction = extract_claims(
            ExtractClaimsInput(
                substance=payload.substance,
                sources_path=retained_path,
                raw_texts_manifest_path=manifest_path,
                raw_texts_dir=papers_dir,
                output_dir=claims_dir,
                provider=payload.provider,
                model=payload.model,
                resume=payload.resume,
            )
        )
        complete(
            "claim_extraction",
            {"proposed": extraction.proposed_claim_count, "failures": extraction.failure_count},
            {"proposed_claims": extraction.proposed_claims_path},
        )
        validation = validate_claims(
            ValidateClaimsInput(
                proposed_claims_path=Path(extraction.proposed_claims_path),
                sources_path=retained_path,
                raw_texts_manifest_path=manifest_path,
                raw_texts_dir=papers_dir,
                output_dir=validation_dir,
                provider=payload.provider,
                model=payload.model,
                resume=payload.resume,
            )
        )
        complete(
            "claim_validation",
            {
                "accepted": validation.accepted_count,
                "rejected": validation.rejected_count,
                "failures": validation.failure_count,
                "held": validation.held_claim_count,
            },
            {"validated_claims": validation.validated_claims_path},
        )
        exported = export_research_report(
            ExportResearchReportInput(
                substance=payload.substance,
                proposed_claims_path=Path(extraction.proposed_claims_path),
                validated_claims_path=Path(validation.validated_claims_path),
                rejected_claims_path=Path(validation.rejected_claims_path),
                sources_path=retained_path,
                output_dir=report_dir,
                held_claims_path=Path(validation.held_claims_path),
                body_adequacy_path=Path(validation.adequacy_path),
                validation_failures_path=Path(validation.failures_path),
            )
        )
        complete(
            "export",
            {"accepted": exported.accepted_count, "rejected": exported.rejected_count, "held": exported.held_count},
            {"report": exported.markdown_path},
        )
        state["status"] = "completed"
        state["completed_at"] = datetime.now(UTC).isoformat()
        _write_atomic_json(run_path, state)
        return RunResearchPipelineOutput(
            run_id=run_id,
            run_dir=str(run_dir),
            status="completed",
            stage_counts={
                "candidates": retrieval.unique_candidate_count,
                "retained": len(retained),
                "full_text": sum(item.status == "full_text_found" for item in records),
                "proposed_claims": extraction.proposed_claim_count,
                "accepted_claims": exported.accepted_count,
            },
            report_dir=exported.report_dir,
            markdown_path=exported.markdown_path,
            json_path=exported.json_path,
            csv_path=exported.csv_path,
            terminal_summary_markdown=exported.terminal_summary_markdown,
            terminal_table_markdown=exported.terminal_table_markdown,
            report_link_markdown=exported.report_link_markdown,
            report_directory_link_markdown=exported.report_directory_link_markdown,
            report_location_markdown=exported.report_location_markdown,
            terminal_response_markdown=exported.terminal_response_markdown,
        )
    except Exception as exc:
        state["status"] = "failed"
        state["failed_at"] = datetime.now(UTC).isoformat()
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_atomic_json(run_path, state)
        raise


def run_research_pipeline(payload: RunResearchPipelineInput) -> RunResearchPipelineOutput:
    """Compatibility wrapper for callers that cannot run the Pi-managed expansion loop."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = payload.run_id or f"{timestamp}_{_slug(payload.substance)}"
    run_dir = (payload.output_root / run_id).expanduser().resolve()
    max_rounds = {"light": 3, "standard": 5, "deep": 10}[payload.depth]
    target_count = payload.target_count or {"light": 10, "standard": 25, "deep": 40}[payload.depth]
    terms = list(dict.fromkeys([payload.substance, *payload.aliases]))
    entity = " OR ".join(f'"{term}"' for term in terms)
    query_tiers = [
        f"({entity}) AND (randomized OR placebo OR clinical trial OR crossover) AND (oral OR consumed)",
        f"({entity}) AND (systematic review OR meta-analysis) AND human AND oral",
        f"({entity}) AND (cohort OR observational OR prospective) AND (dietary OR ingestion)",
        f"({entity}) AND (efficacy OR safety OR adverse effects) AND human consumption",
        f"({entity}) AND (sleep OR mood OR cognition OR quality of life) AND human oral",
        f"({entity}) AND (cardiovascular OR metabolic OR gastrointestinal) AND human intake",
        f"({entity}) AND (immune OR inflammatory OR pain) AND human consumption",
        f"({entity}) AND supplement AND (dose OR dosage) AND clinical outcomes",
        f"({entity}) AND human health AND (trial OR review OR cohort) NOT animal",
    ]
    checkpoint: AdvanceResearchPipelineOutput | None = None
    expansion_queries: list[str] = []
    while True:
        checkpoint = advance_research_pipeline(
            AdvanceResearchPipelineInput(
                substance=payload.substance,
                target_stage="claim_validation",
                aliases=payload.aliases,
                expansion_queries=expansion_queries,
                requested_count=target_count,
                depth=payload.depth,
                provider=payload.provider,
                model=payload.model,
                workspace_root=payload.output_root.parent,
                run_dir_override=run_dir,
                resume=payload.resume,
                max_expansion_rounds=max_rounds,
            )
        )
        if not checkpoint.expansion_recommended:
            break
        tier_index = min(checkpoint.expansion_round - 1, len(query_tiers) - 1)
        expansion_queries = [query_tiers[tier_index]]

    if checkpoint is None or not checkpoint.proposed_claims_path:
        raise RuntimeError("adaptive_pipeline_did_not_reach_claim_validation")
    validation_dir = run_dir / "validation"
    report_dir = run_dir / "report"
    exported = export_research_report(
        ExportResearchReportInput(
            substance=payload.substance,
            proposed_claims_path=Path(checkpoint.proposed_claims_path),
            validated_claims_path=Path(checkpoint.validated_claims_path),
            rejected_claims_path=Path(checkpoint.rejected_claims_path),
            sources_path=Path(checkpoint.retained_sources_path),
            output_dir=report_dir,
            held_claims_path=Path(checkpoint.held_claims_path)
            if checkpoint.held_claims_path
            else validation_dir / "held_claims.json",
            body_adequacy_path=Path(checkpoint.body_adequacy_path)
            if checkpoint.body_adequacy_path
            else validation_dir / "body_adequacy.json",
            validation_failures_path=Path(checkpoint.validation_failures_path)
            if checkpoint.validation_failures_path
            else validation_dir / "validation_failures.json",
            retrieval_expansion_path=Path(checkpoint.expansion_state_path)
            if checkpoint.expansion_state_path
            else None,
        )
    )
    run_state = {
        "run_id": run_id,
        "substance": payload.substance,
        "aliases": payload.aliases,
        "depth": payload.depth,
        "status": "completed",
        "requested_usable_full_texts": target_count,
        "retrieval_target_met": checkpoint.target_met,
        "retrieval_stop_reason": checkpoint.stop_reason,
        "expansion_rounds": checkpoint.expansion_round,
        "counts": {
            "candidates": checkpoint.candidates_count,
            "retained": checkpoint.retained_count,
            "usable_full_texts": checkpoint.full_text_retrieved_count,
            "proposed_claims": checkpoint.proposed_claim_count,
            "accepted_claims": exported.accepted_count,
        },
        "artifacts": {
            "retrieval_expansion": checkpoint.expansion_state_path,
            "report": exported.markdown_path,
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_atomic_json(run_dir / "run.json", run_state)
    return RunResearchPipelineOutput(
        run_id=run_id,
        run_dir=str(run_dir),
        status="completed",
        stage_counts=run_state["counts"],
        report_dir=exported.report_dir,
        markdown_path=exported.markdown_path,
        json_path=exported.json_path,
        csv_path=exported.csv_path,
        terminal_summary_markdown=exported.terminal_summary_markdown,
        terminal_table_markdown=exported.terminal_table_markdown,
        report_link_markdown=exported.report_link_markdown,
        report_directory_link_markdown=exported.report_directory_link_markdown,
        report_location_markdown=exported.report_location_markdown,
        terminal_response_markdown=exported.terminal_response_markdown,
    )


TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "advance_research_pipeline": AdvanceResearchPipelineInput,
    "inspect_research_state": InspectResearchStateInput,
    "extract_claims": ExtractClaimsInput,
    "validate_claims": ValidateClaimsInput,
    "export_research_report": ExportResearchReportInput,
    "run_research_pipeline": RunResearchPipelineInput,
}


def execute_tool(name: str, raw_payload: dict[str, Any]) -> BaseModel:
    if name == "advance_research_pipeline":
        return advance_research_pipeline(AdvanceResearchPipelineInput.model_validate(raw_payload))
    if name == "inspect_research_state":
        return inspect_research_state(InspectResearchStateInput.model_validate(raw_payload))
    if name == "extract_claims":
        return extract_claims(ExtractClaimsInput.model_validate(raw_payload))
    if name == "validate_claims":
        return validate_claims(ValidateClaimsInput.model_validate(raw_payload))
    if name == "export_research_report":
        return export_research_report(ExportResearchReportInput.model_validate(raw_payload))
    if name == "run_research_pipeline":
        return run_research_pipeline(RunResearchPipelineInput.model_validate(raw_payload))
    raise ValueError(f"unknown_workflow_tool:{name}")
