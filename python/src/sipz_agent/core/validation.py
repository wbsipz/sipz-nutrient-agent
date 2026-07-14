from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable, Literal
from uuid import uuid4

import orjson
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from sipz_agent.core.artifacts import model_dump_jsonable, write_json
from sipz_agent.core.claim_proposal import body_text_for_record
from sipz_agent.core.models import LlmProvider
from sipz_agent.core.quote_grounding import ground_quote
from sipz_agent.core.retrieval import truncate_text
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.claims import ProposedClaim, SupportingQuote, ValidatedClaim
from sipz_agent.schemas.raw_texts import RawTextRecord

SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])
PROPOSED_CLAIMS_ADAPTER = TypeAdapter(list[ProposedClaim])
VALIDATED_CLAIMS_ADAPTER = TypeAdapter(list[ValidatedClaim])

ValidationProgress = Callable[[int, int, CandidateCitation, int], None]
VALIDATION_VERDICTS = {
    "supported",
    "supported_with_limitations",
    "unsupported",
    "over_scoped",
}


class ValidatorQuote(BaseModel):
    quote: str = Field(min_length=1)
    section: str | None = None
    reason: str = Field(min_length=1)
    evidence_role: Literal[
        "direct_result",
        "review_synthesis",
        "methods_or_context",
        "background_or_other",
    ] = "direct_result"


class ValidatorDecision(BaseModel):
    proposed_claim_id: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    validated_statement: str = Field(default="", max_length=1200)
    support_level: str = Field(min_length=1)
    claim_scope: str = ""
    supporting_quotes: list[ValidatorQuote] = Field(default_factory=list)
    entity_match: Literal[
        "exact_or_alias",
        "different_species",
        "different_entity",
        "unclear",
    ] = "unclear"
    limitations: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        for verdict in VALIDATION_VERDICTS:
            if normalized == verdict or normalized.startswith(f"{verdict} "):
                return verdict
        return "unsupported"

    @field_validator("validated_statement", "claim_scope", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return ""
        return value


class PaperValidationResponse(BaseModel):
    decisions: list[ValidatorDecision]


PAPER_VALIDATION_ADAPTER = TypeAdapter(PaperValidationResponse)


class QuoteRepairResponse(BaseModel):
    supporting_quotes: list[ValidatorQuote] = Field(default_factory=list)


QUOTE_REPAIR_ADAPTER = TypeAdapter(QuoteRepairResponse)


class BodySanitizationError(ValueError):
    def __init__(self, message: str = "body_sanitization_failed", *, preview: str | None = None) -> None:
        super().__init__(message)
        self.preview = preview


def _normalized_text_with_offsets(value: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    offsets: list[int] = []
    index = 0
    previous_was_space = False
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    while index < len(value):
        char = value[index]
        if char == "-":
            next_index = index + 1
            while next_index < len(value) and value[next_index].isspace():
                next_index += 1
            previous = normalized[-1] if normalized else ""
            following = value[next_index] if next_index < len(value) else ""
            if previous.isalnum() and following.isalnum() and next_index > index + 1:
                index = next_index
                continue
        expanded = ligatures.get(char, char).lower()
        for expanded_char in expanded:
            if expanded_char.isspace():
                if normalized and not previous_was_space:
                    normalized.append(" ")
                    offsets.append(index)
                previous_was_space = True
            else:
                normalized.append(expanded_char)
                offsets.append(index)
                previous_was_space = False
        index += 1
    while normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def _normalized_value(value: str) -> str:
    return _normalized_text_with_offsets(value)[0].strip()


def remove_normalized_span(text: str, value: str | None) -> tuple[str, bool]:
    if not value or not value.strip():
        return text, False
    normalized_text, offsets = _normalized_text_with_offsets(text)
    normalized_value = _normalized_value(value)
    if not normalized_value:
        return text, False
    start = normalized_text.find(normalized_value)
    if start < 0:
        return text, False
    end = start + len(normalized_value) - 1
    original_start = offsets[start]
    original_end = offsets[end] + 1
    return f"{text[:original_start]} {text[original_end:]}".strip(), True


def remove_all_normalized_spans(text: str, value: str | None) -> tuple[str, bool]:
    removed_any = False
    while True:
        text, removed = remove_normalized_span(text, value)
        if not removed:
            return text, removed_any
        removed_any = True


def body_after_reliable_start(text: str) -> str | None:
    preamble = text[:20_000]
    patterns = [
        r"(?i)(?:^|\s)1(?:\.0)?[.)]?\s+introduction(?:\s|$)",
        r"(?i)(?:^|\s)i\s*n\s*t\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n(?:\s|$)",
        r"(?i)(?:^|\s)introduction(?:\s|$)",
        r"(?i)(?:^|\s)wst[eę]p(?:\s|$)",
        r"(?:^|\s)1\.\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿĀ-ž][^.!?]{2,160}(?=\s+[A-Z])",
    ]
    matches = [match for pattern in patterns if (match := re.search(pattern, preamble))]
    if not matches:
        return None
    match = min(matches, key=lambda item: item.start())
    body = text[match.start():].strip()
    return body if len(body) >= 500 else None


def publisher_preview_only(text: str) -> bool:
    preamble = text[:10_000].casefold()
    if "this is a preview of subscription content" in preamble:
        return True
    return "restricted access" in preamble and "purchase article" in preamble


def remove_abstract_like_preamble(text: str) -> str:
    if publisher_preview_only(text):
        return text
    patterns = [
        r"(?is)\babstract\s+.*?(?=\b(?:keywords?|introduction|background|methods?|materials\s+and\s+methods|study\s+design)\b)",
        r"(?is)\bsummary\s+.*?(?=\b(?:keywords?|introduction|background|methods?|materials\s+and\s+methods|study\s+design)\b)",
        r"(?is)\bstreszczenie\s+.*?(?=\b(?:słowa\s+kluczowe|slowa\s+kluczowe|wst[eę]p|introduction|background)\b)",
    ]
    preamble = text[:20_000]
    best_end = 0
    for pattern in patterns:
        match = re.search(pattern, preamble)
        if match and match.start() < 5_000:
            best_end = max(best_end, match.end())
    if not best_end:
        return text
    return text[best_end:].lstrip()


def flexible_heading_pattern(label: str) -> str:
    if label == "materials and methods":
        return r"materials?\s+(?:and|&)\s+methods?"
    if " " in label:
        return r"\s+".join(flexible_heading_pattern(part) for part in label.split())
    return r"\s*".join(re.escape(char) for char in label)


def section_heading_regex(labels: list[str]) -> str:
    alternatives = "|".join(flexible_heading_pattern(label) for label in labels)
    return rf"(?im)(?:^|\n|\s)(?:\d+(?:\.\d+)*[.)]?\s*)?(?:{alternatives})(?:\s|:|$)"


def body_fragment_from_sections(text: str) -> str | None:
    if publisher_preview_only(text):
        return None
    truncated = remove_references_section(remove_abstract_like_preamble(text))
    start_patterns = [
        section_heading_regex(["materials and methods", "methods", "study design", "participants"]),
        section_heading_regex(["results"]),
        section_heading_regex(["discussion", "conclusion", "conclusions"]),
    ]
    starts = [
        match.start()
        for pattern in start_patterns
        for match in re.finditer(pattern, truncated)
        if match.start() >= 300
    ]
    if not starts:
        return None
    candidate = truncated[min(starts):].strip()
    lowered = candidate.casefold()
    has_methods = bool(re.search(section_heading_regex(["materials and methods", "methods", "study design", "participants"]), candidate))
    has_results = bool(re.search(section_heading_regex(["results"]), candidate))
    has_discussion = bool(re.search(section_heading_regex(["discussion", "conclusion", "conclusions"]), candidate))
    has_health_result_text = any(
        marker in lowered
        for marker in [
            "participants",
            "subjects",
            "randomized",
            "consumed",
            "ingested",
            "supplement",
            "outcome",
        ]
    )
    if len(candidate) >= 1_200 and has_results and (has_methods or has_discussion) and has_health_result_text:
        return candidate
    if len(candidate) >= 2_000 and sum([has_methods, has_results, has_discussion]) >= 2:
        return candidate
    return None


def body_after_structured_start(text: str) -> str | None:
    if publisher_preview_only(text):
        return None

    truncated = remove_references_section(remove_abstract_like_preamble(text))
    patterns = [
        r"(?i)(?:^|\s)background\s+",
        r"(?i)(?:^|\s)b\s*a\s*c\s*k\s*g\s*r\s*o\s*u\s*n\s*d\s+",
        r"(?i)(?:^|\s)materials\s+and\s+methods\s+",
        r"(?i)(?:^|\s)m\s*e\s*t\s*h\s*o\s*d\s*s\s+",
        r"(?i)(?:^|\s)methods\s+",
        r"(?i)(?:^|\s)study\s+design(?:\s+and\s+participants)?\s+",
        r"(?i)(?:^|\s)participants\s+",
    ]
    candidates = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, truncated)
        if match.start() >= 300
    ]
    for match in sorted(candidates, key=lambda item: item.start()):
        candidate = truncated[match.start():].strip()
        near_heading_list = candidate[:500].casefold()
        if "results discussion references" in near_heading_list:
            continue
        if "references references export references" in near_heading_list:
            continue
        has_results = re.search(r"(?i)(?:^|\s)results\s+", candidate[800:])
        has_discussion_or_conclusion = re.search(
            r"(?i)(?:^|\s)(?:discussion|conclusions?)\s+",
            candidate[1_500:],
        )
        if has_results and has_discussion_or_conclusion and len(candidate) >= 2_000:
            return candidate
    return None


def remove_references_section(text: str) -> str:
    patterns = [
        r"(?im)^\s*(?:\d+[.)]?\s*)?(?:references|bibliography)\s*$",
        r"\bReferences\s+(?=[A-Z][A-Za-z'’-]+(?:\s+[A-Z]{1,4})?,)",
        r"\bBibliography\s+(?=[A-Z][A-Za-z'’-]+)",
    ]
    candidates = [
        match.start()
        for pattern in patterns
        for match in re.finditer(pattern, text)
        if match.start() >= 500
    ]
    if not candidates:
        return text
    return text[: min(candidates)].rstrip()


def remove_terminal_conclusion_section(text: str) -> str:
    matches = list(
        re.finditer(
            r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?:conclusion|conclusions)\s*$",
            text,
        )
    )
    if not matches or matches[-1].start() < len(text) * 0.45:
        return text
    return text[: matches[-1].start()].rstrip()


def remove_validator_excluded_sections(text: str) -> str:
    return remove_terminal_conclusion_section(remove_references_section(text))


def title_like_quote(quote: str, title: str | None) -> bool:
    if not title:
        return False
    normalized_quote = _normalized_value(quote)
    normalized_title = _normalized_value(title)
    if not normalized_quote or not normalized_title:
        return False
    quote_words = normalized_quote.split()
    if len(quote_words) < 6:
        return False
    if normalized_quote == normalized_title:
        return True
    if normalized_title.startswith(normalized_quote):
        return True
    return (
        normalized_quote in normalized_title
        and len(normalized_quote) >= len(normalized_title) * 0.5
    )


def sanitize_paper_body(
    *,
    body_text: str,
    citation: CandidateCitation,
) -> str:
    sanitized, abstract_removed = remove_all_normalized_spans(body_text, citation.abstract)
    sanitized, _ = remove_all_normalized_spans(sanitized, citation.title)
    sanitized = remove_abstract_like_preamble(sanitized)

    if citation.abstract and abstract_removed:
        sanitized = remove_validator_excluded_sections(sanitized)
        if len(sanitized) >= 500:
            return sanitized

    fallback = body_after_reliable_start(sanitized)
    if fallback:
        fallback = remove_validator_excluded_sections(fallback)
        if len(fallback) >= 500:
            return fallback
    structured_fallback = body_after_structured_start(sanitized)
    if structured_fallback:
        structured_fallback = remove_validator_excluded_sections(structured_fallback)
        if len(structured_fallback) >= 500:
            return structured_fallback
    fragment_fallback = body_fragment_from_sections(sanitized)
    if fragment_fallback:
        fragment_fallback = remove_validator_excluded_sections(fragment_fallback)
        if len(fragment_fallback) >= 500:
            return fragment_fallback
    preview = remove_validator_excluded_sections(sanitized)
    preview = truncate_text(" ".join(preview.split()), max_chars=4_000)
    raise BodySanitizationError("body_sanitization_failed", preview=preview)


def validation_prompt(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    claims: list[ProposedClaim],
    sanitized_body: str,
) -> str:
    claim_payload = [
        {
            "proposed_claim_id": claim.id,
            "statement": claim.statement,
            "compound": claim.compound,
            "effect": claim.effect,
            "direction": claim.direction,
            "population": claim.population,
            "dose_or_exposure": claim.dose_or_exposure,
            "outcome": claim.outcome,
            "study_type": claim.study_type,
            "evidence_type": claim.evidence_type,
            "intake_route": claim.intake_route,
            "exposure_category": claim.exposure_category,
        }
        for claim in claims
    ]
    return f"""You are the body-only claim validator for the Sipz nutrient research agent.

Evaluate every proposed claim using only the supplied paper body. The paper title and abstract have
been removed and must not be inferred as evidence.

Check separately:
1. Text support: does the body support the statement?
2. Evidence scope: are route, population, dose, outcome, and certainty stated at the right strength?

Rules:
- Human health claims must be supported by human evidence, not animal, in-vitro, or theory alone.
- Reject a claim as unsupported when its underlying evidence is animal-only or in-vitro-only,
  including when that evidence is summarized by a review. A review does not convert preclinical
  evidence into human evidence.
- For animal-only, in-vitro-only, or otherwise preclinical-only support, verdict must be
  unsupported. Do not use supported_with_limitations.
- The relevant exposure must be oral consumption.
- Distinguish natural-food exposure from supplement-level exposure.
- Do not accept pharmaceutical, injected, or topical effects as oral nutrient health effects.
- Treat biological species as distinct entities. A related species is not an alias. If the claim's
  evidence concerns a different species than the requested target, set entity_match to
  different_species and return unsupported. A genus-level review may discuss several species, but
  each accepted claim must be supported specifically for the requested species or explicit target.
- Decide whether the paper provides evidence for the requested nutrient/bioactive itself, not merely
  for a formulation containing it. If the intervention combines the requested target with another
  active ingredient, return unsupported unless this paper isolates the requested target's
  contribution. This remains unsupported even when the proposed claim transparently names the
  combination; a combination claim is eligible only when the requested research target is that
  complete formulation. Do not preserve or create a combination claim merely to accept it.
- Use supported_with_limitations and provide a narrower validated_statement when the original is
  directionally supported but too broad or too certain.
- Use over_scoped when a defensible narrower human oral claim cannot be produced.
- For unsupported or over_scoped decisions, return a short rejection sentence for
  validated_statement and claim_scope; do not return null or an empty string.
- Every supported decision must include at least one short, exact quote copied from the supplied body.
- An exact quote must directly report the claimed result or a review's synthesis of that result.
  Study counts, eligibility text, methods, background, and statements that an outcome was measured
  do not prove the direction or magnitude of a health effect. Label each quote's evidence_role.
- Do not use the paper title, citation text, article header, or navigation text as a supporting quote.
- Supporting quotes should come from body sections such as Methods, Results, or Discussion.
- Quotes must not come from references or describe another paper without making clear this paper is a review.
- Keep every string under 600 characters and return JSON only.

Return:
{{"decisions":[{{
  "proposed_claim_id":"...",
  "verdict":"supported|supported_with_limitations|unsupported|over_scoped",
  "validated_statement":"...",
  "support_level":"human_systematic_review|human_rct|human_observational|human_mechanistic|review_author_interpretation|unsupported",
  "entity_match":"exact_or_alias|different_species|different_entity|unclear",
  "claim_scope":"...",
  "supporting_quotes":[{{"quote":"exact body quote","section":"Results","reason":"...","evidence_role":"direct_result|review_synthesis|methods_or_context|background_or_other"}}],
  "limitations":["..."],
  "reasoning":"..."
}}]}}

Nutrient/bioactive: {nutrient_name}
Citation ID: {citation.id}
DOI: {citation.doi or "unknown"}
Proposed claims:
{orjson.dumps(claim_payload, option=orjson.OPT_INDENT_2).decode("utf-8")}

Sanitized paper body:
{sanitized_body}
"""


RESULT_SIGNAL_PATTERN = re.compile(
    r"\b(?:significant(?:ly)?|improv(?:e|ed|ement|ing)|increas(?:e|ed|ing)|"
    r"decreas(?:e|ed|ing)|reduc(?:e|ed|tion|ing)|lower(?:ed)?|higher|greater|"
    r"longer|shorter|worsen(?:ed|ing)?|alleviat(?:e|ed|ion)|reliev(?:e|ed)|"
    r"prevent(?:ed|ion)|associated|association|difference|benefit|harm|risk|"
    r"odds|hazard|incidence|prevalence|effect size|confidence interval|"
    r"relative risk|odds ratio|hazard ratio|p\s*[<=>]|rr\s*[=:]|or\s*[=:]|hr\s*[=:])\b",
    re.IGNORECASE,
)


def quote_reports_result(quote: SupportingQuote) -> bool:
    if quote.evidence_role not in {"direct_result", "review_synthesis"}:
        return False
    return bool(RESULT_SIGNAL_PATTERN.search(quote.quote))


def substantive_grounded_quote_count(
    *,
    quotes: list[SupportingQuote],
    citation_title: str | None,
) -> int:
    grounded_quote_count = sum(1 for quote in quotes if quote.match_status != "not_found")
    title_like_grounded_quotes = sum(
        1
        for quote in quotes
        if quote.match_status != "not_found" and title_like_quote(quote.quote, citation_title)
    )
    semantically_supportive_quotes = sum(
        1
        for quote in quotes
        if quote.match_status != "not_found"
        and not title_like_quote(quote.quote, citation_title)
        and quote_reports_result(quote)
    )
    return min(grounded_quote_count - title_like_grounded_quotes, semantically_supportive_quotes)


def acceptable_grounded_quotes(
    *,
    quotes: list[SupportingQuote],
    citation_title: str | None,
) -> list[SupportingQuote]:
    return [
        quote
        for quote in quotes
        if quote.match_status != "not_found"
        and not title_like_quote(quote.quote, citation_title)
        and quote_reports_result(quote)
    ]


def grounded_supporting_quotes(
    *,
    validator_quotes: list[ValidatorQuote],
    citation_title: str | None,
    sanitized_body: str,
) -> list[SupportingQuote]:
    quotes: list[SupportingQuote] = []
    for proposed_quote in validator_quotes:
        grounding = ground_quote(proposed_quote.quote, sanitized_body)
        quotes.append(
            SupportingQuote(
                quote=proposed_quote.quote,
                section=proposed_quote.section,
                reason=proposed_quote.reason,
                match_status=grounding.match_status,
                evidence_role=proposed_quote.evidence_role,
            )
        )
    return quotes


def quote_repair_terms(*values: str | None) -> list[str]:
    terms: list[str] = []
    stopwords = {
        "about",
        "after",
        "also",
        "because",
        "between",
        "claim",
        "could",
        "effect",
        "from",
        "health",
        "human",
        "into",
        "oral",
        "paper",
        "results",
        "study",
        "that",
        "their",
        "there",
        "these",
        "this",
        "were",
        "when",
        "with",
    }
    for value in values:
        if not value:
            continue
        for term in re.findall(r"[A-Za-z][A-Za-z-]{3,}", value.casefold()):
            if term not in stopwords and term not in terms:
                terms.append(term)
    return terms[:24]


def quote_repair_excerpt(
    *,
    sanitized_body: str,
    claim: Any,
    decision: ValidatorDecision,
    max_chars: int = 12_000,
) -> str:
    if len(sanitized_body) <= max_chars:
        return sanitized_body
    terms = quote_repair_terms(
        claim.statement,
        getattr(claim, "effect", None),
        getattr(claim, "outcome", None),
        getattr(claim, "dose_or_exposure", None) or getattr(claim, "dose_or_serving", None),
        decision.validated_statement,
        decision.claim_scope,
        decision.reasoning,
        " ".join(quote.quote for quote in decision.supporting_quotes),
    )
    if not terms:
        return sanitized_body[:max_chars]
    chunk_size = max_chars
    step = max_chars // 2
    best_start = 0
    best_score = -1
    lowered = sanitized_body.casefold()
    numeric_anchors = list(
        dict.fromkeys(
            re.findall(
                r"\b\d+(?:\.\d+)?\b",
                " ".join(
                    value
                    for value in [
                        claim.statement,
                        getattr(claim, "outcome", None),
                        decision.validated_statement,
                        decision.claim_scope,
                    ]
                    if value
                ),
            )
        )
    )
    decimal_anchors = [anchor for anchor in numeric_anchors if "." in anchor]
    if decimal_anchors:
        passages: list[str] = []
        seen_starts: set[int] = set()
        for anchor in decimal_anchors:
            for match in re.finditer(re.escape(anchor), lowered):
                start = max(0, match.start() - 900)
                if any(abs(start - previous) < 500 for previous in seen_starts):
                    continue
                seen_starts.add(start)
                passages.append(sanitized_body[start : start + 2_400])
                if sum(len(passage) for passage in passages) >= max_chars:
                    break
            if sum(len(passage) for passage in passages) >= max_chars:
                break
        if passages:
            return "\n\n--- EXACT BODY PASSAGE ---\n\n".join(passages)[:max_chars]
    for start in range(0, max(len(sanitized_body) - chunk_size + 1, 1), step):
        chunk = lowered[start : start + chunk_size]
        # Repeated keywords in reference lists should not outweigh a result passage.
        score = sum(term in chunk for term in terms)
        score += sum(
            (20 if "." in anchor else 2) for anchor in numeric_anchors if anchor in chunk
        )
        if score > best_score:
            best_score = score
            best_start = start
    return sanitized_body[best_start : best_start + chunk_size]


def quote_repair_prompt(
    *,
    claim: Any,
    decision: ValidatorDecision,
    body_excerpt: str,
) -> str:
    failed_quotes = [
        {
            "quote": quote.quote,
            "section": quote.section,
            "reason": quote.reason,
        }
        for quote in decision.supporting_quotes
    ]
    payload = {
        "proposed_claim_id": claim.id,
        "statement": claim.statement,
        "validated_statement": decision.validated_statement,
        "claim_scope": decision.claim_scope,
        "failed_quotes": failed_quotes,
    }
    return f"""Return exact copied quote from this body excerpt only.

The previous quote either did not match the body exactly or did not directly report the claimed
result. Do not paraphrase. Do not infer. A study count, method, measured outcome, or background
statement is not result evidence. If no exact result-bearing quote is present, return an empty list.

Return JSON only:
{{"supporting_quotes":[{{"quote":"exact copied text from excerpt","section":"Results|Discussion","reason":"why this exact quote reports the claimed result","evidence_role":"direct_result|review_synthesis"}}]}}

Claim context:
{orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")}

Body excerpt:
{body_excerpt}
"""


def repair_supporting_quotes_once(
    *,
    claim: Any,
    decision: ValidatorDecision,
    citation_title: str | None,
    sanitized_body: str,
    provider: LlmProvider,
) -> list[SupportingQuote] | None:
    if decision.verdict not in {"supported", "supported_with_limitations"}:
        return None
    body_excerpt = quote_repair_excerpt(
        sanitized_body=sanitized_body,
        claim=claim,
        decision=decision,
    )
    try:
        response = provider.complete_json(
            quote_repair_prompt(claim=claim, decision=decision, body_excerpt=body_excerpt),
            QUOTE_REPAIR_ADAPTER,
        )
    except Exception:
        return None
    repaired_quotes = grounded_supporting_quotes(
        validator_quotes=response.supporting_quotes,
        citation_title=citation_title,
        sanitized_body=sanitized_body,
    )
    if substantive_grounded_quote_count(
        quotes=repaired_quotes,
        citation_title=citation_title,
    ) > 0:
        return repaired_quotes
    return None


def grounded_validation(
    *,
    claim: ProposedClaim,
    decision: ValidatorDecision,
    citation_id: str,
    citation_title: str | None,
    sanitized_body: str,
    provider: LlmProvider | None = None,
) -> ValidatedClaim:
    quotes = grounded_supporting_quotes(
        validator_quotes=decision.supporting_quotes,
        citation_title=citation_title,
        sanitized_body=sanitized_body,
    )
    model_supported = decision.verdict in {"supported", "supported_with_limitations"}
    entity_matches = decision.entity_match == "exact_or_alias"
    grounded_quote_count = sum(1 for quote in quotes if quote.match_status != "not_found")
    has_grounded_quote = grounded_quote_count > 0
    has_substantive_grounded_quote = (
        substantive_grounded_quote_count(quotes=quotes, citation_title=citation_title) > 0
    )
    if model_supported and not has_substantive_grounded_quote and provider is not None:
        repaired_quotes = repair_supporting_quotes_once(
            claim=claim,
            decision=decision,
            citation_title=citation_title,
            sanitized_body=sanitized_body,
            provider=provider,
        )
        if repaired_quotes is not None:
            quotes = repaired_quotes
            grounded_quote_count = sum(1 for quote in quotes if quote.match_status != "not_found")
            has_grounded_quote = grounded_quote_count > 0
            has_substantive_grounded_quote = True
    has_validated_statement = bool(decision.validated_statement.strip())
    has_claim_scope = bool(decision.claim_scope.strip())
    accepted = (
        model_supported
        and entity_matches
        and has_substantive_grounded_quote
        and has_validated_statement
        and has_claim_scope
    )
    if accepted:
        verdict = decision.verdict
    elif model_supported and not entity_matches:
        verdict = "unsupported"
    elif model_supported:
        verdict = "quote_not_found"
    else:
        verdict = decision.verdict
    limitations = list(decision.limitations)
    if model_supported and not has_validated_statement:
        limitations.append("The validator did not return a validated statement.")
    if model_supported and not has_claim_scope:
        limitations.append("The validator did not return a claim scope.")
    if model_supported and not entity_matches:
        limitations.append(
            "The supporting evidence concerns a different species or entity than the requested target."
        )
    if model_supported and not has_grounded_quote:
        limitations.append("No validator quote could be grounded in the sanitized paper body.")
    elif model_supported and not has_substantive_grounded_quote:
        limitations.append(
            "No grounded quote directly reported the claimed result; the available exact text was "
            "the paper title or article header, methodological context, or background evidence."
        )
    validated_statement = decision.validated_statement.strip()
    if not validated_statement:
        validated_statement = "Rejected because the paper body did not support this proposed claim."
    claim_scope = decision.claim_scope.strip()
    if not claim_scope:
        claim_scope = "Rejected because the paper body did not support a validated claim scope."
    output_quotes = (
        acceptable_grounded_quotes(quotes=quotes, citation_title=citation_title)
        if accepted
        else quotes
    )

    return ValidatedClaim(
        effect_row_id=str(uuid4()),
        proposed_claim_id=claim.id,
        citation_id=citation_id,
        verdict=verdict,
        support_level=decision.support_level,
        claim_scope=claim_scope,
        validated_statement=validated_statement,
        validator_reasoning=decision.reasoning,
        supporting_quotes=output_quotes,
        limitations=limitations,
        accepted=accepted,
    )


def validate_claims_for_paper(
    *,
    nutrient_name: str,
    citation: CandidateCitation,
    claims: list[ProposedClaim],
    sanitized_body: str,
    provider: LlmProvider,
) -> list[ValidatedClaim]:
    response = provider.complete_json(
        validation_prompt(
            nutrient_name=nutrient_name,
            citation=citation,
            claims=claims,
            sanitized_body=sanitized_body,
        ),
        PAPER_VALIDATION_ADAPTER,
    )
    decisions = {decision.proposed_claim_id: decision for decision in response.decisions}
    results: list[ValidatedClaim] = []
    for claim in claims:
        decision = decisions.get(claim.id)
        if decision is None:
            raise ValueError(f"validator_missing_decision:{claim.id}")
        results.append(
            grounded_validation(
                claim=claim,
                decision=decision,
                citation_id=citation.id,
                citation_title=citation.title,
                sanitized_body=sanitized_body,
                provider=provider,
            )
        )
    return results


def _claim_key(citation_id: str, proposed_claim_id: str) -> tuple[str, str]:
    return (citation_id, proposed_claim_id)


def _read_existing_validations(
    path: Path,
    proposed_keys: set[tuple[str, str]],
) -> list[ValidatedClaim]:
    if not path.exists():
        return []
    try:
        claims = VALIDATED_CLAIMS_ADAPTER.validate_python(orjson.loads(path.read_bytes()))
    except Exception:
        return []
    return [
        claim
        for claim in claims
        if _claim_key(claim.citation_id, claim.proposed_claim_id) in proposed_keys
    ]


def _write_validation_summary(
    path: Path,
    accepted: list[ValidatedClaim],
    rejected: list[ValidatedClaim],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        "# Claim Validation Summary",
        "",
        f"Accepted claims: {len(accepted)}",
        f"Rejected claims: {len(rejected)}",
        f"Paper failures: {len(failures)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_validation_artifacts(
    *,
    out_dir: Path,
    accepted: list[ValidatedClaim],
    rejected: list[ValidatedClaim],
    failures: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "validated_claims.json", model_dump_jsonable(accepted))
    write_json(out_dir / "rejected_claims.json", model_dump_jsonable(rejected))
    write_json(out_dir / "validation_failures.json", failures)
    _write_validation_summary(out_dir / "validation_summary.md", accepted, rejected, failures)


def safe_validation_filename(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return safe[:120] or "source"


def validation_failure_code(exc: Exception) -> str:
    if isinstance(exc, BodySanitizationError):
        return "body_sanitization_failed"
    message = str(exc)
    if message.startswith("paper_exceeds_validation_context:"):
        return "paper_exceeds_validation_context"
    if message in {
        "citation_not_found",
        "raw_text_record_not_found",
        "full_text_not_found",
    }:
        return message
    return type(exc).__name__


def write_sanitized_body_preview(
    *,
    out_dir: Path,
    citation_id: str,
    body_text: str | None,
    exc: Exception,
) -> str | None:
    if not isinstance(exc, BodySanitizationError):
        return None
    preview = exc.preview
    if not preview and body_text:
        preview = truncate_text(" ".join(remove_references_section(body_text).split()), max_chars=4_000)
    if not preview:
        return None
    preview_dir = out_dir / "validation_body_previews" / safe_validation_filename(citation_id)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "sanitized_body_preview.txt"
    preview_path.write_text(preview, encoding="utf-8")
    return str(preview_path.relative_to(out_dir))


def _append_audit_event(out_dir: Path, event: dict[str, Any]) -> None:
    with (out_dir / "audit_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(orjson.dumps(event).decode("utf-8") + "\n")


def _update_packet_counts(out_dir: Path, accepted: int, rejected: int) -> None:
    packet_path = out_dir / "packet.json"
    if not packet_path.exists():
        return
    packet = orjson.loads(packet_path.read_bytes())
    packet["counts"]["validated_claims"] = accepted
    packet["counts"]["rejected_claims"] = rejected
    packet["completed_at"] = datetime.now(UTC).isoformat()
    write_json(packet_path, packet)


def validate_proposed_claims(
    *,
    proposed_claims_path: Path,
    sources_path: Path,
    raw_texts_manifest_path: Path,
    raw_texts_dir: Path,
    out_dir: Path,
    provider: LlmProvider,
    update_run_packet: bool = False,
    progress: ValidationProgress | None = None,
    max_body_chars: int = 750_000,
    resume: bool = True,
) -> tuple[list[ValidatedClaim], list[ValidatedClaim], list[dict[str, Any]]]:
    proposed = PROPOSED_CLAIMS_ADAPTER.validate_python(
        orjson.loads(proposed_claims_path.read_bytes())
    )
    sources = SOURCES_ADAPTER.validate_python(orjson.loads(sources_path.read_bytes()))
    raw_records = RAW_TEXTS_ADAPTER.validate_python(
        orjson.loads(raw_texts_manifest_path.read_bytes())
    )
    proposed_keys = {_claim_key(claim.citation_id, claim.id) for claim in proposed}
    accepted = (
        _read_existing_validations(out_dir / "validated_claims.json", proposed_keys)
        if resume
        else []
    )
    rejected = (
        _read_existing_validations(out_dir / "rejected_claims.json", proposed_keys)
        if resume
        else []
    )
    completed_keys = {
        _claim_key(claim.citation_id, claim.proposed_claim_id) for claim in accepted + rejected
    }
    existing_failures: list[dict[str, Any]] = []
    failure_path = out_dir / "validation_failures.json"
    if resume and failure_path.exists():
        try:
            existing_failures = orjson.loads(failure_path.read_bytes())
        except Exception:
            existing_failures = []

    citations = {citation.id: citation for citation in sources}
    records = {record.source_id: record for record in raw_records}
    claims_by_source: dict[str, list[ProposedClaim]] = {}
    for claim in proposed:
        if _claim_key(claim.citation_id, claim.id) not in completed_keys:
            claims_by_source.setdefault(claim.citation_id, []).append(claim)

    papers = list(claims_by_source.items())
    total = len(papers)
    failures = [
        failure
        for failure in existing_failures
        if failure.get("citation_id") not in claims_by_source
    ]
    if update_run_packet:
        _append_audit_event(
            out_dir,
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "claim_validation_started",
                "papers": total,
                "claims": sum(len(items) for _, items in papers),
            },
        )

    for index, (citation_id, paper_claims) in enumerate(papers, start=1):
        citation = citations.get(citation_id)
        record = records.get(citation_id)
        body_text: str | None = None
        if citation is not None and progress is not None:
            progress(index, total, citation, len(paper_claims))
        try:
            if citation is None:
                raise ValueError("citation_not_found")
            if record is None:
                raise ValueError("raw_text_record_not_found")
            body_text = body_text_for_record(raw_texts_dir, record)
            if not body_text:
                raise ValueError("full_text_not_found")
            sanitized = sanitize_paper_body(body_text=body_text, citation=citation)
            if len(sanitized) > max_body_chars:
                raise ValueError(
                    f"paper_exceeds_validation_context:{len(sanitized)}>{max_body_chars}"
                )
            paper_results = validate_claims_for_paper(
                nutrient_name=paper_claims[0].nutrient_name,
                citation=citation,
                claims=paper_claims,
                sanitized_body=sanitized,
                provider=provider,
            )
            accepted.extend(result for result in paper_results if result.accepted)
            rejected.extend(result for result in paper_results if not result.accepted)
            failures = [
                failure for failure in failures if failure.get("citation_id") != citation_id
            ]
        except Exception as exc:
            failure_code = validation_failure_code(exc)
            preview_path = write_sanitized_body_preview(
                out_dir=out_dir,
                citation_id=citation_id,
                body_text=body_text,
                exc=exc,
            )
            failures = [
                failure for failure in failures if failure.get("citation_id") != citation_id
            ]
            failure = {
                "citation_id": citation_id,
                "proposed_claim_ids": [claim.id for claim in paper_claims],
                "failure_code": failure_code,
                "error": str(exc) or type(exc).__name__,
            }
            if preview_path:
                failure["sanitized_body_preview_path"] = preview_path
            failures.append(failure)
            if update_run_packet:
                _append_audit_event(
                    out_dir,
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "event": "claim_validation_paper_failed",
                        "citation_id": citation_id,
                        "failure_code": failure_code,
                        "error": str(exc) or type(exc).__name__,
                    },
                )
        _write_validation_artifacts(
            out_dir=out_dir,
            accepted=accepted,
            rejected=rejected,
            failures=failures,
        )

    if not papers:
        _write_validation_artifacts(
            out_dir=out_dir,
            accepted=accepted,
            rejected=rejected,
            failures=failures,
        )
    if update_run_packet:
        _update_packet_counts(out_dir, len(accepted), len(rejected))
        _append_audit_event(
            out_dir,
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": "claim_validation_completed",
                "accepted": len(accepted),
                "rejected": len(rejected),
                "failures": len(failures),
            },
        )
    return accepted, rejected, failures


def validate_claim(
    claim: ProposedClaim,
    citation: CandidateCitation,
    body_text_without_abstract: str,
) -> ValidatedClaim:
    if claim.proposed_effect_slug == "dental_caries_prevention":
        quote = "Water fluoridation probably reduces caries experience in children"
    else:
        quote = "This exact supporting quote is intentionally absent from the paper body"

    grounded = ground_quote(quote, body_text_without_abstract)
    accepted = claim.proposed_effect_slug == "dental_caries_prevention" and grounded.found

    return ValidatedClaim(
        effect_row_id=str(uuid4()),
        proposed_claim_id=claim.id,
        citation_id=citation.id,
        verdict="supported_with_limitations" if accepted else "quote_not_found",
        support_level="human_systematic_review" if accepted else "unsupported",
        claim_scope=(
            "Human evidence for dental caries prevention from oral fluoride exposure; "
            "benefit estimate varies by population and exposure context."
            if accepted
            else "Rejected because the validator quote could not be grounded in the body text."
        ),
        supporting_quotes=[
            SupportingQuote(
                quote=quote,
                section="Results" if accepted else "Unknown",
                reason=(
                    "Supports reduced caries risk."
                    if accepted
                    else "Demonstrates quote-grounding rejection."
                ),
                match_status=grounded.match_status,
            )
        ],
        limitations=(
            [
                "Effect size may differ between older and contemporary studies.",
                "Evidence depends on exposure route, population, and baseline fluoride availability.",
            ]
            if accepted
            else ["The required supporting quote was not found in the paper body."]
        ),
        accepted=accepted,
    )
