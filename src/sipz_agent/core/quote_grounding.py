from dataclasses import dataclass

from sipz_agent.schemas.claims import QuoteMatchStatus


@dataclass(frozen=True)
class QuoteGroundingResult:
    found: bool
    match_status: QuoteMatchStatus


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalize_ligatures(value: str) -> str:
    return (
        value.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("ﬀ", "ff")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
    )


def normalize_hyphenation(value: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        if value[i] == "-":
            j = i + 1
            while j < len(value) and value[j].isspace():
                j += 1
            if out and out[-1].isalnum() and j < len(value) and value[j].isalnum():
                i = j
                continue
        out.append(value[i])
        i += 1
    return "".join(out)


def normalize_for_tier3(value: str) -> str:
    return normalize_whitespace(normalize_hyphenation(normalize_ligatures(value)))


def ground_quote(quote: str, body_text: str) -> QuoteGroundingResult:
    quote = quote.strip()
    if not quote or not body_text:
        return QuoteGroundingResult(found=False, match_status="not_found")

    if quote in body_text:
        return QuoteGroundingResult(found=True, match_status="exact")

    if normalize_whitespace(quote) in normalize_whitespace(body_text):
        return QuoteGroundingResult(found=True, match_status="normalized_whitespace")

    if normalize_for_tier3(quote) in normalize_for_tier3(body_text):
        return QuoteGroundingResult(found=True, match_status="dehyphenated_ligature_normalized")

    return QuoteGroundingResult(found=False, match_status="not_found")
