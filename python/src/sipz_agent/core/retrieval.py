from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import threading
import time
from typing import Any, Literal
from urllib import error, parse, request
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from sipz_agent.core.config import load_config
from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import CandidateCitation

HTTP_MAX_ATTEMPTS = 4
HTTP_BACKOFF_SECONDS = 1.0
HTTP_MAX_RETRY_DELAY_SECONDS = 10.0
HOST_MIN_INTERVAL_SECONDS = {
    "eutils.ncbi.nlm.nih.gov": 0.4,
    "api.semanticscholar.org": 1.0,
}
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_REQUEST_TIMES: dict[str, float] = {}
_REQUEST_THROTTLE_LOCK = threading.Lock()


class DemoNutrient(BaseModel):
    id: str
    name: str


class DemoCorpus(BaseModel):
    nutrient: DemoNutrient
    citations: list[CandidateCitation]


class CandidateFinderOutput(BaseModel):
    nutrient_id: str | None
    normalized_nutrient_name: str
    citations: list[CandidateCitation]


class CandidatePageOutput(BaseModel):
    citations: list[CandidateCitation]
    raw_candidate_count: int = 0
    source_failures: list[str] = Field(default_factory=list)


class VisiblePageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip_depth += 1
            return
        if tag == "meta":
            attributes = dict(attrs)
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            content = attributes.get("content")
            if content and name in {"description", "og:description", "twitter:description"}:
                self.chunks.append(content)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if len(text) >= 20:
            self.chunks.append(text)


def slugify(value: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", value.strip().lower()))


def load_demo_corpus(nutrient_name: str) -> DemoCorpus:
    root = Path(__file__).resolve().parents[3]
    path = root / "examples" / "demo-corpus" / f"{slugify(nutrient_name)}.json"
    return DemoCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def retry_delay_seconds(exc: error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return min(HTTP_MAX_RETRY_DELAY_SECONDS, max(0.0, float(retry_after)))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                return min(
                    HTTP_MAX_RETRY_DELAY_SECONDS,
                    max(0.0, (retry_at - datetime.now(UTC)).total_seconds()),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return min(HTTP_MAX_RETRY_DELAY_SECONDS, HTTP_BACKOFF_SECONDS * (2**attempt))


def throttle_request(url: str) -> None:
    host = parse.urlparse(url).netloc.lower()
    minimum_interval = HOST_MIN_INTERVAL_SECONDS.get(host)
    if minimum_interval is None:
        return
    with _REQUEST_THROTTLE_LOCK:
        now = time.monotonic()
        delay = minimum_interval - (now - _REQUEST_TIMES.get(host, 0.0))
        if delay > 0:
            time.sleep(delay)
        _REQUEST_TIMES[host] = time.monotonic()


def urlopen_with_retry(req: request.Request, timeout: int):
    for attempt in range(HTTP_MAX_ATTEMPTS):
        throttle_request(req.full_url)
        try:
            return request.urlopen(req, timeout=timeout)
        except error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUS_CODES or attempt == HTTP_MAX_ATTEMPTS - 1:
                raise
            time.sleep(retry_delay_seconds(exc, attempt))
    raise RuntimeError("http_retry_loop_exhausted")


def json_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    req = request.Request(url, headers=headers or {"User-Agent": "sipz-nutrient-agent/0.1"})
    with urlopen_with_retry(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def text_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    req = request.Request(url, headers=headers or {"User-Agent": "sipz-nutrient-agent/0.1"})
    with urlopen_with_retry(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def truncate_text(value: str, max_chars: int = 4000) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def visible_page_summary_from_html(html_text: str, max_chars: int = 4000) -> str | None:
    parser = VisiblePageTextParser()
    try:
        parser.feed(html_text)
    except Exception:
        return None
    seen: set[str] = set()
    unique_chunks: list[str] = []
    for chunk in parser.chunks:
        normalized = " ".join(chunk.split())
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(normalized)
    summary = truncate_text(" ".join(unique_chunks), max_chars=max_chars)
    return summary or None


def fetch_visible_page_summary(url: str | None) -> str | None:
    if not url:
        return None
    headers = {
        "User-Agent": "sipz-nutrient-agent/0.1",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        return visible_page_summary_from_html(text_get(url, headers=headers, timeout=15))
    except Exception:
        return None


def candidate_landing_url(citation: CandidateCitation) -> str | None:
    if citation.url:
        return str(citation.url)
    if citation.doi:
        return f"https://doi.org/{citation.doi}"
    return None


def xml_body_text(xml_text: str) -> str | None:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    paragraphs = [" ".join(text.split()) for text in root.itertext() if text and text.strip()]
    body = "\n".join(paragraphs).strip()
    return body or None


def pubmed_abstracts_from_xml(xml_text: str) -> dict[str, str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return {}

    abstracts: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        abstract_parts = [
            " ".join(part.itertext()).strip()
            for part in article.findall(".//Article/Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part).strip()
        if abstract:
            abstracts[pmid_el.text.strip()] = abstract
    return abstracts


def fetch_pubmed_abstracts(pmids: list[str], email: str | None) -> dict[str, str]:
    if not pmids:
        return {}
    params: dict[str, str | int] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    api_key = load_config().ncbi_api_key
    if api_key:
        params["api_key"] = api_key
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + query_string(params)
    try:
        return pubmed_abstracts_from_xml(text_get(url))
    except Exception as exc:
        raise RuntimeError(f"ncbi_request_failed:{type(exc).__name__}") from None


def crossref_url(url: str, params: dict[str, str | int] | None = None) -> str:
    query_params = dict(params or {})
    mailto = load_config().crossref_mailto
    if mailto:
        query_params["mailto"] = mailto
    if not query_params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query_string(query_params)}"


def fetch_crossref_abstract_by_doi(doi: str) -> str | None:
    try:
        payload = json_get(crossref_url(f"https://api.crossref.org/works/{parse.quote(doi)}"))
    except Exception as exc:
        raise RuntimeError(f"crossref_request_failed:{type(exc).__name__}") from None
    return text_from_markup(payload.get("message", {}).get("abstract"))


def fetch_openalex_abstract_by_doi(doi: str) -> str | None:
    payload = json_get(
        openalex_url(f"https://api.openalex.org/works/https://doi.org/{parse.quote(doi)}")
    )
    return openalex_abstract_from_inverted_index(payload.get("abstract_inverted_index"))


def openalex_url(url: str, params: dict[str, str | int] | None = None) -> str:
    query_params = dict(params or {})
    api_key = load_config().openalex_api_key
    if api_key:
        query_params["api_key"] = api_key
    if not query_params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query_string(query_params)}"


def nested_get(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def elsevier_headers(api_key: str) -> dict[str, str]:
    return {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
        "User-Agent": "sipz-nutrient-agent/0.1",
    }


def fetch_elsevier_json(url: str, api_key: str) -> dict[str, Any] | None:
    try:
        return json_get(url, headers=elsevier_headers(api_key))
    except Exception:
        return None


def extract_elsevier_description(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    candidates = [
        nested_get(payload, ["full-text-retrieval-response", "coredata", "dc:description"]),
        nested_get(payload, ["abstracts-retrieval-response", "coredata", "dc:description"]),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            abstract = text_from_markup(candidate)
            if abstract:
                return abstract
    return None


def fetch_elsevier_abstract_by_doi(doi: str) -> str | None:
    api_key = load_config().elsevier_api_key
    if not api_key:
        return None

    escaped_doi = parse.quote(doi, safe="")
    urls = [
        f"https://api.elsevier.com/content/article/doi/{escaped_doi}?httpAccept=application/json",
        f"https://api.elsevier.com/content/abstract/doi/{escaped_doi}?httpAccept=application/json",
    ]
    for url in urls:
        abstract = extract_elsevier_description(fetch_elsevier_json(url, api_key))
        if abstract:
            return abstract
    return None


class DoiMetadata(BaseModel):
    abstract: str | None = None
    pmid: str | None = None


def fetch_europe_pmc_metadata_by_doi(doi: str) -> DoiMetadata:
    query = f'DOI:"{doi}"'
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query_string(
        {"query": query, "format": "json", "pageSize": 1}
    )
    payload = json_get(url)
    results = payload.get("resultList", {}).get("result", [])
    if not results:
        return DoiMetadata()
    item = results[0]
    return DoiMetadata(
        abstract=item.get("abstractText"),
        pmid=str(item.get("pmid")) if item.get("pmid") else None,
    )


def query_string(params: dict[str, str | int]) -> str:
    return parse.urlencode(params)


def year_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    normalized = doi.strip()
    normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    return normalized.lower() or None


def first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def text_from_markup(value: str | None) -> str | None:
    if not value:
        return None
    without_tags = re.sub(r"<[^>]+>", " ", value)
    normalized = " ".join(unescape(without_tags).split())
    return normalized or None


def openalex_abstract_from_inverted_index(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    positioned_words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned_words.append((position, word))
    if not positioned_words:
        return None
    return " ".join(word for _, word in sorted(positioned_words))


def candidate_query(nutrient_name: str) -> str:
    return f"{nutrient_name.strip()} health human"


def resolved_candidate_query(nutrient_name: str, query_override: str | None) -> str:
    return " ".join((query_override or candidate_query(nutrient_name)).split())


def natural_language_query(query: str) -> str:
    without_negatives = strip_negative_query_clauses(query)
    without_operators = re.sub(
        r"\b(?:AND|OR|NOT)\b",
        " ",
        without_negatives,
        flags=re.IGNORECASE,
    )
    without_syntax = re.sub(r'[()"]', " ", without_operators)
    return " ".join(without_syntax.split())


def strip_negative_query_clauses(query: str) -> str:
    stripped = re.sub(
        r"\bNOT\s*\([^)]*\)",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"\bNOT\s+\"[^\"]+\"",
        " ",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"\bNOT\s+\S+",
        " ",
        stripped,
        flags=re.IGNORECASE,
    )
    return " ".join(stripped.split())


def query_for_source(nutrient_name: str, query_override: str | None, source: str) -> str:
    query = resolved_candidate_query(nutrient_name, query_override)
    if source in {"pubmed", "europe_pmc"}:
        return query
    if source == "firecrawl":
        return re.sub(
            r"\bAND\b",
            " ",
            strip_negative_query_clauses(query),
            flags=re.IGNORECASE,
        )
    return natural_language_query(query)


def depth_limit(depth: StudyDepth) -> int:
    if depth == "light":
        return 5
    if depth == "deep":
        return 25
    return 10


def selection_reason_for_source(
    *,
    source_label: str,
    query: str,
    has_abstract: bool = False,
    has_body_text: bool = False,
    has_doi: bool = False,
    has_pmid: bool = False,
) -> str:
    evidence_parts = []
    if has_abstract:
        evidence_parts.append("abstract metadata")
    if has_body_text:
        evidence_parts.append("full-text body text")
    if has_doi:
        evidence_parts.append("DOI metadata")
    if has_pmid:
        evidence_parts.append("PMID metadata")
    evidence_text = f" Includes {', '.join(evidence_parts)}." if evidence_parts else ""
    return f"Selected because {source_label} returned this record for the candidate query '{query}'.{evidence_text}"


def find_pubmed_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    query = query_for_source(nutrient_name, query_override, "pubmed")
    limit = page_size or depth_limit(depth)
    config = load_config()
    email = config.ncbi_email
    base_params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "retstart": page_index * limit,
        "sort": "relevance",
    }
    if email:
        base_params["email"] = email
    if config.ncbi_api_key:
        base_params["api_key"] = config.ncbi_api_key
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + query_string(
        base_params
    )
    try:
        search = json_get(search_url)
    except Exception as exc:
        raise RuntimeError(f"ncbi_request_failed:{type(exc).__name__}") from None
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        # Still call summary in tests and to keep the code path stable for empty results.
        ids = []
    summary_params: dict[str, str | int] = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
    }
    if email:
        summary_params["email"] = email
    if config.ncbi_api_key:
        summary_params["api_key"] = config.ncbi_api_key
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + query_string(
        summary_params
    )
    try:
        summary = json_get(summary_url)
    except Exception as exc:
        raise RuntimeError(f"ncbi_request_failed:{type(exc).__name__}") from None
    result = summary.get("result", {})
    try:
        abstracts = fetch_pubmed_abstracts([str(pmid) for pmid in ids], email)
    except Exception:
        abstracts = {}

    citations: list[CandidateCitation] = []
    for pmid in result.get("uids", []):
        item = result.get(pmid, {})
        doi = None
        for article_id in item.get("articleids", []):
            if article_id.get("idtype") == "doi":
                doi = article_id.get("value")
                break
        title = item.get("title") or f"PubMed record {pmid}"
        citations.append(
            CandidateCitation(
                id=f"pmid:{pmid}",
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=normalize_doi(doi),
                pmid=str(pmid),
                year=year_from_text(item.get("pubdate")),
                source="pubmed",
                retrieval_query=query,
                selection_reason=selection_reason_for_source(
                    source_label="PubMed",
                    query=query,
                    has_abstract=bool(abstracts.get(str(pmid))),
                    has_doi=bool(doi),
                    has_pmid=True,
                ),
                abstract=abstracts.get(str(pmid)),
            )
        )
    return citations


def europe_pmc_full_text_url(item: dict[str, Any]) -> str | None:
    urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
    if not urls:
        return None
    return urls[0].get("url")


def fetch_europe_pmc_body_text(item: dict[str, Any]) -> str | None:
    record_id = item.get("id")
    if not record_id:
        return None
    source = item.get("source") or ("PMC" if str(record_id).upper().startswith("PMC") else "MED")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{record_id}/fullTextXML"
    try:
        return xml_body_text(text_get(url))
    except Exception:
        return europe_pmc_full_text_url(item)


def find_europe_pmc_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    query = query_for_source(nutrient_name, query_override, "europe_pmc")
    limit = page_size or depth_limit(depth)
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query_string(
        {
            "query": query,
            "format": "json",
            "pageSize": limit,
            "page": page_index + 1,
        }
    )
    payload = json_get(url)
    citations: list[CandidateCitation] = []
    for item in payload.get("resultList", {}).get("result", []):
        record_id = item.get("pmid") or item.get("id") or item.get("doi")
        title = item.get("title")
        if not record_id or not title:
            continue
        pmid = item.get("pmid")
        body_text = fetch_europe_pmc_body_text(item)
        citations.append(
            CandidateCitation(
                id=f"europepmc:{record_id}",
                title=title,
                url=(f"https://europepmc.org/article/MED/{pmid}" if pmid else None),
                doi=normalize_doi(item.get("doi")),
                pmid=str(pmid) if pmid else None,
                year=year_from_text(str(item.get("pubYear") or "")),
                source="europe_pmc",
                retrieval_query=query,
                selection_reason=selection_reason_for_source(
                    source_label="Europe PMC",
                    query=query,
                    has_abstract=bool(item.get("abstractText")),
                    has_body_text=bool(body_text and not str(body_text).startswith("http")),
                    has_doi=bool(item.get("doi")),
                    has_pmid=bool(pmid),
                ),
                abstract=item.get("abstractText"),
                body_text=body_text,
            )
        )
    return citations


def find_openalex_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    query = query_for_source(nutrient_name, query_override, "openalex")
    limit = page_size or depth_limit(depth)
    url = openalex_url(
        "https://api.openalex.org/works",
        {
            "search": query,
            "per-page": limit,
            "page": page_index + 1,
            "sort": "relevance_score:desc",
        },
    )
    try:
        payload = json_get(url)
    except Exception as exc:
        # HTTPError includes the request URL, which contains the API key.
        raise RuntimeError(f"openalex_request_failed:{type(exc).__name__}") from None
    citations: list[CandidateCitation] = []
    for item in payload.get("results", []):
        record_id = item.get("id")
        title = item.get("display_name")
        if not record_id or not title:
            continue
        location = item.get("primary_location") or {}
        landing_url = location.get("landing_page_url") or item.get("id")
        if isinstance(landing_url, str) and not landing_url.startswith(("http://", "https://")):
            landing_url = None
        citations.append(
            CandidateCitation(
                id=f"openalex:{str(record_id).rsplit('/', 1)[-1]}",
                title=title,
                url=landing_url,
                doi=normalize_doi(item.get("doi")),
                year=item.get("publication_year"),
                source="openalex",
                retrieval_query=query,
                selection_reason=selection_reason_for_source(
                    source_label="OpenAlex",
                    query=query,
                    has_abstract=bool(openalex_abstract_from_inverted_index(item.get("abstract_inverted_index"))),
                    has_doi=bool(item.get("doi")),
                ),
                abstract=openalex_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            )
        )
    return citations


def find_semantic_scholar_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    query = query_for_source(nutrient_name, query_override, "semantic_scholar")
    limit = page_size or depth_limit(depth)
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + query_string(
        {
            "query": query,
            "limit": limit,
            "offset": page_index * limit,
            "fields": "title,year,abstract,url,externalIds",
        }
    )
    payload = json_get(url)
    citations: list[CandidateCitation] = []
    for item in payload.get("data", []):
        paper_id = item.get("paperId")
        title = item.get("title")
        if not paper_id or not title:
            continue
        external = item.get("externalIds") or {}
        pmid = external.get("PubMed")
        citations.append(
            CandidateCitation(
                id=f"semanticscholar:{paper_id}",
                title=title,
                url=item.get("url"),
                doi=normalize_doi(external.get("DOI")),
                pmid=str(pmid) if pmid else None,
                year=item.get("year"),
                source="semantic_scholar",
                retrieval_query=query,
                selection_reason=selection_reason_for_source(
                    source_label="Semantic Scholar",
                    query=query,
                    has_abstract=bool(item.get("abstract")),
                    has_doi=bool(external.get("DOI")),
                    has_pmid=bool(pmid),
                ),
                abstract=item.get("abstract"),
            )
        )
    return citations


def find_crossref_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    query = query_for_source(nutrient_name, query_override, "crossref")
    limit = page_size or depth_limit(depth)
    url = crossref_url(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": query,
            "rows": limit,
            "offset": page_index * limit,
            "sort": "relevance",
        },
    )
    try:
        payload = json_get(url)
    except Exception as exc:
        raise RuntimeError(f"crossref_request_failed:{type(exc).__name__}") from None
    citations: list[CandidateCitation] = []
    for item in payload.get("message", {}).get("items", []):
        doi = normalize_doi(item.get("DOI"))
        title = first(item.get("title"))
        if not doi or not title:
            continue
        date_parts = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or []
        )
        year = None
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        citations.append(
            CandidateCitation(
                id=f"doi:{doi}",
                title=title,
                url=item.get("URL") or f"https://doi.org/{doi}",
                doi=doi,
                year=year,
                source="crossref",
                retrieval_query=query,
                selection_reason=selection_reason_for_source(
                    source_label="Crossref",
                    query=query,
                    has_abstract=bool(text_from_markup(item.get("abstract"))),
                    has_doi=True,
                ),
                abstract=text_from_markup(item.get("abstract")),
            )
        )
    return citations


def firecrawl_search_url(api_url: str) -> str:
    normalized = api_url.rstrip("/")
    if normalized.endswith("/v1/search"):
        return normalized
    return f"{normalized}/v1/search"


def find_firecrawl_candidates(
    nutrient_name: str,
    depth: StudyDepth,
    page_index: int = 0,
    page_size: int | None = None,
    query_override: str | None = None,
) -> list[CandidateCitation]:
    if page_index > 0:
        return []
    config = load_config()
    if not config.firecrawl_api_key:
        return []
    search_terms = (
        query_for_source(nutrient_name, query_override, "firecrawl")
        if query_override
        else f'"{nutrient_name}" health human'
    )
    query = f"site:pubmed.ncbi.nlm.nih.gov OR site:europepmc.org {search_terms}"
    payload = {"query": query, "limit": page_size or depth_limit(depth)}
    req = request.Request(
        firecrawl_search_url(config.firecrawl_api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.firecrawl_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen_with_retry(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    citations: list[CandidateCitation] = []
    for index, item in enumerate(data.get("data", []), start=1):
        title = item.get("title")
        url = item.get("url")
        if not title or not url:
            continue
        citations.append(
            CandidateCitation(
                id=f"firecrawl:{slugify(title) or index}",
                title=title,
                url=url,
                source="firecrawl",
                retrieval_query=query,
                selection_reason=(
                    "Selected because Firecrawl site-search returned this result while looking "
                    "for nutrient-health papers on literature sites."
                ),
                abstract=item.get("description"),
            )
        )
    return citations


def collect_safely(collector, nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    try:
        return collector(nutrient_name, depth)
    except Exception:
        return []


def citation_key(citation: CandidateCitation) -> str:
    return (
        normalize_doi(citation.doi)
        or (f"pmid:{citation.pmid}" if citation.pmid else None)
        or citation.title.strip().lower()
    )


def deduplicate_citations(citations: list[CandidateCitation]) -> list[CandidateCitation]:
    seen: set[str] = set()
    deduped: list[CandidateCitation] = []
    for citation in citations:
        key = citation_key(citation)
        if key in seen:
            for index, existing in enumerate(deduped):
                existing_key = citation_key(existing)
                if existing_key != key:
                    continue
                deduped[index] = existing.model_copy(
                    update={
                        "abstract": existing.abstract or citation.abstract,
                        "page_summary": existing.page_summary or citation.page_summary,
                        "body_text": existing.body_text or citation.body_text,
                        "doi": existing.doi or citation.doi,
                        "pmid": existing.pmid or citation.pmid,
                        "url": existing.url or citation.url,
                        "selection_reason": " ".join(
                            part
                            for part in [
                                existing.selection_reason,
                                (
                                    f"Merged additional metadata from {citation.source}."
                                    if citation.abstract or citation.page_summary or citation.body_text
                                    else None
                                ),
                            ]
                            if part
                        )
                        or None,
                    }
                )
                break
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def add_page_summaries_to_missing_abstracts(
    citations: list[CandidateCitation],
) -> list[CandidateCitation]:
    enriched: list[CandidateCitation] = []
    for citation in citations:
        if citation.abstract or citation.page_summary:
            enriched.append(citation)
            continue

        page_summary = fetch_visible_page_summary(candidate_landing_url(citation))
        if not page_summary:
            enriched.append(citation)
            continue

        enriched.append(
            citation.model_copy(
                update={
                    "page_summary": page_summary,
                    "selection_reason": (
                        f"{citation.selection_reason or 'Selected by retrieval.'} "
                        "No API abstract was available; captured visible landing-page text "
                        "for light screening context."
                    ),
                }
            )
        )
    return enriched


def enrich_missing_abstracts_from_metadata(
    citations: list[CandidateCitation],
    disabled_sources: set[str] | None = None,
) -> list[CandidateCitation]:
    email = load_config().ncbi_email
    disabled = disabled_sources if disabled_sources is not None else set()
    enriched: list[CandidateCitation] = []
    for citation in citations:
        abstract = citation.abstract
        pmid = citation.pmid
        metadata_sources: list[str] = []

        if not abstract and citation.doi and "crossref" not in disabled:
            try:
                abstract = fetch_crossref_abstract_by_doi(citation.doi)
            except Exception:
                disabled.add("crossref")
                abstract = None
            if abstract:
                metadata_sources.append("Crossref DOI detail")

        if not abstract and citation.doi and "openalex" not in disabled:
            try:
                abstract = fetch_openalex_abstract_by_doi(citation.doi)
            except Exception:
                disabled.add("openalex")
                abstract = None
            if abstract:
                metadata_sources.append("OpenAlex DOI detail")

        if not abstract and citation.doi:
            abstract = fetch_elsevier_abstract_by_doi(citation.doi)
            if abstract:
                metadata_sources.append("Elsevier DOI detail")

        if citation.doi and (not abstract or not pmid) and "europe_pmc" not in disabled:
            try:
                europe_pmc = fetch_europe_pmc_metadata_by_doi(citation.doi)
            except Exception:
                disabled.add("europe_pmc")
                europe_pmc = DoiMetadata()
            if not abstract and europe_pmc.abstract:
                abstract = europe_pmc.abstract
                metadata_sources.append("Europe PMC DOI detail")
            if not pmid and europe_pmc.pmid:
                pmid = europe_pmc.pmid
                metadata_sources.append("Europe PMC PMID lookup")

        if not abstract and pmid and "pubmed" not in disabled:
            try:
                abstract = fetch_pubmed_abstracts([pmid], email).get(pmid)
            except Exception:
                disabled.add("pubmed")
                abstract = None
            if abstract:
                metadata_sources.append("PubMed PMID detail")

        if abstract or pmid:
            enriched.append(
                citation.model_copy(
                    update={
                        "abstract": abstract,
                        "pmid": pmid,
                        "selection_reason": " ".join(
                            part
                            for part in [
                                citation.selection_reason,
                                (
                                    "Added missing metadata from "
                                    f"{', '.join(metadata_sources)}."
                                    if metadata_sources
                                    else None
                                ),
                            ]
                            if part
                        )
                        or None,
                    }
                )
            )
        else:
            enriched.append(citation)

    return enriched


def find_live_candidate_page(
    nutrient_name: str,
    depth: StudyDepth,
    *,
    page_index: int,
    page_size: int = 10,
    queries: list[str] | None = None,
    disabled_collectors: set[str] | None = None,
    disabled_metadata_sources: set[str] | None = None,
    collector_mode: Literal["primary", "fallback", "all"] = "all",
) -> CandidatePageOutput:
    primary_collectors = [
        find_pubmed_candidates,
        find_europe_pmc_candidates,
        find_openalex_candidates,
        find_crossref_candidates,
    ]
    fallback_collectors = [
        find_semantic_scholar_candidates,
        find_firecrawl_candidates,
    ]
    collectors = {
        "primary": primary_collectors,
        "fallback": fallback_collectors,
        "all": primary_collectors + fallback_collectors,
    }[collector_mode]
    planned_queries = list(
        dict.fromkeys(
            " ".join(query.split())
            for query in (queries or [candidate_query(nutrient_name)])
            if query.strip()
        )
    )
    planned_queries = planned_queries[:page_size] or [candidate_query(nutrient_name)]
    base_size, remainder = divmod(page_size, len(planned_queries))
    query_sizes = [
        base_size + (1 if index < remainder else 0)
        for index in range(len(planned_queries))
    ]
    citations: list[CandidateCitation] = []
    source_failures: list[str] = []
    for collector in collectors:
        collector_name = collector.__name__
        if disabled_collectors is not None and collector_name in disabled_collectors:
            continue
        for query, query_size in zip(planned_queries, query_sizes, strict=True):
            try:
                citations.extend(
                    collector(
                        nutrient_name,
                        depth,
                        page_index=page_index,
                        page_size=query_size,
                        query_override=query,
                    )
                )
            except Exception as exc:
                source_failures.append(
                    f"{collector_name} [{query}]: {type(exc).__name__}: {exc}"
                )
                if disabled_collectors is not None:
                    disabled_collectors.add(collector_name)
                break
    deduped = deduplicate_citations(citations)
    enriched = enrich_missing_abstracts_from_metadata(
        deduped,
        disabled_sources=disabled_metadata_sources,
    )
    return CandidatePageOutput(
        citations=add_page_summaries_to_missing_abstracts(enriched),
        raw_candidate_count=len(citations),
        source_failures=source_failures,
    )


def find_live_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    return find_live_candidate_page(
        nutrient_name,
        depth,
        page_index=0,
        page_size=depth_limit(depth),
    ).citations


def find_candidate_papers(
    nutrient_name: str,
    depth: StudyDepth,
    demo: bool,
) -> CandidateFinderOutput:
    if not demo:
        return CandidateFinderOutput(
            nutrient_id=None,
            normalized_nutrient_name=nutrient_name.strip(),
            citations=find_live_candidates(nutrient_name, depth),
        )

    corpus = load_demo_corpus(nutrient_name)
    return CandidateFinderOutput(
        nutrient_id=corpus.nutrient.id,
        normalized_nutrient_name=corpus.nutrient.name,
        citations=corpus.citations,
    )
