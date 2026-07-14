from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import json
import re
import tarfile
from typing import Any, cast, get_args
from urllib import error, parse, request
from xml.etree import ElementTree

import orjson
from pydantic import TypeAdapter
from pypdf import PdfReader

from sipz_agent.core.artifacts import model_dump_jsonable, write_json
from sipz_agent.core.config import load_config
from sipz_agent.core.retrieval import (
    elsevier_headers,
    crossref_url,
    json_get,
    openalex_url,
    text_get,
    truncate_text,
    visible_page_summary_from_html,
    xml_body_text,
)
from sipz_agent.schemas.artifacts import Packet
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.raw_texts import (
    FullTextRetrievalAttempt,
    FullTextRetrievalMethod,
    RawTextRecord,
)

SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
PACKET_ADAPTER = TypeAdapter(Packet)
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])
FULL_TEXT_ATTEMPTS_ADAPTER = TypeAdapter(list[FullTextRetrievalAttempt])
PUBLISHER_FULL_TEXT_MARKERS = [
    "introduction",
    "materials and methods",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
]
ACCESS_DENIED_MARKERS = [
    "access denied",
    "performing security verification",
    "you don't have permission",
    "captcha",
    "recaptcha",
    "just a moment",
    "cloudflare",
]
PAYWALL_MARKERS = [
    "payment required",
    "purchase access",
    "rent this article",
    "subscribe to access",
    "institutional access",
    "log in to access",
    "this is a preview of subscription content",
    "preview of subscription content",
]
FULL_TEXT_STATUS_FIELDS = [
    "full_text_found",
    "abstract_only",
    "page_summary_only",
    "blocked",
    "blocked_by_cloudflare",
    "paywalled",
    "pdf_parse_failed",
    "no_oa_location",
    "not_available",
    "retrieval_error",
]
FULL_TEXT_RETRIEVAL_METHODS = set(get_args(FullTextRetrievalMethod))
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Connection": "keep-alive",
}
PUBLISHER_HTML_HEADERS = BROWSER_HEADERS | {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
PDF_HEADERS = BROWSER_HEADERS | {
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
}
XML_HEADERS = BROWSER_HEADERS | {
    "Accept": "application/xml,text/xml,*/*;q=0.8",
}


@dataclass(frozen=True)
class FullTextCandidate:
    url: str
    method: str
    content_type: str
    access_evidence: str
    oa_url: str | None = None
    license: str | None = None
    manuscript_version: str | None = None
    oa_host_type: str | None = None
    oa_status: str | None = None


class FullTextResult:
    def __init__(
        self,
        *,
        record: RawTextRecord,
        body_text: str | None = None,
        attempts: list[FullTextRetrievalAttempt] | None = None,
    ) -> None:
        self.record = record
        self.body_text = body_text
        self.attempts = attempts or []


@dataclass(frozen=True)
class ManualFullTextIngestionResult:
    run_dir: Path
    ingested: list[str]
    missing_assumed_paywalled: list[str]
    failed: list[str]
    unchanged: list[str]
    counts: dict[str, int]


class FullTextRetrievalFailure(Exception):
    def __init__(self, status: str, notes: str, *, http_status: int | None = None) -> None:
        super().__init__(notes)
        self.status = status
        self.notes = notes
        self.http_status = http_status


class PublisherArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if len(text) >= 2:
            self.chunks.append(text)


def safe_source_filename(source_id: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", source_id).strip("._-")
    if not safe:
        safe = f"source_{index}"
    return f"{index:03d}_{safe[:120]}.txt"


def looks_like_url(value: str | None) -> bool:
    return bool(value and value.strip().lower().startswith(("http://", "https://")))


def normalized_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = doi.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    return value.lower() or None


def method_for_full_text_url(url: str, content_type: str, default_method: str) -> str:
    lowered_url = url.lower()
    lowered_type = content_type.lower()
    if "onlinelibrary.wiley.com" in lowered_url and "full-xml" in lowered_url:
        return "wiley_full_xml"
    if "mdpi.com" in lowered_url and lowered_url.rstrip("/").endswith("/xml"):
        return "mdpi_xml"
    if "mdpi.com" in lowered_url and "/pdf" in lowered_url:
        return "mdpi_pdf"
    if "xml" in lowered_type or "full-xml" in lowered_url:
        return "crossref_full_xml"
    if "pdf" in lowered_type or "/pdf" in lowered_url or lowered_url.endswith(".pdf"):
        return default_method if default_method.endswith("_pdf") else "crossref_pdf"
    return default_method


def coerce_full_text_method(method: str | None) -> FullTextRetrievalMethod:
    value = (method or "").strip()
    if value in FULL_TEXT_RETRIEVAL_METHODS:
        return cast(FullTextRetrievalMethod, value)
    return "none"


def unique_candidates(candidates: list[FullTextCandidate]) -> list[FullTextCandidate]:
    seen: set[str] = set()
    unique: list[FullTextCandidate] = []
    for candidate in candidates:
        key = candidate.url
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def is_access_denied_text(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in ACCESS_DENIED_MARKERS)


def is_paywall_text(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in PAYWALL_MARKERS)


def bytes_get(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> bytes:
    req = request.Request(url, headers=headers or BROWSER_HEADERS)
    with request.urlopen(req, timeout=timeout) as response:
        return response.read()


def mdpi_article_base_url_from_doi(doi: str | None) -> str | None:
    normalized = normalized_doi(doi)
    if not normalized:
        return None
    match = re.fullmatch(r"10\.3390/([a-z]+)(\d{2})(\d{2})(\d+)", normalized)
    if not match:
        return None
    journal, volume, issue, article = match.groups()
    issns = {
        "nu": "2072-6643",
    }
    issn = issns.get(journal)
    if not issn:
        return None
    return f"https://www.mdpi.com/{issn}/{int(volume)}/{int(issue)}/{int(article)}"


def mdpi_candidates_for_doi(doi: str | None) -> list[FullTextCandidate]:
    base_url = mdpi_article_base_url_from_doi(doi)
    if not base_url:
        return []
    return [
        FullTextCandidate(
            url=base_url,
            method="publisher_page",
            content_type="html",
            access_evidence="Derived MDPI article HTML URL from DOI.",
            oa_url=base_url,
            license=None,
        ),
        FullTextCandidate(
            url=f"{base_url}/xml",
            method="mdpi_xml",
            content_type="xml",
            access_evidence="Derived MDPI article XML URL from DOI.",
            oa_url=f"{base_url}/xml",
            license=None,
        ),
        FullTextCandidate(
            url=f"{base_url}/pdf?download=1",
            method="mdpi_pdf",
            content_type="pdf",
            access_evidence="Derived MDPI direct PDF download URL from DOI.",
            oa_url=f"{base_url}/pdf?download=1",
            license=None,
        ),
        FullTextCandidate(
            url=f"{base_url}/pdf",
            method="mdpi_pdf",
            content_type="pdf",
            access_evidence="Derived MDPI PDF URL from DOI.",
            oa_url=f"{base_url}/pdf",
            license=None,
        ),
    ]


def source_landing_urls(citation: CandidateCitation) -> list[str]:
    urls = []
    if citation.doi:
        urls.append(f"https://doi.org/{citation.doi}")
    mdpi_base_url = mdpi_article_base_url_from_doi(citation.doi)
    if mdpi_base_url:
        urls.append(mdpi_base_url)
    if citation.url:
        urls.append(str(citation.url))
    return list(dict.fromkeys(urls))


def is_elsevier_candidate(citation: CandidateCitation) -> bool:
    doi = (citation.doi or "").lower()
    if doi.startswith("10.1016/"):
        return True
    if citation.url:
        host = parse.urlparse(str(citation.url)).netloc.lower()
        return "sciencedirect.com" in host or "elsevier.com" in host
    return False


def firecrawl_scrape_url(api_url: str) -> str:
    normalized = api_url.rstrip("/")
    if normalized.endswith("/v1/scrape"):
        return normalized
    return f"{normalized}/v1/scrape"


def is_literature_index_url(url: str) -> bool:
    host = parse.urlparse(url).netloc.lower()
    return any(
        index_host in host
        for index_host in [
            "pubmed.ncbi.nlm.nih.gov",
            "europepmc.org",
            "crossref.org",
            "openalex.org",
            "semanticscholar.org",
        ]
    )


def looks_like_full_article_text(text: str | None) -> bool:
    if (
        not text
        or len(text) < 1500
        or is_access_denied_text(text)
        or is_paywall_text(text)
    ):
        return False
    lowered = text.lower()
    marker_count = sum(1 for marker in PUBLISHER_FULL_TEXT_MARKERS if marker in lowered)
    return marker_count >= 2


def looks_like_elsevier_full_article_text(text: str | None) -> bool:
    if not looks_like_full_article_text(text):
        return False
    lowered = text.lower()
    return "introduction" in lowered and any(
        marker in lowered
        for marker in ["methods", "materials and methods", "discussion", "references"]
    )


def extract_publisher_body_text_from_html(html_text: str) -> str | None:
    parser = PublisherArticleTextParser()
    try:
        parser.feed(html_text)
    except Exception:
        return None
    text = truncate_text(" ".join(parser.chunks), max_chars=250_000)
    if not looks_like_full_article_text(text):
        return None
    return text


def fetch_publisher_body_text(url: str) -> str | None:
    html_text = text_get(
        url,
        headers=PUBLISHER_HTML_HEADERS,
        timeout=30,
    )
    if is_access_denied_text(html_text):
        raise FullTextRetrievalFailure(
            "blocked_by_cloudflare",
            "Publisher page returned access-denied, CAPTCHA, or security-verification content.",
        )
    if is_paywall_text(html_text):
        raise FullTextRetrievalFailure("paywalled", "Publisher page appears to require payment or login.")
    return extract_publisher_body_text_from_html(html_text)


def extract_firecrawl_text(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    data = payload.get("data", payload)
    candidates = [
        data.get("markdown") if isinstance(data, dict) else None,
        data.get("content") if isinstance(data, dict) else None,
        data.get("html") if isinstance(data, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        if is_access_denied_text(candidate):
            raise FullTextRetrievalFailure(
                "blocked_by_cloudflare",
                "Firecrawl response contained access-denied, CAPTCHA, or security-verification content.",
            )
        if is_paywall_text(candidate):
            raise FullTextRetrievalFailure("paywalled", "Firecrawl response appears to require payment or login.")
        text = visible_page_summary_from_html(candidate, max_chars=250_000) if "<" in candidate else truncate_text(candidate, max_chars=250_000)
        if looks_like_full_article_text(text):
            return text
    return None


def fetch_firecrawl_body_text(url: str) -> str | None:
    config = load_config()
    if not config.firecrawl_api_key:
        return None
    payload = {"url": url, "formats": ["markdown"]}
    req = request.Request(
        firecrawl_scrape_url(config.firecrawl_api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.firecrawl_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError:
        raise
    except Exception:
        return None
    return extract_firecrawl_text(payload)


def extract_pdf_body_text(pdf_bytes: bytes) -> str | None:
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise FullTextRetrievalFailure("pdf_parse_failed", "Downloaded content was not a PDF.")
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise FullTextRetrievalFailure("pdf_parse_failed", "PDF downloaded but could not be parsed.")
    text = truncate_text(text, max_chars=250_000)
    if not looks_like_full_article_text(text):
        raise FullTextRetrievalFailure("pdf_parse_failed", "PDF parsed but did not contain usable article body text.")
    return text


def fetch_pdf_body_text(url: str) -> str | None:
    return extract_pdf_body_text(
        bytes_get(
            url,
            headers=PDF_HEADERS,
            timeout=60,
        )
    )


def fetch_xml_body_text(url: str) -> str | None:
    xml_text = text_get(
        url,
        headers=XML_HEADERS,
        timeout=60,
    )
    if is_access_denied_text(xml_text):
        raise FullTextRetrievalFailure(
            "blocked_by_cloudflare",
            "XML endpoint returned access-denied, CAPTCHA, or security-verification content.",
        )
    if is_paywall_text(xml_text):
        raise FullTextRetrievalFailure("paywalled", "XML endpoint appears to require payment or login.")
    text = xml_body_text(xml_text)
    text = truncate_text(text or "", max_chars=250_000)
    if not looks_like_full_article_text(text):
        return None
    return text


def fetch_openalex_work_by_doi(doi: str) -> dict[str, Any] | None:
    url = openalex_url(
        f"https://api.openalex.org/works/https://doi.org/{parse.quote(doi, safe='/')}"
    )
    try:
        return json_get(url, timeout=30)
    except Exception:
        return None


def fetch_crossref_work_by_doi(doi: str) -> dict[str, Any] | None:
    url = crossref_url(f"https://api.crossref.org/works/{parse.quote(doi, safe='')}")
    try:
        return json_get(url, timeout=30).get("message")
    except Exception:
        return None


def pmcid_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/(?:articles?/)?(PMC\d+)", url, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    numeric_match = re.search(r"/(?:articles?/)?(\d{5,})(?:/|$)", url, flags=re.IGNORECASE)
    parsed = parse.urlparse(url)
    if ("pmc" in parsed.netloc.lower() or "/pmc/" in parsed.path.lower()) and numeric_match:
        return f"PMC{numeric_match.group(1)}"
    return None


def fetch_pmcid_for_citation(citation: CandidateCitation) -> str | None:
    ids = [value for value in [citation.pmid, normalized_doi(citation.doi)] if value]
    if not ids:
        return None
    url = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?" + parse.urlencode(
        {"ids": ",".join(ids), "format": "json"}
    )
    try:
        payload = json_get(url, timeout=30)
    except Exception:
        return None
    for record in payload.get("records", []):
        pmcid = record.get("pmcid")
        if pmcid:
            return str(pmcid).upper()
    return None


def pmc_oa_links(pmcid: str) -> list[tuple[str, str]]:
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?" + parse.urlencode({"id": pmcid})
    try:
        root = ElementTree.fromstring(text_get(url, timeout=30))
    except Exception:
        return []
    links: list[tuple[str, str]] = []
    for link in root.findall(".//link"):
        href = link.attrib.get("href")
        fmt = link.attrib.get("format") or ""
        if href:
            links.append((href, fmt.lower()))
    return links


def pmc_download_url_variants(url: str) -> list[str]:
    variants = [url]
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        https_url = "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix("ftp://ftp.ncbi.nlm.nih.gov/")
        variants.append(https_url)
        variants.append(https_url.replace("/pub/pmc/", "/pub/pmc/deprecated/", 1))
    elif "ftp.ncbi.nlm.nih.gov/pub/pmc/" in url:
        variants.append(url.replace("/pub/pmc/", "/pub/pmc/deprecated/", 1))
    return list(dict.fromkeys(variants))


def extract_tgz_nxml_body_text(data: bytes) -> str | None:
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
            members = [
                member for member in archive.getmembers() if member.isfile() and member.name.endswith(".nxml")
            ]
            if not members:
                return None
            member = sorted(members, key=lambda item: len(item.name))[0]
            extracted = archive.extractfile(member)
            if extracted is None:
                return None
            xml_text = extracted.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    text = xml_body_text(xml_text)
    text = truncate_text(text or "", max_chars=250_000)
    if not looks_like_full_article_text(text):
        return None
    return text


def fetch_pmc_oa_body_text(pmcid: str, prefer_pdf: bool = False) -> tuple[str | None, str | None, str]:
    links = pmc_oa_links(pmcid)
    ordered = sorted(links, key=lambda item: 0 if (prefer_pdf and item[1] == "pdf") or item[1] == "tgz" else 1)
    attempted: list[str] = []
    for href, fmt in ordered:
        for download_url in pmc_download_url_variants(href):
            attempted.append(download_url)
            try:
                data = bytes_get(download_url, timeout=90)
            except Exception:
                continue
            if fmt == "pdf" or download_url.lower().endswith(".pdf"):
                try:
                    text = extract_pdf_body_text(data)
                except FullTextRetrievalFailure:
                    text = None
                if text:
                    return text, download_url, "pmc_oa_pdf"
                continue
            text = extract_tgz_nxml_body_text(data)
            if text:
                return text, download_url, "pmc_oa_xml"
    return None, None, ", ".join(attempted)


def openalex_candidates(payload: dict[str, Any] | None) -> list[FullTextCandidate]:
    if not payload:
        return []
    candidates: list[FullTextCandidate] = []
    oa = payload.get("open_access") or {}
    locations = [payload.get("best_oa_location"), payload.get("primary_location")]
    locations.extend(payload.get("locations") or [])
    for location in locations:
        if not isinstance(location, dict):
            continue
        license_value = location.get("license") or oa.get("oa_status")
        landing_url = location.get("landing_page_url")
        pdf_url = location.get("pdf_url")
        pmcid = pmcid_from_url(landing_url)
        if pmcid:
            candidates.append(
                FullTextCandidate(
                    url=f"pmc:{pmcid}",
                    method="pmc_oa_xml",
                    content_type="pmc",
                    access_evidence="OpenAlex OA location points to PubMed Central.",
                    oa_url=landing_url,
                    license=license_value,
                )
            )
        if pdf_url:
            candidates.append(
                FullTextCandidate(
                    url=pdf_url,
                    method=method_for_full_text_url(pdf_url, "application/pdf", "openalex_pdf"),
                    content_type="pdf",
                    access_evidence="OpenAlex OA location provides a PDF URL.",
                    oa_url=pdf_url,
                    license=license_value,
                )
            )
    oa_url = oa.get("oa_url")
    if isinstance(oa_url, str) and looks_like_url(oa_url):
        candidates.append(
            FullTextCandidate(
                url=oa_url,
                method=method_for_full_text_url(oa_url, "application/pdf" if ".pdf" in oa_url else "", "openalex_pdf"),
                content_type="pdf" if ".pdf" in oa_url or "/pdf" in oa_url else "html",
                access_evidence="OpenAlex open_access.oa_url is available.",
                oa_url=oa_url,
                license=oa.get("oa_status"),
            )
        )
    return unique_candidates(candidates)


def crossref_candidates(payload: dict[str, Any] | None) -> list[FullTextCandidate]:
    if not payload:
        return []
    license_value = None
    if payload.get("license"):
        license_value = payload["license"][0].get("URL")
    candidates: list[FullTextCandidate] = []
    for link in payload.get("link") or []:
        url = link.get("URL")
        if not url:
            continue
        if parse.urlparse(url).netloc.lower() == "api.elsevier.com":
            # Crossref exposes Elsevier API endpoints as links, but they are not PDFs or
            # publisher pages. The authenticated Elsevier DOI path handles them separately.
            continue
        content_type = link.get("content-type") or ""
        method = method_for_full_text_url(url, content_type, "crossref_pdf")
        content_kind = "xml" if method in {"crossref_full_xml", "wiley_full_xml"} else "pdf"
        candidates.append(
            FullTextCandidate(
                url=url,
                method=method,
                content_type=content_kind,
                access_evidence="Crossref metadata provides a full-text link.",
                oa_url=url,
                license=license_value,
            )
        )
    return unique_candidates(candidates)


UNPAYWALL_VERSION_PRIORITY = {
    "publishedVersion": 0,
    "acceptedVersion": 1,
    "submittedVersion": 2,
}


def fetch_unpaywall_work_by_doi(doi: str) -> dict[str, Any] | None:
    email = load_config().unpaywall_email
    if not email:
        return None
    url = f"https://api.unpaywall.org/v2/{parse.quote(doi, safe='/')}?" + parse.urlencode(
        {"email": email}
    )
    try:
        return json_get(url, timeout=30)
    except error.HTTPError as exc:
        if exc.code in {404, 422}:
            return None
        return None
    except Exception:
        return None


def unpaywall_candidates(payload: dict[str, Any] | None) -> list[FullTextCandidate]:
    if not payload or not payload.get("is_oa"):
        return []
    locations: list[dict[str, Any]] = []
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    locations.extend(
        location
        for location in payload.get("oa_locations") or []
        if isinstance(location, dict)
    )
    oa_status = payload.get("oa_status")
    candidates: list[tuple[tuple[int, int, int], FullTextCandidate]] = []
    for index, location in enumerate(locations):
        version = location.get("version")
        host_type = location.get("host_type")
        license_value = location.get("license")
        evidence = (
            "Unpaywall reports an open-access location"
            f" (version={version or 'unknown'}, host={host_type or 'unknown'},"
            f" oa_status={oa_status or 'unknown'})."
        )
        version_priority = UNPAYWALL_VERSION_PRIORITY.get(str(version), 3)
        pdf_url = location.get("url_for_pdf")
        if isinstance(pdf_url, str) and looks_like_url(pdf_url):
            candidates.append(
                (
                    (version_priority, 0, index),
                    FullTextCandidate(
                        url=pdf_url,
                        method="unpaywall_pdf",
                        content_type="pdf",
                        access_evidence=evidence,
                        oa_url=pdf_url,
                        license=license_value,
                        manuscript_version=version,
                        oa_host_type=host_type,
                        oa_status=oa_status,
                    ),
                )
            )
        landing_url = location.get("url_for_landing_page") or location.get("url")
        if isinstance(landing_url, str) and looks_like_url(landing_url):
            candidates.append(
                (
                    (version_priority, 1, index),
                    FullTextCandidate(
                        url=landing_url,
                        method="unpaywall_landing_page",
                        content_type="html",
                        access_evidence=evidence,
                        oa_url=landing_url,
                        license=license_value,
                        manuscript_version=version,
                        oa_host_type=host_type,
                        oa_status=oa_status,
                    ),
                )
            )
    candidates.sort(key=lambda item: item[0])
    return unique_candidates([candidate for _, candidate in candidates])


def resolve_full_text_candidates(citation: CandidateCitation) -> list[FullTextCandidate]:
    doi = normalized_doi(citation.doi)
    candidates: list[FullTextCandidate] = []
    if doi:
        candidates.extend(mdpi_candidates_for_doi(doi))
        candidates.extend(unpaywall_candidates(fetch_unpaywall_work_by_doi(doi)))
        candidates.extend(openalex_candidates(fetch_openalex_work_by_doi(doi)))
        candidates.extend(crossref_candidates(fetch_crossref_work_by_doi(doi)))
    pmcid = fetch_pmcid_for_citation(citation)
    if pmcid:
        candidates.insert(
            0,
            FullTextCandidate(
                url=f"pmc:{pmcid}",
                method="pmc_oa_xml",
                content_type="pmc",
                access_evidence="PMC ID Converter mapped this source to a PMCID.",
                oa_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                license=None,
            ),
        )
    return unique_candidates(candidates)


def fetch_resolved_candidate_body_text(candidate: FullTextCandidate) -> tuple[str | None, str | None, str]:
    if candidate.url.startswith("pmc:"):
        pmcid = candidate.url.split(":", 1)[1]
        text, resolved_url, method = fetch_pmc_oa_body_text(pmcid)
        if text:
            return text, resolved_url, method or candidate.method
        return None, resolved_url, candidate.method
    if candidate.content_type == "xml" or candidate.method in {"crossref_full_xml", "wiley_full_xml", "mdpi_xml"}:
        return fetch_xml_body_text(candidate.url), candidate.url, candidate.method
    if candidate.content_type == "pdf" or candidate.method.endswith("_pdf"):
        return fetch_pdf_body_text(candidate.url), candidate.url, candidate.method
    return fetch_publisher_body_text(candidate.url), candidate.url, candidate.method


def nested_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [" ".join(value.split())] if value.strip() else []
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(nested_values(item))
        return values
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            if key in {"coredata", "link"}:
                continue
            values.extend(nested_values(item))
        return values
    return []


def extract_elsevier_body_text(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    original_text = payload.get("full-text-retrieval-response", {}).get("originalText")
    if isinstance(original_text, str):
        text = xml_body_text(original_text) or re.sub(r"<[^>]+>", " ", original_text)
    else:
        text = " ".join(nested_values(original_text))
    text = truncate_text(text, max_chars=250_000) if text else ""
    return text if looks_like_elsevier_full_article_text(text) else None


def fetch_elsevier_body_text_by_doi(doi: str) -> str | None:
    api_key = load_config().elsevier_api_key
    if not api_key:
        return None
    escaped = parse.quote(doi, safe="/")
    base_url = f"https://api.elsevier.com/content/article/doi/{escaped}"
    last_http_error: error.HTTPError | None = None
    received_response = False
    for url in [
        f"{base_url}?httpAccept=application/json",
        base_url,
    ]:
        try:
            payload = json_get(url, headers=elsevier_headers(api_key), timeout=60)
        except error.HTTPError as exc:
            last_http_error = exc
            continue
        except Exception:
            continue
        received_response = True
        text = extract_elsevier_body_text(payload)
        if text:
            return text

    xml_headers = elsevier_headers(api_key) | {"Accept": "application/xml,text/xml"}
    try:
        xml_text = text_get(
            f"{base_url}?httpAccept=text/xml",
            headers=xml_headers,
            timeout=60,
        )
        received_response = True
    except error.HTTPError as exc:
        last_http_error = exc
        xml_text = ""
    except Exception:
        xml_text = ""
    if xml_text:
        text = xml_body_text(xml_text)
        text = truncate_text(text or "", max_chars=250_000)
        if looks_like_elsevier_full_article_text(text):
            return text

    if not received_response and last_http_error is not None:
        raise last_http_error
    return None


def europe_pmc_fulltext_url(source: str, record_id: str) -> str:
    return f"https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{record_id}/fullTextXML"


def fetch_europe_pmc_full_text(source: str, record_id: str) -> str | None:
    try:
        return xml_body_text(text_get(europe_pmc_fulltext_url(source, record_id), timeout=45))
    except Exception:
        return None


def find_europe_pmc_record(citation: CandidateCitation) -> tuple[str, str] | None:
    if citation.id.startswith("europepmc:"):
        record_id = citation.id.split(":", 1)[1]
        source = "PMC" if record_id.upper().startswith("PMC") else "MED"
        return source, record_id
    if citation.pmid:
        return "MED", citation.pmid
    if not citation.doi:
        return None
    query = f'DOI:"{citation.doi}"'
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + parse.urlencode(
        {"query": query, "format": "json", "pageSize": 1}
    )
    try:
        payload = json_get(url, timeout=30)
    except Exception:
        return None
    results = payload.get("resultList", {}).get("result", [])
    if not results:
        return None
    item = results[0]
    record_id = item.get("pmcid") or item.get("pmid") or item.get("id")
    if not record_id:
        return None
    source = item.get("source") or ("PMC" if str(record_id).upper().startswith("PMC") else "MED")
    return str(source), str(record_id)


def classify_http_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, FullTextRetrievalFailure):
        return exc.status, exc.notes
    if isinstance(exc, error.HTTPError):
        if exc.code in {401, 402}:
            return "paywalled", f"HTTP {exc.code}: access requires payment or credentials"
        if exc.code == 403:
            return "blocked", "HTTP 403: publisher blocked automated access"
        return "retrieval_error", f"HTTP {exc.code}: {exc.reason}"
    return "retrieval_error", f"{type(exc).__name__}: {exc}"


def base_record(citation: CandidateCitation) -> dict[str, Any]:
    return {
        "source_id": citation.id,
        "title": citation.title,
        "doi": citation.doi,
        "pmid": citation.pmid,
        "url": str(citation.url) if citation.url else None,
    }


def http_status_for_exception(exc: Exception) -> int | None:
    if isinstance(exc, FullTextRetrievalFailure):
        return exc.http_status
    if isinstance(exc, error.HTTPError):
        return exc.code
    return None


def is_pdf_candidate(candidate: FullTextCandidate) -> bool:
    return candidate.content_type == "pdf" or candidate.method.endswith("_pdf")


def status_for_missing_candidate_text(candidate: FullTextCandidate) -> tuple[str, str]:
    if is_pdf_candidate(candidate):
        return "pdf_parse_failed", "PDF candidate did not yield usable article body text."
    return "not_available", "Candidate did not yield usable article body text."


def best_terminal_failure(attempts: list[FullTextRetrievalAttempt]) -> FullTextRetrievalAttempt | None:
    for status in [
        "blocked_by_cloudflare",
        "paywalled",
        "pdf_parse_failed",
        "blocked",
        "retrieval_error",
    ]:
        for attempt in attempts:
            if attempt.status == status:
                return attempt
    return None


def retrieve_full_text_for_source(citation: CandidateCitation) -> FullTextResult:
    attempts: list[FullTextRetrievalAttempt] = []

    def add_attempt(
        *,
        method: str,
        url: str | None,
        status: str,
        http_status: int | None = None,
        content_type: str | None = None,
        resolved_url: str | None = None,
        oa_url: str | None = None,
        license: str | None = None,
        access_evidence: str | None = None,
        manuscript_version: str | None = None,
        oa_host_type: str | None = None,
        oa_status: str | None = None,
        text_char_count: int = 0,
        notes: str | None = None,
    ) -> FullTextRetrievalAttempt:
        safe_method = coerce_full_text_method(method)
        if safe_method != method:
            method_note = f"Original retrieval method was invalid or empty: {method!r}."
            notes = f"{notes} {method_note}" if notes else method_note
        attempt = FullTextRetrievalAttempt(
            source_id=citation.id,
            attempt_index=len(attempts) + 1,
            method=safe_method,
            url=url,
            status=status,
            http_status=http_status,
            content_type=content_type,
            resolved_url=resolved_url,
            oa_url=oa_url,
            license=license,
            access_evidence=access_evidence,
            manuscript_version=manuscript_version,
            oa_host_type=oa_host_type,
            oa_status=oa_status,
            text_char_count=text_char_count,
            notes=notes,
        )
        attempts.append(attempt)
        return attempt

    def attempted_urls() -> list[str]:
        return [attempt.url for attempt in attempts if attempt.url]

    if citation.body_text and not looks_like_url(citation.body_text):
        text = truncate_text(citation.body_text, max_chars=250_000)
        add_attempt(
            method="existing_body_text",
            url=None,
            status="full_text_found",
            text_char_count=len(text),
            notes="Used body text already present on source record.",
        )
        return FullTextResult(
            body_text=text,
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status="full_text_found",
                retrieval_method="existing_body_text",
                text_char_count=len(text),
                notes="Used body text already present on source record.",
            ),
        )

    publisher_error: Exception | None = None
    for url in source_landing_urls(citation):
        if is_literature_index_url(url):
            continue
        try:
            text = fetch_publisher_body_text(url)
        except Exception as exc:
            publisher_error = exc
            status, notes = classify_http_error(exc)
            add_attempt(
                method="publisher_page",
                url=url,
                status=status,
                http_status=http_status_for_exception(exc),
                content_type="html",
                notes=notes,
            )
            continue
        if text:
            text = truncate_text(text, max_chars=250_000)
            add_attempt(
                method="publisher_page",
                url=url,
                status="full_text_found",
                content_type="html",
                resolved_url=url,
                text_char_count=len(text),
                notes="Retrieved article body text from publisher landing page.",
            )
            return FullTextResult(
                body_text=text,
                attempts=attempts,
                record=RawTextRecord(
                    **base_record(citation),
                    status="full_text_found",
                    retrieval_method="publisher_page",
                    resolved_url=url,
                    attempted_urls=attempted_urls(),
                    text_char_count=len(text),
                    notes=f"Retrieved article body text from publisher page: {url}",
                ),
            )
        add_attempt(
            method="publisher_page",
            url=url,
            status="not_available",
            content_type="html",
            notes="Publisher landing page did not contain usable article body text.",
        )

    resolved_candidates = resolve_full_text_candidates(citation)
    if not resolved_candidates:
        add_attempt(
            method="none",
            url=None,
            status="no_oa_location",
            notes=(
                "No Unpaywall, OpenAlex, Crossref, or PMC open-access full-text location "
                "was available."
            ),
        )
    for candidate in resolved_candidates:
        try:
            text, resolved_url, method = fetch_resolved_candidate_body_text(candidate)
        except Exception as exc:
            status, notes = classify_http_error(exc)
            add_attempt(
                method=candidate.method,
                url=candidate.url,
                status=status,
                http_status=http_status_for_exception(exc),
                content_type=candidate.content_type,
                oa_url=candidate.oa_url,
                license=candidate.license,
                access_evidence=candidate.access_evidence,
                manuscript_version=candidate.manuscript_version,
                oa_host_type=candidate.oa_host_type,
                oa_status=candidate.oa_status,
                notes=notes,
            )
            continue
        if text:
            text = truncate_text(text, max_chars=250_000)
            add_attempt(
                method=method,
                url=candidate.url,
                status="full_text_found",
                content_type=candidate.content_type,
                resolved_url=resolved_url,
                oa_url=candidate.oa_url,
                license=candidate.license,
                access_evidence=candidate.access_evidence,
                manuscript_version=candidate.manuscript_version,
                oa_host_type=candidate.oa_host_type,
                oa_status=candidate.oa_status,
                text_char_count=len(text),
                notes="Retrieved body text from resolved open-access link.",
            )
            return FullTextResult(
                body_text=text,
                attempts=attempts,
                record=RawTextRecord(
                    **base_record(citation),
                    status="full_text_found",
                    retrieval_method=method,
                    resolved_url=resolved_url,
                    oa_url=candidate.oa_url,
                    license=candidate.license,
                    access_evidence=candidate.access_evidence,
                    manuscript_version=candidate.manuscript_version,
                    oa_host_type=candidate.oa_host_type,
                    oa_status=candidate.oa_status,
                    attempted_urls=attempted_urls(),
                    text_char_count=len(text),
                    notes=f"Retrieved body text from resolved open-access link: {resolved_url}",
                ),
            )
        status, notes = status_for_missing_candidate_text(candidate)
        add_attempt(
            method=method,
            url=candidate.url,
            status=status,
            content_type=candidate.content_type,
            resolved_url=resolved_url,
            oa_url=candidate.oa_url,
            license=candidate.license,
            access_evidence=candidate.access_evidence,
            manuscript_version=candidate.manuscript_version,
            oa_host_type=candidate.oa_host_type,
            oa_status=candidate.oa_status,
            notes=notes,
        )

    europe_pmc_record = find_europe_pmc_record(citation)
    if europe_pmc_record:
        source, record_id = europe_pmc_record
        method = "pubmed_central" if source.upper() == "PMC" else "europe_pmc_fulltext_xml"
        url = europe_pmc_fulltext_url(source, record_id)
        try:
            text = fetch_europe_pmc_full_text(source, record_id)
        except Exception as exc:
            publisher_error = exc
            status, notes = classify_http_error(exc)
            add_attempt(
                method=method,
                url=url,
                status=status,
                http_status=http_status_for_exception(exc),
                content_type="xml",
                notes=notes,
            )
        else:
            if text:
                text = truncate_text(text, max_chars=250_000)
                add_attempt(
                    method=method,
                    url=url,
                    status="full_text_found",
                    content_type="xml",
                    resolved_url=url,
                    text_char_count=len(text),
                    notes=f"Retrieved full-text XML from Europe PMC record {source}/{record_id}.",
                )
                return FullTextResult(
                    body_text=text,
                    attempts=attempts,
                    record=RawTextRecord(
                        **base_record(citation),
                        status="full_text_found",
                        retrieval_method=method,
                        resolved_url=url,
                        attempted_urls=attempted_urls(),
                        text_char_count=len(text),
                        notes=f"Retrieved full-text XML from Europe PMC record {source}/{record_id}.",
                    ),
                )
            add_attempt(
                method=method,
                url=url,
                status="not_available",
                content_type="xml",
                notes=f"Europe PMC record {source}/{record_id} did not yield usable body text.",
            )

    if citation.doi and is_elsevier_candidate(citation):
        text = None
        elsevier_url = f"https://api.elsevier.com/content/article/doi/{parse.quote(citation.doi, safe='/')}"
        try:
            text = fetch_elsevier_body_text_by_doi(citation.doi)
        except Exception as exc:
            publisher_error = exc
            status, notes = classify_http_error(exc)
            add_attempt(
                method="elsevier_article_api",
                url=elsevier_url,
                status=status,
                http_status=http_status_for_exception(exc),
                content_type="api",
                notes=notes,
            )
        if text:
            text = truncate_text(text, max_chars=250_000)
            add_attempt(
                method="elsevier_article_api",
                url=elsevier_url,
                status="full_text_found",
                content_type="api",
                text_char_count=len(text),
                notes="Retrieved body text from Elsevier article API.",
            )
            return FullTextResult(
                body_text=text,
                attempts=attempts,
                record=RawTextRecord(
                    **base_record(citation),
                    status="full_text_found",
                    retrieval_method="elsevier_article_api",
                    attempted_urls=attempted_urls(),
                    text_char_count=len(text),
                    notes="Retrieved body text from Elsevier article API.",
                ),
            )
        if not any(attempt.method == "elsevier_article_api" for attempt in attempts):
            add_attempt(
                method="elsevier_article_api",
                url=elsevier_url,
                status="not_available",
                content_type="api",
                notes="Elsevier article API did not return usable body text.",
            )

    for url in source_landing_urls(citation):
        if is_literature_index_url(url):
            continue
        try:
            text = fetch_firecrawl_body_text(url)
        except Exception as exc:
            publisher_error = exc
            status, notes = classify_http_error(exc)
            add_attempt(
                method="firecrawl_scrape",
                url=url,
                status=status,
                http_status=http_status_for_exception(exc),
                content_type="html",
                notes=notes,
            )
            continue
        if text:
            text = truncate_text(text, max_chars=250_000)
            add_attempt(
                method="firecrawl_scrape",
                url=url,
                status="full_text_found",
                content_type="html",
                resolved_url=url,
                text_char_count=len(text),
                notes="Retrieved article body text through Firecrawl scrape.",
            )
            return FullTextResult(
                body_text=text,
                attempts=attempts,
                record=RawTextRecord(
                    **base_record(citation),
                    status="full_text_found",
                    retrieval_method="firecrawl_scrape",
                    resolved_url=url,
                    attempted_urls=attempted_urls(),
                    text_char_count=len(text),
                    notes=f"Retrieved article body text through Firecrawl scrape: {url}",
                ),
            )
        add_attempt(
            method="firecrawl_scrape",
            url=url,
            status="not_available",
            content_type="html",
            notes="Firecrawl did not return usable article body text or is not configured.",
        )

    terminal_failure = best_terminal_failure(attempts)
    if terminal_failure:
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status=terminal_failure.status,
                retrieval_method=terminal_failure.method,
                resolved_url=terminal_failure.resolved_url,
                oa_url=terminal_failure.oa_url,
                license=terminal_failure.license,
                access_evidence=terminal_failure.access_evidence,
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes=terminal_failure.notes,
            ),
        )

    if citation.abstract:
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status="abstract_only",
                retrieval_method="none",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes="Only abstract text is available; not stored as body text.",
            )
        )

    if citation.page_summary:
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status="page_summary_only",
                retrieval_method="publisher_page",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes="Only landing-page summary text is available; not stored as body text.",
            )
        )

    try:
        page_summary = None
        if citation.url:
            html_text = text_get(
                str(citation.url),
                headers=PUBLISHER_HTML_HEADERS,
                timeout=15,
            )
            if is_access_denied_text(html_text):
                raise FullTextRetrievalFailure(
                    "blocked_by_cloudflare",
                    "Publisher page returned access-denied, CAPTCHA, or security-verification content.",
                )
            page_summary = visible_page_summary_from_html(html_text)
    except Exception as exc:
        status, notes = classify_http_error(exc)
        add_attempt(
            method="publisher_page",
            url=str(citation.url) if citation.url else None,
            status=status,
            http_status=http_status_for_exception(exc),
            content_type="html",
            notes=notes,
        )
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status=status,
                retrieval_method="publisher_page",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes=notes,
            )
        )
    if page_summary:
        add_attempt(
            method="publisher_page",
            url=str(citation.url) if citation.url else None,
            status="page_summary_only",
            content_type="html",
            notes="Captured publisher landing-page text only; not stored as body text.",
        )
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status="page_summary_only",
                retrieval_method="publisher_page",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes="Captured publisher landing-page text only; not stored as body text.",
            )
        )

    if publisher_error:
        status, notes = classify_http_error(publisher_error)
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status=status,
                retrieval_method="publisher_page",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes=notes,
            )
        )

    no_oa_attempt = next((attempt for attempt in attempts if attempt.status == "no_oa_location"), None)
    if no_oa_attempt:
        return FullTextResult(
            attempts=attempts,
            record=RawTextRecord(
                **base_record(citation),
                status="no_oa_location",
                retrieval_method="none",
                attempted_urls=attempted_urls(),
                text_char_count=0,
                notes=no_oa_attempt.notes,
            ),
        )

    return FullTextResult(
        attempts=attempts,
        record=RawTextRecord(
            **base_record(citation),
            status="not_available",
            retrieval_method="none",
            attempted_urls=attempted_urls(),
            text_char_count=0,
            notes="No full body text was available from configured retrieval methods.",
        )
    )


def raw_text_counts(records: list[RawTextRecord]) -> dict[str, int]:
    return {status: sum(1 for record in records if record.status == status) for status in FULL_TEXT_STATUS_FIELDS}


def write_raw_texts_markdown(path: Path, records: list[RawTextRecord]) -> None:
    lines = ["# Raw Text Retrieval", ""]
    if not records:
        lines.extend(["No sources were processed.", ""])
    for index, record in enumerate(records, start=1):
        lines.append(f"## {index}. {record.title}")
        lines.append("")
        lines.append(f"- Source ID: {record.source_id}")
        if record.doi:
            lines.append(f"- DOI: {record.doi}")
        if record.pmid:
            lines.append(f"- PMID: {record.pmid}")
        if record.url:
            lines.append(f"- URL: {record.url}")
        lines.append(f"- Status: {record.status}")
        lines.append(f"- Retrieval method: {record.retrieval_method}")
        if record.resolved_url:
            lines.append(f"- Resolved URL: {record.resolved_url}")
        if record.oa_url:
            lines.append(f"- OA URL: {record.oa_url}")
        if record.license:
            lines.append(f"- License: {record.license}")
        if record.access_evidence:
            lines.append(f"- Access evidence: {record.access_evidence}")
        if record.attempted_urls:
            lines.append(f"- Attempted URLs: {', '.join(record.attempted_urls)}")
        lines.append(f"- Text chars: {record.text_char_count}")
        if record.text_path:
            lines.append(f"- Text path: {record.text_path}")
        if record.notes:
            lines.append(f"- Notes: {record.notes}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def manual_full_text_needed(record: RawTextRecord) -> bool:
    return record.status != "full_text_found"


def manual_pdf_filename(record: RawTextRecord) -> str:
    if record.pmid:
        return f"pmid-{record.pmid}.pdf"
    if record.doi:
        safe_doi = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalized_doi(record.doi) or record.doi).strip("-")
        if safe_doi:
            return f"doi-{safe_doi[:120]}.pdf"
    safe_source = re.sub(r"[^a-zA-Z0-9._-]+", "-", record.source_id).strip("-")
    return f"{safe_source[:120] or 'source'}.pdf"


def manual_queue_folder_name(index: int, record: RawTextRecord) -> str:
    base = record.pmid and f"pmid-{record.pmid}"
    if not base and record.doi:
        safe_doi = re.sub(r"[^a-zA-Z0-9._-]+", "-", normalized_doi(record.doi) or record.doi).strip("-")
        base = f"doi-{safe_doi[:80]}" if safe_doi else None
    if not base:
        base = re.sub(r"[^a-zA-Z0-9._-]+", "-", record.source_id).strip("-") or "source"
    return f"{index:02d}_{base[:100]}"


def manual_queue_links(source: CandidateCitation | None, record: RawTextRecord) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    if record.doi:
        links.append({"label": "DOI", "url": f"https://doi.org/{normalized_doi(record.doi) or record.doi}"})
    if record.pmid:
        links.append({"label": "PubMed", "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/"})
    url = str(source.url) if source and source.url else record.url
    if url:
        links.append({"label": "Source URL", "url": url})
    for attempted in record.attempted_urls:
        if looks_like_url(attempted) and all(link["url"] != attempted for link in links):
            links.append({"label": "Attempted URL", "url": attempted})
    return links


def write_manual_queue_item_readme(path: Path, item: dict[str, Any]) -> None:
    lines = [
        f"# {item['title']}",
        "",
        f"- Source ID: `{item['source_id']}`",
        f"- Current status: `{item['status']}`",
        f"- Save PDF as: `{item['save_as']}`",
    ]
    if item.get("doi"):
        lines.append(f"- DOI: `{item['doi']}`")
    if item.get("pmid"):
        lines.append(f"- PMID: `{item['pmid']}`")
    if item.get("notes"):
        lines.append(f"- Retrieval notes: {item['notes']}")
    if item.get("links"):
        lines.extend(["", "## Links", ""])
        for link in item["links"]:
            lines.append(f"- {link['label']}: {link['url']}")
    lines.extend(
        [
            "",
            "## Instructions",
            "",
            f"Save the full paper PDF in this folder using the exact filename `{item['save_as']}`.",
            "If you cannot access the paper, leave this folder without a PDF; ingestion will treat it as paywalled/unavailable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manual_full_text_queue(
    run_dir: Path,
    *,
    sources: list[CandidateCitation],
    records: list[RawTextRecord],
) -> list[dict[str, Any]]:
    source_by_id = {source.id: source for source in sources}
    queue_items: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not manual_full_text_needed(record):
            continue
        folder = Path("manual_full_text_queue") / manual_queue_folder_name(index, record)
        item = {
            "source_id": record.source_id,
            "title": record.title,
            "status": record.status,
            "doi": record.doi,
            "pmid": record.pmid,
            "folder": str(folder),
            "save_as": manual_pdf_filename(record),
            "links": manual_queue_links(source_by_id.get(record.source_id), record),
            "notes": record.notes,
        }
        queue_items.append(item)

    queue_dir = run_dir / "manual_full_text_queue"
    if not queue_items:
        if queue_dir.exists():
            write_json(queue_dir / "manual_full_text_manifest.json", {"run": str(run_dir), "count": 0, "items": []})
            (queue_dir / "README.md").write_text(
                "# Manual Full Text Queue\n\nNo papers currently need manual PDF download.\n",
                encoding="utf-8",
            )
        return []

    queue_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        "# Manual Full Text Queue",
        "",
        f"Papers needing manual PDF download: {len(queue_items)}",
        "",
        "Save each PDF into its listed folder using the exact filename shown.",
        "If a paper is paywalled or otherwise inaccessible, leave that paper's folder empty.",
        "",
    ]
    for item in queue_items:
        item_dir = run_dir / item["folder"]
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / ".gitkeep").touch()
        write_manual_queue_item_readme(item_dir / "README.md", item)
        readme_lines.extend(
            [
                f"## {item['title']}",
                "",
                f"- Folder: `{item['folder']}`",
                f"- Save as: `{item['save_as']}`",
                f"- Status: `{item['status']}`",
            ]
        )
        if item.get("doi"):
            readme_lines.append(f"- DOI: `{item['doi']}`")
        if item.get("pmid"):
            readme_lines.append(f"- PMID: `{item['pmid']}`")
        if item.get("links"):
            readme_lines.append("- Links:")
            for link in item["links"]:
                readme_lines.append(f"  - {link['label']}: {link['url']}")
        readme_lines.append("")

    (queue_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    write_json(queue_dir / "manual_full_text_manifest.json", {"run": str(run_dir), "count": len(queue_items), "items": queue_items})
    return queue_items


def append_audit_event(run_dir: Path, event: dict[str, Any]) -> None:
    audit_path = run_dir / "audit_log.jsonl"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(orjson.dumps(event).decode("utf-8") + "\n")


def update_packet_counts(run_dir: Path, counts: dict[str, int]) -> None:
    packet_path = run_dir / "packet.json"
    packet = PACKET_ADAPTER.validate_python(orjson.loads(packet_path.read_bytes()))
    packet_data = packet.model_dump(mode="json")
    packet_data["counts"].update(counts)
    packet_data["completed_at"] = datetime.now(UTC).isoformat()
    write_json(packet_path, packet_data)


def update_optional_ingredient_packet_counts(run_dir: Path, counts: dict[str, int]) -> None:
    packet_path = run_dir / "ingredient_packet.json"
    if not packet_path.exists():
        return
    packet_data = orjson.loads(packet_path.read_bytes())
    packet_data.setdefault("counts", {}).update(counts)
    packet_data["completed_at"] = datetime.now(UTC).isoformat()
    write_json(packet_path, packet_data)


def load_full_text_attempts(run_dir: Path) -> list[FullTextRetrievalAttempt]:
    attempts_path = run_dir / "full_text_retrieval_attempts.json"
    if not attempts_path.exists():
        return []
    return FULL_TEXT_ATTEMPTS_ADAPTER.validate_python(orjson.loads(attempts_path.read_bytes()))


def next_attempt_index_by_source(attempts: list[FullTextRetrievalAttempt]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for attempt in attempts:
        current = indexes.get(attempt.source_id, 0)
        indexes[attempt.source_id] = max(current, attempt.attempt_index)
    return {source_id: index + 1 for source_id, index in indexes.items()}


def ingest_manual_full_text_queue(run_dir: str | Path) -> ManualFullTextIngestionResult:
    run_path = Path(run_dir)
    manifest_path = run_path / "manual_full_text_queue" / "manual_full_text_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manual full-text manifest: {manifest_path}")
    raw_texts_path = run_path / "raw_texts.json"
    if not raw_texts_path.exists():
        raise FileNotFoundError(f"Missing raw text manifest: {raw_texts_path}")

    raw_dir = run_path / "raw_texts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = orjson.loads(manifest_path.read_bytes())
    records = RAW_TEXTS_ADAPTER.validate_python(orjson.loads(raw_texts_path.read_bytes()))
    attempts = load_full_text_attempts(run_path)
    next_indexes = next_attempt_index_by_source(attempts)
    record_by_source_id = {record.source_id: index for index, record in enumerate(records)}
    ingested: list[str] = []
    missing_assumed_paywalled: list[str] = []
    failed: list[str] = []
    unchanged: list[str] = []

    append_audit_event(
        run_path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "manual_full_text_ingestion_started",
            "manifest": str(manifest_path.relative_to(run_path)),
            "item_count": len(manifest.get("items", [])),
        },
    )

    for item in manifest.get("items", []):
        source_id = str(item.get("source_id") or "")
        if source_id not in record_by_source_id:
            failed.append(source_id or "<missing-source-id>")
            continue
        record_index = record_by_source_id[source_id]
        record = records[record_index]
        folder = str(item.get("folder") or "")
        save_as = str(item.get("save_as") or "")
        relative_pdf_path = Path(folder) / save_as
        pdf_path = run_path / relative_pdf_path
        attempt_index = next_indexes.get(source_id, 1)
        next_indexes[source_id] = attempt_index + 1

        if pdf_path.exists():
            try:
                body_text = extract_pdf_body_text(pdf_path.read_bytes())
            except FullTextRetrievalFailure as exc:
                status = exc.status if exc.status in FULL_TEXT_STATUS_FIELDS else "pdf_parse_failed"
                updated_record = record.model_copy(
                    update={
                        "status": status,
                        "retrieval_method": "manual_pdf",
                        "resolved_url": str(relative_pdf_path),
                        "attempted_urls": [*record.attempted_urls, str(relative_pdf_path)],
                        "text_char_count": 0,
                        "text_path": None,
                        "notes": f"Manual PDF was supplied but could not be used: {exc.notes}",
                    }
                )
                records[record_index] = updated_record
                attempts.append(
                    FullTextRetrievalAttempt(
                        source_id=source_id,
                        attempt_index=attempt_index,
                        method="manual_pdf",
                        url=str(relative_pdf_path),
                        status=status,  # type: ignore[arg-type]
                        content_type="application/pdf",
                        text_char_count=0,
                        notes=exc.notes,
                    )
                )
                failed.append(source_id)
                continue

            filename = safe_source_filename(source_id, record_index + 1)
            text_path = raw_dir / filename
            text_path.write_text(body_text or "", encoding="utf-8")
            updated_record = record.model_copy(
                update={
                    "status": "full_text_found",
                    "retrieval_method": "manual_pdf",
                    "resolved_url": str(relative_pdf_path),
                    "attempted_urls": [*record.attempted_urls, str(relative_pdf_path)],
                    "text_path": str(Path("raw_texts") / filename),
                    "text_char_count": len(body_text or ""),
                    "notes": f"Loaded body text from manually supplied PDF: {relative_pdf_path}",
                }
            )
            records[record_index] = updated_record
            attempts.append(
                FullTextRetrievalAttempt(
                    source_id=source_id,
                    attempt_index=attempt_index,
                    method="manual_pdf",
                    url=str(relative_pdf_path),
                    status="full_text_found",
                    content_type="application/pdf",
                    text_char_count=len(body_text or ""),
                    notes="Manual PDF parsed successfully.",
                )
            )
            ingested.append(source_id)
            continue

        if record.status == "full_text_found":
            unchanged.append(source_id)
            continue

        updated_record = record.model_copy(
            update={
                "status": "paywalled",
                "retrieval_method": record.retrieval_method,
                "attempted_urls": [*record.attempted_urls, str(relative_pdf_path)],
                "text_char_count": 0,
                "text_path": None,
                "notes": (
                    "Manual full-text queue PDF was not supplied; treating source as "
                    "paywalled or otherwise unavailable by user review."
                ),
            }
        )
        records[record_index] = updated_record
        attempts.append(
            FullTextRetrievalAttempt(
                source_id=source_id,
                attempt_index=attempt_index,
                method="manual_pdf",
                url=str(relative_pdf_path),
                status="paywalled",
                text_char_count=0,
                notes="Expected manual PDF was not supplied; treated as paywalled/unavailable.",
            )
        )
        missing_assumed_paywalled.append(source_id)

    write_json(raw_texts_path, model_dump_jsonable(records))
    write_json(run_path / "full_text_retrieval_attempts.json", model_dump_jsonable(attempts))
    write_raw_texts_markdown(run_path / "raw_texts.md", records)
    counts = raw_text_counts(records)
    update_packet_counts(run_path, counts)
    update_optional_ingredient_packet_counts(run_path, counts)
    append_audit_event(
        run_path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "manual_full_text_ingestion_completed",
            "ingested": ingested,
            "missing_assumed_paywalled": missing_assumed_paywalled,
            "failed": failed,
            "unchanged": unchanged,
            "counts": counts,
        },
    )
    return ManualFullTextIngestionResult(
        run_dir=run_path,
        ingested=ingested,
        missing_assumed_paywalled=missing_assumed_paywalled,
        failed=failed,
        unchanged=unchanged,
        counts=counts,
    )


def retrieve_full_text_for_run(run_dir: str | Path, max_workers: int = 4) -> list[RawTextRecord]:
    run_path = Path(run_dir)
    sources = SOURCES_ADAPTER.validate_python(orjson.loads((run_path / "sources.json").read_bytes()))
    raw_dir = run_path / "raw_texts"
    raw_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    append_audit_event(
        run_path,
        {
            "ts": started_at,
            "event": "full_text_retrieval_started",
            "source_count": len(sources),
            "max_workers": max_workers,
        },
    )

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        results = list(executor.map(retrieve_full_text_for_source, sources))

    records: list[RawTextRecord] = []
    attempts: list[FullTextRetrievalAttempt] = []
    for index, result in enumerate(results, start=1):
        record = result.record
        if result.body_text:
            filename = safe_source_filename(record.source_id, index)
            text_path = raw_dir / filename
            text_path.write_text(result.body_text, encoding="utf-8")
            record = record.model_copy(
                update={
                    "text_path": str(Path("raw_texts") / filename),
                    "text_char_count": len(result.body_text),
                }
            )
        records.append(record)
        attempts.extend(result.attempts)

    write_json(run_path / "raw_texts.json", model_dump_jsonable(records))
    write_json(run_path / "full_text_retrieval_attempts.json", model_dump_jsonable(attempts))
    write_raw_texts_markdown(run_path / "raw_texts.md", records)
    manual_queue_items = write_manual_full_text_queue(run_path, sources=sources, records=records)
    counts = raw_text_counts(records)
    update_packet_counts(run_path, counts)
    append_audit_event(
        run_path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "full_text_retrieval_completed",
            "counts": counts,
            "manual_full_text_queue_count": len(manual_queue_items),
        },
    )
    return records
