from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any
from urllib import parse, request
from xml.etree import ElementTree

from pydantic import BaseModel

from sipz_agent.core.config import load_config
from sipz_agent.schemas.artifacts import StudyDepth
from sipz_agent.schemas.citations import CandidateCitation


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


def slugify(value: str) -> str:
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", value.strip().lower()))


def load_demo_corpus(nutrient_name: str) -> DemoCorpus:
    root = Path(__file__).resolve().parents[3]
    path = root / "examples" / "demo-corpus" / f"{slugify(nutrient_name)}.json"
    return DemoCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def json_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    req = request.Request(url, headers=headers or {"User-Agent": "sipz-nutrient-agent/0.1"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def text_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    req = request.Request(url, headers=headers or {"User-Agent": "sipz-nutrient-agent/0.1"})
    with request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def xml_body_text(xml_text: str) -> str | None:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    paragraphs = [" ".join(text.split()) for text in root.itertext() if text and text.strip()]
    body = "\n".join(paragraphs).strip()
    return body or None


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


def candidate_query(nutrient_name: str) -> str:
    return f"{nutrient_name.strip()} health human"


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


def find_pubmed_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    query = candidate_query(nutrient_name)
    limit = depth_limit(depth)
    email = load_config().ncbi_email
    base_params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
    }
    if email:
        base_params["email"] = email
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + query_string(
        base_params
    )
    search = json_get(search_url)
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
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + query_string(
        summary_params
    )
    summary = json_get(summary_url)
    result = summary.get("result", {})

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
                    has_doi=bool(doi),
                    has_pmid=True,
                ),
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


def find_europe_pmc_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    query = candidate_query(nutrient_name)
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query_string(
        {"query": query, "format": "json", "pageSize": depth_limit(depth)}
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


def find_openalex_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    query = candidate_query(nutrient_name)
    url = "https://api.openalex.org/works?" + query_string(
        {"search": query, "per-page": depth_limit(depth), "sort": "relevance_score:desc"}
    )
    payload = json_get(url)
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
                    has_doi=bool(item.get("doi")),
                ),
            )
        )
    return citations


def find_semantic_scholar_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    query = candidate_query(nutrient_name)
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + query_string(
        {
            "query": query,
            "limit": depth_limit(depth),
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


def find_crossref_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    query = candidate_query(nutrient_name)
    url = "https://api.crossref.org/works?" + query_string(
        {"query.bibliographic": query, "rows": depth_limit(depth), "sort": "relevance"}
    )
    payload = json_get(url)
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
                    has_abstract=bool(item.get("abstract")),
                    has_doi=True,
                ),
                abstract=item.get("abstract"),
            )
        )
    return citations


def find_firecrawl_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    config = load_config()
    if not config.firecrawl_api_key:
        return []
    query = f'site:pubmed.ncbi.nlm.nih.gov OR site:europepmc.org "{nutrient_name}" health human'
    payload = {"query": query, "limit": depth_limit(depth)}
    req = request.Request(
        "https://api.firecrawl.dev/v1/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.firecrawl_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:
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


def deduplicate_citations(citations: list[CandidateCitation]) -> list[CandidateCitation]:
    seen: set[str] = set()
    deduped: list[CandidateCitation] = []
    for citation in citations:
        key = normalize_doi(citation.doi) or (f"pmid:{citation.pmid}" if citation.pmid else None)
        if key is None:
            key = citation.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def find_live_candidates(nutrient_name: str, depth: StudyDepth) -> list[CandidateCitation]:
    collectors = [
        find_pubmed_candidates,
        find_europe_pmc_candidates,
        find_openalex_candidates,
        find_semantic_scholar_candidates,
        find_crossref_candidates,
        find_firecrawl_candidates,
    ]
    citations: list[CandidateCitation] = []
    for collector in collectors:
        citations.extend(collect_safely(collector, nutrient_name, depth))
    return deduplicate_citations(citations)


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
