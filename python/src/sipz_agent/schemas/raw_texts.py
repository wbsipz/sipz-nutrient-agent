from typing import Literal

from pydantic import BaseModel, Field


FullTextStatus = Literal[
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

FullTextRetrievalMethod = Literal[
    "existing_body_text",
    "europe_pmc_fulltext_xml",
    "pubmed_central",
    "pmc_oa_xml",
    "pmc_oa_pdf",
    "crossref_full_xml",
    "crossref_pdf",
    "openalex_pdf",
    "wiley_full_xml",
    "mdpi_pdf",
    "mdpi_xml",
    "unpaywall_pdf",
    "unpaywall_landing_page",
    "manual_pdf",
    "elsevier_article_api",
    "publisher_page",
    "firecrawl_scrape",
    "none",
]


class RawTextRecord(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None
    status: FullTextStatus
    retrieval_method: FullTextRetrievalMethod
    resolved_url: str | None = None
    oa_url: str | None = None
    license: str | None = None
    access_evidence: str | None = None
    manuscript_version: str | None = None
    oa_host_type: str | None = None
    oa_status: str | None = None
    attempted_urls: list[str] = Field(default_factory=list)
    text_path: str | None = None
    text_char_count: int = Field(ge=0)
    notes: str | None = None


class FullTextRetrievalAttempt(BaseModel):
    source_id: str = Field(min_length=1)
    attempt_index: int = Field(ge=1)
    method: FullTextRetrievalMethod
    url: str | None = None
    status: FullTextStatus
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    resolved_url: str | None = None
    oa_url: str | None = None
    license: str | None = None
    access_evidence: str | None = None
    manuscript_version: str | None = None
    oa_host_type: str | None = None
    oa_status: str | None = None
    text_char_count: int = Field(default=0, ge=0)
    notes: str | None = None
