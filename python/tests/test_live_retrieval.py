import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from urllib import error
from urllib.parse import parse_qs, urlparse

import pytest

from sipz_agent.core.retrieval import (
    DoiMetadata,
    HTTP_MAX_RETRY_DELAY_SECONDS,
    add_page_summaries_to_missing_abstracts,
    deduplicate_citations,
    enrich_missing_abstracts_from_metadata,
    find_candidate_papers,
    find_crossref_candidates,
    find_firecrawl_candidates,
    find_live_candidate_page,
    find_openalex_candidates,
    find_pubmed_candidates,
    fetch_crossref_abstract_by_doi,
    fetch_openalex_abstract_by_doi,
    fetch_pubmed_abstracts,
    json_get,
    openalex_abstract_from_inverted_index,
    query_for_source,
    retry_delay_seconds,
    extract_elsevier_description,
    pubmed_abstracts_from_xml,
    text_from_markup,
    throttle_request,
    visible_page_summary_from_html,
)
from sipz_agent.schemas.citations import CandidateCitation


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return self.text.encode("utf-8")


def test_retry_delay_caps_large_retry_after_values() -> None:
    numeric = error.HTTPError(
        "https://example.org",
        429,
        "Too Many Requests",
        {"Retry-After": "7200"},
        None,
    )
    future_date = error.HTTPError(
        "https://example.org",
        429,
        "Too Many Requests",
        {"Retry-After": format_datetime(datetime.now(UTC) + timedelta(hours=2))},
        None,
    )

    assert retry_delay_seconds(numeric, 0) == HTTP_MAX_RETRY_DELAY_SECONDS
    assert retry_delay_seconds(future_date, 0) == HTTP_MAX_RETRY_DELAY_SECONDS


def test_candidate_page_disables_failed_collector_for_remaining_queries_and_pages(
    monkeypatch,
) -> None:
    calls = []

    def failing_collector(
        nutrient_name,
        depth,
        page_index=0,
        page_size=None,
        query_override=None,
    ):
        calls.append((page_index, query_override))
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(
        "sipz_agent.core.retrieval.find_pubmed_candidates",
        failing_collector,
    )
    for collector_name in [
        "find_europe_pmc_candidates",
        "find_openalex_candidates",
        "find_semantic_scholar_candidates",
        "find_crossref_candidates",
        "find_firecrawl_candidates",
    ]:
        monkeypatch.setattr(
            f"sipz_agent.core.retrieval.{collector_name}",
            lambda *args, **kwargs: [],
        )

    disabled = set()
    first = find_live_candidate_page(
        "luteolin",
        "standard",
        page_index=0,
        page_size=3,
        queries=["query one", "query two", "query three"],
        disabled_collectors=disabled,
    )
    second = find_live_candidate_page(
        "luteolin",
        "standard",
        page_index=1,
        page_size=3,
        queries=["query one", "query two", "query three"],
        disabled_collectors=disabled,
    )

    assert calls == [(0, "query one")]
    assert disabled == {"failing_collector"}
    assert len(first.source_failures) == 1
    assert second.source_failures == []


def test_metadata_enrichment_disables_source_after_first_failure(monkeypatch) -> None:
    calls = []

    def failing_crossref(doi):
        calls.append(doi)
        raise TimeoutError("metadata timeout")

    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_crossref_abstract_by_doi",
        failing_crossref,
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_openalex_abstract_by_doi",
        lambda doi: None,
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_elsevier_abstract_by_doi",
        lambda doi: None,
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_europe_pmc_metadata_by_doi",
        lambda doi: DoiMetadata(),
    )
    citations = [
        CandidateCitation(
            id=f"doi:{index}",
            title=f"Paper {index}",
            doi=f"10.1000/{index}",
            source="crossref",
            retrieval_query="query",
        )
        for index in range(3)
    ]
    disabled = set()

    enrich_missing_abstracts_from_metadata(citations, disabled_sources=disabled)

    assert calls == ["10.1000/0"]
    assert disabled == {"crossref"}


def test_pubmed_abstracts_from_xml_maps_abstracts_by_pmid() -> None:
    abstracts = pubmed_abstracts_from_xml(
        """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>111</PMID>
              <Article>
                <Abstract>
                  <AbstractText Label="BACKGROUND">Magnesium is relevant to humans.</AbstractText>
                  <AbstractText Label="RESULTS">Blood pressure changed in adults.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """
    )

    assert abstracts == {"111": "Magnesium is relevant to humans. Blood pressure changed in adults."}


def test_openalex_abstract_from_inverted_index_reconstructs_text() -> None:
    abstract = openalex_abstract_from_inverted_index(
        {"Magnesium": [0], "supports": [1], "human": [2], "health": [3]}
    )

    assert abstract == "Magnesium supports human health"


def test_text_from_markup_strips_crossref_jats_abstract() -> None:
    abstract = text_from_markup(
        "<jats:p>Kaempferol and quercetin have therapeutic potential for human health.</jats:p>"
    )

    assert abstract == "Kaempferol and quercetin have therapeutic potential for human health."


def test_extract_elsevier_description_reads_article_description() -> None:
    abstract = extract_elsevier_description(
        {
            "full-text-retrieval-response": {
                "coredata": {
                    "dc:description": (
                        "Background Quercetin has been associated with health benefits. "
                        "Scope and approach This review discusses delivery systems."
                    )
                }
            }
        }
    )

    assert abstract == (
        "Background Quercetin has been associated with health benefits. "
        "Scope and approach This review discusses delivery systems."
    )


def test_visible_page_summary_from_html_extracts_visible_text() -> None:
    summary = visible_page_summary_from_html(
        """
        <html>
          <head>
            <meta name="description" content="Publisher summary about magnesium health.">
            <style>.hidden { display: none; }</style>
          </head>
          <body>
            <nav>This navigation text should be ignored even though it is long.</nav>
            <h1>Magnesium and human health outcomes</h1>
            <p>This page describes a human study of magnesium and blood pressure outcomes.</p>
            <script>This script text should be ignored even though it is long.</script>
          </body>
        </html>
        """
    )

    assert "Publisher summary about magnesium health" in summary
    assert "human study of magnesium and blood pressure" in summary
    assert "navigation text" not in summary
    assert "script text" not in summary


def test_add_page_summaries_to_missing_abstracts_uses_landing_page(monkeypatch) -> None:
    def fake_fetch(url):
        assert url == "https://example.org/article"
        return "Visible publisher page summary about magnesium and human health."

    citation = CandidateCitation(
        id="publisher:1",
        title="Magnesium and health",
        url="https://example.org/article",
        source="publisher",
        retrieval_query="magnesium health human",
        selection_reason="Selected by search.",
    )
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_visible_page_summary", fake_fetch)

    enriched = add_page_summaries_to_missing_abstracts([citation])

    assert enriched[0].page_summary == "Visible publisher page summary about magnesium and human health."
    assert enriched[0].abstract is None
    assert "No API abstract was available" in enriched[0].selection_reason


def test_add_page_summaries_to_missing_abstracts_uses_doi_when_url_missing(monkeypatch) -> None:
    seen = {}

    def fake_fetch(url):
        seen["url"] = url
        return "DOI landing page text."

    citation = CandidateCitation(
        id="doi:10.1000/example",
        title="Magnesium and health",
        doi="10.1000/example",
        source="crossref",
        retrieval_query="magnesium health human",
    )
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_visible_page_summary", fake_fetch)

    enriched = add_page_summaries_to_missing_abstracts([citation])

    assert seen["url"] == "https://doi.org/10.1000/example"
    assert enriched[0].page_summary == "DOI landing page text."


def test_live_retrieval_collects_candidates_from_literature_apis(monkeypatch) -> None:
    seen_urls: list[str] = []

    def fake_urlopen(request, timeout):
        url = request.full_url
        seen_urls.append(url)
        host = urlparse(url).netloc
        path = urlparse(url).path
        query = parse_qs(urlparse(url).query)

        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esearch.fcgi"):
            assert "magnesium health human" in query["term"][0]
            assert query["retstart"] == ["0"]
            return FakeResponse({"esearchresult": {"idlist": ["111"]}})
        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esummary.fcgi"):
            return FakeResponse(
                {
                    "result": {
                        "uids": ["111"],
                        "111": {
                            "title": "Magnesium intake and cardiometabolic health",
                            "pubdate": "2021 Jan",
                            "articleids": [{"idtype": "doi", "value": "10.1000/pubmed"}],
                        },
                    }
                }
            )
        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/efetch.fcgi"):
            return TextResponse(
                """
                <PubmedArticleSet>
                  <PubmedArticle>
                    <MedlineCitation>
                      <PMID>111</PMID>
                      <Article>
                        <Abstract>
                          <AbstractText>
                            Magnesium intake was studied in relation to human cardiometabolic health.
                          </AbstractText>
                        </Abstract>
                      </Article>
                    </MedlineCitation>
                  </PubmedArticle>
                </PubmedArticleSet>
                """
            )
        if host == "www.ebi.ac.uk" and path.endswith("/search"):
            assert query["page"] == ["1"]
            return FakeResponse(
                {
                    "resultList": {
                        "result": [
                            {
                                "id": "PMC123",
                                "title": "Magnesium and blood pressure trial",
                                "doi": "10.1000/epmc",
                                "pmid": "222",
                                "pubYear": "2022",
                                "abstractText": "A human trial of magnesium and blood pressure.",
                                "fullTextUrlList": {
                                    "fullTextUrl": [{"url": "https://example.org/fulltext"}]
                                },
                            }
                        ]
                    }
                }
            )
        if host == "www.ebi.ac.uk" and path.endswith("/fullTextXML"):
            return TextResponse(
                "<article><body><p>Full text paragraph about magnesium and blood pressure.</p></body></article>"
            )
        if host == "api.openalex.org":
            assert query["page"] == ["1"]
            return FakeResponse(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "display_name": "Magnesium status and health outcomes",
                            "doi": "https://doi.org/10.1000/openalex",
                            "publication_year": 2020,
                            "primary_location": {"landing_page_url": "https://example.org/openalex"},
                        }
                    ]
                }
            )
        if host == "api.semanticscholar.org":
            assert query["offset"] == ["0"]
            return FakeResponse(
                {
                    "data": [
                        {
                            "paperId": "S2-1",
                            "title": "Magnesium supplementation systematic review",
                            "year": 2023,
                            "abstract": "Systematic review abstract.",
                            "url": "https://example.org/s2",
                            "externalIds": {"DOI": "10.1000/s2", "PubMed": "333"},
                        }
                    ]
                }
            )
        if host == "api.crossref.org":
            assert query["offset"] == ["0"]
            return FakeResponse(
                {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1000/crossref",
                                "title": ["Magnesium and inflammation"],
                                "published-print": {"date-parts": [[2019]]},
                                "URL": "https://example.org/crossref",
                            }
                        ]
                    }
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = find_candidate_papers("magnesium", depth="standard", demo=False)

    assert result.nutrient_id is None
    assert result.normalized_nutrient_name == "magnesium"
    assert len(result.citations) == 5
    assert {citation.source for citation in result.citations} == {
        "pubmed",
        "europe_pmc",
        "openalex",
        "semantic_scholar",
        "crossref",
    }
    assert result.citations[0].id == "pmid:111"
    assert result.citations[0].doi == "10.1000/pubmed"
    assert "human cardiometabolic health" in result.citations[0].abstract
    assert "PubMed" in result.citations[0].selection_reason
    assert "abstract metadata" in result.citations[0].selection_reason
    assert "magnesium health human" in result.citations[0].selection_reason
    assert result.citations[1].body_text == "Full text paragraph about magnesium and blood pressure."
    assert "Europe PMC" in result.citations[1].selection_reason
    assert "full-text body" in result.citations[1].selection_reason
    assert any("api.crossref.org" in url for url in seen_urls)


def test_live_retrieval_page_uses_source_pagination_parameters(monkeypatch) -> None:
    seen_queries: dict[str, dict[str, list[str]]] = {}

    def fake_urlopen(request, timeout):
        url = request.full_url
        host = urlparse(url).netloc
        path = urlparse(url).path
        query = parse_qs(urlparse(url).query)

        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esearch.fcgi"):
            seen_queries["pubmed"] = query
            return FakeResponse({"esearchresult": {"idlist": []}})
        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esummary.fcgi"):
            return FakeResponse({"result": {"uids": []}})
        if host == "www.ebi.ac.uk" and path.endswith("/search"):
            seen_queries["europe_pmc"] = query
            return FakeResponse({"resultList": {"result": []}})
        if host == "api.openalex.org":
            seen_queries["openalex"] = query
            return FakeResponse({"results": []})
        if host == "api.semanticscholar.org":
            seen_queries["semantic_scholar"] = query
            return FakeResponse({"data": []})
        if host == "api.crossref.org":
            seen_queries["crossref"] = query
            return FakeResponse({"message": {"items": []}})
        raise AssertionError(url)

    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = find_live_candidate_page(
        "magnesium",
        "standard",
        page_index=2,
        page_size=10,
    )

    assert result.citations == []
    assert seen_queries["pubmed"]["retstart"] == ["20"]
    assert seen_queries["europe_pmc"]["page"] == ["3"]
    assert seen_queries["openalex"]["page"] == ["3"]
    assert seen_queries["semantic_scholar"]["offset"] == ["20"]
    assert seen_queries["crossref"]["offset"] == ["20"]


def test_source_specific_query_formatting_removes_boolean_syntax_for_natural_search() -> None:
    query = (
        "(anethole OR trans-anethole) AND (human OR clinical OR trial) "
        "NOT (in vitro OR mouse OR rat)"
    )

    assert query_for_source("anethole", query, "pubmed") == query
    assert query_for_source("anethole", query, "europe_pmc") == query
    assert query_for_source("anethole", query, "openalex") == (
        "anethole trans-anethole human clinical trial"
    )
    assert query_for_source("anethole", query, "semantic_scholar") == (
        "anethole trans-anethole human clinical trial"
    )
    assert query_for_source("anethole", query, "crossref") == (
        "anethole trans-anethole human clinical trial"
    )
    assert "mouse" not in query_for_source("anethole", query, "firecrawl")


def test_openalex_search_uses_configured_api_key(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")

    def fake_json_get(url: str, **kwargs):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fake_json_get)

    assert find_openalex_candidates("magnesium", "light") == []
    assert parse_qs(urlparse(captured["url"]).query)["api_key"] == ["openalex-secret"]


def test_openalex_abstract_enrichment_uses_configured_api_key(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")

    def fake_json_get(url: str, **kwargs):
        captured["url"] = url
        return {"abstract_inverted_index": {"Human": [0], "trial": [1]}}

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fake_json_get)

    assert fetch_openalex_abstract_by_doi("10.1234/example") == "Human trial"
    assert parse_qs(urlparse(captured["url"]).query)["api_key"] == ["openalex-secret"]


def test_openalex_search_failure_does_not_expose_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")

    def fail(url: str, **kwargs):
        raise error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fail)

    with pytest.raises(RuntimeError) as exc_info:
        find_openalex_candidates("magnesium", "light")

    assert "openalex-secret" not in str(exc_info.value)
    assert str(exc_info.value) == "openalex_request_failed:HTTPError"


def test_primary_collector_mode_does_not_call_semantic_scholar(monkeypatch) -> None:
    def empty(*args, **kwargs):
        return []

    for name in [
        "find_pubmed_candidates",
        "find_europe_pmc_candidates",
        "find_openalex_candidates",
        "find_crossref_candidates",
    ]:
        monkeypatch.setattr(f"sipz_agent.core.retrieval.{name}", empty)

    def unexpected(*args, **kwargs):
        raise AssertionError("Semantic Scholar must not run in the primary tier")

    monkeypatch.setattr(
        "sipz_agent.core.retrieval.find_semantic_scholar_candidates", unexpected
    )

    page = find_live_candidate_page(
        "magnesium", "light", page_index=0, collector_mode="primary"
    )

    assert page.citations == []
    assert page.source_failures == []


def test_pubmed_search_and_summary_use_ncbi_credentials(monkeypatch) -> None:
    urls: list[str] = []
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-secret")
    monkeypatch.setenv("NCBI_EMAIL", "research@example.com")

    def fake_json_get(url: str, **kwargs):
        urls.append(url)
        if "esearch.fcgi" in url:
            return {"esearchresult": {"idlist": []}}
        return {"result": {"uids": []}}

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fake_json_get)

    assert find_pubmed_candidates("magnesium", "light") == []
    for url in urls:
        query = parse_qs(urlparse(url).query)
        assert query["api_key"] == ["ncbi-secret"]
        assert query["email"] == ["research@example.com"]


def test_pubmed_abstract_fetch_uses_ncbi_credentials(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-secret")

    def fake_text_get(url: str, **kwargs):
        captured["url"] = url
        return "<PubmedArticleSet />"

    monkeypatch.setattr("sipz_agent.core.retrieval.text_get", fake_text_get)

    assert fetch_pubmed_abstracts(["123"], "research@example.com") == {}
    query = parse_qs(urlparse(captured["url"]).query)
    assert query["api_key"] == ["ncbi-secret"]
    assert query["email"] == ["research@example.com"]


def test_ncbi_failure_does_not_expose_api_key(monkeypatch) -> None:
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-secret")

    def fail(url: str, **kwargs):
        raise error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fail)

    with pytest.raises(RuntimeError) as exc_info:
        find_pubmed_candidates("magnesium", "light")

    assert str(exc_info.value) == "ncbi_request_failed:HTTPError"
    assert "ncbi-secret" not in str(exc_info.value)


def test_crossref_search_and_doi_enrichment_use_mailto(monkeypatch) -> None:
    urls: list[str] = []
    monkeypatch.setenv("CROSSREF_MAILTO", "research@example.com")

    def fake_json_get(url: str, **kwargs):
        urls.append(url)
        return {"message": {"items": []}} if urlparse(url).path == "/works" else {"message": {}}

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fake_json_get)

    assert find_crossref_candidates("magnesium", "light") == []
    assert fetch_crossref_abstract_by_doi("10.1234/example") is None
    assert all(
        parse_qs(urlparse(url).query)["mailto"] == ["research@example.com"] for url in urls
    )


def test_crossref_failure_does_not_expose_mailto(monkeypatch) -> None:
    monkeypatch.setenv("CROSSREF_MAILTO", "private@example.com")

    def fail(url: str, **kwargs):
        raise error.HTTPError(url, 429, "Rate limited", None, None)

    monkeypatch.setattr("sipz_agent.core.retrieval.json_get", fail)

    with pytest.raises(RuntimeError) as exc_info:
        find_crossref_candidates("magnesium", "light")

    assert str(exc_info.value) == "crossref_request_failed:HTTPError"
    assert "private@example.com" not in str(exc_info.value)


def test_json_get_retries_retryable_http_errors(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            from urllib.error import HTTPError

            raise HTTPError(
                url=request.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "0"},
                fp=None,
            )
        return FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("sipz_agent.core.retrieval.throttle_request", lambda url: None)
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))

    assert json_get("https://api.semanticscholar.org/test") == {"ok": True}
    assert calls["count"] == 2
    assert sleeps == [0.0]


def test_throttle_request_waits_for_rate_limited_hosts(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("sipz_agent.core.retrieval._REQUEST_TIMES", {"api.semanticscholar.org": 10.0})
    monkeypatch.setattr("time.monotonic", lambda: 10.25)
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(round(delay, 2)))

    throttle_request("https://api.semanticscholar.org/graph/v1/paper/search")

    assert sleeps == [0.75]


def test_live_retrieval_splits_per_source_page_budget_across_planned_queries(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def fake_collector(
        nutrient_name,
        depth,
        page_index=0,
        page_size=None,
        query_override=None,
    ):
        calls.append((query_override, page_index, page_size))
        return []

    for name in [
        "find_pubmed_candidates",
        "find_europe_pmc_candidates",
        "find_openalex_candidates",
        "find_semantic_scholar_candidates",
        "find_crossref_candidates",
        "find_firecrawl_candidates",
    ]:
        monkeypatch.setattr(f"sipz_agent.core.retrieval.{name}", fake_collector)

    find_live_candidate_page(
        "anethole",
        "standard",
        page_index=1,
        page_size=10,
        queries=["anethole health human", "trans-anethole human", "p-propenylanisole human"],
    )

    assert calls[:3] == [
        ("anethole health human", 1, 4),
        ("trans-anethole human", 1, 3),
        ("p-propenylanisole human", 1, 3),
    ]
    assert len(calls) == 18
    assert sum(call[2] for call in calls[:3]) == 10


def test_live_retrieval_deduplicates_by_doi(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        host = urlparse(request.full_url).netloc
        path = urlparse(request.full_url).path
        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esearch.fcgi"):
            return FakeResponse({"esearchresult": {"idlist": []}})
        if host == "eutils.ncbi.nlm.nih.gov" and path.endswith("/esummary.fcgi"):
            return FakeResponse({"result": {"uids": []}})
        if host == "www.ebi.ac.uk":
            return FakeResponse({"resultList": {"result": []}})
        if host == "api.openalex.org":
            return FakeResponse(
                {"results": [{"id": "W1", "display_name": "First", "doi": "https://doi.org/10.1000/same"}]}
            )
        if host == "api.semanticscholar.org":
            return FakeResponse(
                {"data": [{"paperId": "S1", "title": "Duplicate", "externalIds": {"DOI": "10.1000/same"}}]}
            )
        if host == "api.crossref.org":
            return FakeResponse({"message": {"items": []}})
        raise AssertionError(request.full_url)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = find_candidate_papers("magnesium", depth="standard", demo=False)

    assert len(result.citations) == 1
    assert result.citations[0].title == "First"


def test_deduplicate_citations_merges_abstract_from_later_duplicate() -> None:
    first = CandidateCitation(
        id="openalex:W1",
        title="Bioactivity and Therapeutic Potential of Kaempferol and Quercetin",
        doi="10.3390/plants11192623",
        source="openalex",
        retrieval_query="kaempferol health human",
        selection_reason="Selected by OpenAlex.",
    )
    duplicate = CandidateCitation(
        id="doi:10.3390/plants11192623",
        title="Bioactivity and Therapeutic Potential of Kaempferol and Quercetin",
        doi="10.3390/plants11192623",
        source="crossref",
        retrieval_query="kaempferol health human",
        selection_reason="Selected by Crossref.",
        abstract="Kaempferol and quercetin in plants have therapeutic potential for human health.",
    )

    deduped = deduplicate_citations([first, duplicate])

    assert len(deduped) == 1
    assert deduped[0].id == "openalex:W1"
    assert "therapeutic potential for human health" in deduped[0].abstract
    assert "Merged additional metadata from crossref" in deduped[0].selection_reason


def test_enrich_missing_abstracts_uses_crossref_doi_detail(monkeypatch) -> None:
    citation = CandidateCitation(
        id="doi:10.3390/plants11192623",
        title="Kaempferol and quercetin",
        doi="10.3390/plants11192623",
        source="crossref",
        retrieval_query="quercetin health human",
        selection_reason="Selected by Crossref search.",
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_crossref_abstract_by_doi",
        lambda doi: "Crossref detail abstract about quercetin and human health.",
    )
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_openalex_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_elsevier_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_europe_pmc_metadata_by_doi",
        lambda doi: type("Metadata", (), {"abstract": None, "pmid": None})(),
    )

    enriched = enrich_missing_abstracts_from_metadata([citation])

    assert enriched[0].abstract == "Crossref detail abstract about quercetin and human health."
    assert "Crossref DOI detail" in enriched[0].selection_reason


def test_enrich_missing_abstracts_uses_elsevier_doi_detail(monkeypatch) -> None:
    citation = CandidateCitation(
        id="doi:10.1016/j.tifs.2016.07.004",
        title="Quercetin review",
        doi="10.1016/j.tifs.2016.07.004",
        source="crossref",
        retrieval_query="quercetin health human",
    )
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_crossref_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_openalex_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_elsevier_abstract_by_doi",
        lambda doi: "Elsevier abstract about quercetin delivery systems and human health.",
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_europe_pmc_metadata_by_doi",
        lambda doi: type("Metadata", (), {"abstract": None, "pmid": None})(),
    )

    enriched = enrich_missing_abstracts_from_metadata([citation])

    assert enriched[0].abstract == "Elsevier abstract about quercetin delivery systems and human health."
    assert "Elsevier DOI detail" in enriched[0].selection_reason


def test_enrich_missing_abstracts_uses_europe_pmc_pmid_then_pubmed(monkeypatch) -> None:
    citation = CandidateCitation(
        id="doi:10.1016/j.fct.2007.05.015",
        title="Safety of quercetin",
        doi="10.1016/j.fct.2007.05.015",
        source="crossref",
        retrieval_query="quercetin health human",
    )
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_crossref_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_openalex_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.retrieval.fetch_elsevier_abstract_by_doi", lambda doi: None)
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_europe_pmc_metadata_by_doi",
        lambda doi: type("Metadata", (), {"abstract": None, "pmid": "17698276"})(),
    )
    monkeypatch.setattr(
        "sipz_agent.core.retrieval.fetch_pubmed_abstracts",
        lambda pmids, email: {
            "17698276": "PubMed abstract for the quercetin safety review.",
        },
    )

    enriched = enrich_missing_abstracts_from_metadata([citation])

    assert enriched[0].pmid == "17698276"
    assert enriched[0].abstract == "PubMed abstract for the quercetin safety review."
    assert "Europe PMC PMID lookup" in enriched[0].selection_reason
    assert "PubMed PMID detail" in enriched[0].selection_reason


def test_firecrawl_uses_configured_local_api_url(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "data": [
                    {
                        "title": "Magnesium paper from local Firecrawl",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                        "description": "Local search result.",
                    }
                ]
            }
        )

    monkeypatch.setenv("FIRECRAWL_API_KEY", "local-key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    citations = find_firecrawl_candidates("magnesium", depth="light")

    assert captured["url"] == "http://localhost:3002/v1/search"
    assert captured["headers"]["Authorization"] == "Bearer local-key"
    assert captured["body"]["query"] == 'site:pubmed.ncbi.nlm.nih.gov OR site:europepmc.org "magnesium" health human'
    assert citations[0].source == "firecrawl"
    assert citations[0].title == "Magnesium paper from local Firecrawl"


def test_firecrawl_defaults_to_local_api_url(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse({"data": []})

    monkeypatch.setenv("FIRECRAWL_API_KEY", "local-key")
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    citations = find_firecrawl_candidates("magnesium", depth="light")

    assert captured["url"] == "http://localhost:3002/v1/search"
    assert citations == []
