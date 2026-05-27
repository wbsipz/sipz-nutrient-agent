import json
from urllib.parse import parse_qs, urlparse

from sipz_agent.core.retrieval import find_candidate_papers


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
        if host == "www.ebi.ac.uk" and path.endswith("/search"):
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
    assert "PubMed" in result.citations[0].selection_reason
    assert "magnesium health human" in result.citations[0].selection_reason
    assert result.citations[1].body_text == "Full text paragraph about magnesium and blood pressure."
    assert "Europe PMC" in result.citations[1].selection_reason
    assert "full-text body" in result.citations[1].selection_reason
    assert any("api.crossref.org" in url for url in seen_urls)


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
