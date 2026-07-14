import json
from urllib.error import HTTPError

from typer.testing import CliRunner

from sipz_agent.cli import app
from sipz_agent.core.artifacts import write_json
from sipz_agent.core.full_text import (
    FullTextCandidate,
    FullTextRetrievalFailure,
    fetch_pdf_body_text,
    fetch_xml_body_text,
    crossref_candidates,
    extract_elsevier_body_text,
    fetch_elsevier_body_text_by_doi,
    fetch_crossref_work_by_doi,
    fetch_openalex_work_by_doi,
    fetch_unpaywall_work_by_doi,
    mdpi_article_base_url_from_doi,
    mdpi_candidates_for_doi,
    openalex_candidates,
    pmc_download_url_variants,
    pmcid_from_url,
    ingest_manual_full_text_queue,
    resolve_full_text_candidates,
    retrieve_full_text_for_run,
    retrieve_full_text_for_source,
    unpaywall_candidates,
)
from sipz_agent.schemas.artifacts import Packet, PacketCounts, PacketInput, PacketModel
from sipz_agent.schemas.citations import CandidateCitation


def citation(
    *,
    id: str = "source:1",
    title: str = "Magnesium and human health",
    url: str | None = "https://example.org/paper",
    doi: str | None = None,
    pmid: str | None = None,
    abstract: str | None = None,
    page_summary: str | None = None,
    body_text: str | None = None,
) -> CandidateCitation:
    return CandidateCitation(
        id=id,
        title=title,
        url=url,
        doi=doi,
        pmid=pmid,
        source="test",
        retrieval_query="magnesium human health",
        selection_reason="Test source.",
        abstract=abstract,
        page_summary=page_summary,
        body_text=body_text,
    )


def write_minimal_run(tmp_path, sources: list[CandidateCitation]):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    packet = Packet(
        run_id="run",
        input=PacketInput(nutrient_name="magnesium", depth="light", demo=False),
        model=PacketModel(provider="heuristic", model_name="heuristic"),
        status="completed",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:00+00:00",
        counts=PacketCounts(
            candidate_citations=len(sources),
            screened_sources=len(sources),
            proposed_claims=0,
            validated_claims=0,
            rejected_claims=0,
            effect_rows=0,
        ),
    )
    write_json(run_dir / "packet.json", packet.model_dump(mode="json"))
    write_json(run_dir / "sources.json", [source.model_dump(mode="json") for source in sources])
    (run_dir / "audit_log.jsonl").write_text("", encoding="utf-8")
    return run_dir


def test_retrieve_full_text_for_run_writes_raw_text_manifest_and_counts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    run_dir = write_minimal_run(
        tmp_path,
        [
            citation(
                id="pmid:1",
                body_text="Full paper body text about magnesium and human blood pressure outcomes.",
            ),
            citation(id="doi:2", abstract="Abstract only about magnesium and human health."),
        ],
    )

    records = retrieve_full_text_for_run(run_dir)

    raw_manifest = json.loads((run_dir / "raw_texts.json").read_text(encoding="utf-8"))
    packet = json.loads((run_dir / "packet.json").read_text(encoding="utf-8"))
    audit_log = (run_dir / "audit_log.jsonl").read_text(encoding="utf-8")

    assert [record.status for record in records] == ["full_text_found", "abstract_only"]
    attempts = json.loads((run_dir / "full_text_retrieval_attempts.json").read_text(encoding="utf-8"))
    assert attempts[0]["status"] == "full_text_found"
    assert any(attempt["status"] == "no_oa_location" for attempt in attempts)
    assert raw_manifest[0]["text_path"].startswith("raw_texts/")
    assert (run_dir / raw_manifest[0]["text_path"]).read_text(encoding="utf-8").startswith("Full paper body text")
    assert raw_manifest[1]["text_path"] is None
    assert packet["counts"]["full_text_found"] == 1
    assert packet["counts"]["abstract_only"] == 1
    queue_manifest = json.loads(
        (run_dir / "manual_full_text_queue" / "manual_full_text_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert queue_manifest["count"] == 1
    assert queue_manifest["items"][0]["source_id"] == "doi:2"
    assert queue_manifest["items"][0]["folder"].startswith("manual_full_text_queue/02_")
    assert (run_dir / queue_manifest["items"][0]["folder"] / "README.md").exists()
    assert "full_text_retrieval_started" in audit_log
    assert "full_text_retrieval_completed" in audit_log
    assert "manual_full_text_queue_count" in audit_log


def test_retrieve_full_text_for_run_skips_manual_queue_when_all_full_text(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    run_dir = write_minimal_run(
        tmp_path,
        [
            citation(
                id="pmid:1",
                body_text="Full paper body text about magnesium and human blood pressure outcomes.",
            ),
        ],
    )

    retrieve_full_text_for_run(run_dir)

    assert not (run_dir / "manual_full_text_queue").exists()


def test_ingest_manual_full_text_queue_updates_records_attempts_and_counts(monkeypatch, tmp_path) -> None:
    run_dir = write_minimal_run(
        tmp_path,
        [
            citation(id="pmid:1", title="Manual acai review", pmid="1"),
            citation(id="pmid:2", title="Paywalled acai paper", pmid="2"),
        ],
    )
    write_json(
        run_dir / "raw_texts.json",
        [
            {
                "source_id": "pmid:1",
                "title": "Manual acai review",
                "pmid": "1",
                "status": "blocked_by_cloudflare",
                "retrieval_method": "firecrawl_scrape",
                "attempted_urls": ["https://example.org/one"],
                "text_char_count": 0,
            },
            {
                "source_id": "pmid:2",
                "title": "Paywalled acai paper",
                "pmid": "2",
                "status": "blocked",
                "retrieval_method": "crossref_pdf",
                "attempted_urls": ["https://example.org/two"],
                "text_char_count": 0,
            },
        ],
    )
    write_json(
        run_dir / "full_text_retrieval_attempts.json",
        [
            {
                "source_id": "pmid:1",
                "attempt_index": 1,
                "method": "firecrawl_scrape",
                "url": "https://example.org/one",
                "status": "blocked_by_cloudflare",
                "text_char_count": 0,
            }
        ],
    )
    queue_dir = run_dir / "manual_full_text_queue"
    supplied_dir = queue_dir / "01_pmid-1"
    missing_dir = queue_dir / "02_pmid-2"
    supplied_dir.mkdir(parents=True)
    missing_dir.mkdir(parents=True)
    (supplied_dir / "pmid-1.pdf").write_bytes(b"%PDF test")
    write_json(
        queue_dir / "manual_full_text_manifest.json",
        {
            "run": str(run_dir),
            "count": 2,
            "items": [
                {
                    "source_id": "pmid:1",
                    "folder": "manual_full_text_queue/01_pmid-1",
                    "save_as": "pmid-1.pdf",
                },
                {
                    "source_id": "pmid:2",
                    "folder": "manual_full_text_queue/02_pmid-2",
                    "save_as": "pmid-2.pdf",
                },
            ],
        },
    )
    monkeypatch.setattr(
        "sipz_agent.core.full_text.extract_pdf_body_text",
        lambda _: "Introduction\nMethods\nResults\nDiscussion\nAcai body text",
    )

    result = ingest_manual_full_text_queue(run_dir)

    raw_manifest = json.loads((run_dir / "raw_texts.json").read_text(encoding="utf-8"))
    attempts = json.loads((run_dir / "full_text_retrieval_attempts.json").read_text(encoding="utf-8"))
    packet = json.loads((run_dir / "packet.json").read_text(encoding="utf-8"))
    audit_log = (run_dir / "audit_log.jsonl").read_text(encoding="utf-8")

    assert result.ingested == ["pmid:1"]
    assert result.missing_assumed_paywalled == ["pmid:2"]
    assert raw_manifest[0]["status"] == "full_text_found"
    assert raw_manifest[0]["retrieval_method"] == "manual_pdf"
    assert (run_dir / raw_manifest[0]["text_path"]).read_text(encoding="utf-8").startswith("Introduction")
    assert raw_manifest[1]["status"] == "paywalled"
    assert attempts[-2]["method"] == "manual_pdf"
    assert attempts[-2]["attempt_index"] == 2
    assert attempts[-2]["status"] == "full_text_found"
    assert attempts[-1]["status"] == "paywalled"
    assert packet["counts"]["full_text_found"] == 1
    assert packet["counts"]["paywalled"] == 1
    assert "manual_full_text_ingestion_completed" in audit_log


def test_retrieve_full_text_marks_page_summary_as_fallback_without_storing_text(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(
        citation(
            abstract=None,
            page_summary="Visible landing page says magnesium was studied in adults.",
        )
    )

    assert result.record.status == "page_summary_only"
    assert result.record.retrieval_method == "publisher_page"
    assert result.body_text is None


def test_retrieve_full_text_rejects_subscription_preview_as_full_article(
    monkeypatch,
) -> None:
    preview = (
        "<html><body><article><h1>Human oral trial</h1><h2>Abstract</h2>"
        "<p>Methods and results from the abstract. "
        + ("Participants consumed the intervention. " * 80)
        + "</p><p>This is a preview of subscription content, log in via an institution "
        "to check access.</p><h2>References</h2>"
        + ("<p>Reference article with methods and discussion.</p>" * 60)
        + "</article></body></html>"
    )
    monkeypatch.setattr("sipz_agent.core.full_text.text_get", lambda *args, **kwargs: preview)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(
        citation(
            doi="10.1007/subscription-preview",
            abstract="The abstract reports a human oral trial.",
        )
    )

    assert result.record.status == "paywalled"
    assert result.body_text is None
    assert result.attempts[0].status == "paywalled"


def test_retrieve_full_text_reports_publisher_payment_required(monkeypatch) -> None:
    def raise_payment_required(*args, **kwargs):
        raise HTTPError(
            url="https://example.org/paper",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("sipz_agent.core.full_text.text_get", raise_payment_required)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(citation(abstract=None, page_summary=None))

    assert result.record.status == "paywalled"
    assert result.record.retrieval_method == "publisher_page"
    assert "HTTP 402" in (result.record.notes or "")


def test_retrieve_full_text_uses_doi_publisher_page_before_elsevier(monkeypatch) -> None:
    calls: list[str] = []

    def fake_text_get(url, **kwargs):
        calls.append(url)
        return """
        <html><body>
          <article>
            <h2>Abstract</h2><p>Quercetin was studied in humans.</p>
            <h2>Introduction</h2><p>Quercetin has been assessed for human health outcomes.</p>
            <h2>Methods</h2><p>Participants consumed quercetin in a randomized human trial.</p>
            <h2>Results</h2><p>Researchers measured clinically relevant outcomes.</p>
            <h2>Discussion</h2><p>The authors discuss limitations and implications.</p>
            <h2>References</h2><p>Reference list.</p>
          </article>
        </body></html>
        """ + (" Human article body text." * 120)

    def fail_elsevier(doi):
        raise AssertionError("Elsevier should not be used for non-Elsevier DOI prefixes")

    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.text_get", fake_text_get)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_elsevier_body_text_by_doi", fail_elsevier)

    result = retrieve_full_text_for_source(
        citation(
            url="https://pubmed.ncbi.nlm.nih.gov/35458696/",
            doi="10.3390/molecules27082498",
        )
    )

    assert result.record.status == "full_text_found"
    assert result.record.retrieval_method == "publisher_page"
    assert calls == ["https://doi.org/10.3390/molecules27082498"]
    assert result.body_text and "Quercetin was studied in humans" in result.body_text


def test_retrieve_full_text_uses_europe_pmc_fulltext_xml(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: ("MED", "123"))
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_europe_pmc_full_text",
        lambda source, record_id: "Europe PMC full text body about magnesium in a human clinical trial.",
    )

    result = retrieve_full_text_for_source(citation(pmid="123"))

    assert result.record.status == "full_text_found"
    assert result.record.retrieval_method == "europe_pmc_fulltext_xml"
    assert result.body_text and result.body_text.startswith("Europe PMC full text body")


def test_retrieve_full_text_uses_elsevier_article_api(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_elsevier_body_text_by_doi",
        lambda doi: "Elsevier article body text about magnesium human health outcomes. " * 30,
    )

    result = retrieve_full_text_for_source(citation(doi="10.1016/j.test.2026.01.001"))

    assert result.record.status == "full_text_found"
    assert result.record.retrieval_method == "elsevier_article_api"
    assert result.body_text and "Elsevier article body text" in result.body_text


def test_retrieve_full_text_uses_resolved_open_access_link_before_abstract(monkeypatch) -> None:
    candidate_link = FullTextCandidate(
        url="https://onlinelibrary.wiley.com/doi/full-xml/10.1002/cam4.1411",
        method="wiley_full_xml",
        content_type="xml",
        access_evidence="Crossref metadata provides a full-text link.",
        oa_url="https://onlinelibrary.wiley.com/doi/full-xml/10.1002/cam4.1411",
        license="https://creativecommons.org/licenses/by/4.0/",
    )
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [candidate_link])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_resolved_candidate_body_text",
        lambda candidate: (
            "Introduction Methods Results Discussion References " + ("human quercetin trial " * 200),
            candidate.url,
            candidate.method,
        ),
    )

    result = retrieve_full_text_for_source(
        citation(doi="10.1002/cam4.1411", abstract="Abstract should not be the stopping point.")
    )

    assert result.record.status == "full_text_found"
    assert result.record.retrieval_method == "wiley_full_xml"
    assert result.record.license == "https://creativecommons.org/licenses/by/4.0/"
    assert result.record.access_evidence == "Crossref metadata provides a full-text link."
    assert candidate_link.url in result.record.attempted_urls


def test_retrieve_full_text_marks_pdf_candidate_parse_failure_before_abstract(monkeypatch) -> None:
    candidate_link = FullTextCandidate(
        url="https://www.mdpi.com/1420-3049/27/8/2498/pdf",
        method="mdpi_pdf",
        content_type="pdf",
        access_evidence="Crossref metadata provides a full-text link.",
        oa_url="https://www.mdpi.com/1420-3049/27/8/2498/pdf",
        license="https://creativecommons.org/licenses/by/4.0/",
    )
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [candidate_link])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_resolved_candidate_body_text",
        lambda candidate: (None, None, candidate.method),
    )

    result = retrieve_full_text_for_source(
        citation(doi="10.3390/molecules27082498", abstract="This source has an abstract.")
    )

    assert result.record.status == "pdf_parse_failed"
    assert result.record.retrieval_method == "mdpi_pdf"
    assert result.record.oa_url == candidate_link.oa_url
    assert candidate_link.url in result.record.attempted_urls
    assert any(attempt.status == "pdf_parse_failed" for attempt in result.attempts)


def test_retrieve_full_text_handles_empty_pmc_download_method(monkeypatch) -> None:
    candidate_link = FullTextCandidate(
        url="pmc:PMC123",
        method="pmc_oa_xml",
        content_type="pmc",
        access_evidence="OpenAlex OA location points to PubMed Central.",
        oa_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
        license=None,
    )
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", lambda url: None)
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [candidate_link])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_pmc_oa_body_text", lambda pmcid: (None, None, ""))
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(
        citation(doi="10.1234/example", abstract="This source has an abstract.")
    )

    assert result.record.status == "abstract_only"
    assert result.record.retrieval_method == "none"
    assert any(attempt.method == "pmc_oa_xml" for attempt in result.attempts)


def test_retrieve_full_text_attempts_doi_landing_before_metadata_candidates(monkeypatch) -> None:
    calls: list[str] = []
    candidate_link = FullTextCandidate(
        url="https://example.org/full.xml",
        method="crossref_full_xml",
        content_type="xml",
        access_evidence="Crossref metadata provides a full-text link.",
    )

    def fake_publisher(url: str) -> None:
        calls.append(f"publisher:{url}")
        return None

    def fake_fetch_candidate(candidate: FullTextCandidate):
        calls.append(f"candidate:{candidate.url}")
        return "Introduction Methods Results Discussion References " + ("human trial " * 200), candidate.url, candidate.method

    monkeypatch.setattr("sipz_agent.core.full_text.fetch_publisher_body_text", fake_publisher)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [candidate_link])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_resolved_candidate_body_text", fake_fetch_candidate)
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(
        citation(
            doi="10.1234/example",
            url="https://publisher.example/article",
        )
    )

    assert result.record.status == "full_text_found"
    assert calls == [
        "publisher:https://doi.org/10.1234/example",
        "publisher:https://publisher.example/article",
        "candidate:https://example.org/full.xml",
    ]
    assert [attempt.method for attempt in result.attempts] == [
        "publisher_page",
        "publisher_page",
        "crossref_full_xml",
    ]


def test_retrieve_full_text_records_cloudflare_block(monkeypatch) -> None:
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_publisher_body_text",
        lambda url: (_ for _ in ()).throw(
            FullTextRetrievalFailure(
                "blocked_by_cloudflare",
                "Publisher page returned access-denied, CAPTCHA, or security-verification content.",
            )
        ),
    )
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(citation(abstract=None, page_summary=None))

    assert result.record.status == "blocked_by_cloudflare"
    assert result.record.retrieval_method == "publisher_page"
    assert result.attempts[0].status == "blocked_by_cloudflare"


def test_retrieve_full_text_records_no_oa_location_when_no_full_text_path(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.find_europe_pmc_record", lambda _: None)
    monkeypatch.setattr("sipz_agent.core.full_text.resolve_full_text_candidates", lambda _: [])
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_firecrawl_body_text", lambda url: None)

    result = retrieve_full_text_for_source(
        citation(url=None, doi=None, abstract=None, page_summary=None)
    )

    assert result.record.status == "no_oa_location"
    assert result.record.retrieval_method == "none"
    assert any(attempt.status == "no_oa_location" for attempt in result.attempts)


def test_crossref_candidates_extract_wiley_xml_and_mdpi_pdf_links() -> None:
    candidates = crossref_candidates(
        {
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
            "link": [
                {
                    "URL": "https://onlinelibrary.wiley.com/doi/full-xml/10.1002/cam4.1411",
                    "content-type": "application/xml",
                },
                {
                    "URL": "https://www.mdpi.com/1420-3049/27/8/2498/pdf",
                    "content-type": "unspecified",
                },
            ],
        }
    )

    assert [candidate.method for candidate in candidates] == ["wiley_full_xml", "mdpi_pdf"]
    assert candidates[0].license == "https://creativecommons.org/licenses/by/4.0/"


def test_crossref_candidates_skip_elsevier_api_links() -> None:
    candidates = crossref_candidates(
        {
            "link": [
                {
                    "URL": (
                        "https://api.elsevier.com/content/article/PII:S0924224416301817"
                        "?httpAccept=text/plain"
                    ),
                    "content-type": "text/plain",
                }
            ]
        }
    )

    assert candidates == []


def test_mdpi_doi_derives_html_xml_and_pdf_candidates(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_unpaywall_work_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_openalex_work_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_crossref_work_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_pmcid_for_citation", lambda citation: None)

    base_url = mdpi_article_base_url_from_doi("10.3390/nu15040989")
    candidates = mdpi_candidates_for_doi("10.3390/nu15040989")
    resolved = resolve_full_text_candidates(citation(doi="10.3390/nu15040989"))

    assert base_url == "https://www.mdpi.com/2072-6643/15/4/989"
    assert [candidate.url for candidate in candidates] == [
        "https://www.mdpi.com/2072-6643/15/4/989",
        "https://www.mdpi.com/2072-6643/15/4/989/xml",
        "https://www.mdpi.com/2072-6643/15/4/989/pdf?download=1",
        "https://www.mdpi.com/2072-6643/15/4/989/pdf",
    ]
    assert [candidate.method for candidate in candidates] == [
        "publisher_page",
        "mdpi_xml",
        "mdpi_pdf",
        "mdpi_pdf",
    ]
    assert [candidate.url for candidate in resolved[:4]] == [candidate.url for candidate in candidates]


def test_unpaywall_candidates_rank_versions_and_preserve_provenance() -> None:
    candidates = unpaywall_candidates(
        {
            "is_oa": True,
            "oa_status": "hybrid",
            "best_oa_location": {
                "url_for_pdf": "https://repository.example/submitted.pdf",
                "url_for_landing_page": "https://repository.example/submitted",
                "version": "submittedVersion",
                "host_type": "repository",
                "license": "cc-by",
            },
            "oa_locations": [
                {
                    "url_for_pdf": "https://publisher.example/published.pdf",
                    "url_for_landing_page": "https://publisher.example/article",
                    "version": "publishedVersion",
                    "host_type": "publisher",
                    "license": "cc-by",
                },
                {
                    "url_for_pdf": "https://repository.example/accepted.pdf",
                    "version": "acceptedVersion",
                    "host_type": "repository",
                    "license": None,
                },
            ],
        }
    )

    assert [candidate.manuscript_version for candidate in candidates] == [
        "publishedVersion",
        "publishedVersion",
        "acceptedVersion",
        "submittedVersion",
        "submittedVersion",
    ]
    assert candidates[0].method == "unpaywall_pdf"
    assert candidates[0].oa_host_type == "publisher"
    assert candidates[0].oa_status == "hybrid"
    assert candidates[0].license == "cc-by"


def test_unpaywall_candidates_deduplicate_best_location() -> None:
    location = {
        "url_for_pdf": "https://repository.example/paper.pdf",
        "version": "acceptedVersion",
        "host_type": "repository",
    }

    candidates = unpaywall_candidates(
        {
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": location,
            "oa_locations": [location],
        }
    )

    assert len(candidates) == 1


def test_fetch_unpaywall_requires_email(monkeypatch) -> None:
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.setattr("sipz_agent.core.config.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "sipz_agent.core.full_text.json_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call API")),
    )

    assert fetch_unpaywall_work_by_doi("10.1234/example") is None


def test_resolve_full_text_candidates_includes_unpaywall(monkeypatch) -> None:
    monkeypatch.setattr(
        "sipz_agent.core.full_text.fetch_unpaywall_work_by_doi",
        lambda doi: {
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": {
                "url_for_pdf": "https://repository.example/paper.pdf",
                "version": "acceptedVersion",
                "host_type": "repository",
            },
        },
    )
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_openalex_work_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_crossref_work_by_doi", lambda doi: None)
    monkeypatch.setattr("sipz_agent.core.full_text.fetch_pmcid_for_citation", lambda citation: None)

    resolved = resolve_full_text_candidates(citation(doi="10.1234/example"))

    assert len(resolved) == 1
    assert resolved[0].method == "unpaywall_pdf"
    assert resolved[0].manuscript_version == "acceptedVersion"


def test_pdf_and_xml_fetches_use_browser_like_headers(monkeypatch) -> None:
    calls: list[tuple[str, dict | None]] = []

    def fake_bytes_get(url, headers=None, timeout=45):
        calls.append((url, headers))
        raise FullTextRetrievalFailure("pdf_parse_failed", "stop")

    def fake_text_get(url, headers=None, timeout=30):
        calls.append((url, headers))
        return "<article><body>short</body></article>"

    monkeypatch.setattr("sipz_agent.core.full_text.bytes_get", fake_bytes_get)
    monkeypatch.setattr("sipz_agent.core.full_text.text_get", fake_text_get)

    try:
        fetch_pdf_body_text("https://www.mdpi.com/2072-6643/15/4/989/pdf?download=1")
    except FullTextRetrievalFailure:
        pass
    fetch_xml_body_text("https://www.mdpi.com/2072-6643/15/4/989/xml")

    pdf_headers = calls[0][1] or {}
    xml_headers = calls[1][1] or {}
    assert "Mozilla/5.0" in pdf_headers["User-Agent"]
    assert "application/pdf" in pdf_headers["Accept"]
    assert "Mozilla/5.0" in xml_headers["User-Agent"]
    assert "application/xml" in xml_headers["Accept"]


def test_extract_elsevier_body_text_reads_xml_original_text() -> None:
    body = (
        "<article><body><sec><title>Introduction</title><p>"
        + ("Anethole was evaluated in human health research. " * 30)
        + "</p></sec><sec><title>Methods</title><p>"
        + ("Participants consumed the test material orally. " * 30)
        + "</p></sec></body></article>"
    )

    text = extract_elsevier_body_text(
        {"full-text-retrieval-response": {"originalText": body}}
    )

    assert text is not None
    assert "<article>" not in text
    assert "Participants consumed" in text


def test_elsevier_article_api_does_not_send_invalid_full_view(monkeypatch) -> None:
    urls: list[str] = []
    monkeypatch.setenv("ELSEVIER_API_KEY", "test-key")

    def fake_json_get(url, headers=None, timeout=30):
        urls.append(url)
        return {
            "full-text-retrieval-response": {
                "originalText": {"xocs:doc": {"xocs:meta": "metadata only"}}
            }
        }

    def fake_text_get(url, headers=None, timeout=30):
        urls.append(url)
        return "<article><metadata>metadata only</metadata></article>"

    monkeypatch.setattr("sipz_agent.core.full_text.json_get", fake_json_get)
    monkeypatch.setattr("sipz_agent.core.full_text.text_get", fake_text_get)

    text = fetch_elsevier_body_text_by_doi("10.1016/j.example.2026.01.001")

    assert text is None
    assert all("view=FULL" not in url for url in urls)
    assert any("httpAccept=application/json" in url for url in urls)
    assert any("httpAccept=text/xml" in url for url in urls)


def test_openalex_doi_lookup_uses_configured_api_key(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")

    def fake_json_get(url: str, **kwargs):
        captured["url"] = url
        return {"id": "https://openalex.org/W1"}

    monkeypatch.setattr("sipz_agent.core.full_text.json_get", fake_json_get)

    assert fetch_openalex_work_by_doi("10.1234/example") == {"id": "https://openalex.org/W1"}
    assert "api_key=openalex-secret" in captured["url"]


def test_crossref_doi_lookup_uses_configured_mailto(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("CROSSREF_MAILTO", "research@example.com")

    def fake_json_get(url: str, **kwargs):
        captured["url"] = url
        return {"message": {"DOI": "10.1234/example"}}

    monkeypatch.setattr("sipz_agent.core.full_text.json_get", fake_json_get)

    assert fetch_crossref_work_by_doi("10.1234/example") == {"DOI": "10.1234/example"}
    assert "mailto=research%40example.com" in captured["url"]


def test_openalex_candidates_include_pubmed_central_location_before_pdf() -> None:
    candidates = openalex_candidates(
        {
            "open_access": {"oa_status": "gold"},
            "best_oa_location": {
                "landing_page_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9032170",
                "pdf_url": "https://www.mdpi.com/1420-3049/27/8/2498/pdf",
                "license": "cc-by",
            },
        }
    )

    assert candidates[0].url == "pmc:PMC9032170"
    assert candidates[0].method == "pmc_oa_xml"
    assert candidates[1].method == "mdpi_pdf"


def test_pmc_helpers_handle_openalex_numeric_pmc_urls_and_deprecated_downloads() -> None:
    assert pmcid_from_url("https://www.ncbi.nlm.nih.gov/pmc/articles/9032170") == "PMC9032170"
    assert pmc_download_url_variants("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/ca/94/x.tar.gz") == [
        "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/ca/94/x.tar.gz",
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/ca/94/x.tar.gz",
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_package/ca/94/x.tar.gz",
    ]


def test_cli_retrieve_text_command_processes_existing_run(tmp_path) -> None:
    run_dir = write_minimal_run(
        tmp_path,
        [citation(id="pmid:1", body_text="Full paper body text about magnesium outcomes.")],
    )
    runner = CliRunner()

    result = runner.invoke(app, ["retrieve-text", "--run", str(run_dir)])

    assert result.exit_code == 0
    assert "full_text_found: 1" in result.stdout
    assert (run_dir / "raw_texts.json").exists()
    assert (run_dir / "full_text_retrieval_attempts.json").exists()


def test_cli_study_retrieve_full_text_flag_writes_raw_text_artifacts(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "study",
            "fluoride",
            "--demo",
            "--retrieve-full-text",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    run_dir = next(tmp_path.iterdir())
    packet = json.loads((run_dir / "packet.json").read_text(encoding="utf-8"))
    assert (run_dir / "raw_texts.json").exists()
    assert (run_dir / "raw_texts.md").exists()
    assert (run_dir / "full_text_retrieval_attempts.json").exists()
    assert (run_dir / "raw_texts").is_dir()
    assert packet["counts"]["full_text_found"] == 2
