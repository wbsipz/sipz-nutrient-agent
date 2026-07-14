#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any
from urllib import error, parse, request

import orjson
from pydantic import TypeAdapter

from sipz_agent.core.artifacts import model_dump_jsonable, write_json
from sipz_agent.core.full_text import (
    PDF_HEADERS,
    PUBLISHER_HTML_HEADERS,
    FullTextRetrievalFailure,
    append_audit_event,
    classify_http_error,
    extract_pdf_body_text,
    extract_publisher_body_text_from_html,
    load_full_text_attempts,
    next_attempt_index_by_source,
    raw_text_counts,
    safe_source_filename,
    update_optional_ingredient_packet_counts,
    update_packet_counts,
    write_manual_full_text_queue,
    write_raw_texts_markdown,
    is_access_denied_text,
    is_paywall_text,
)
from sipz_agent.schemas.citations import CandidateCitation
from sipz_agent.schemas.raw_texts import FullTextRetrievalAttempt, RawTextRecord


DEFAULT_INGREDIENTS = [
    "almond",
    "almond creamer",
    "almond milk",
    "aloe vera",
    "amazake",
]
SOURCES_ADAPTER = TypeAdapter(list[CandidateCitation])
RAW_TEXTS_ADAPTER = TypeAdapter(list[RawTextRecord])
REQUEST_TIMES: dict[str, float] = {}
HOST_COOLDOWNS_UNTIL: dict[str, float] = {}
HOST_MIN_INTERVAL_SECONDS = {
    "api.unpaywall.org": 1.0,
    "doi.org": 2.0,
    "mdpi.com": 5.0,
    "ncbi.nlm.nih.gov": 3.0,
    "europepmc.org": 3.0,
    "ebi.ac.uk": 3.0,
    "doaj.org": 3.0,
    "jstage.jst.go.jp": 5.0,
    "medrxiv.org": 4.0,
}


def normalized_host(url: str) -> str:
    return parse.urlparse(url).netloc.lower().removeprefix("www.")


def host_min_interval_seconds(host: str, default_seconds: float) -> float:
    for configured_host, seconds in HOST_MIN_INTERVAL_SECONDS.items():
        normalized = configured_host.removeprefix("www.")
        if host == normalized or host.endswith(f".{normalized}"):
            return seconds
    return default_seconds


def gentle_throttle(url: str, *, default_delay_seconds: float, jitter_seconds: float) -> None:
    host = normalized_host(url)
    if not host:
        return
    now = time.monotonic()
    interval = host_min_interval_seconds(host, default_delay_seconds)
    cooldown_delay = max(0.0, HOST_COOLDOWNS_UNTIL.get(host, 0.0) - now)
    interval_delay = interval - (now - REQUEST_TIMES.get(host, 0.0))
    delay = max(cooldown_delay, interval_delay)
    if delay > 0:
        time.sleep(delay + random.uniform(0.0, jitter_seconds))
    REQUEST_TIMES[host] = time.monotonic()


def note_cooldown(url: str, seconds: float, *, jitter_seconds: float) -> None:
    host = normalized_host(url)
    if not host:
        return
    until = time.monotonic() + seconds + random.uniform(0.0, jitter_seconds)
    HOST_COOLDOWNS_UNTIL[host] = max(HOST_COOLDOWNS_UNTIL.get(host, 0.0), until)


def gentle_urlopen(
    url: str,
    *,
    headers: dict[str, str] | None,
    timeout: int,
    default_delay_seconds: float,
    jitter_seconds: float,
    blocked_cooldown_seconds: float,
    max_attempts: int,
):
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        gentle_throttle(url, default_delay_seconds=default_delay_seconds, jitter_seconds=jitter_seconds)
        req = request.Request(url, headers=headers or {"User-Agent": "sipz-nutrient-agent/0.1"})
        try:
            return request.urlopen(req, timeout=timeout)
        except error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    cooldown = float(retry_after) if retry_after else blocked_cooldown_seconds
                except ValueError:
                    cooldown = blocked_cooldown_seconds
                note_cooldown(url, min(max(cooldown, 0.0), 120.0), jitter_seconds=jitter_seconds)
            elif exc.code == 403:
                note_cooldown(url, blocked_cooldown_seconds, jitter_seconds=jitter_seconds)
                raise
            if attempt == max_attempts - 1 or exc.code not in {429, 500, 502, 503, 504}:
                raise
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
        time.sleep(min(10.0, (2**attempt) + random.uniform(0.0, jitter_seconds)))
    if last_error:
        raise last_error
    raise RuntimeError("gentle_urlopen_failed")


def gentle_read_bytes(
    url: str,
    *,
    headers: dict[str, str] | None,
    timeout: int,
    default_delay_seconds: float,
    jitter_seconds: float,
    blocked_cooldown_seconds: float,
    max_attempts: int,
) -> bytes:
    with gentle_urlopen(
        url,
        headers=headers,
        timeout=timeout,
        default_delay_seconds=default_delay_seconds,
        jitter_seconds=jitter_seconds,
        blocked_cooldown_seconds=blocked_cooldown_seconds,
        max_attempts=max_attempts,
    ) as response:
        return response.read()


def gentle_read_text(
    url: str,
    *,
    headers: dict[str, str] | None,
    timeout: int,
    default_delay_seconds: float,
    jitter_seconds: float,
    blocked_cooldown_seconds: float,
    max_attempts: int,
) -> str:
    return gentle_read_bytes(
        url,
        headers=headers,
        timeout=timeout,
        default_delay_seconds=default_delay_seconds,
        jitter_seconds=jitter_seconds,
        blocked_cooldown_seconds=blocked_cooldown_seconds,
        max_attempts=max_attempts,
    ).decode("utf-8", errors="replace")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "ingredient"


def latest_run_for_ingredient(runs_root: Path, ingredient: str) -> Path:
    slug = slugify(ingredient)
    matches = sorted(runs_root.glob(f"*_{slug}"), key=lambda path: path.name, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No run found for ingredient '{ingredient}' with slug '{slug}'")
    return matches[0]


def ingredient_name_from_run_dir(run_dir: Path) -> str:
    match = re.match(r".*?_(.+)$", run_dir.name)
    if not match:
        return run_dir.name
    return match.group(1).replace("-", " ")


def discover_run_dirs(runs_root: Path, *, all_runs: bool, ingredients: list[str]) -> list[tuple[str, Path]]:
    if all_runs:
        discovered: list[tuple[str, Path]] = []
        for run_dir in sorted(runs_root.glob("*")):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "sources.json").exists() or not (run_dir / "raw_texts.json").exists():
                continue
            discovered.append((ingredient_name_from_run_dir(run_dir), run_dir))
        return discovered
    return [(ingredient, latest_run_for_ingredient(runs_root, ingredient)) for ingredient in ingredients]


def load_sources(run_dir: Path) -> list[CandidateCitation]:
    return SOURCES_ADAPTER.validate_python(orjson.loads((run_dir / "sources.json").read_bytes()))


def load_raw_texts(run_dir: Path) -> list[RawTextRecord]:
    return RAW_TEXTS_ADAPTER.validate_python(orjson.loads((run_dir / "raw_texts.json").read_bytes()))


def unpaywall_record(
    doi: str,
    email: str,
    *,
    default_delay_seconds: float,
    jitter_seconds: float,
    blocked_cooldown_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    url = "https://api.unpaywall.org/v2/" + parse.quote(doi, safe="") + "?" + parse.urlencode({"email": email})
    text = gentle_read_text(
        url,
        headers={"User-Agent": "sipz-nutrient-agent/0.1 (mailto:%s)" % email},
        timeout=30,
        default_delay_seconds=default_delay_seconds,
        jitter_seconds=jitter_seconds,
        blocked_cooldown_seconds=blocked_cooldown_seconds,
        max_attempts=max_attempts,
    )
    return json.loads(text)


def location_label(location: dict[str, Any]) -> str:
    host_type = location.get("host_type") or "unknown_host"
    version = location.get("version") or "unknown_version"
    return f"Unpaywall OA location ({host_type}, {version})."


def candidate_urls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    locations = []
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    for location in payload.get("oa_locations") or []:
        if isinstance(location, dict):
            locations.append(location)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for location in locations:
        license_value = location.get("license")
        for kind, key in [
            ("unpaywall_pdf", "url_for_pdf"),
            ("unpaywall_landing_page", "url_for_landing_page"),
            ("unpaywall_landing_page", "url"),
        ]:
            url = location.get(key)
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            dedupe_key = (kind, url)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "method": kind,
                    "url": url,
                    "license": license_value,
                    "access_evidence": location_label(location),
                }
            )
    return candidates


def fetch_candidate_body(
    candidate: dict[str, Any],
    *,
    default_delay_seconds: float,
    jitter_seconds: float,
    blocked_cooldown_seconds: float,
    max_attempts: int,
) -> str | None:
    url = candidate["url"]
    if candidate["method"] == "unpaywall_pdf":
        data = gentle_read_bytes(
            url,
            headers=PDF_HEADERS,
            timeout=60,
            default_delay_seconds=default_delay_seconds,
            jitter_seconds=jitter_seconds,
            blocked_cooldown_seconds=blocked_cooldown_seconds,
            max_attempts=max_attempts,
        )
        return extract_pdf_body_text(data)

    html_text = gentle_read_text(
        url,
        headers=PUBLISHER_HTML_HEADERS,
        timeout=30,
        default_delay_seconds=default_delay_seconds,
        jitter_seconds=jitter_seconds,
        blocked_cooldown_seconds=blocked_cooldown_seconds,
        max_attempts=max_attempts,
    )
    if is_access_denied_text(html_text):
        note_cooldown(url, blocked_cooldown_seconds, jitter_seconds=jitter_seconds)
        raise FullTextRetrievalFailure(
            "blocked_by_cloudflare",
            "Publisher page returned access-denied, CAPTCHA, or security-verification content.",
        )
    if is_paywall_text(html_text):
        raise FullTextRetrievalFailure("paywalled", "Publisher page appears to require payment or login.")
    return extract_publisher_body_text_from_html(html_text)


def source_doi(source: CandidateCitation | None, record: RawTextRecord) -> str | None:
    return record.doi or (source.doi if source else None)


def attempt_unpaywall_for_run(
    run_dir: Path,
    email: str,
    *,
    overwrite: bool = False,
    default_delay_seconds: float = 2.0,
    jitter_seconds: float = 2.0,
    blocked_cooldown_seconds: float = 30.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    sources = load_sources(run_dir)
    records = load_raw_texts(run_dir)
    source_by_id = {source.id: source for source in sources}
    attempts = load_full_text_attempts(run_dir)
    next_indexes = next_attempt_index_by_source(attempts)
    raw_dir = run_dir / "raw_texts"
    raw_dir.mkdir(exist_ok=True)

    report: dict[str, Any] = {
        "run": str(run_dir),
        "started_at": datetime.now(UTC).isoformat(),
        "email": email,
        "records_considered": 0,
        "doi_records": 0,
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "unpaywall_trial_started",
            "email": email,
        },
    )

    for index, record in enumerate(records):
        report["records_considered"] += 1
        if record.status == "full_text_found" and not overwrite:
            report["skipped"].append(
                {
                    "source_id": record.source_id,
                    "reason": "already_full_text_found",
                }
            )
            continue

        source = source_by_id.get(record.source_id)
        doi = source_doi(source, record)
        if not doi:
            report["skipped"].append({"source_id": record.source_id, "reason": "missing_doi"})
            continue
        report["doi_records"] += 1

        try:
            payload = unpaywall_record(
                doi,
                email,
                default_delay_seconds=default_delay_seconds,
                jitter_seconds=jitter_seconds,
                blocked_cooldown_seconds=blocked_cooldown_seconds,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            status, notes = classify_http_error(exc)
            attempt_index = next_indexes.get(record.source_id, 1)
            next_indexes[record.source_id] = attempt_index + 1
            attempts.append(
                FullTextRetrievalAttempt(
                    source_id=record.source_id,
                    attempt_index=attempt_index,
                    method="none",
                    url=f"https://api.unpaywall.org/v2/{doi}",
                    status=status,  # type: ignore[arg-type]
                    http_status=exc.code if isinstance(exc, error.HTTPError) else None,
                    text_char_count=0,
                    notes=f"Unpaywall metadata lookup failed: {notes}",
                )
            )
            report["failed"].append(
                {
                    "source_id": record.source_id,
                    "doi": doi,
                    "status": status,
                    "notes": notes,
                }
            )
            continue

        urls = candidate_urls(payload)
        if not urls:
            report["skipped"].append(
                {
                    "source_id": record.source_id,
                    "doi": doi,
                    "reason": "unpaywall_no_oa_url",
                    "is_oa": payload.get("is_oa"),
                }
            )
            continue

        found_text: str | None = None
        found_candidate: dict[str, Any] | None = None
        candidate_failures: list[dict[str, Any]] = []
        for candidate in urls:
            attempt_index = next_indexes.get(record.source_id, 1)
            next_indexes[record.source_id] = attempt_index + 1
            try:
                found_text = fetch_candidate_body(
                    candidate,
                    default_delay_seconds=default_delay_seconds,
                    jitter_seconds=jitter_seconds,
                    blocked_cooldown_seconds=blocked_cooldown_seconds,
                    max_attempts=max_attempts,
                )
            except Exception as exc:
                status, notes = classify_http_error(exc)
                if isinstance(exc, FullTextRetrievalFailure):
                    status = exc.status
                    notes = exc.notes
                attempts.append(
                    FullTextRetrievalAttempt(
                        source_id=record.source_id,
                        attempt_index=attempt_index,
                        method=candidate["method"],
                        url=candidate["url"],
                        status=status,  # type: ignore[arg-type]
                        http_status=exc.code if isinstance(exc, error.HTTPError) else None,
                        resolved_url=candidate["url"],
                        oa_url=candidate["url"],
                        license=candidate.get("license"),
                        access_evidence=candidate.get("access_evidence"),
                        text_char_count=0,
                        notes=notes,
                    )
                )
                candidate_failures.append(
                    {
                        "method": candidate["method"],
                        "url": candidate["url"],
                        "status": status,
                        "notes": notes,
                    }
                )
                continue

            if found_text:
                attempts.append(
                    FullTextRetrievalAttempt(
                        source_id=record.source_id,
                        attempt_index=attempt_index,
                        method=candidate["method"],
                        url=candidate["url"],
                        status="full_text_found",
                        resolved_url=candidate["url"],
                        oa_url=candidate["url"],
                        license=candidate.get("license"),
                        access_evidence=candidate.get("access_evidence"),
                        text_char_count=len(found_text),
                        notes="Retrieved body text through Unpaywall OA location.",
                    )
                )
                found_candidate = candidate
                break

            attempts.append(
                FullTextRetrievalAttempt(
                    source_id=record.source_id,
                    attempt_index=attempt_index,
                    method=candidate["method"],
                    url=candidate["url"],
                    status="not_available",
                    resolved_url=candidate["url"],
                    oa_url=candidate["url"],
                    license=candidate.get("license"),
                    access_evidence=candidate.get("access_evidence"),
                    text_char_count=0,
                    notes="Unpaywall OA URL did not yield usable article body text.",
                )
            )
            candidate_failures.append(
                {
                    "method": candidate["method"],
                    "url": candidate["url"],
                    "status": "not_available",
                    "notes": "Unpaywall OA URL did not yield usable article body text.",
                }
            )

        if not found_text or not found_candidate:
            report["failed"].append(
                {
                    "source_id": record.source_id,
                    "doi": doi,
                    "candidate_count": len(urls),
                    "failures": candidate_failures,
                }
            )
            continue

        filename = safe_source_filename(record.source_id, index + 1)
        text_path = raw_dir / filename
        text_path.write_text(found_text, encoding="utf-8")
        records[index] = record.model_copy(
            update={
                "status": "full_text_found",
                "retrieval_method": found_candidate["method"],
                "resolved_url": found_candidate["url"],
                "oa_url": found_candidate["url"],
                "license": found_candidate.get("license"),
                "access_evidence": found_candidate.get("access_evidence"),
                "attempted_urls": [*record.attempted_urls, *[candidate["url"] for candidate in urls]],
                "text_path": str(Path("raw_texts") / filename),
                "text_char_count": len(found_text),
                "notes": "Retrieved body text through Unpaywall OA location.",
            }
        )
        report["successful"].append(
            {
                "source_id": record.source_id,
                "doi": doi,
                "method": found_candidate["method"],
                "url": found_candidate["url"],
                "text_char_count": len(found_text),
            }
        )

    counts = raw_text_counts(records)
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["raw_text_status_counts"] = counts
    report["summary"] = {
        "successful": len(report["successful"]),
        "failed": len(report["failed"]),
        "skipped": len(report["skipped"]),
    }

    write_json(run_dir / "raw_texts.json", model_dump_jsonable(records))
    write_json(run_dir / "full_text_retrieval_attempts.json", model_dump_jsonable(attempts))
    write_raw_texts_markdown(run_dir / "raw_texts.md", records)
    update_packet_counts(run_dir, counts)
    update_optional_ingredient_packet_counts(run_dir, counts)
    write_manual_full_text_queue(run_dir, sources=sources, records=records)
    write_json(run_dir / "unpaywall_retrieval_report.json", report)
    append_audit_event(
        run_dir,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "unpaywall_trial_completed",
            "successful": len(report["successful"]),
            "failed": len(report["failed"]),
            "skipped": len(report["skipped"]),
            "counts": counts,
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try Unpaywall OA full-text retrieval for existing ingredient runs."
    )
    parser.add_argument(
        "ingredients",
        nargs="*",
        default=DEFAULT_INGREDIENTS,
        help="Ingredient names to process. Defaults to the almond/aloe/amazake trial set unless --all-runs is used.",
    )
    parser.add_argument(
        "--ingredients-file",
        type=Path,
        help="Newline-delimited ingredient names to process. Overrides positional ingredients.",
    )
    parser.add_argument("--runs-root", default="ingredient_runs")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process every existing ingredient run under --runs-root that has sources.json and raw_texts.json.",
    )
    parser.add_argument("--email", default=os.environ.get("UNPAYWALL_EMAIL"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retry sources that already have full text. Default only fills missing/blocked records.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=2.0,
        help="Minimum delay between requests to the same host for this follow-up script.",
    )
    parser.add_argument(
        "--jitter-seconds",
        type=float,
        default=3.0,
        help="Maximum random extra delay added to host waits.",
    )
    parser.add_argument(
        "--blocked-cooldown-seconds",
        type=float,
        default=45.0,
        help="Cooldown for a host after 403/429/access-denied responses.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum HTTP attempts per Unpaywall or OA URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.email:
        raise SystemExit("Set UNPAYWALL_EMAIL or pass --email. Unpaywall requires a real contact email.")
    if args.delay_seconds < 0 or args.jitter_seconds < 0 or args.blocked_cooldown_seconds < 0:
        raise SystemExit("Delay, jitter, and cooldown values must be non-negative.")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1.")

    ingredients = args.ingredients
    if args.ingredients_file is not None:
        ingredients = [
            line.strip()
            for line in args.ingredients_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not ingredients:
            raise SystemExit(f"No ingredients found in {args.ingredients_file}.")

    runs_root = Path(args.runs_root)
    run_items = discover_run_dirs(
        runs_root,
        all_runs=args.all_runs,
        ingredients=ingredients,
    )
    if not run_items:
        raise SystemExit(f"No eligible runs found under {runs_root}.")
    aggregate = {
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "all_runs" if args.all_runs else "ingredients",
        "ingredients": [ingredient for ingredient, _ in run_items],
        "runs": [],
    }
    total_success = 0
    total_failed = 0
    total_skipped = 0

    for ingredient, run_dir in run_items:
        print(f"=== {ingredient}: {run_dir} ===")
        report = attempt_unpaywall_for_run(
            run_dir,
            args.email,
            overwrite=args.overwrite,
            default_delay_seconds=args.delay_seconds,
            jitter_seconds=args.jitter_seconds,
            blocked_cooldown_seconds=args.blocked_cooldown_seconds,
            max_attempts=args.max_attempts,
        )
        summary = report["summary"]
        total_success += summary["successful"]
        total_failed += summary["failed"]
        total_skipped += summary["skipped"]
        aggregate["runs"].append(
            {
                "ingredient": ingredient,
                "run": str(run_dir),
                "summary": summary,
                "raw_text_status_counts": report["raw_text_status_counts"],
            }
        )
        print(
            "success={successful} failed={failed} skipped={skipped} statuses={statuses}".format(
                successful=summary["successful"],
                failed=summary["failed"],
                skipped=summary["skipped"],
                statuses=dict(Counter(report["raw_text_status_counts"])),
            )
        )

    aggregate["completed_at"] = datetime.now(UTC).isoformat()
    aggregate["summary"] = {
        "successful": total_success,
        "failed": total_failed,
        "skipped": total_skipped,
    }
    out_path = runs_root / "unpaywall_trial_summary.json"
    write_json(out_path, aggregate)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
