"""Site keyword audit using Playwright.

This script crawls a website starting from a base URL and inspects each page
for the presence of specific keywords in the page title, body text, and
metadata. It outputs a structured summary of all keyword matches that it finds.

Example:
    python site_keyword_audit.py https://fandbsports.com \
        --keyword wheel --keyword rim --keyword bike --max-pages 100

The script only visits pages that share the same domain as the base URL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from playwright.async_api import Browser, Error, Page, async_playwright


DEFAULT_KEYWORDS = [
    "wheel",
    "wheels",
    "rim",
    "rims",
    "bike",
    "bikes",
    "mountain",
    "mountain bike",
]


@dataclass
class Match:
    """Represents a single keyword match in a page."""

    keyword: str
    location: str
    snippet: str


@dataclass
class PageReport:
    """Represents all matches found on a single page."""

    url: str
    matches: List[Match] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "matches": [match.__dict__ for match in self.matches],
        }


def normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower()


def normalize_url(
    reference_url: str, raw_url: str, allowed_netlocs: Sequence[str]
) -> Optional[str]:
    """Resolve a raw URL relative to ``reference_url`` if it is in-domain."""

    if not raw_url:
        return None

    joined = urljoin(reference_url, raw_url)
    parsed_joined = urlparse(joined)
    if parsed_joined.scheme not in {"http", "https"}:
        return None

    if parsed_joined.netloc not in allowed_netlocs:
        return None

    cleaned, _ = urldefrag(parsed_joined.geturl())
    return cleaned


def find_snippets(text: str, keyword: str, *, context: int = 60) -> Iterable[str]:
    """Yield snippets of text surrounding each keyword occurrence."""

    lower_text = text.lower()
    lower_keyword = keyword.lower()
    start = 0
    while True:
        index = lower_text.find(lower_keyword, start)
        if index == -1:
            break
        snippet_start = max(0, index - context)
        snippet_end = min(len(text), index + len(keyword) + context)
        snippet = " ".join(text[snippet_start:snippet_end].split())
        yield snippet
        start = index + len(keyword)


async def extract_links(
    page: Page, base_url: str, allowed_netlocs: Sequence[str]
) -> Set[str]:
    hrefs: List[str] = await page.eval_on_selector_all(
        "a[href]", "elements => elements.map(el => el.getAttribute('href'))"
    )
    links: Set[str] = set()
    for href in hrefs:
        normalized = normalize_url(base_url, href, allowed_netlocs)
        if normalized:
            links.add(normalized)
    return links


async def get_page_text(page: Page) -> Tuple[str, str]:
    title = await page.title()
    body_text = await page.evaluate("document.body ? document.body.innerText : ''")
    return title or "", body_text or ""


async def get_metadata(page: Page) -> List[Tuple[str, str]]:
    meta_entries: List[Tuple[str, str]] = await page.evaluate(
        """
        () => Array.from(document.querySelectorAll('meta'))
            .map(meta => [
                meta.getAttribute('name') || meta.getAttribute('property') || '',
                meta.getAttribute('content') || ''
            ])
        """
    )
    return meta_entries


async def analyze_page(
    page: Page,
    url: str,
    keywords: Iterable[str],
    allowed_netlocs: Sequence[str],
) -> Tuple[PageReport, Set[str]]:
    await page.goto(url, wait_until="networkidle")
    links = await extract_links(page, url, allowed_netlocs)

    title, body_text = await get_page_text(page)
    metadata = await get_metadata(page)

    matches: List[Match] = []
    combined_sources = [("title", title), ("body", body_text)]

    for meta_name, meta_content in metadata:
        label = f"meta:{meta_name or 'unnamed'}"
        combined_sources.append((label, meta_content))

    for keyword in keywords:
        for location, text in combined_sources:
            if not text:
                continue
            for snippet in find_snippets(text, keyword):
                matches.append(Match(keyword=keyword, location=location, snippet=snippet))

    return PageReport(url=url, matches=matches), links


async def crawl_site(
    base_url: str,
    keywords: Iterable[str],
    *,
    max_pages: Optional[int] = None,
    delay: float = 0.0,
) -> List[PageReport]:
    normalized_keywords = [normalize_keyword(k) for k in keywords if k.strip()]
    base_url = urldefrag(base_url)[0]
    base_parsed = urlparse(base_url)
    base_netloc = base_parsed.netloc
    allowed_netlocs = {base_netloc}
    if base_netloc.startswith("www."):
        allowed_netlocs.add(base_netloc[4:])
    else:
        allowed_netlocs.add(f"www.{base_netloc}")

    allowed_netlocs_seq: Sequence[str] = tuple(sorted(allowed_netlocs))

    queue: deque[str] = deque([base_url])
    visited: Set[str] = set()
    reports: List[PageReport] = []

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch()
        context = await browser.new_context()

        try:
            while queue:
                current_url = queue.popleft()
                if current_url in visited:
                    continue
                if max_pages is not None and len(visited) >= max_pages:
                    break

                visited.add(current_url)
                page = await context.new_page()
                try:
                    report, links = await analyze_page(
                        page,
                        current_url,
                        normalized_keywords,
                        allowed_netlocs_seq,
                    )
                except Error as exc:
                    print(f"Failed to analyze {current_url}: {exc}", file=sys.stderr)
                else:
                    if report.matches:
                        reports.append(report)
                    for link in links:
                        if link not in visited:
                            queue.append(link)
                finally:
                    await page.close()
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            await context.close()
            await browser.close()

    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_url",
        help="The starting URL for the crawl. Only pages on this domain are visited.",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        action="append",
        dest="keywords",
        help=(
            "Keyword to search for. Can be specified multiple times. Defaults to a set "
            "of common wheel-related terms."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit on the number of pages to crawl.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between page visits to avoid overwhelming the site.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report with all matches.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    base_url = args.base_url
    keywords = args.keywords or DEFAULT_KEYWORDS

    reports = await crawl_site(
        base_url,
        keywords,
        max_pages=args.max_pages,
        delay=args.delay,
    )

    if not reports:
        print("No keyword matches found.")
    else:
        for report in reports:
            print(f"\nURL: {report.url}")
            for match in report.matches:
                print(f"  - [{match.location}] '{match.keyword}': {match.snippet}")

    if args.output:
        data = [report.to_dict() for report in reports]
        args.output.write_text(json.dumps(data, indent=2))
        print(f"\nReport written to {args.output}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalized_base = normalize_url(args.base_url, args.base_url, [urlparse(args.base_url).netloc])
    if not normalized_base:
        parser.error("base_url must be an HTTP(S) URL and include a domain.")
    args.base_url = normalized_base
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
