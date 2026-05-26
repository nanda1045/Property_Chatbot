#!/usr/bin/env python3
"""Scrape configured public property pages into property-scoped JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

DEFAULT_SOURCE_CONFIG = Path("config/property_sources.json")
DEFAULT_OUTPUT_DIR = Path("Data/unstructured")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


@dataclass(frozen=True)
class PageSection:
    heading: str
    heading_level: int
    content: str


@dataclass(frozen=True)
class ScrapedPage:
    property_code: str
    property_name: str
    address: str | None
    source_url: str
    page_type: str
    title: str | None
    description: str | None
    content: str
    sections: list[PageSection]
    links: list[str]
    content_hash: str
    scraped_at: str


class SectionAwareHTMLParser(HTMLParser):
    """Extract visible text while preserving heading-based website sections."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "div",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.description: str | None = None
        self.links: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.heading_tag: str | None = None
        self.heading_parts: list[str] = []
        self.section_heading = "Page Overview"
        self.section_level = 0
        self.section_parts: list[str] = []
        self.completed_sections: list[PageSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs}

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if tag == "title":
            self.in_title = True
            return

        if tag == "meta":
            name = (attrs_dict.get("name") or attrs_dict.get("property") or "").lower()
            if name in {"description", "og:description"} and attrs_dict.get("content"):
                self.description = attrs_dict["content"]
            return

        if tag == "a":
            href = attrs_dict.get("href")
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                self.links.append(href)
            self._append_newline()
            return

        if tag in self.HEADING_TAGS:
            self._finish_section()
            self.heading_tag = tag
            self.heading_parts = []
            return

        if tag == "img":
            alt = attrs_dict.get("alt")
            if alt and not is_boilerplate_line(alt):
                self._append_text(f"Image: {alt}")
            return

        if tag in self.BLOCK_TAGS:
            self._append_newline()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "title":
            self.in_title = False
            return
        if tag == self.heading_tag:
            heading = normalize_spaces(" ".join(self.heading_parts))
            if heading and not is_boilerplate_line(heading):
                self.section_heading = heading
                self.section_level = int(tag[1])
            self.heading_tag = None
            self.heading_parts = []
            self._append_newline()
            return
        if tag in self.BLOCK_TAGS:
            self._append_newline()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        cleaned = normalize_spaces(data)
        if not cleaned:
            return
        if self.in_title:
            self.title_parts.append(cleaned)
        elif self.heading_tag:
            self.heading_parts.append(cleaned)
        else:
            self._append_text(cleaned)

    def close(self) -> None:
        super().close()
        self._finish_section()

    def _append_text(self, text: str) -> None:
        self.parts.append(text)
        self.section_parts.append(text)

    def _append_newline(self) -> None:
        self.parts.append("\n")
        self.section_parts.append("\n")

    def _finish_section(self) -> None:
        content = parts_to_text(self.section_parts)
        if len(content) >= 80:
            self.completed_sections.append(
                PageSection(
                    heading=self.section_heading,
                    heading_level=self.section_level,
                    content=content,
                )
            )
        self.section_parts = []

    @property
    def title(self) -> str | None:
        title = normalize_spaces(" ".join(self.title_parts))
        return title or None

    @property
    def text(self) -> str:
        return parts_to_text(self.parts)

    @property
    def sections(self) -> list[PageSection]:
        return self.completed_sections


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


BOILERPLATE_EXACT = {
    "×",
    "available",
    "community fee guide",
    "contact",
    "contact us",
    "e-brochure",
    "email us",
    "fee guide",
    "floor plans",
    "floorplans",
    "home",
    "amenities",
    "neighborhood",
    "gallery",
    "residents",
    "faqs",
    "building history",
    "privacy policy",
    "site map",
    "web accessibility",
    "all rights reserved.",
    "i agree",
    "how we use your data",
    "show more",
    "find your home",
    "book a tour",
    "apply now",
    "virtual tour",
    "self-guided tour",
    "check availability",
    "call us at",
    "map",
    "our address",
    "page overview",
    "pet policy",
    "loading icon",
    "welcome home",
}
BOILERPLATE_PREFIXES = (
    "skip to main content",
    "this site uses cookies",
    "by using our site",
    "© copyright",
    "© 2026",
    "all pets being kept within household",
    "prohibited dog breeds include",
    "image: ",
)


def is_boilerplate_line(value: str) -> bool:
    text = normalize_spaces(value)
    lowered = text.lower()
    if not lowered:
        return True
    if lowered in BOILERPLATE_EXACT:
        return True
    if any(lowered.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
        if lowered.startswith("image: ") and not should_keep_image_alt(text):
            return True
        return not lowered.startswith("image: ")
    if any(
        phrase in lowered
        for phrase in [
            "additional fees",
            "all dimensions are approximate",
            "artist's rendering",
            "artist’s rendering",
            "base rent",
            "contact a representative",
            "fee list",
            "floor plans are artist",
            "floorplans are artist",
            "monthly leasing prices",
            "move-in",
            "move-out",
            "not all features are available",
            "optional services",
            "prices and availability",
            "required monthly fees",
            "subject to change",
            "variable or usage-based",
        ]
    ):
        return True
    if re.fullmatch(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", lowered):
        return True
    if "logo" in lowered and lowered.startswith("image:"):
        return True
    return False


def should_keep_image_alt(value: str) -> bool:
    lowered = value.lower()
    return not any(term in lowered for term in ["logo", "loading icon"])


def parts_to_text(parts: list[str]) -> str:
    lines = []
    previous = ""
    for line in re.split(r"\n+", "\n".join(parts)):
        line = normalize_spaces(line)
        if not line or is_boilerplate_line(line):
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)


def read_sources(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as source_file:
        return json.load(source_file)


def selected_sources(sources: dict[str, dict], codes: list[str] | None) -> dict[str, dict]:
    if not codes:
        return {
            code: config
            for code, config in sources.items()
            if config.get("primary_site") and config.get("seed_paths")
        }

    selected = {}
    missing = []
    for code in codes:
        normalized = code.lower()
        config = sources.get(normalized)
        if not config:
            missing.append(code)
            continue
        if not config.get("primary_site"):
            missing.append(code)
            continue
        selected[normalized] = config

    if missing:
        print(f"Skipping unknown or source-less property codes: {', '.join(missing)}")

    return selected


def build_seed_urls(config: dict, max_pages: int | None) -> list[tuple[str, str]]:
    base_url = config["primary_site"]
    urls = []
    for seed_path in config.get("seed_paths", []):
        source_url = urljoin(base_url, seed_path)
        page_type = classify_page_type(seed_path)
        urls.append((source_url, page_type))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for source_url, page_type in urls:
        canonical = source_url.rstrip("/") + "/"
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append((source_url, page_type))

    if max_pages is not None:
        return deduped[:max_pages]
    return deduped


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if not path.endswith("/"):
        path += "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", "", ""))


def normalized_host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def is_floorplan_listing_url(url: str) -> bool:
    return urlparse(url).path.rstrip("/").endswith("/floorplans")


def floorplan_detail_links(
    page: ScrapedPage,
    base_url: str,
    max_links: int,
) -> list[str]:
    if not is_floorplan_listing_url(page.source_url):
        return []

    base_host = normalized_host(base_url)
    listing_canonical = canonical_url(page.source_url)
    discovered: list[str] = []
    seen = {listing_canonical}

    for href in page.links:
        absolute = canonical_url(urljoin(page.source_url, href))
        parsed = urlparse(absolute)
        if normalized_host(absolute) != base_host:
            continue
        if "/floorplans/" not in parsed.path:
            continue
        if "ebrochure" in parsed.path.lower():
            continue
        if absolute in seen:
            continue
        if re.search(r"\.(?:pdf|jpg|jpeg|png|webp|gif|svg)$", parsed.path, re.IGNORECASE):
            continue
        seen.add(absolute)
        discovered.append(absolute)
        if len(discovered) >= max_links:
            break

    return discovered


def classify_page_type(seed_path: str) -> str:
    cleaned = seed_path.strip("/")
    if not cleaned:
        return "home"
    return cleaned.split("/", 1)[0]


def fetch_html(url: str, timeout_seconds: int, allow_insecure_ssl: bool) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": USER_AGENT,
        },
    )
    ssl_context = ssl._create_unverified_context() if allow_insecure_ssl else None
    with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
        raw = response.read()
        final_url = response.geturl()
        content_type = response.headers.get_content_charset() or "utf-8"
    return raw.decode(content_type, errors="replace"), final_url


def extract_page(
    property_code: str,
    config: dict,
    source_url: str,
    page_type: str,
    timeout_seconds: int,
    allow_insecure_ssl: bool,
) -> ScrapedPage | None:
    try:
        html, final_url = fetch_html(source_url, timeout_seconds, allow_insecure_ssl)
    except HTTPError as exc:
        print(f"[{property_code}] HTTP {exc.code}: {source_url}")
        return None
    except URLError as exc:
        print(f"[{property_code}] URL error: {source_url} ({exc.reason})")
        return None
    except TimeoutError:
        print(f"[{property_code}] Timeout: {source_url}")
        return None

    parser = SectionAwareHTMLParser()
    parser.feed(html)
    parser.close()
    content = parser.text
    if len(content) < 120:
        if page_type == "floorplans" and parser.links:
            content = "\n".join(
                value for value in [parser.title, parser.description, content] if value
            )
        else:
            print(f"[{property_code}] Too little text after extraction: {source_url}")
            return None

    scraped_at = datetime.now(UTC).isoformat()
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return ScrapedPage(
        property_code=property_code,
        property_name=config["property_name"],
        address=config.get("address"),
        source_url=final_url,
        page_type=page_type,
        title=parser.title,
        description=parser.description,
        content=content,
        sections=parser.sections,
        links=parser.links,
        content_hash=content_hash,
        scraped_at=scraped_at,
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def page_to_dict(page: ScrapedPage) -> dict:
    return {
        "property_code": page.property_code,
        "property_name": page.property_name,
        "address": page.address,
        "source_url": page.source_url,
        "page_type": page.page_type,
        "title": page.title,
        "description": page.description,
        "content": page.content,
        "sections": [
            {
                "heading": section.heading,
                "heading_level": section.heading_level,
                "content": section.content,
            }
            for section in page.sections
        ],
        "links": page.links,
        "content_hash": page.content_hash,
        "scraped_at": page.scraped_at,
    }


def chunk_text(content: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in content.split("\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for index in range(1, len(chunks)):
        previous_tail = chunks[index - 1][-overlap:].strip()
        overlapped.append(f"{previous_tail}\n{chunks[index]}".strip())
    return overlapped


def chunks_for_page(page: ScrapedPage, chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    section_chunks = []
    for section_index, section in enumerate(page.sections, start=1):
        section_content = f"{section.heading}\n{section.content}".strip()
        for split_index, chunk in enumerate(
            chunk_text(section_content, chunk_size, overlap),
            start=1,
        ):
            section_chunks.append((section_index, split_index, section, chunk))

    if not section_chunks:
        fallback_section = PageSection("Page Overview", 0, page.content)
        section_chunks = [
            (1, index, fallback_section, chunk)
            for index, chunk in enumerate(chunk_text(page.content, chunk_size, overlap), start=1)
        ]

    for index, (section_index, split_index, section, chunk) in enumerate(section_chunks, start=1):
        chunk_hash = hashlib.sha256(
            (
                f"{page.property_code}:{page.source_url}:{section_index}:"
                f"{split_index}:{chunk}"
            ).encode()
        ).hexdigest()
        chunks.append(
            {
                "id": chunk_hash[:24],
                "property_code": page.property_code,
                "property_name": page.property_name,
                "address": page.address,
                "source_url": page.source_url,
                "page_type": page.page_type,
                "section_heading": section.heading,
                "section_index": section_index,
                "section_split_index": split_index,
                "chunk_index": index,
                "chunk_strategy": "html_section_v1",
                "title": page.title,
                "content": chunk,
                "scraped_at": page.scraped_at,
            }
        )
    return chunks


def scrape(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = read_sources(Path(args.sources))
    chosen_sources = selected_sources(sources, args.codes)
    pages: list[ScrapedPage] = []

    for property_code, config in chosen_sources.items():
        seed_urls = build_seed_urls(config, args.max_pages_per_property)
        pending_urls = list(seed_urls)
        seen_urls: set[str] = set()
        floorplan_detail_count = 0
        print(f"[{property_code}] Scraping from {config['primary_site']}")
        while pending_urls:
            source_url, page_type = pending_urls.pop(0)
            canonical = canonical_url(source_url)
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            page = extract_page(
                property_code=property_code,
                config=config,
                source_url=source_url,
                page_type=page_type,
                timeout_seconds=args.timeout_seconds,
                allow_insecure_ssl=args.allow_insecure_ssl,
            )
            if page:
                pages.append(page)
                print(f"[{property_code}] OK {page.page_type}: {page.source_url}")
                if args.crawl_floorplan_details and page_type == "floorplans":
                    remaining = args.max_floorplan_detail_pages - floorplan_detail_count
                    if remaining > 0:
                        detail_urls = floorplan_detail_links(
                            page=page,
                            base_url=config["primary_site"],
                            max_links=remaining,
                        )
                        for detail_url in detail_urls:
                            if canonical_url(detail_url) in seen_urls:
                                continue
                            pending_urls.append((detail_url, "floorplans"))
                            floorplan_detail_count += 1
            if args.delay_seconds:
                time.sleep(args.delay_seconds)

    raw_path = output_dir / "property_pages.jsonl"
    chunk_path = output_dir / "property_chunks.jsonl"
    manifest_path = output_dir / "scrape_manifest.json"

    raw_count = write_jsonl(raw_path, (page_to_dict(page) for page in pages))
    chunk_rows = [
        chunk
        for page in pages
        for chunk in chunks_for_page(page, args.chunk_size, args.chunk_overlap)
    ]
    chunk_count = write_jsonl(chunk_path, chunk_rows)

    manifest = {
        "scraped_at": datetime.now(UTC).isoformat(),
        "source_config": str(args.sources),
        "property_codes": sorted({page.property_code for page in pages}),
        "page_count": raw_count,
        "chunk_count": chunk_count,
        "raw_pages_path": str(raw_path),
        "chunks_path": str(chunk_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {raw_count} page(s) to {raw_path}")
    print(f"Wrote {chunk_count} chunk(s) to {chunk_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0 if raw_count else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=str(DEFAULT_SOURCE_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--codes",
        nargs="*",
        help="Property codes to scrape. Defaults to all with sources.",
    )
    parser.add_argument("--max-pages-per-property", type=int, default=4)
    parser.add_argument(
        "--crawl-floorplan-details",
        action="store_true",
        help="Follow same-domain links below /floorplans/ to capture individual plan pages.",
    )
    parser.add_argument("--max-floorplan-detail-pages", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help=(
            "Disable TLS certificate verification for local scraping environments "
            "with missing CA roots."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(scrape(parse_args()))
