"""
HTML page parsing and content block extraction.

Parses HTML into structured content blocks with semantic
classification for downstream processing.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag, NavigableString

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class BlockType(str, Enum):
    """Types of content blocks."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    LINK = "link"
    IMAGE = "image"
    NAVIGATION = "navigation"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    FORM = "form"
    UNKNOWN = "unknown"


@dataclass
class ContentBlock:
    """
    A block of content extracted from a page.

    Represents a semantic unit like a heading, paragraph, or list.
    """

    block_type: BlockType
    text: str
    html: str = ""
    tag_name: str = ""
    level: int = 0  # For headings (1-6)
    links: list[dict] = field(default_factory=list)  # [{url, text}]
    attributes: dict = field(default_factory=dict)
    children: list["ContentBlock"] = field(default_factory=list)
    position: int = 0  # Order in document

    @property
    def word_count(self) -> int:
        """Count words in text."""
        return len(self.text.split())

    @property
    def is_substantive(self) -> bool:
        """Check if block has meaningful content."""
        return self.word_count >= 3 or self.block_type == BlockType.IMAGE


@dataclass
class ParsedPage:
    """
    Fully parsed page with extracted content blocks.

    Provides structured access to page content for extraction.
    """

    url: str
    title: str
    blocks: list[ContentBlock] = field(default_factory=list)
    headings: list[ContentBlock] = field(default_factory=list)
    paragraphs: list[ContentBlock] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    tables: list[ContentBlock] = field(default_factory=list)
    main_content_blocks: list[ContentBlock] = field(default_factory=list)
    language: str | None = None

    @property
    def full_text(self) -> str:
        """Get all text content joined."""
        return "\n\n".join(
            block.text for block in self.blocks if block.text.strip()
        )

    @property
    def main_text(self) -> str:
        """Get main content text only."""
        return "\n\n".join(
            block.text for block in self.main_content_blocks if block.text.strip()
        )

    @property
    def word_count(self) -> int:
        """Total word count."""
        return sum(block.word_count for block in self.blocks)

    @property
    def heading_outline(self) -> list[tuple[int, str]]:
        """Get document outline as (level, text) tuples."""
        return [(h.level, h.text) for h in self.headings]


class PageParser:
    """
    Parses HTML pages into structured content blocks.

    Extracts semantic structure from HTML, identifying headings,
    paragraphs, lists, tables, and other content types.

    Example:
        >>> parser = PageParser()
        >>> parsed = parser.parse(html, url="https://example.com")
        >>> for block in parsed.blocks:
        ...     print(f"{block.block_type}: {block.text[:50]}")
    """

    # Tags to completely remove
    REMOVE_TAGS = {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "video",
        "audio",
        "map",
        "object",
        "embed",
    }

    # Tags that indicate navigation/chrome
    NAV_TAGS = {"nav", "header", "footer", "aside"}

    # Tags that indicate main content
    CONTENT_TAGS = {"article", "main", "section"}

    # Block-level tags to process
    BLOCK_TAGS = {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "table",
        "blockquote",
        "pre",
        "code",
        "div",
        "figure",
        "figcaption",
    }

    def __init__(
        self,
        extract_links: bool = True,
        extract_images: bool = True,
        min_text_length: int = 10,
    ) -> None:
        """
        Initialize page parser.

        Args:
            extract_links: Whether to extract links
            extract_images: Whether to extract images
            min_text_length: Minimum text length for a block
        """
        self.extract_links = extract_links
        self.extract_images = extract_images
        self.min_text_length = min_text_length

    def parse(self, html: str, url: str = "") -> ParsedPage:
        """
        Parse HTML into structured content.

        Args:
            html: HTML content to parse
            url: Base URL for resolving relative links

        Returns:
            ParsedPage with extracted blocks
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags
        self._remove_tags(soup)

        # Extract basic info
        title = self._extract_title(soup)
        language = self._extract_language(soup)

        # Extract content blocks
        blocks = list(self._extract_blocks(soup, url))

        # Categorize blocks
        headings = [b for b in blocks if b.block_type == BlockType.HEADING]
        paragraphs = [b for b in blocks if b.block_type == BlockType.PARAGRAPH]
        tables = [b for b in blocks if b.block_type == BlockType.TABLE]

        # Extract links and images
        links = self._extract_all_links(
            soup, url) if self.extract_links else []
        images = self._extract_all_images(
            soup, url) if self.extract_images else []

        # Identify main content blocks
        main_blocks = self._identify_main_content(blocks, soup)

        return ParsedPage(
            url=url,
            title=title,
            blocks=blocks,
            headings=headings,
            paragraphs=paragraphs,
            links=links,
            images=images,
            tables=tables,
            main_content_blocks=main_blocks,
            language=language,
        )

    def _remove_tags(self, soup: BeautifulSoup) -> None:
        """Remove unwanted tags from soup."""
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Also remove hidden elements
        for tag in soup.find_all(attrs={"hidden": True}):
            tag.decompose()
        for tag in soup.find_all(attrs={"aria-hidden": "true"}):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile(r"\b(hidden|invisible|sr-only)\b")):
            tag.decompose()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try title tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return self._get_text(h1).strip()

        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        return ""

    def _extract_language(self, soup: BeautifulSoup) -> str | None:
        """Extract page language."""
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag["lang"]
        return None

    def _extract_blocks(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> Iterator[ContentBlock]:
        """Extract content blocks from soup."""
        position = 0

        # Find body or use whole soup
        body = soup.find("body") or soup

        for element in body.descendants:
            if not isinstance(element, Tag):
                continue

            tag_name = element.name.lower()

            # Skip if parent already processed
            if self._is_inside_processed_block(element):
                continue

            block = None

            # Headings
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                block = self._parse_heading(element, base_url)

            # Paragraphs
            elif tag_name == "p":
                block = self._parse_paragraph(element, base_url)

            # Lists
            elif tag_name in ("ul", "ol"):
                block = self._parse_list(element, base_url)

            # Tables
            elif tag_name == "table":
                block = self._parse_table(element, base_url)

            # Blockquotes
            elif tag_name == "blockquote":
                block = self._parse_quote(element, base_url)

            # Code blocks
            elif tag_name == "pre":
                block = self._parse_code(element)

            # Images (standalone)
            elif tag_name == "img":
                block = self._parse_image(element, base_url)

            # Divs with substantial text (fallback)
            elif tag_name == "div":
                text = self._get_text(element)
                if len(text) >= self.min_text_length and not element.find(self.BLOCK_TAGS):
                    block = ContentBlock(
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        html=str(element),
                        tag_name=tag_name,
                        position=position,
                    )

            if block and (block.text.strip() or block.block_type == BlockType.IMAGE):
                block.position = position
                position += 1
                yield block

    def _is_inside_processed_block(self, element: Tag) -> bool:
        """Check if element is inside an already processed block."""
        for parent in element.parents:
            if parent.name in ("ul", "ol", "table", "blockquote", "pre"):
                return True
        return False

    def _parse_heading(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse heading element."""
        level = int(element.name[1])  # h1 -> 1, h2 -> 2, etc.
        text = self._get_text(element)
        links = self._extract_links_from_element(element, base_url)

        return ContentBlock(
            block_type=BlockType.HEADING,
            text=text,
            html=str(element),
            tag_name=element.name,
            level=level,
            links=links,
        )

    def _parse_paragraph(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse paragraph element."""
        text = self._get_text(element)
        links = self._extract_links_from_element(element, base_url)

        return ContentBlock(
            block_type=BlockType.PARAGRAPH,
            text=text,
            html=str(element),
            tag_name="p",
            links=links,
        )

    def _parse_list(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse list element."""
        items = []
        for li in element.find_all("li", recursive=False):
            item_text = self._get_text(li)
            items.append(item_text)

        text = "\n".join(f"• {item}" for item in items)
        links = self._extract_links_from_element(element, base_url)

        children = [
            ContentBlock(
                block_type=BlockType.LIST_ITEM,
                text=item,
                tag_name="li",
            )
            for item in items
        ]

        return ContentBlock(
            block_type=BlockType.LIST,
            text=text,
            html=str(element),
            tag_name=element.name,
            links=links,
            children=children,
        )

    def _parse_table(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse table element."""
        rows = []

        for tr in element.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                cells.append(self._get_text(cell))
            if cells:
                rows.append(cells)

        # Format as text
        if rows:
            text = "\n".join(" | ".join(row) for row in rows)
        else:
            text = ""

        links = self._extract_links_from_element(element, base_url)

        return ContentBlock(
            block_type=BlockType.TABLE,
            text=text,
            html=str(element),
            tag_name="table",
            links=links,
            attributes={"rows": len(rows), "cols": len(
                rows[0]) if rows else 0},
        )

    def _parse_quote(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse blockquote element."""
        text = self._get_text(element)
        links = self._extract_links_from_element(element, base_url)

        # Check for citation
        cite = element.find("cite")
        citation = self._get_text(cite) if cite else None

        return ContentBlock(
            block_type=BlockType.QUOTE,
            text=text,
            html=str(element),
            tag_name="blockquote",
            links=links,
            attributes={"citation": citation} if citation else {},
        )

    def _parse_code(self, element: Tag) -> ContentBlock:
        """Parse code block element."""
        # Get raw text preserving whitespace
        text = element.get_text()

        # Try to detect language
        code_elem = element.find("code")
        language = None
        if code_elem:
            classes = code_elem.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    language = cls[9:]
                    break
                elif cls.startswith("lang-"):
                    language = cls[5:]
                    break

        return ContentBlock(
            block_type=BlockType.CODE,
            text=text,
            html=str(element),
            tag_name="pre",
            attributes={"language": language} if language else {},
        )

    def _parse_image(self, element: Tag, base_url: str) -> ContentBlock:
        """Parse image element."""
        src = element.get("src", "")
        if src and base_url:
            src = urljoin(base_url, src)

        alt = element.get("alt", "")
        title = element.get("title", "")

        return ContentBlock(
            block_type=BlockType.IMAGE,
            text=alt or title or "",
            html=str(element),
            tag_name="img",
            attributes={
                "src": src,
                "alt": alt,
                "title": title,
            },
        )

    def _get_text(self, element: Tag) -> str:
        """Get clean text from element."""
        text = element.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_links_from_element(
        self,
        element: Tag,
        base_url: str,
    ) -> list[dict]:
        """Extract links from an element."""
        links = []

        for a in element.find_all("a", href=True):
            href = a["href"]

            # Skip non-http links
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            # Resolve relative URLs
            if base_url:
                href = urljoin(base_url, href)

            text = self._get_text(a)

            links.append({
                "url": href,
                "text": text,
                "title": a.get("title", ""),
            })

        return links

    def _extract_all_links(self, soup: BeautifulSoup, base_url: str) -> list[dict]:
        """Extract all links from page."""
        links = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if href.startswith(("javascript:", "mailto:", "tel:")):
                continue

            if base_url:
                href = urljoin(base_url, href)

            # Remove fragment
            if "#" in href:
                href = href.split("#")[0]

            if not href or href in seen:
                continue

            seen.add(href)
            text = self._get_text(a)

            links.append({
                "url": href,
                "text": text,
                "title": a.get("title", ""),
                "rel": a.get("rel", []),
            })

        return links

    def _extract_all_images(self, soup: BeautifulSoup, base_url: str) -> list[dict]:
        """Extract all images from page."""
        images = []
        seen = set()

        for img in soup.find_all("img"):
            src = img.get("src", "")

            if not src:
                continue

            if base_url:
                src = urljoin(base_url, src)

            if src in seen:
                continue

            seen.add(src)

            images.append({
                "src": src,
                "alt": img.get("alt", ""),
                "title": img.get("title", ""),
                "width": img.get("width"),
                "height": img.get("height"),
            })

        return images

    def _identify_main_content(
        self,
        blocks: list[ContentBlock],
        soup: BeautifulSoup,
    ) -> list[ContentBlock]:
        """
        Identify blocks that are part of main content.

        Uses heuristics to filter out navigation, sidebars, footers.
        """
        # Find main content container
        main_container = (
            soup.find("main")
            or soup.find("article")
            or soup.find(role="main")
            or soup.find(id=re.compile(r"(main|content|article)", re.I))
            or soup.find(class_=re.compile(r"(main|content|article)", re.I))
        )

        if main_container:
            # Get text from main container
            main_text = self._get_text(main_container)

            # Filter blocks to those whose text appears in main content
            main_blocks = []
            for block in blocks:
                if block.text and block.text in main_text:
                    main_blocks.append(block)
            return main_blocks

        # Fallback: use heuristics
        # Exclude navigation, footer, sidebar blocks
        main_blocks = []
        for block in blocks:
            # Skip very short blocks
            if block.word_count < 5 and block.block_type != BlockType.HEADING:
                continue

            # Skip blocks that look like navigation
            if block.block_type == BlockType.LIST:
                # Navigation often has many short list items with links
                if len(block.links) > 5 and block.word_count < 50:
                    continue

            main_blocks.append(block)

        return main_blocks
