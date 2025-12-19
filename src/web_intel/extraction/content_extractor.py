"""
Main content extraction and detection.

Identifies and extracts the primary content from web pages,
filtering out navigation, ads, and other non-content elements.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from bs4 import BeautifulSoup, Tag

from web_intel.extraction.page_parser import PageParser, ParsedPage, ContentBlock, BlockType
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedContent:
    """
    Extracted main content from a page.

    Contains cleaned, structured content ready for LLM processing.
    """

    url: str
    title: str
    main_text: str
    summary_text: str  # First ~500 chars for quick preview
    headings: list[tuple[int, str]]  # (level, text)
    paragraphs: list[str]
    links: list[dict]
    word_count: int
    language: str | None = None
    content_type: str = "article"  # article, listing, form, etc.
    quality_score: float = 0.0  # 0-1 content quality estimate

    @property
    def is_substantive(self) -> bool:
        """Check if content is substantive enough for processing."""
        return self.word_count >= 50 and len(self.paragraphs) >= 1


class MainContentDetector:
    """
    Detects the main content area of a web page.

    Uses multiple heuristics including:
    - Semantic HTML5 tags (article, main)
    - Common CSS class/ID patterns
    - Text density analysis
    - Link density analysis

    Example:
        >>> detector = MainContentDetector()
        >>> main_element = detector.find_main_content(soup)
        >>> if main_element:
        ...     content = main_element.get_text()
    """

    # Patterns indicating main content
    MAIN_PATTERNS = [
        re.compile(r"\b(main|content|article|post|entry|body)\b", re.I),
    ]

    # Patterns indicating non-content
    NON_CONTENT_PATTERNS = [
        re.compile(
            r"\b(nav|navigation|menu|sidebar|footer|header|comment|ad|advertisement|social|share|related|popular|trending)\b", re.I),
    ]

    # Minimum text length for a valid content block
    MIN_CONTENT_LENGTH = 100

    def __init__(self) -> None:
        """Initialize main content detector."""
        pass

    def find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        """
        Find the main content element in the page.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Tag containing main content, or None
        """
        # Try semantic HTML5 elements first
        main = soup.find("main")
        if main and self._has_substantial_text(main):
            return main

        article = soup.find("article")
        if article and self._has_substantial_text(article):
            return article

        # Try role="main"
        role_main = soup.find(attrs={"role": "main"})
        if role_main and self._has_substantial_text(role_main):
            return role_main

        # Try common ID/class patterns
        for pattern in self.MAIN_PATTERNS:
            # Check IDs
            element = soup.find(id=pattern)
            if element and self._has_substantial_text(element):
                return element

            # Check classes
            element = soup.find(class_=pattern)
            if element and self._has_substantial_text(element):
                return element

        # Fallback: find element with highest text density
        return self._find_by_text_density(soup)

    def _has_substantial_text(self, element: Tag) -> bool:
        """Check if element has enough text content."""
        text = element.get_text(strip=True)
        return len(text) >= self.MIN_CONTENT_LENGTH

    def _find_by_text_density(self, soup: BeautifulSoup) -> Tag | None:
        """Find content element using text density heuristic."""
        body = soup.find("body")
        if not body:
            return None

        best_element = None
        best_score = 0

        # Check divs and sections
        for element in body.find_all(["div", "section"]):
            score = self._calculate_content_score(element)

            if score > best_score:
                best_score = score
                best_element = element

        return best_element if best_score > 0.3 else None

    def _calculate_content_score(self, element: Tag) -> float:
        """
        Calculate content score for an element.

        Higher scores indicate more likely to be main content.
        """
        # Skip if matches non-content patterns
        element_id = element.get("id", "")
        element_class = " ".join(element.get("class", []))

        for pattern in self.NON_CONTENT_PATTERNS:
            if pattern.search(element_id) or pattern.search(element_class):
                return 0.0

        # Get text and link counts
        text = element.get_text(strip=True)
        text_length = len(text)

        if text_length < self.MIN_CONTENT_LENGTH:
            return 0.0

        # Count links
        links = element.find_all("a")
        link_text_length = sum(len(a.get_text(strip=True)) for a in links)

        # Calculate link density (lower is better for content)
        link_density = link_text_length / text_length if text_length > 0 else 1.0

        # Count paragraphs (more is better)
        paragraphs = element.find_all("p")
        paragraph_count = len(paragraphs)

        # Calculate score
        # Prefer: longer text, lower link density, more paragraphs
        score = 0.0

        # Text length component (up to 0.4)
        score += min(0.4, text_length / 5000)

        # Link density component (up to 0.3, lower is better)
        score += 0.3 * (1 - link_density)

        # Paragraph component (up to 0.3)
        score += min(0.3, paragraph_count * 0.03)

        return score


class ContentExtractor:
    """
    Extracts and cleans main content from web pages.

    Combines parsing, main content detection, and cleaning
    into a single extraction pipeline.

    Example:
        >>> extractor = ContentExtractor()
        >>> content = extractor.extract(html, url)
        >>> print(f"Title: {content.title}")
        >>> print(f"Words: {content.word_count}")
    """

    def __init__(
        self,
        min_word_count: int = 20,
        max_summary_length: int = 500,
    ) -> None:
        """
        Initialize content extractor.

        Args:
            min_word_count: Minimum words for valid content
            max_summary_length: Maximum summary text length
        """
        self.min_word_count = min_word_count
        self.max_summary_length = max_summary_length
        self.parser = PageParser()
        self.detector = MainContentDetector()

    def extract(self, html: str, url: str = "") -> ExtractedContent:
        """
        Extract main content from HTML.

        Args:
            html: HTML content
            url: Page URL

        Returns:
            ExtractedContent with cleaned main content
        """
        # Parse the page
        parsed = self.parser.parse(html, url)

        # Get main text
        main_text = parsed.main_text or parsed.full_text

        # Clean the text
        main_text = self._clean_text(main_text)

        # Generate summary
        summary_text = self._generate_summary(main_text)

        # Extract paragraph texts
        paragraphs = [
            block.text for block in parsed.main_content_blocks
            if block.block_type == BlockType.PARAGRAPH and block.text.strip()
        ]

        # Determine content type
        content_type = self._detect_content_type(parsed)

        # Calculate quality score
        quality_score = self._calculate_quality_score(parsed, main_text)

        # Filter links to main content only
        main_links = self._filter_main_links(parsed)

        return ExtractedContent(
            url=url,
            title=parsed.title,
            main_text=main_text,
            summary_text=summary_text,
            headings=parsed.heading_outline,
            paragraphs=paragraphs,
            links=main_links,
            word_count=len(main_text.split()),
            language=parsed.language,
            content_type=content_type,
            quality_score=quality_score,
        )

    def extract_from_parsed(self, parsed: ParsedPage) -> ExtractedContent:
        """
        Extract content from already parsed page.

        Args:
            parsed: ParsedPage from PageParser

        Returns:
            ExtractedContent
        """
        main_text = parsed.main_text or parsed.full_text
        main_text = self._clean_text(main_text)
        summary_text = self._generate_summary(main_text)

        paragraphs = [
            block.text for block in parsed.main_content_blocks
            if block.block_type == BlockType.PARAGRAPH and block.text.strip()
        ]

        content_type = self._detect_content_type(parsed)
        quality_score = self._calculate_quality_score(parsed, main_text)
        main_links = self._filter_main_links(parsed)

        return ExtractedContent(
            url=parsed.url,
            title=parsed.title,
            main_text=main_text,
            summary_text=summary_text,
            headings=parsed.heading_outline,
            paragraphs=paragraphs,
            links=main_links,
            word_count=len(main_text.split()),
            language=parsed.language,
            content_type=content_type,
            quality_score=quality_score,
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip
        text = text.strip()

        return text

    def _generate_summary(self, text: str) -> str:
        """Generate a summary preview of the text."""
        if len(text) <= self.max_summary_length:
            return text

        # Find a good break point
        summary = text[: self.max_summary_length]

        # Try to break at sentence
        last_period = summary.rfind(".")
        if last_period > self.max_summary_length * 0.6:
            summary = summary[: last_period + 1]
        else:
            # Break at word
            last_space = summary.rfind(" ")
            if last_space > 0:
                summary = summary[:last_space] + "..."

        return summary

    def _detect_content_type(self, parsed: ParsedPage) -> str:
        """Detect the type of content on the page."""
        # Check for listing pages
        if len(parsed.tables) > 2:
            return "listing"

        # Check for forms
        # (Would need to track forms in parser)

        # Check ratio of headings to paragraphs
        if len(parsed.headings) > len(parsed.paragraphs):
            return "index"

        # Default to article
        return "article"

    def _calculate_quality_score(self, parsed: ParsedPage, main_text: str) -> float:
        """Calculate content quality score (0-1)."""
        score = 0.0

        # Word count component (up to 0.3)
        word_count = len(main_text.split())
        score += min(0.3, word_count / 1000)

        # Structure component (up to 0.3)
        # Presence of headings, paragraphs, lists
        has_headings = len(parsed.headings) > 0
        has_paragraphs = len(parsed.paragraphs) > 2
        has_structure = len(parsed.main_content_blocks) > 3

        if has_headings:
            score += 0.1
        if has_paragraphs:
            score += 0.1
        if has_structure:
            score += 0.1

        # Title presence (0.1)
        if parsed.title:
            score += 0.1

        # Language detected (0.1)
        if parsed.language:
            score += 0.1

        # Low link density in main content (0.2)
        total_words = sum(
            block.word_count for block in parsed.main_content_blocks
        )
        total_link_words = sum(
            len(link.get("text", "").split())
            for block in parsed.main_content_blocks
            for link in block.links
        )

        if total_words > 0:
            link_density = total_link_words / total_words
            score += 0.2 * (1 - min(1.0, link_density * 2))

        return min(1.0, score)

    def _filter_main_links(self, parsed: ParsedPage) -> list[dict]:
        """Filter links to only those in main content."""
        main_link_urls = set()

        for block in parsed.main_content_blocks:
            for link in block.links:
                main_link_urls.add(link["url"])

        return [
            link for link in parsed.links
            if link["url"] in main_link_urls
        ]
