"""
Metadata extraction from web pages.

Extracts structured metadata including:
- Standard HTML meta tags
- Open Graph protocol
- Twitter Cards
- JSON-LD structured data
- Schema.org markup
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StructuredData:
    """
    JSON-LD or microdata structured data from a page.

    Contains schema.org or other structured data formats.
    """

    data_type: str  # e.g., "Article", "Product", "Organization"
    data: dict
    source: str = "json-ld"  # json-ld, microdata, rdfa

    @property
    def schema_type(self) -> str | None:
        """Get schema.org @type if present."""
        return self.data.get("@type")


@dataclass
class PageMetadata:
    """
    Comprehensive metadata extracted from a page.

    Combines multiple metadata sources into a unified structure.
    """

    # Basic metadata
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    author: str = ""
    language: str = ""
    canonical_url: str = ""

    # Dates
    published_date: datetime | None = None
    modified_date: datetime | None = None

    # Open Graph
    og_title: str = ""
    og_description: str = ""
    og_image: str = ""
    og_type: str = ""
    og_site_name: str = ""

    # Twitter
    twitter_card: str = ""
    twitter_title: str = ""
    twitter_description: str = ""
    twitter_image: str = ""

    # Structured data
    structured_data: list[StructuredData] = field(default_factory=list)

    # Technical
    charset: str = ""
    viewport: str = ""
    robots: str = ""

    # Custom metadata
    custom: dict = field(default_factory=dict)

    @property
    def best_title(self) -> str:
        """Get best available title."""
        return self.og_title or self.twitter_title or self.title

    @property
    def best_description(self) -> str:
        """Get best available description."""
        return self.og_description or self.twitter_description or self.description

    @property
    def best_image(self) -> str:
        """Get best available image."""
        return self.og_image or self.twitter_image

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "author": self.author,
            "language": self.language,
            "canonical_url": self.canonical_url,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "og": {
                "title": self.og_title,
                "description": self.og_description,
                "image": self.og_image,
                "type": self.og_type,
                "site_name": self.og_site_name,
            },
            "twitter": {
                "card": self.twitter_card,
                "title": self.twitter_title,
                "description": self.twitter_description,
                "image": self.twitter_image,
            },
            "structured_data_count": len(self.structured_data),
        }


class MetadataExtractor:
    """
    Extracts metadata from HTML pages.

    Parses multiple metadata formats and combines them
    into a unified PageMetadata structure.

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(html)
        >>> print(f"Title: {metadata.best_title}")
        >>> print(f"Description: {metadata.best_description}")
    """

    # Date formats to try when parsing
    DATE_FORMATS = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%d %B %Y",
        "%m/%d/%Y",
    ]

    def __init__(self) -> None:
        """Initialize metadata extractor."""
        pass

    def extract(self, html: str) -> PageMetadata:
        """
        Extract all metadata from HTML.

        Args:
            html: HTML content

        Returns:
            PageMetadata with extracted data
        """
        soup = BeautifulSoup(html, "html.parser")
        metadata = PageMetadata()

        # Extract basic meta tags
        self._extract_basic_meta(soup, metadata)

        # Extract Open Graph
        self._extract_open_graph(soup, metadata)

        # Extract Twitter Cards
        self._extract_twitter_cards(soup, metadata)

        # Extract structured data
        self._extract_structured_data(soup, metadata)

        # Extract dates
        self._extract_dates(soup, metadata)

        # Extract language
        self._extract_language(soup, metadata)

        return metadata

    def _extract_basic_meta(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract basic HTML meta tags."""
        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata.title = title_tag.get_text(strip=True)

        # Description
        desc = soup.find("meta", attrs={"name": "description"})
        if desc:
            metadata.description = desc.get("content", "")

        # Keywords
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords:
            content = keywords.get("content", "")
            metadata.keywords = [k.strip()
                                 for k in content.split(",") if k.strip()]

        # Author
        author = soup.find("meta", attrs={"name": "author"})
        if author:
            metadata.author = author.get("content", "")

        # Canonical URL
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if canonical:
            metadata.canonical_url = canonical.get("href", "")

        # Charset
        charset = soup.find("meta", attrs={"charset": True})
        if charset:
            metadata.charset = charset.get("charset", "")
        else:
            charset = soup.find("meta", attrs={"http-equiv": "Content-Type"})
            if charset:
                content = charset.get("content", "")
                if "charset=" in content:
                    metadata.charset = content.split("charset=")[-1].strip()

        # Viewport
        viewport = soup.find("meta", attrs={"name": "viewport"})
        if viewport:
            metadata.viewport = viewport.get("content", "")

        # Robots
        robots = soup.find("meta", attrs={"name": "robots"})
        if robots:
            metadata.robots = robots.get("content", "")

    def _extract_open_graph(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract Open Graph metadata."""
        og_tags = soup.find_all("meta", property=re.compile(r"^og:"))

        for tag in og_tags:
            prop = tag.get("property", "")
            content = tag.get("content", "")

            if prop == "og:title":
                metadata.og_title = content
            elif prop == "og:description":
                metadata.og_description = content
            elif prop == "og:image":
                metadata.og_image = content
            elif prop == "og:type":
                metadata.og_type = content
            elif prop == "og:site_name":
                metadata.og_site_name = content

    def _extract_twitter_cards(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract Twitter Card metadata."""
        twitter_tags = soup.find_all(
            "meta", attrs={"name": re.compile(r"^twitter:")})

        for tag in twitter_tags:
            name = tag.get("name", "")
            content = tag.get("content", "")

            if name == "twitter:card":
                metadata.twitter_card = content
            elif name == "twitter:title":
                metadata.twitter_title = content
            elif name == "twitter:description":
                metadata.twitter_description = content
            elif name == "twitter:image":
                metadata.twitter_image = content

    def _extract_structured_data(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract JSON-LD structured data."""
        scripts = soup.find_all("script", type="application/ld+json")

        for script in scripts:
            try:
                text = script.get_text(strip=True)
                if not text:
                    continue

                data = json.loads(text)

                # Handle arrays
                if isinstance(data, list):
                    for item in data:
                        self._add_structured_data(item, metadata)
                else:
                    self._add_structured_data(data, metadata)

            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON-LD: {e}")
            except Exception as e:
                logger.debug(f"Error extracting structured data: {e}")

    def _add_structured_data(self, data: dict, metadata: PageMetadata) -> None:
        """Add a structured data item to metadata."""
        if not isinstance(data, dict):
            return

        data_type = data.get("@type", "Unknown")

        # Handle array of types
        if isinstance(data_type, list):
            data_type = data_type[0] if data_type else "Unknown"

        metadata.structured_data.append(
            StructuredData(
                data_type=data_type,
                data=data,
                source="json-ld",
            )
        )

    def _extract_dates(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract publication and modification dates."""
        # Try article:published_time
        published = soup.find("meta", property="article:published_time")
        if published:
            metadata.published_date = self._parse_date(
                published.get("content", ""))

        # Try article:modified_time
        modified = soup.find("meta", property="article:modified_time")
        if modified:
            metadata.modified_date = self._parse_date(
                modified.get("content", ""))

        # Try schema.org dates from structured data
        for sd in metadata.structured_data:
            if not metadata.published_date and "datePublished" in sd.data:
                metadata.published_date = self._parse_date(
                    sd.data["datePublished"])

            if not metadata.modified_date and "dateModified" in sd.data:
                metadata.modified_date = self._parse_date(
                    sd.data["dateModified"])

        # Try time elements
        if not metadata.published_date:
            time_elem = soup.find("time", attrs={"datetime": True})
            if time_elem:
                metadata.published_date = self._parse_date(
                    time_elem["datetime"])

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse a date string into datetime."""
        if not date_str:
            return None

        # Clean the string
        date_str = date_str.strip()

        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try ISO format as fallback
        try:
            # Handle timezone offset format
            date_str = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", date_str)
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        logger.debug(f"Could not parse date: {date_str}")
        return None

    def _extract_language(self, soup: BeautifulSoup, metadata: PageMetadata) -> None:
        """Extract page language."""
        # Try html lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            metadata.language = html_tag["lang"]
            return

        # Try Content-Language meta
        lang_meta = soup.find("meta", attrs={"http-equiv": "Content-Language"})
        if lang_meta:
            metadata.language = lang_meta.get("content", "")
            return

        # Try og:locale
        og_locale = soup.find("meta", property="og:locale")
        if og_locale:
            # Convert og:locale format (e.g., en_US) to language code
            locale = og_locale.get("content", "")
            if locale:
                metadata.language = locale.split("_")[0]

    def extract_article_info(self, metadata: PageMetadata) -> dict:
        """
        Extract article-specific information from metadata.

        Args:
            metadata: PageMetadata to extract from

        Returns:
            Dictionary with article info
        """
        info = {
            "title": metadata.best_title,
            "description": metadata.best_description,
            "author": metadata.author,
            "published_date": metadata.published_date,
            "modified_date": metadata.modified_date,
            "image": metadata.best_image,
        }

        # Try to get more from structured data
        for sd in metadata.structured_data:
            if sd.data_type in ("Article", "NewsArticle", "BlogPosting"):
                data = sd.data

                if not info["author"] and "author" in data:
                    author = data["author"]
                    if isinstance(author, dict):
                        info["author"] = author.get("name", "")
                    elif isinstance(author, str):
                        info["author"] = author

                if "headline" in data and not info["title"]:
                    info["title"] = data["headline"]

                if "image" in data and not info["image"]:
                    img = data["image"]
                    if isinstance(img, str):
                        info["image"] = img
                    elif isinstance(img, dict):
                        info["image"] = img.get("url", "")
                    elif isinstance(img, list) and img:
                        first = img[0]
                        info["image"] = first if isinstance(
                            first, str) else first.get("url", "")

                break

        return info

    def extract_product_info(self, metadata: PageMetadata) -> dict | None:
        """
        Extract product information from structured data.

        Args:
            metadata: PageMetadata to extract from

        Returns:
            Dictionary with product info, or None if not a product page
        """
        for sd in metadata.structured_data:
            if sd.data_type == "Product":
                data = sd.data
                return {
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "image": data.get("image", ""),
                    "brand": data.get("brand", {}).get("name", "") if isinstance(data.get("brand"), dict) else data.get("brand", ""),
                    "sku": data.get("sku", ""),
                    "price": self._extract_price(data),
                    "availability": self._extract_availability(data),
                    "rating": self._extract_rating(data),
                }

        return None

    def _extract_price(self, data: dict) -> dict | None:
        """Extract price from product data."""
        offers = data.get("offers", {})

        if isinstance(offers, list):
            offers = offers[0] if offers else {}

        if not offers:
            return None

        return {
            "amount": offers.get("price"),
            "currency": offers.get("priceCurrency"),
        }

    def _extract_availability(self, data: dict) -> str:
        """Extract availability from product data."""
        offers = data.get("offers", {})

        if isinstance(offers, list):
            offers = offers[0] if offers else {}

        availability = offers.get("availability", "")

        # Simplify schema.org URLs
        if "schema.org" in availability:
            availability = availability.split("/")[-1]

        return availability

    def _extract_rating(self, data: dict) -> dict | None:
        """Extract rating from product data."""
        rating = data.get("aggregateRating", {})

        if not rating:
            return None

        return {
            "value": rating.get("ratingValue"),
            "count": rating.get("reviewCount") or rating.get("ratingCount"),
            "best": rating.get("bestRating", 5),
        }
