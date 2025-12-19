"""
Navigation pattern detection for exploration.

Identifies common navigation patterns like pagination, infinite scroll,
tabs, and categorization to help the agent navigate efficiently.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, parse_qs

from playwright.async_api import Page

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class NavigationPattern(str, Enum):
    """Types of navigation patterns detected on pages."""

    PAGINATION_NUMBERED = "pagination_numbered"  # Page 1, 2, 3...
    PAGINATION_NEXT_PREV = "pagination_next_prev"  # Next/Previous only
    INFINITE_SCROLL = "infinite_scroll"  # Content loads on scroll
    LOAD_MORE_BUTTON = "load_more_button"  # Click to load more
    TABBED_CONTENT = "tabbed_content"  # Tab navigation
    ACCORDION = "accordion"  # Expandable sections
    CATEGORY_LISTING = "category_listing"  # Category/tag pages
    SEARCH_RESULTS = "search_results"  # Search result pages
    HIERARCHICAL = "hierarchical"  # Parent/child navigation
    ALPHABETICAL = "alphabetical"  # A-Z navigation
    DATE_BASED = "date_based"  # Archive by date
    NONE = "none"  # No clear pattern


@dataclass
class PaginationInfo:
    """
    Information about detected pagination.

    Helps the agent understand how to navigate paginated content.
    """

    pattern: NavigationPattern
    current_page: int | None = None
    total_pages: int | None = None
    next_url: str | None = None
    prev_url: str | None = None
    page_urls: list[str] = field(default_factory=list)
    has_more: bool = False

    @property
    def is_first_page(self) -> bool:
        """Check if this is the first page."""
        return self.current_page == 1 or self.prev_url is None

    @property
    def is_last_page(self) -> bool:
        """Check if this is the last page."""
        if self.total_pages and self.current_page:
            return self.current_page >= self.total_pages
        return self.next_url is None and not self.has_more


class NavigationPatternDetector:
    """
    Detects navigation patterns on web pages.

    Analyzes page structure to identify pagination, infinite scroll,
    and other navigation mechanisms.

    Example:
        >>> detector = NavigationPatternDetector()
        >>> patterns = await detector.detect_patterns(page)
        >>> if patterns.pagination:
        ...     print(f"Found pagination: {patterns.pagination.pattern}")
    """

    # Regex patterns for pagination detection
    PAGE_NUMBER_PATTERNS = [
        r"[?&]page=(\d+)",
        r"[?&]p=(\d+)",
        r"/page/(\d+)",
        r"/p/(\d+)",
        r"-page-(\d+)",
        r"_page_(\d+)",
    ]

    def __init__(self) -> None:
        """Initialize pattern detector."""
        self._page_patterns = [re.compile(
            p, re.IGNORECASE) for p in self.PAGE_NUMBER_PATTERNS]

    async def detect_patterns(self, page: Page) -> dict:
        """
        Detect all navigation patterns on the page.

        Args:
            page: Playwright page instance

        Returns:
            Dictionary with detected patterns
        """
        results = {
            "pagination": None,
            "infinite_scroll": False,
            "load_more": False,
            "tabs": False,
            "accordion": False,
            "patterns": [],
        }

        # Check pagination
        pagination = await self._detect_pagination(page)
        if pagination:
            results["pagination"] = pagination
            results["patterns"].append(pagination.pattern)

        # Check infinite scroll
        if await self._detect_infinite_scroll(page):
            results["infinite_scroll"] = True
            results["patterns"].append(NavigationPattern.INFINITE_SCROLL)

        # Check load more
        if await self._detect_load_more(page):
            results["load_more"] = True
            results["patterns"].append(NavigationPattern.LOAD_MORE_BUTTON)

        # Check tabs
        if await self._detect_tabs(page):
            results["tabs"] = True
            results["patterns"].append(NavigationPattern.TABBED_CONTENT)

        # Check accordion
        if await self._detect_accordion(page):
            results["accordion"] = True
            results["patterns"].append(NavigationPattern.ACCORDION)

        return results

    async def _detect_pagination(self, page: Page) -> PaginationInfo | None:
        """Detect pagination pattern on page."""
        current_url = page.url

        # Extract pagination info from page
        pagination_data = await page.evaluate("""
            () => {
                const result = {
                    numbered: [],
                    nextUrl: null,
                    prevUrl: null,
                    currentPage: null,
                    totalPages: null,
                };
                
                // Find pagination container
                const containers = document.querySelectorAll(
                    '.pagination, .pager, [aria-label*="pagination"], nav[role="navigation"]'
                );
                
                for (const container of containers) {
                    // Find page number links
                    const links = container.querySelectorAll('a');
                    for (const link of links) {
                        const href = link.getAttribute('href');
                        const text = (link.textContent || '').trim();
                        
                        if (!href) continue;
                        
                        // Check for next/prev
                        const rel = link.getAttribute('rel') || '';
                        const ariaLabel = (link.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (rel === 'next' || ariaLabel.includes('next') || 
                            text.toLowerCase().includes('next')) {
                            result.nextUrl = href;
                        }
                        if (rel === 'prev' || ariaLabel.includes('prev') || 
                            text.toLowerCase().includes('previous')) {
                            result.prevUrl = href;
                        }
                        
                        // Check for page numbers
                        const pageNum = parseInt(text, 10);
                        if (!isNaN(pageNum) && pageNum > 0 && pageNum < 1000) {
                            result.numbered.push({ page: pageNum, url: href });
                        }
                    }
                    
                    // Find current page indicator
                    const current = container.querySelector(
                        '.current, .active, [aria-current="page"]'
                    );
                    if (current) {
                        const num = parseInt(current.textContent || '', 10);
                        if (!isNaN(num)) {
                            result.currentPage = num;
                        }
                    }
                }
                
                // Try to find total from "Page X of Y" pattern
                const pageText = document.body.innerText;
                const match = pageText.match(/page\\s+(\\d+)\\s+of\\s+(\\d+)/i);
                if (match) {
                    result.currentPage = result.currentPage || parseInt(match[1], 10);
                    result.totalPages = parseInt(match[2], 10);
                }
                
                return result;
            }
        """)

        # No pagination found
        if not pagination_data["numbered"] and not pagination_data["nextUrl"]:
            # Check URL for page number as fallback
            current_page = self._extract_page_from_url(current_url)
            if current_page and current_page > 1:
                return PaginationInfo(
                    pattern=NavigationPattern.PAGINATION_NUMBERED,
                    current_page=current_page,
                )
            return None

        # Build pagination info
        pattern = NavigationPattern.PAGINATION_NUMBERED
        if pagination_data["numbered"]:
            page_urls = [p["url"] for p in sorted(
                pagination_data["numbered"], key=lambda x: x["page"])]
        else:
            pattern = NavigationPattern.PAGINATION_NEXT_PREV
            page_urls = []

        # Resolve relative URLs
        base_url = current_url
        next_url = pagination_data["nextUrl"]
        prev_url = pagination_data["prevUrl"]

        if next_url and not next_url.startswith("http"):
            from urllib.parse import urljoin
            next_url = urljoin(base_url, next_url)
        if prev_url and not prev_url.startswith("http"):
            from urllib.parse import urljoin
            prev_url = urljoin(base_url, prev_url)

        return PaginationInfo(
            pattern=pattern,
            current_page=pagination_data["currentPage"] or self._extract_page_from_url(
                current_url) or 1,
            total_pages=pagination_data["totalPages"],
            next_url=next_url,
            prev_url=prev_url,
            page_urls=page_urls,
            has_more=next_url is not None,
        )

    def _extract_page_from_url(self, url: str) -> int | None:
        """Extract page number from URL."""
        for pattern in self._page_patterns:
            match = pattern.search(url)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        return None

    async def _detect_infinite_scroll(self, page: Page) -> bool:
        """Detect if page uses infinite scroll."""
        # Check for common infinite scroll indicators
        result = await page.evaluate("""
            () => {
                // Check for infinite scroll libraries/attributes
                if (document.querySelector('[data-infinite-scroll]')) return true;
                if (document.querySelector('.infinite-scroll')) return true;
                if (document.querySelector('[data-behavior="infinite-scroll"]')) return true;
                
                // Check for scroll event listeners that trigger loading
                // This is a heuristic - can't directly detect event listeners
                const loadingIndicators = document.querySelectorAll(
                    '.loading, .spinner, [data-loading], .load-more-spinner'
                );
                if (loadingIndicators.length > 0) return true;
                
                return false;
            }
        """)

        return result

    async def _detect_load_more(self, page: Page) -> bool:
        """Detect load more button."""
        result = await page.evaluate("""
            () => {
                const selectors = [
                    'button:has-text("Load more")',
                    'button:has-text("Show more")',
                    'a:has-text("Load more")',
                    'a:has-text("Show more")',
                    '[class*="load-more"]',
                    '[class*="loadmore"]',
                    '[data-action="load-more"]',
                ];
                
                for (const selector of selectors) {
                    try {
                        // Use basic selectors that work in evaluate
                        const elements = document.querySelectorAll(
                            '[class*="load-more"], [class*="loadmore"], [data-action="load-more"]'
                        );
                        if (elements.length > 0) return true;
                    } catch (e) {}
                }
                
                // Check button/link text
                const buttons = document.querySelectorAll('button, a');
                for (const btn of buttons) {
                    const text = (btn.textContent || '').toLowerCase();
                    if (text.includes('load more') || text.includes('show more')) {
                        return true;
                    }
                }
                
                return false;
            }
        """)

        return result

    async def _detect_tabs(self, page: Page) -> bool:
        """Detect tabbed content."""
        result = await page.evaluate("""
            () => {
                // Check for tab-related elements
                const tabSelectors = [
                    '[role="tablist"]',
                    '[role="tab"]',
                    '.tabs',
                    '.tab-content',
                    '[data-toggle="tab"]',
                    '.nav-tabs',
                ];
                
                for (const selector of tabSelectors) {
                    if (document.querySelector(selector)) return true;
                }
                
                return false;
            }
        """)

        return result

    async def _detect_accordion(self, page: Page) -> bool:
        """Detect accordion/collapsible content."""
        result = await page.evaluate("""
            () => {
                const selectors = [
                    '[aria-expanded]',
                    '.accordion',
                    '[data-toggle="collapse"]',
                    'details',
                    '.collapsible',
                    '.expandable',
                ];
                
                for (const selector of selectors) {
                    const elements = document.querySelectorAll(selector);
                    if (elements.length >= 2) return true;  // Need multiple for accordion
                }
                
                return false;
            }
        """)

        return result

    async def get_pagination_urls(
        self,
        page: Page,
        max_pages: int = 10,
    ) -> list[str]:
        """
        Get URLs for paginated content.

        Args:
            page: Playwright page instance
            max_pages: Maximum pages to return

        Returns:
            List of pagination URLs
        """
        pagination = await self._detect_pagination(page)

        if not pagination:
            return []

        urls = []
        current_url = page.url

        # If we have explicit page URLs
        if pagination.page_urls:
            return pagination.page_urls[:max_pages]

        # Generate URLs from pattern
        if pagination.next_url:
            urls.append(pagination.next_url)

            # Try to generate more URLs based on pattern
            if pagination.current_page:
                base_pattern = self._find_page_pattern(
                    current_url, pagination.current_page)
                if base_pattern:
                    for i in range(pagination.current_page + 1, pagination.current_page + max_pages):
                        if pagination.total_pages and i > pagination.total_pages:
                            break
                        url = base_pattern.replace("{page}", str(i))
                        if url not in urls:
                            urls.append(url)

        return urls[:max_pages]

    def _find_page_pattern(self, url: str, current_page: int) -> str | None:
        """Find the URL pattern for pagination."""
        for pattern in self._page_patterns:
            match = pattern.search(url)
            if match:
                # Replace page number with placeholder
                start, end = match.span(1)
                return url[:start] + "{page}" + url[end:]
        return None
