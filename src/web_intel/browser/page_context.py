"""
Page context wrapper with utilities for content extraction.

Provides a high-level interface for page interaction and content retrieval,
handling common patterns like waiting for load, extracting content, and
managing page state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page, Response

from web_intel.core.exceptions import NavigationError, PageLoadError
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """
    Extracted content from a web page.

    Contains both raw HTML and processed text content,
    along with metadata about the page.
    """

    url: str
    final_url: str  # After redirects
    title: str
    html: str
    text: str
    status_code: int
    content_type: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    load_time_ms: float = 0.0
    links: list[str] = field(default_factory=list)
    meta_description: str = ""
    meta_keywords: list[str] = field(default_factory=list)
    canonical_url: str | None = None
    language: str | None = None

    @property
    def domain(self) -> str:
        """Extract domain from final URL."""
        parsed = urlparse(self.final_url)
        return parsed.netloc

    @property
    def is_html(self) -> bool:
        """Check if content type indicates HTML."""
        return "text/html" in self.content_type.lower()


class PageContext:
    """
    Wrapper around Playwright Page with utility methods.

    Provides consistent error handling, content extraction,
    and page state management.

    Example:
        >>> page = await context.new_page()
        >>> ctx = PageContext(page)
        >>> content = await ctx.navigate_and_extract("https://example.com")
    """

    def __init__(self, page: Page) -> None:
        """
        Initialize page context.

        Args:
            page: Playwright Page instance
        """
        self.page = page
        self._last_response: Response | None = None

    @property
    def current_url(self) -> str:
        """Get the current page URL."""
        return self.page.url

    async def navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
    ) -> Response | None:
        """
        Navigate to URL and wait for page load.

        Args:
            url: Target URL to navigate to
            wait_until: Load state to wait for:
                - "domcontentloaded": DOM is ready
                - "load": Full page load including resources
                - "networkidle": No network activity for 500ms

        Returns:
            Response object if available

        Raises:
            NavigationError: If navigation fails or times out
        """
        import time

        start_time = time.perf_counter()

        try:
            logger.debug(f"Navigating to: {url}")

            response = await self.page.goto(
                url,
                wait_until=wait_until,
            )

            self._last_response = response

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Navigation complete in {elapsed:.0f}ms")

            # Check for error responses
            if response and response.status >= 400:
                raise NavigationError(
                    f"HTTP {response.status} error",
                    url=url,
                    status_code=response.status,
                    retry_after=5.0 if response.status == 429 else None,
                )

            return response

        except NavigationError:
            raise
        except Exception as e:
            error_msg = str(e)

            # Detect timeout errors
            if "timeout" in error_msg.lower():
                raise NavigationError(
                    f"Navigation timeout: {error_msg}",
                    url=url,
                    retry_after=10.0,
                ) from e

            # Detect network errors
            if any(x in error_msg.lower() for x in ["net::", "dns", "connection"]):
                raise NavigationError(
                    f"Network error: {error_msg}",
                    url=url,
                    retry_after=5.0,
                ) from e

            raise NavigationError(
                f"Navigation failed: {error_msg}",
                url=url,
            ) from e

    async def extract_content(self) -> PageContent:
        """
        Extract content from the current page.

        Returns:
            PageContent with extracted data

        Raises:
            PageLoadError: If content extraction fails
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get basic page info
            url = self.page.url
            title = await self.page.title()

            # Get HTML content
            html = await self.page.content()

            # Extract visible text
            text = await self._extract_visible_text()

            # Get response info
            status_code = 200
            content_type = "text/html"

            if self._last_response:
                status_code = self._last_response.status
                content_type = self._last_response.headers.get(
                    "content-type", "text/html"
                )

            # Extract links
            links = await self._extract_links()

            # Extract meta information
            meta = await self._extract_meta()

            elapsed = (time.perf_counter() - start_time) * 1000

            return PageContent(
                url=url,
                final_url=url,
                title=title,
                html=html,
                text=text,
                status_code=status_code,
                content_type=content_type,
                load_time_ms=elapsed,
                links=links,
                meta_description=meta.get("description", ""),
                meta_keywords=meta.get("keywords", []),
                canonical_url=meta.get("canonical"),
                language=meta.get("language"),
            )

        except Exception as e:
            raise PageLoadError(
                f"Failed to extract content: {e}",
                url=self.page.url,
            ) from e

    async def navigate_and_extract(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
    ) -> PageContent:
        """
        Navigate to URL and extract page content.

        Combines navigation and extraction into a single operation.

        Args:
            url: Target URL
            wait_until: Load state to wait for

        Returns:
            Extracted PageContent

        Raises:
            NavigationError: If navigation fails
            PageLoadError: If extraction fails
        """
        await self.navigate(url, wait_until)
        content = await self.extract_content()
        content.url = url  # Store original URL
        return content

    async def _extract_visible_text(self) -> str:
        """
        Extract visible text content from page.

        Removes script, style, and hidden elements.
        """
        # JavaScript to extract clean text
        text = await self.page.evaluate("""
            () => {
                // Clone body to avoid modifying the actual page
                const clone = document.body.cloneNode(true);
                
                // Remove non-visible elements
                const removeSelectors = [
                    'script', 'style', 'noscript', 'iframe',
                    'svg', 'canvas', 'video', 'audio',
                    '[hidden]', '[aria-hidden="true"]',
                    '.hidden', '.invisible', '.sr-only'
                ];
                
                removeSelectors.forEach(selector => {
                    clone.querySelectorAll(selector).forEach(el => el.remove());
                });
                
                // Get text and normalize whitespace
                return clone.innerText
                    .replace(/\\s+/g, ' ')
                    .trim();
            }
        """)

        return text or ""

    async def _extract_links(self) -> list[str]:
        """
        Extract all href links from the page.

        Returns absolute URLs, filtering out javascript: and mailto: links.
        """
        base_url = self.page.url

        links = await self.page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href');
                    if (href && !href.startsWith('javascript:') && 
                        !href.startsWith('mailto:') && !href.startsWith('tel:')) {
                        links.push(href);
                    }
                });
                return [...new Set(links)];  // Deduplicate
            }
        """)

        # Convert to absolute URLs
        absolute_links = []
        for link in links:
            try:
                absolute = urljoin(base_url, link)
                # Remove fragments
                if "#" in absolute:
                    absolute = absolute.split("#")[0]
                if absolute:
                    absolute_links.append(absolute)
            except Exception:
                continue

        return list(set(absolute_links))

    async def _extract_meta(self) -> dict:
        """Extract meta tags and other page metadata."""
        meta = await self.page.evaluate("""
            () => {
                const result = {};
                
                // Meta description
                const desc = document.querySelector('meta[name="description"]');
                if (desc) result.description = desc.getAttribute('content') || '';
                
                // Meta keywords
                const kw = document.querySelector('meta[name="keywords"]');
                if (kw) {
                    const content = kw.getAttribute('content') || '';
                    result.keywords = content.split(',').map(k => k.trim()).filter(k => k);
                }
                
                // Canonical URL
                const canonical = document.querySelector('link[rel="canonical"]');
                if (canonical) result.canonical = canonical.getAttribute('href');
                
                // Language
                const html = document.documentElement;
                result.language = html.getAttribute('lang') || null;
                
                return result;
            }
        """)

        return meta or {}

    async def wait_for_selector(
        self,
        selector: str,
        timeout_ms: int | None = None,
        state: str = "visible",
    ) -> bool:
        """
        Wait for an element to appear on the page.

        Args:
            selector: CSS selector to wait for
            timeout_ms: Maximum wait time (uses default if None)
            state: Element state to wait for (visible, hidden, attached)

        Returns:
            True if element found, False if timeout
        """
        try:
            await self.page.wait_for_selector(
                selector,
                timeout=timeout_ms,
                state=state,
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the page."""
        try:
            await self.page.close()
        except Exception as e:
            logger.warning(f"Error closing page: {e}")
