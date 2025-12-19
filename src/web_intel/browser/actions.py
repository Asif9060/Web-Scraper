"""
Common browser actions and utilities.

Provides reusable functions for common browser operations like
link extraction, content retrieval, and safe interactions.
"""

from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page, ElementHandle

from web_intel.core.exceptions import BrowserError, PageLoadError
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


async def extract_links(
    page: Page,
    base_url: str | None = None,
    same_domain_only: bool = False,
) -> list[dict[str, str]]:
    """
    Extract all links from a page with their text and attributes.

    Args:
        page: Playwright Page instance
        base_url: Base URL for resolving relative links (defaults to page URL)
        same_domain_only: If True, only return links to the same domain

    Returns:
        List of dicts with 'url', 'text', and 'rel' keys
    """
    if base_url is None:
        base_url = page.url

    base_domain = urlparse(base_url).netloc

    links_data = await page.evaluate("""
        () => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.getAttribute('href');
                if (href && !href.startsWith('javascript:') && 
                    !href.startsWith('mailto:') && !href.startsWith('tel:') &&
                    !href.startsWith('#')) {
                    links.push({
                        href: href,
                        text: (a.innerText || a.textContent || '').trim().slice(0, 200),
                        rel: a.getAttribute('rel') || '',
                        title: a.getAttribute('title') || ''
                    });
                }
            });
            return links;
        }
    """)

    result = []
    seen_urls = set()

    for link in links_data:
        try:
            absolute_url = urljoin(base_url, link["href"])

            # Remove fragments
            if "#" in absolute_url:
                absolute_url = absolute_url.split("#")[0]

            if not absolute_url or absolute_url in seen_urls:
                continue

            # Filter by domain if requested
            if same_domain_only:
                link_domain = urlparse(absolute_url).netloc
                if link_domain != base_domain:
                    continue

            seen_urls.add(absolute_url)
            result.append({
                "url": absolute_url,
                "text": link["text"],
                "rel": link["rel"],
                "title": link["title"],
            })

        except Exception:
            continue

    return result


async def extract_text_content(
    page: Page,
    selector: str | None = None,
    include_hidden: bool = False,
) -> str:
    """
    Extract text content from page or specific element.

    Args:
        page: Playwright Page instance
        selector: CSS selector for specific element (None = entire page)
        include_hidden: Whether to include hidden elements

    Returns:
        Extracted text content
    """
    if selector:
        element = await page.query_selector(selector)
        if not element:
            return ""

        if include_hidden:
            return await element.text_content() or ""
        else:
            return await element.inner_text() or ""

    # Full page extraction
    script = """
        (includeHidden) => {
            const clone = document.body.cloneNode(true);
            
            if (!includeHidden) {
                const removeSelectors = [
                    'script', 'style', 'noscript', 'iframe',
                    'svg', 'canvas', 'video', 'audio',
                    '[hidden]', '[aria-hidden="true"]',
                    '.hidden', '.invisible'
                ];
                
                removeSelectors.forEach(selector => {
                    clone.querySelectorAll(selector).forEach(el => el.remove());
                });
            }
            
            return clone.innerText || clone.textContent || '';
        }
    """

    text = await page.evaluate(script, include_hidden)
    return (text or "").strip()


async def wait_for_page_ready(
    page: Page,
    timeout_ms: int = 30000,
    check_network_idle: bool = False,
) -> bool:
    """
    Wait for page to be fully loaded and ready.

    Args:
        page: Playwright Page instance
        timeout_ms: Maximum wait time in milliseconds
        check_network_idle: Also wait for network to be idle

    Returns:
        True if page is ready, False if timeout
    """
    try:
        # Wait for DOM content
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)

        if check_network_idle:
            # Wait for network to settle (no requests for 500ms)
            try:
                await page.wait_for_load_state(
                    "networkidle",
                    # Cap at 10s for network idle
                    timeout=min(timeout_ms, 10000),
                )
            except Exception:
                # Network idle timeout is acceptable
                logger.debug("Network idle timeout, continuing anyway")

        return True

    except Exception as e:
        logger.warning(f"Page ready wait failed: {e}")
        return False


async def safe_click(
    page: Page,
    selector: str,
    timeout_ms: int = 5000,
    force: bool = False,
) -> bool:
    """
    Safely click an element with error handling.

    Args:
        page: Playwright Page instance
        selector: CSS selector for element to click
        timeout_ms: Maximum wait time for element
        force: Force click even if element is not visible

    Returns:
        True if click succeeded, False otherwise
    """
    try:
        # Wait for element to be clickable
        element = await page.wait_for_selector(
            selector,
            timeout=timeout_ms,
            state="visible" if not force else "attached",
        )

        if element is None:
            logger.debug(f"Element not found: {selector}")
            return False

        await element.click(force=force, timeout=timeout_ms)
        logger.debug(f"Clicked element: {selector}")
        return True

    except Exception as e:
        logger.debug(f"Click failed for {selector}: {e}")
        return False


async def take_screenshot(
    page: Page,
    path: Path | str,
    full_page: bool = False,
    selector: str | None = None,
) -> Path:
    """
    Take a screenshot of the page or specific element.

    Args:
        page: Playwright Page instance
        path: Output file path (will add .png if no extension)
        full_page: Capture entire scrollable page
        selector: CSS selector for specific element

    Returns:
        Path to saved screenshot

    Raises:
        BrowserError: If screenshot fails
    """
    path = Path(path)

    # Ensure .png extension
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if selector:
            element = await page.query_selector(selector)
            if not element:
                raise BrowserError(f"Element not found: {selector}")
            await element.screenshot(path=str(path))
        else:
            await page.screenshot(path=str(path), full_page=full_page)

        logger.debug(f"Screenshot saved: {path}")
        return path

    except BrowserError:
        raise
    except Exception as e:
        raise BrowserError(
            f"Screenshot failed: {e}",
            details={"path": str(path)},
        ) from e


async def get_element_attributes(
    page: Page,
    selector: str,
) -> dict[str, str] | None:
    """
    Get all attributes of an element.

    Args:
        page: Playwright Page instance
        selector: CSS selector for element

    Returns:
        Dictionary of attribute name -> value, or None if not found
    """
    result = await page.evaluate(
        """
        (selector) => {
            const el = document.querySelector(selector);
            if (!el) return null;
            
            const attrs = {};
            for (const attr of el.attributes) {
                attrs[attr.name] = attr.value;
            }
            return attrs;
        }
        """,
        selector,
    )

    return result


async def scroll_to_bottom(
    page: Page,
    step_pixels: int = 500,
    delay_ms: int = 100,
    max_scrolls: int = 50,
) -> int:
    """
    Scroll to bottom of page incrementally.

    Useful for triggering lazy-loaded content.

    Args:
        page: Playwright Page instance
        step_pixels: Pixels to scroll per step
        delay_ms: Delay between scrolls
        max_scrolls: Maximum number of scroll steps

    Returns:
        Number of scrolls performed
    """
    scrolls = 0

    for _ in range(max_scrolls):
        # Get current and max scroll positions
        positions = await page.evaluate("""
            () => ({
                current: window.scrollY,
                max: document.documentElement.scrollHeight - window.innerHeight
            })
        """)

        if positions["current"] >= positions["max"]:
            break

        # Scroll down
        await page.evaluate(f"window.scrollBy(0, {step_pixels})")
        scrolls += 1

        if delay_ms > 0:
            await page.wait_for_timeout(delay_ms)

    return scrolls


async def extract_structured_data(page: Page) -> dict:
    """
    Extract structured data (JSON-LD, microdata) from page.

    Returns:
        Dictionary with 'json_ld' and 'open_graph' keys
    """
    data = await page.evaluate("""
        () => {
            const result = {
                json_ld: [],
                open_graph: {}
            };
            
            // Extract JSON-LD
            document.querySelectorAll('script[type="application/ld+json"]')
                .forEach(script => {
                    try {
                        const data = JSON.parse(script.textContent);
                        result.json_ld.push(data);
                    } catch (e) {}
                });
            
            // Extract Open Graph
            document.querySelectorAll('meta[property^="og:"]')
                .forEach(meta => {
                    const prop = meta.getAttribute('property').replace('og:', '');
                    result.open_graph[prop] = meta.getAttribute('content');
                });
            
            return result;
        }
    """)

    return data or {"json_ld": [], "open_graph": {}}
