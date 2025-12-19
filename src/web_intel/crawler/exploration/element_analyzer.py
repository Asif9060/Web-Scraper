"""
Interactive element analysis for exploration.

Detects and classifies clickable elements on a page to help
the exploration agent decide what to interact with.
"""

from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page, ElementHandle

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class ElementType(str, Enum):
    """Types of interactive elements."""

    NAVIGATION_LINK = "navigation_link"  # Main nav, header links
    CONTENT_LINK = "content_link"  # Links within content
    PAGINATION = "pagination"  # Next/prev, page numbers
    TAB = "tab"  # Tab controls
    ACCORDION = "accordion"  # Expandable sections
    DROPDOWN = "dropdown"  # Dropdown menus
    BUTTON = "button"  # Action buttons
    MODAL_TRIGGER = "modal_trigger"  # Opens modal/popup
    FORM_SUBMIT = "form_submit"  # Form submission
    LOAD_MORE = "load_more"  # Load more content
    FILTER = "filter"  # Filter/sort controls
    UNKNOWN = "unknown"


@dataclass
class InteractiveElement:
    """
    An interactive element detected on the page.

    Contains element metadata and scoring for prioritization.
    """

    selector: str
    element_type: ElementType
    text: str
    href: str | None = None
    aria_label: str | None = None
    priority_score: float = 0.5
    is_visible: bool = True
    bounding_box: dict | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def is_link(self) -> bool:
        """Check if element is a navigational link."""
        return self.href is not None and self.element_type in (
            ElementType.NAVIGATION_LINK,
            ElementType.CONTENT_LINK,
            ElementType.PAGINATION,
        )

    @property
    def display_text(self) -> str:
        """Get display text for logging."""
        text = self.text[:50] if self.text else ""
        if self.aria_label and not text:
            text = self.aria_label[:50]
        return text or f"[{self.element_type.value}]"


class ElementAnalyzer:
    """
    Analyzes page elements to find interactive components.

    Detects links, buttons, and other clickable elements,
    classifying them by type and importance.

    Example:
        >>> analyzer = ElementAnalyzer()
        >>> elements = await analyzer.find_interactive_elements(page)
        >>> for elem in elements:
        ...     print(f"{elem.element_type}: {elem.text}")
    """

    # Selectors for different element types
    NAV_SELECTORS = [
        "nav a",
        "header a",
        "[role='navigation'] a",
        ".nav a",
        ".navbar a",
        ".menu a",
        ".navigation a",
    ]

    PAGINATION_SELECTORS = [
        ".pagination a",
        ".pager a",
        "[aria-label*='page']",
        "[aria-label*='Page']",
        "a[rel='next']",
        "a[rel='prev']",
        ".page-numbers a",
        ".pages a",
    ]

    LOAD_MORE_SELECTORS = [
        "button:has-text('Load more')",
        "button:has-text('Show more')",
        "a:has-text('Load more')",
        "a:has-text('Show more')",
        "[class*='load-more']",
        "[class*='loadmore']",
    ]

    TAB_SELECTORS = [
        "[role='tab']",
        ".tab",
        ".tabs button",
        ".tabs a",
        "[data-toggle='tab']",
    ]

    ACCORDION_SELECTORS = [
        "[aria-expanded]",
        ".accordion-toggle",
        ".accordion-header",
        "[data-toggle='collapse']",
        "details summary",
    ]

    def __init__(
        self,
        base_url: str | None = None,
        same_domain_only: bool = True,
    ) -> None:
        """
        Initialize element analyzer.

        Args:
            base_url: Base URL for resolving relative links
            same_domain_only: Only include same-domain links
        """
        self.base_url = base_url
        self.same_domain_only = same_domain_only
        self._base_domain: str | None = None

        if base_url:
            parsed = urlparse(base_url)
            self._base_domain = parsed.netloc.lower()

    async def find_interactive_elements(
        self,
        page: Page,
        include_types: set[ElementType] | None = None,
    ) -> list[InteractiveElement]:
        """
        Find all interactive elements on the page.

        Args:
            page: Playwright page instance
            include_types: Types to include (None = all)

        Returns:
            List of interactive elements sorted by priority
        """
        elements: list[InteractiveElement] = []
        base_url = self.base_url or page.url

        # Find navigation links
        if include_types is None or ElementType.NAVIGATION_LINK in include_types:
            nav_elements = await self._find_by_selectors(
                page, self.NAV_SELECTORS, ElementType.NAVIGATION_LINK, base_url
            )
            elements.extend(nav_elements)

        # Find pagination
        if include_types is None or ElementType.PAGINATION in include_types:
            pagination = await self._find_by_selectors(
                page, self.PAGINATION_SELECTORS, ElementType.PAGINATION, base_url
            )
            elements.extend(pagination)

        # Find load more buttons
        if include_types is None or ElementType.LOAD_MORE in include_types:
            load_more = await self._find_by_selectors(
                page, self.LOAD_MORE_SELECTORS, ElementType.LOAD_MORE, base_url
            )
            elements.extend(load_more)

        # Find tabs
        if include_types is None or ElementType.TAB in include_types:
            tabs = await self._find_by_selectors(
                page, self.TAB_SELECTORS, ElementType.TAB, base_url
            )
            elements.extend(tabs)

        # Find accordions
        if include_types is None or ElementType.ACCORDION in include_types:
            accordions = await self._find_by_selectors(
                page, self.ACCORDION_SELECTORS, ElementType.ACCORDION, base_url
            )
            elements.extend(accordions)

        # Find content links (general links not caught above)
        if include_types is None or ElementType.CONTENT_LINK in include_types:
            content_links = await self._find_content_links(page, base_url, elements)
            elements.extend(content_links)

        # Deduplicate and sort by priority
        elements = self._deduplicate(elements)
        elements.sort(key=lambda e: e.priority_score, reverse=True)

        logger.debug(f"Found {len(elements)} interactive elements")
        return elements

    async def _find_by_selectors(
        self,
        page: Page,
        selectors: list[str],
        element_type: ElementType,
        base_url: str,
    ) -> list[InteractiveElement]:
        """Find elements matching any of the given selectors."""
        elements = []
        seen_selectors = set()

        for selector in selectors:
            try:
                handles = await page.query_selector_all(selector)

                for handle in handles:
                    elem = await self._element_from_handle(
                        handle, element_type, base_url
                    )
                    if elem and elem.selector not in seen_selectors:
                        seen_selectors.add(elem.selector)
                        elements.append(elem)

            except Exception as e:
                logger.debug(f"Selector failed '{selector}': {e}")

        return elements

    async def _find_content_links(
        self,
        page: Page,
        base_url: str,
        existing: list[InteractiveElement],
    ) -> list[InteractiveElement]:
        """Find content links not already captured."""
        existing_hrefs = {e.href for e in existing if e.href}
        elements = []

        try:
            handles = await page.query_selector_all("a[href]")

            for handle in handles:
                elem = await self._element_from_handle(
                    handle, ElementType.CONTENT_LINK, base_url
                )

                if elem and elem.href and elem.href not in existing_hrefs:
                    existing_hrefs.add(elem.href)
                    elements.append(elem)

        except Exception as e:
            logger.debug(f"Content link extraction failed: {e}")

        return elements

    async def _element_from_handle(
        self,
        handle: ElementHandle,
        element_type: ElementType,
        base_url: str,
    ) -> InteractiveElement | None:
        """Convert element handle to InteractiveElement."""
        try:
            # Check visibility
            is_visible = await handle.is_visible()
            if not is_visible:
                return None

            # Get element properties
            props = await handle.evaluate("""
                (el) => ({
                    tagName: el.tagName.toLowerCase(),
                    text: (el.innerText || el.textContent || '').trim().slice(0, 200),
                    href: el.getAttribute('href'),
                    ariaLabel: el.getAttribute('aria-label'),
                    className: el.className || '',
                    id: el.id || '',
                    role: el.getAttribute('role'),
                    type: el.getAttribute('type'),
                })
            """)

            # Build selector
            selector = self._build_selector(props)

            # Resolve href
            href = None
            if props.get("href"):
                href = urljoin(base_url, props["href"])
                # Remove fragments
                if "#" in href:
                    href = href.split("#")[0]

                # Filter external links if configured
                if self.same_domain_only and self._base_domain:
                    href_domain = urlparse(href).netloc.lower()
                    if href_domain != self._base_domain:
                        return None

            # Skip non-http links
            if href and not href.startswith(("http://", "https://")):
                return None

            # Calculate priority score
            priority = self._calculate_priority(props, element_type)

            # Get bounding box
            try:
                bbox = await handle.bounding_box()
            except Exception:
                bbox = None

            return InteractiveElement(
                selector=selector,
                element_type=element_type,
                text=props.get("text", ""),
                href=href,
                aria_label=props.get("ariaLabel"),
                priority_score=priority,
                is_visible=is_visible,
                bounding_box=bbox,
                attributes={
                    "tag": props.get("tagName"),
                    "class": props.get("className"),
                    "id": props.get("id"),
                    "role": props.get("role"),
                },
            )

        except Exception as e:
            logger.debug(f"Failed to analyze element: {e}")
            return None

    def _build_selector(self, props: dict) -> str:
        """Build a CSS selector for the element."""
        tag = props.get("tagName", "")
        elem_id = props.get("id", "")
        class_name = props.get("className", "")

        if elem_id:
            return f"#{elem_id}"

        if class_name:
            # Use first significant class
            classes = [c for c in class_name.split() if len(c) > 2]
            if classes:
                return f"{tag}.{classes[0]}"

        return tag

    def _calculate_priority(self, props: dict, element_type: ElementType) -> float:
        """Calculate priority score for element."""
        score = 0.5

        # Type-based scoring
        type_scores = {
            ElementType.NAVIGATION_LINK: 0.8,
            ElementType.PAGINATION: 0.7,
            ElementType.LOAD_MORE: 0.75,
            ElementType.TAB: 0.6,
            ElementType.ACCORDION: 0.5,
            ElementType.CONTENT_LINK: 0.4,
        }
        score = type_scores.get(element_type, 0.5)

        # Boost for semantic indicators
        text = (props.get("text") or "").lower()
        aria = (props.get("ariaLabel") or "").lower()

        important_terms = ["main", "primary",
                           "home", "about", "product", "service"]
        for term in important_terms:
            if term in text or term in aria:
                score += 0.1
                break

        # Penalize likely noise
        noise_terms = ["social", "share", "tweet",
                       "facebook", "twitter", "instagram"]
        for term in noise_terms:
            if term in text or term in aria:
                score -= 0.2
                break

        return max(0.0, min(1.0, score))

    def _deduplicate(
        self,
        elements: list[InteractiveElement],
    ) -> list[InteractiveElement]:
        """Remove duplicate elements based on href or selector."""
        seen_hrefs: set[str] = set()
        seen_selectors: set[str] = set()
        unique: list[InteractiveElement] = []

        for elem in elements:
            # Check href uniqueness for links
            if elem.href:
                if elem.href in seen_hrefs:
                    continue
                seen_hrefs.add(elem.href)

            # Check selector uniqueness for non-links
            elif elem.selector in seen_selectors:
                continue
            else:
                seen_selectors.add(elem.selector)

            unique.append(elem)

        return unique
