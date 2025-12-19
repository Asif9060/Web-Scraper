"""
Exploration agent for intelligent page discovery.

Coordinates element analysis, pattern detection, and state tracking
to make intelligent decisions about what to explore on a website.
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from playwright.async_api import Page

from web_intel.browser.page_context import PageContext, PageContent
from web_intel.crawler.exploration.element_analyzer import (
    ElementAnalyzer,
    InteractiveElement,
    ElementType,
)
from web_intel.crawler.exploration.navigation_patterns import (
    NavigationPatternDetector,
    NavigationPattern,
    PaginationInfo,
)
from web_intel.crawler.exploration.state_tracker import (
    ExplorationState,
    PageState,
    PageStateType,
)
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExplorationDecision:
    """
    A decision about what to explore next.

    Contains the action to take and why it was chosen.
    """

    action: str  # "navigate", "click", "scroll", "skip"
    target: str | None = None  # URL or selector
    reason: str = ""
    priority: float = 0.5
    element: InteractiveElement | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExplorationResult:
    """
    Result of an exploration action.

    Contains discovered URLs and any new content found.
    """

    success: bool
    discovered_urls: list[str] = field(default_factory=list)
    new_content: bool = False
    content_hash: str | None = None
    error: str | None = None


class ExplorationAgent:
    """
    Intelligent agent for exploring websites.

    Analyzes pages to find interactive elements, detects navigation
    patterns, and makes decisions about what to explore to maximize
    coverage while avoiding loops.

    Example:
        >>> agent = ExplorationAgent(base_url="https://example.com")
        >>> decisions = await agent.analyze_page(page_context)
        >>> for decision in decisions:
        ...     if decision.action == "navigate":
        ...         await crawler.queue_url(decision.target)
    """

    def __init__(
        self,
        base_url: str,
        max_interactions_per_page: int = 5,
        max_visits_per_url: int = 3,
        prioritize_navigation: bool = True,
        explore_pagination: bool = True,
        explore_tabs: bool = True,
        explore_accordions: bool = False,
    ) -> None:
        """
        Initialize exploration agent.

        Args:
            base_url: Base URL of website being explored
            max_interactions_per_page: Max click actions per page
            max_visits_per_url: Max visits to same URL
            prioritize_navigation: Prioritize main nav links
            explore_pagination: Follow pagination links
            explore_tabs: Click tabs to reveal content
            explore_accordions: Expand accordion sections
        """
        self.base_url = base_url
        self.max_interactions_per_page = max_interactions_per_page
        self.prioritize_navigation = prioritize_navigation
        self.explore_pagination = explore_pagination
        self.explore_tabs = explore_tabs
        self.explore_accordions = explore_accordions

        # Initialize components
        self.element_analyzer = ElementAnalyzer(
            base_url=base_url,
            same_domain_only=True,
        )
        self.pattern_detector = NavigationPatternDetector()
        self.state_tracker = ExplorationState(
            max_visits_per_url=max_visits_per_url,
        )

    async def analyze_page(
        self,
        page: Page,
        current_depth: int = 0,
    ) -> list[ExplorationDecision]:
        """
        Analyze page and generate exploration decisions.

        Args:
            page: Playwright page instance
            current_depth: Current crawl depth

        Returns:
            List of exploration decisions sorted by priority
        """
        decisions: list[ExplorationDecision] = []
        current_url = page.url

        # Record visit
        content = await page.content()
        content_hash = self.state_tracker.compute_content_hash(content)
        self.state_tracker.record_visit(
            url=current_url,
            content_hash=content_hash,
            depth=current_depth,
        )

        # Record page state
        state = await self._capture_page_state(page, content_hash)
        is_new_state = self.state_tracker.record_state(state)

        if not is_new_state:
            logger.debug(f"Duplicate page state detected: {current_url}")

        # Detect navigation patterns
        patterns = await self.pattern_detector.detect_patterns(page)

        # Find interactive elements
        elements = await self.element_analyzer.find_interactive_elements(page)

        # Generate decisions for navigation links
        nav_decisions = self._process_navigation_links(elements, current_depth)
        decisions.extend(nav_decisions)

        # Generate decisions for pagination
        if self.explore_pagination and patterns.get("pagination"):
            pagination_decisions = self._process_pagination(
                patterns["pagination"], current_depth
            )
            decisions.extend(pagination_decisions)

        # Generate decisions for interactive elements
        if self.explore_tabs and patterns.get("tabs"):
            tab_decisions = self._process_tabs(elements)
            decisions.extend(tab_decisions)

        if self.explore_accordions and patterns.get("accordion"):
            accordion_decisions = self._process_accordions(elements)
            decisions.extend(accordion_decisions)

        # Filter out loops
        decisions = self._filter_loops(decisions, current_url)

        # Sort by priority
        decisions.sort(key=lambda d: d.priority, reverse=True)

        # Limit interactions
        click_decisions = [d for d in decisions if d.action == "click"]
        nav_decisions = [d for d in decisions if d.action == "navigate"]

        limited_clicks = click_decisions[: self.max_interactions_per_page]
        final_decisions = nav_decisions + limited_clicks

        logger.debug(
            f"Generated {len(final_decisions)} exploration decisions "
            f"({len(nav_decisions)} navigate, {len(limited_clicks)} click)"
        )

        return final_decisions

    async def _capture_page_state(
        self,
        page: Page,
        content_hash: str,
    ) -> PageState:
        """Capture current page state."""
        title = await page.title()

        stats = await page.evaluate("""
            () => ({
                elementCount: document.querySelectorAll('*').length,
                linkCount: document.querySelectorAll('a[href]').length,
                textLength: (document.body.innerText || '').length,
            })
        """)

        return PageState(
            url=page.url,
            content_hash=content_hash,
            title=title,
            element_count=stats["elementCount"],
            link_count=stats["linkCount"],
            text_length=stats["textLength"],
        )

    def _process_navigation_links(
        self,
        elements: list[InteractiveElement],
        current_depth: int,
    ) -> list[ExplorationDecision]:
        """Process navigation and content links."""
        decisions = []

        for elem in elements:
            if not elem.is_link or not elem.href:
                continue

            # Check if should visit
            if not self.state_tracker.should_visit(elem.href):
                continue

            # Determine priority
            priority = elem.priority_score

            if elem.element_type == ElementType.NAVIGATION_LINK:
                if self.prioritize_navigation:
                    priority += 0.2
                reason = "Main navigation link"
            else:
                reason = "Content link"

            decisions.append(
                ExplorationDecision(
                    action="navigate",
                    target=elem.href,
                    reason=reason,
                    priority=priority,
                    element=elem,
                    metadata={"depth": current_depth + 1},
                )
            )

        return decisions

    def _process_pagination(
        self,
        pagination: PaginationInfo,
        current_depth: int,
    ) -> list[ExplorationDecision]:
        """Process pagination links."""
        decisions = []

        # Prioritize next page
        if pagination.next_url and self.state_tracker.should_visit(pagination.next_url):
            decisions.append(
                ExplorationDecision(
                    action="navigate",
                    target=pagination.next_url,
                    reason="Next page in pagination",
                    priority=0.75,
                    metadata={
                        "depth": current_depth,  # Same depth for pagination
                        "pagination": True,
                        "page": (pagination.current_page or 0) + 1,
                    },
                )
            )

        # Add numbered pages
        for url in pagination.page_urls[:5]:  # Limit to first 5 pages
            if self.state_tracker.should_visit(url):
                decisions.append(
                    ExplorationDecision(
                        action="navigate",
                        target=url,
                        reason="Pagination page",
                        priority=0.6,
                        metadata={"depth": current_depth, "pagination": True},
                    )
                )

        return decisions

    def _process_tabs(
        self,
        elements: list[InteractiveElement],
    ) -> list[ExplorationDecision]:
        """Process tab elements."""
        decisions = []

        for elem in elements:
            if elem.element_type != ElementType.TAB:
                continue

            decisions.append(
                ExplorationDecision(
                    action="click",
                    target=elem.selector,
                    reason=f"Tab: {elem.display_text}",
                    priority=0.5,
                    element=elem,
                )
            )

        return decisions

    def _process_accordions(
        self,
        elements: list[InteractiveElement],
    ) -> list[ExplorationDecision]:
        """Process accordion elements."""
        decisions = []

        for elem in elements:
            if elem.element_type != ElementType.ACCORDION:
                continue

            decisions.append(
                ExplorationDecision(
                    action="click",
                    target=elem.selector,
                    reason=f"Accordion: {elem.display_text}",
                    priority=0.4,
                    element=elem,
                )
            )

        return decisions

    def _filter_loops(
        self,
        decisions: list[ExplorationDecision],
        current_url: str,
    ) -> list[ExplorationDecision]:
        """Filter out decisions that would create loops."""
        filtered = []

        for decision in decisions:
            action_key = f"{decision.action}:{decision.target}"

            if self.state_tracker.detect_loop(current_url, action_key):
                logger.debug(f"Filtered loop decision: {action_key}")
                continue

            filtered.append(decision)

        return filtered

    async def execute_click(
        self,
        page: Page,
        decision: ExplorationDecision,
    ) -> ExplorationResult:
        """
        Execute a click decision and capture results.

        Args:
            page: Playwright page instance
            decision: Click decision to execute

        Returns:
            Result of the click action
        """
        if decision.action != "click" or not decision.target:
            return ExplorationResult(
                success=False,
                error="Invalid click decision",
            )

        selector = decision.target
        current_url = page.url

        try:
            # Capture state before click
            content_before = await page.content()
            hash_before = self.state_tracker.compute_content_hash(
                content_before)

            # Perform click
            await page.click(selector, timeout=5000)

            # Wait for any navigation or content change
            await page.wait_for_timeout(500)

            # Check if URL changed
            new_url = page.url
            discovered_urls = []

            if new_url != current_url:
                discovered_urls.append(new_url)
                logger.debug(f"Click triggered navigation to: {new_url}")

            # Check if content changed
            content_after = await page.content()
            hash_after = self.state_tracker.compute_content_hash(content_after)
            new_content = hash_after != hash_before

            # Record the interaction
            self.state_tracker.record_visit(
                url=current_url,
                content_hash=hash_after,
                state_type=PageStateType.AFTER_CLICK,
                action=f"click:{selector}",
            )

            # Extract any new links from changed content
            if new_content:
                links = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('a[href]'))
                        .map(a => a.href)
                        .filter(href => href.startsWith('http'))
                """)
                # Filter to same domain
                from urllib.parse import urlparse
                base_domain = urlparse(self.base_url).netloc.lower()
                for link in links:
                    link_domain = urlparse(link).netloc.lower()
                    if link_domain == base_domain and link not in discovered_urls:
                        if self.state_tracker.should_visit(link):
                            discovered_urls.append(link)

            return ExplorationResult(
                success=True,
                discovered_urls=discovered_urls,
                new_content=new_content,
                content_hash=hash_after,
            )

        except Exception as e:
            logger.warning(f"Click failed on {selector}: {e}")
            return ExplorationResult(
                success=False,
                error=str(e),
            )

    def get_discovered_urls(
        self,
        decisions: list[ExplorationDecision],
    ) -> list[str]:
        """
        Extract all URLs from navigation decisions.

        Args:
            decisions: List of exploration decisions

        Returns:
            List of URLs to navigate to
        """
        urls = []
        for decision in decisions:
            if decision.action == "navigate" and decision.target:
                urls.append(decision.target)
        return urls

    def get_stats(self) -> dict:
        """Get exploration statistics."""
        return self.state_tracker.get_stats()

    def reset(self) -> None:
        """Reset exploration state."""
        self.state_tracker.clear()
