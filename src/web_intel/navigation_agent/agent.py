"""
LLM-powered navigation agent for goal-directed browsing.

Coordinates browser automation with intelligent decision-making
to find specific information on websites.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Awaitable
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page

from web_intel.config import Settings
from web_intel.llm import LocalLLM, GenerationConfig
from web_intel.browser.page_context import PageContext, PageContent
from web_intel.browser.actions import extract_links, extract_text_content
from web_intel.navigation_agent.planner import (
    NavigationPlanner,
    LinkCandidate,
    PageAssessment,
    RelevanceLevel,
)
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class GoalStatus(str, Enum):
    """Status of a navigation goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class NavigationGoal:
    """
    A goal for the navigation agent to achieve.

    Describes what information the agent should find.

    Example:
        >>> goal = NavigationGoal(
        ...     objective="Find pricing information for the enterprise plan",
        ...     keywords=["pricing", "enterprise", "cost"],
        ...     max_pages=10,
        ... )
    """

    objective: str
    keywords: list[str] = field(default_factory=list)
    max_pages: int = 10
    max_depth: int = 3
    timeout_seconds: int = 300
    stop_on_first_match: bool = False

    def __post_init__(self):
        # Extract keywords from objective if not provided
        if not self.keywords:
            # Simple keyword extraction
            stop_words = {"the", "a", "an", "is", "are", "for", "to",
                          "of", "and", "or", "in", "on", "find", "get", "show", "what"}
            words = self.objective.lower().split()
            self.keywords = [
                w for w in words if w not in stop_words and len(w) > 2]


@dataclass
class NavigationStep:
    """
    A single step in navigation execution.

    Records what was done and what was found.
    """

    step_number: int
    url: str
    title: str
    action: str  # "navigate", "assess", "extract"
    assessment: PageAssessment | None = None
    extracted_info: str = ""
    links_found: int = 0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


class StepOutcome(str, Enum):
    """Outcome of a navigation step."""

    SUCCESS = "success"
    PARTIAL = "partial"
    NO_INFO = "no_info"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class NavigationPlan:
    """
    A plan for achieving a navigation goal.

    Contains prioritized links to visit.
    """

    goal: NavigationGoal
    current_url: str
    candidates: list[LinkCandidate] = field(default_factory=list)
    visited_count: int = 0

    def next_link(self) -> LinkCandidate | None:
        """Get next unvisited link with highest priority."""
        for candidate in self.candidates:
            if not candidate.visited:
                return candidate
        return None

    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited."""
        for candidate in self.candidates:
            if candidate.url == url:
                candidate.visited = True
                break
        self.visited_count += 1


@dataclass
class NavigationResult:
    """
    Result of a navigation session.

    Contains all findings and execution history.
    """

    goal: NavigationGoal
    status: GoalStatus
    found_info: list[str] = field(default_factory=list)
    relevant_urls: list[str] = field(default_factory=list)
    steps: list[NavigationStep] = field(default_factory=list)
    pages_visited: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Get duration of navigation."""
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def is_successful(self) -> bool:
        """Check if goal was achieved."""
        return self.status == GoalStatus.ACHIEVED

    def get_combined_findings(self) -> str:
        """Get all findings as combined text."""
        return "\n\n".join(self.found_info) if self.found_info else ""

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        status_emoji = {
            GoalStatus.ACHIEVED: "✓",
            GoalStatus.FAILED: "✗",
            GoalStatus.TIMEOUT: "⏱",
            GoalStatus.IN_PROGRESS: "→",
            GoalStatus.PENDING: "○",
        }

        lines = [
            f"{status_emoji.get(self.status, '?')} Goal: {self.goal.objective}",
            f"Status: {self.status.value}",
            f"Pages visited: {self.pages_visited}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]

        if self.relevant_urls:
            lines.append(f"Relevant pages: {len(self.relevant_urls)}")

        if self.found_info:
            lines.append(f"Findings: {len(self.found_info)} items")

        return "\n".join(lines)


# Type for page callback
PageCallback = Callable[[PageContent], Awaitable[None]]


class NavigationAgent:
    """
    LLM-powered agent for goal-directed web navigation.

    Uses intelligent reasoning to navigate websites and find
    specific information based on user goals.

    Example:
        >>> agent = NavigationAgent.from_settings(settings)
        >>>
        >>> goal = NavigationGoal(
        ...     objective="Find the company's pricing plans",
        ...     max_pages=15,
        ... )
        >>>
        >>> async with BrowserManager(settings.browser) as browser:
        ...     async with await browser.new_context() as context:
        ...         page = await context.new_page()
        ...         page_ctx = PageContext(page)
        ...
        ...         result = await agent.navigate(
        ...             page_ctx=page_ctx,
        ...             start_url="https://example.com",
        ...             goal=goal,
        ...         )
        ...
        ...         if result.is_successful:
        ...             print(result.get_combined_findings())
    """

    def __init__(
        self,
        planner: NavigationPlanner | None = None,
        llm: LocalLLM | None = None,
        max_content_length: int = 5000,
        link_context_chars: int = 100,
    ) -> None:
        """
        Initialize navigation agent.

        Args:
            planner: Navigation planner instance
            llm: Local LLM for reasoning
            max_content_length: Max content to analyze per page
            link_context_chars: Context chars around links
        """
        self.llm = llm
        self.planner = planner or NavigationPlanner(llm=llm)
        self.max_content_length = max_content_length
        self.link_context_chars = link_context_chars

        # Statistics
        self._total_navigations = 0
        self._successful_navigations = 0

        logger.info("NavigationAgent initialized")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        llm: LocalLLM | None = None,
    ) -> "NavigationAgent":
        """
        Create NavigationAgent from settings.

        Args:
            settings: Application settings
            llm: Optional pre-configured LLM

        Returns:
            Configured NavigationAgent
        """
        if llm is None and settings.local_llm.enabled:
            llm = LocalLLM.from_settings(settings)

        planner = NavigationPlanner(llm=llm)

        return cls(planner=planner, llm=llm)

    async def navigate(
        self,
        page_ctx: PageContext,
        start_url: str,
        goal: NavigationGoal,
        on_page: PageCallback | None = None,
    ) -> NavigationResult:
        """
        Navigate website to achieve goal.

        Args:
            page_ctx: Browser page context
            start_url: Starting URL
            goal: Navigation goal to achieve
            on_page: Optional callback for each page

        Returns:
            NavigationResult with findings
        """
        self._total_navigations += 1

        result = NavigationResult(
            goal=goal,
            status=GoalStatus.IN_PROGRESS,
        )

        visited_urls: set[str] = set()
        plan = NavigationPlan(goal=goal, current_url=start_url)
        base_domain = urlparse(start_url).netloc

        logger.info(f"Starting navigation: {goal.objective}")

        try:
            # Navigate to start URL
            await page_ctx.navigate(start_url)
            current_depth = 0

            while result.pages_visited < goal.max_pages:
                current_url = page_ctx.page.url

                # Skip if already visited
                if current_url in visited_urls:
                    next_link = plan.next_link()
                    if next_link:
                        await page_ctx.navigate(next_link.url)
                        plan.mark_visited(next_link.url)
                        continue
                    else:
                        break

                visited_urls.add(current_url)
                result.pages_visited += 1

                # Extract page content
                content = await self._extract_page_content(page_ctx)
                title = await page_ctx.page.title()

                # Callback
                if on_page:
                    page_content = PageContent(
                        url=current_url,
                        title=title,
                        html="",
                        text_content=content[:1000],
                        status_code=200,
                    )
                    await on_page(page_content)

                # Assess page
                assessment = self.planner.assess_page(
                    goal=goal.objective,
                    title=title,
                    url=current_url,
                    content=content,
                )

                # Record step
                step = NavigationStep(
                    step_number=result.pages_visited,
                    url=current_url,
                    title=title,
                    action="assess",
                    assessment=assessment,
                )

                # Collect findings
                if assessment.is_relevant:
                    result.relevant_urls.append(current_url)

                    if assessment.extracted_content:
                        result.found_info.append(
                            f"[{title}]\n{assessment.extracted_content}"
                        )
                        step.extracted_info = assessment.extracted_content

                    if goal.stop_on_first_match and assessment.is_goal_complete:
                        logger.info(f"Goal achieved at: {current_url}")
                        result.status = GoalStatus.ACHIEVED
                        result.steps.append(step)
                        break

                result.steps.append(step)

                # Check timeout
                if result.duration_seconds > goal.timeout_seconds:
                    logger.warning("Navigation timeout")
                    result.status = GoalStatus.TIMEOUT
                    break

                # Get links from current page
                if assessment.should_continue and current_depth < goal.max_depth:
                    links = await self._extract_links(page_ctx, base_domain)
                    step.links_found = len(links)

                    # Filter unvisited
                    new_links = [l for l in links if l.url not in visited_urls]

                    # Prioritize
                    prioritized = self.planner.prioritize_links(
                        goal=goal.objective,
                        current_page=f"{title} ({current_url})",
                        links=new_links,
                    )

                    # Add to plan
                    for link in prioritized:
                        if link.url not in {c.url for c in plan.candidates}:
                            plan.candidates.append(link)

                    # Re-sort all candidates
                    plan.candidates.sort(
                        key=lambda l: l.priority_score, reverse=True)

                # Navigate to next promising link
                next_link = plan.next_link()
                if next_link:
                    logger.debug(
                        f"Navigating to: {next_link.text} ({next_link.priority_score:.2f})"
                    )
                    try:
                        await page_ctx.navigate(next_link.url)
                        plan.mark_visited(next_link.url)
                        current_depth += 1
                    except Exception as e:
                        logger.warning(f"Navigation failed: {e}")
                        plan.mark_visited(next_link.url)
                        continue
                else:
                    logger.info("No more links to explore")
                    break

            # Finalize result
            if result.status == GoalStatus.IN_PROGRESS:
                if result.found_info:
                    result.status = GoalStatus.ACHIEVED
                    self._successful_navigations += 1
                else:
                    result.status = GoalStatus.FAILED

            result.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Navigation error: {e}")
            result.status = GoalStatus.FAILED
            result.completed_at = datetime.now(timezone.utc)

        logger.info(
            f"Navigation complete: {result.status.value}, "
            f"pages={result.pages_visited}, findings={len(result.found_info)}"
        )

        return result

    async def quick_search(
        self,
        page_ctx: PageContext,
        start_url: str,
        query: str,
        max_pages: int = 5,
    ) -> str:
        """
        Quick search for specific information.

        Simplified interface for common searches.

        Args:
            page_ctx: Browser page context
            start_url: Starting URL
            query: What to search for
            max_pages: Maximum pages to check

        Returns:
            Found information or empty string
        """
        goal = NavigationGoal(
            objective=query,
            max_pages=max_pages,
            stop_on_first_match=True,
        )

        result = await self.navigate(page_ctx, start_url, goal)

        return result.get_combined_findings()

    async def _extract_page_content(self, page_ctx: PageContext) -> str:
        """Extract text content from page."""
        try:
            content = await extract_text_content(page_ctx.page)
            return content[: self.max_content_length]
        except Exception as e:
            logger.warning(f"Content extraction failed: {e}")
            return ""

    async def _extract_links(
        self,
        page_ctx: PageContext,
        base_domain: str,
    ) -> list[LinkCandidate]:
        """Extract and filter links from page."""
        candidates = []

        try:
            raw_links = await extract_links(page_ctx.page)

            for link in raw_links:
                url = link.get("href", "")
                text = link.get("text", "").strip()

                if not url or not text:
                    continue

                # Make absolute
                if not url.startswith(("http://", "https://")):
                    url = urljoin(page_ctx.page.url, url)

                # Filter same domain
                link_domain = urlparse(url).netloc
                if link_domain != base_domain:
                    continue

                # Skip common non-content links
                skip_patterns = [
                    "/login", "/signup", "/register", "/cart",
                    "/checkout", "javascript:", "mailto:", "tel:",
                    "#", "/search"
                ]
                if any(p in url.lower() for p in skip_patterns):
                    continue

                candidates.append(
                    LinkCandidate(
                        url=url,
                        text=text[:100],
                        context="",  # Could extract surrounding text
                    )
                )

        except Exception as e:
            logger.warning(f"Link extraction failed: {e}")

        return candidates

    def get_stats(self) -> dict:
        """Get navigation statistics."""
        return {
            "total_navigations": self._total_navigations,
            "successful_navigations": self._successful_navigations,
            "success_rate": (
                self._successful_navigations / max(self._total_navigations, 1)
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_navigations = 0
        self._successful_navigations = 0

    def __repr__(self) -> str:
        return (
            f"NavigationAgent(navigations={self._total_navigations}, "
            f"success_rate={self.get_stats()['success_rate']:.1%})"
        )
