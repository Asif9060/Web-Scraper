"""
Navigation planning using LLM reasoning.

Assesses pages and prioritizes links based on
how likely they are to contain goal-relevant content.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from web_intel.llm import LocalLLM, GenerationConfig
from web_intel.llm.prompt_templates import PromptTemplate
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class NavigationPrompts:
    """Prompts for navigation decision-making."""

    ASSESS_PAGE = PromptTemplate(
        name="assess_page",
        system=(
            "You are a web navigation assistant. Analyze the page content and determine "
            "if it contains the information the user is looking for. Be precise and factual."
        ),
        user=(
            "Goal: {goal}\n\n"
            "Page Title: {title}\n"
            "Page URL: {url}\n\n"
            "Page Content Summary:\n{content}\n\n"
            "Questions:\n"
            "1. Does this page contain information relevant to the goal? (yes/partially/no)\n"
            "2. What relevant information was found? (brief list or 'none')\n"
            "3. Should we continue exploring from this page? (yes/no)\n\n"
            "Answer each question on a separate line:"
        ),
    )

    PRIORITIZE_LINKS = PromptTemplate(
        name="prioritize_links",
        system=(
            "You are a web navigation assistant. Given a goal and a list of links, "
            "identify which links are most likely to lead to the desired information. "
            "Consider link text, context, and typical website structures."
        ),
        user=(
            "Goal: {goal}\n\n"
            "Current Page: {current_page}\n\n"
            "Available Links:\n{links}\n\n"
            "Rate each link from 0-10 based on how likely it leads to goal-relevant content.\n"
            "Format: [link_number]: [score] - [brief reason]\n"
            "Only list the top 5 most promising links:"
        ),
    )

    SHOULD_EXPLORE = PromptTemplate(
        name="should_explore",
        system=(
            "You are a web navigation assistant deciding whether to explore a link. "
            "Make practical decisions based on the goal and link information."
        ),
        user=(
            "Goal: {goal}\n\n"
            "Link Text: {link_text}\n"
            "Link URL: {link_url}\n"
            "Context around link: {context}\n\n"
            "Question: Should we visit this link to find information about the goal?\n"
            "Answer with 'yes' or 'no' followed by a brief reason:"
        ),
    )

    EXTRACT_GOAL_INFO = PromptTemplate(
        name="extract_goal_info",
        system=(
            "You are an information extraction assistant. Extract only the specific "
            "information requested. Be precise and quote directly when possible."
        ),
        user=(
            "Goal: {goal}\n\n"
            "Page Content:\n{content}\n\n"
            "Extract the information relevant to the goal. If the information is not found, "
            "say 'Not found on this page'.\n\n"
            "Extracted information:"
        ),
    )


@dataclass
class LinkCandidate:
    """
    A candidate link for navigation.

    Contains the link details and relevance scoring.
    """

    url: str
    text: str
    context: str = ""  # Text around the link
    priority_score: float = 0.5
    reason: str = ""
    visited: bool = False

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinkCandidate):
            return False
        return self.url == other.url


class RelevanceLevel(str, Enum):
    """How relevant a page is to the goal."""

    HIGH = "yes"
    PARTIAL = "partially"
    LOW = "no"


@dataclass
class PageAssessment:
    """
    Assessment of a page's relevance to a goal.

    Produced by analyzing page content against the navigation goal.
    """

    url: str
    title: str
    relevance: RelevanceLevel
    found_info: list[str] = field(default_factory=list)
    should_continue: bool = True
    extracted_content: str = ""
    confidence: float = 0.5

    @property
    def is_relevant(self) -> bool:
        """Check if page has any relevant content."""
        return self.relevance in (RelevanceLevel.HIGH, RelevanceLevel.PARTIAL)

    @property
    def is_goal_complete(self) -> bool:
        """Check if goal appears to be satisfied."""
        return self.relevance == RelevanceLevel.HIGH and bool(self.extracted_content)


class NavigationPlanner:
    """
    Plans navigation paths using LLM reasoning.

    Assesses pages, prioritizes links, and determines
    the best exploration path to achieve a goal.

    Example:
        >>> planner = NavigationPlanner(llm)
        >>> assessment = planner.assess_page(
        ...     goal="Find pricing information",
        ...     title="Products - Company",
        ...     url="https://example.com/products",
        ...     content="Our products include..."
        ... )
        >>> if assessment.should_continue:
        ...     candidates = planner.prioritize_links(goal, links)
    """

    def __init__(
        self,
        llm: LocalLLM | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        """
        Initialize navigation planner.

        Args:
            llm: Local LLM instance for reasoning
            generation_config: Generation settings
        """
        self.llm = llm
        self.config = generation_config or GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,
        )

        self._assessment_cache: dict[str, PageAssessment] = {}
        logger.info("NavigationPlanner initialized")

    def assess_page(
        self,
        goal: str,
        title: str,
        url: str,
        content: str,
        use_cache: bool = True,
    ) -> PageAssessment:
        """
        Assess if a page contains goal-relevant information.

        Args:
            goal: Navigation goal description
            title: Page title
            url: Page URL
            content: Page content (summarized)
            use_cache: Whether to use cached assessments

        Returns:
            PageAssessment with relevance analysis
        """
        cache_key = f"{goal}:{url}"
        if use_cache and cache_key in self._assessment_cache:
            logger.debug(f"Using cached assessment for {url}")
            return self._assessment_cache[cache_key]

        if self.llm is None:
            return self._heuristic_assessment(goal, title, url, content)

        # Truncate content for prompt
        content_summary = content[:2000] if len(content) > 2000 else content

        prompt = NavigationPrompts.ASSESS_PAGE.format_user(
            goal=goal,
            title=title,
            url=url,
            content=content_summary,
        )

        try:
            result = self.llm.generate(
                prompt=prompt,
                config=self.config,
                system_prompt=NavigationPrompts.ASSESS_PAGE.system,
            )

            assessment = self._parse_assessment(result.text, url, title)

            # Extract specific goal info if relevant
            if assessment.is_relevant:
                assessment.extracted_content = self._extract_goal_info(
                    goal, content_summary
                )

            if use_cache:
                self._assessment_cache[cache_key] = assessment

            return assessment

        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}, using heuristic")
            return self._heuristic_assessment(goal, title, url, content)

    def _parse_assessment(
        self,
        response: str,
        url: str,
        title: str,
    ) -> PageAssessment:
        """Parse LLM response into PageAssessment."""
        lines = response.strip().split("\n")

        relevance = RelevanceLevel.LOW
        found_info = []
        should_continue = True

        for line in lines:
            line_lower = line.lower().strip()

            # Parse relevance
            if "yes" in line_lower and "relevant" in line_lower:
                relevance = RelevanceLevel.HIGH
            elif "partially" in line_lower:
                relevance = RelevanceLevel.PARTIAL
            elif "no" in line_lower and ("relevant" in line_lower or line_lower.startswith("1")):
                relevance = RelevanceLevel.LOW

            # Parse found info
            if line.startswith("2.") or "found:" in line_lower:
                info_part = line.split(":", 1)[-1].strip()
                if info_part.lower() not in ("none", "n/a", ""):
                    found_info = [i.strip()
                                  for i in info_part.split(",") if i.strip()]

            # Parse continue decision
            if line.startswith("3.") or "continue" in line_lower:
                should_continue = "yes" in line_lower

        confidence = 0.8 if relevance == RelevanceLevel.HIGH else 0.5 if relevance == RelevanceLevel.PARTIAL else 0.3

        return PageAssessment(
            url=url,
            title=title,
            relevance=relevance,
            found_info=found_info,
            should_continue=should_continue,
            confidence=confidence,
        )

    def _heuristic_assessment(
        self,
        goal: str,
        title: str,
        url: str,
        content: str,
    ) -> PageAssessment:
        """Fallback heuristic assessment without LLM."""
        goal_terms = set(goal.lower().split())
        content_lower = content.lower()
        title_lower = title.lower()

        # Count goal term matches
        matches = sum(
            1 for term in goal_terms if term in content_lower or term in title_lower)
        match_ratio = matches / max(len(goal_terms), 1)

        if match_ratio >= 0.6:
            relevance = RelevanceLevel.HIGH
        elif match_ratio >= 0.3:
            relevance = RelevanceLevel.PARTIAL
        else:
            relevance = RelevanceLevel.LOW

        return PageAssessment(
            url=url,
            title=title,
            relevance=relevance,
            found_info=[],
            should_continue=True,
            confidence=0.4,  # Lower confidence for heuristic
        )

    def prioritize_links(
        self,
        goal: str,
        current_page: str,
        links: list[LinkCandidate],
        max_links: int = 10,
    ) -> list[LinkCandidate]:
        """
        Prioritize links based on goal relevance.

        Args:
            goal: Navigation goal
            current_page: Current page title/URL
            links: Available links to prioritize
            max_links: Maximum links to return

        Returns:
            Links sorted by priority (highest first)
        """
        if not links:
            return []

        # Filter already visited
        unvisited = [l for l in links if not l.visited]
        if not unvisited:
            return []

        if self.llm is None:
            return self._heuristic_prioritize(goal, unvisited)[:max_links]

        # Format links for prompt
        link_descriptions = []
        for i, link in enumerate(unvisited[:20], 1):  # Limit to 20 for prompt
            desc = f"{i}. [{link.text}] -> {link.url}"
            if link.context:
                desc += f" (context: {link.context[:50]}...)"
            link_descriptions.append(desc)

        links_text = "\n".join(link_descriptions)

        prompt = NavigationPrompts.PRIORITIZE_LINKS.format_user(
            goal=goal,
            current_page=current_page,
            links=links_text,
        )

        try:
            result = self.llm.generate(
                prompt=prompt,
                config=self.config,
                system_prompt=NavigationPrompts.PRIORITIZE_LINKS.system,
            )

            prioritized = self._parse_link_priorities(
                result.text, unvisited[:20])
            return prioritized[:max_links]

        except Exception as e:
            logger.warning(f"LLM prioritization failed: {e}, using heuristic")
            return self._heuristic_prioritize(goal, unvisited)[:max_links]

    def _parse_link_priorities(
        self,
        response: str,
        links: list[LinkCandidate],
    ) -> list[LinkCandidate]:
        """Parse LLM response to extract link priorities."""
        # Pattern: [number]: [score] - [reason]
        pattern = r"(\d+)[:\.]?\s*(\d+(?:\.\d+)?)\s*[-–]?\s*(.+)?"

        scored_links = []
        for match in re.finditer(pattern, response):
            try:
                link_num = int(match.group(1)) - 1
                score = float(match.group(2)) / 10.0  # Normalize to 0-1
                reason = match.group(3) or ""

                if 0 <= link_num < len(links):
                    link = links[link_num]
                    link.priority_score = score
                    link.reason = reason.strip()
                    scored_links.append(link)
            except (ValueError, IndexError):
                continue

        # Sort by score descending
        scored_links.sort(key=lambda l: l.priority_score, reverse=True)
        return scored_links

    def _heuristic_prioritize(
        self,
        goal: str,
        links: list[LinkCandidate],
    ) -> list[LinkCandidate]:
        """Fallback heuristic link prioritization."""
        goal_terms = set(goal.lower().split())

        for link in links:
            link_text_lower = link.text.lower()
            url_lower = link.url.lower()
            context_lower = link.context.lower() if link.context else ""

            # Score based on keyword matches
            score = 0.0
            for term in goal_terms:
                if term in link_text_lower:
                    score += 0.3
                if term in url_lower:
                    score += 0.2
                if term in context_lower:
                    score += 0.1

            # Boost common navigation patterns
            nav_patterns = {
                "pricing": 0.3,
                "price": 0.3,
                "contact": 0.2,
                "about": 0.1,
                "product": 0.2,
                "service": 0.2,
                "feature": 0.2,
                "faq": 0.15,
                "help": 0.1,
                "support": 0.1,
                "docs": 0.15,
                "documentation": 0.15,
            }

            for pattern, boost in nav_patterns.items():
                if pattern in goal.lower() and pattern in link_text_lower:
                    score += boost

            link.priority_score = min(score, 1.0)
            link.reason = "Heuristic match"

        links.sort(key=lambda l: l.priority_score, reverse=True)
        return links

    def _extract_goal_info(self, goal: str, content: str) -> str:
        """Extract goal-specific information from content."""
        if self.llm is None:
            return ""

        prompt = NavigationPrompts.EXTRACT_GOAL_INFO.format_user(
            goal=goal,
            content=content[:3000],  # Limit content
        )

        try:
            result = self.llm.generate(
                prompt=prompt,
                config=GenerationConfig(max_new_tokens=300, temperature=0.1),
                system_prompt=NavigationPrompts.EXTRACT_GOAL_INFO.system,
            )

            extracted = result.text.strip()
            if "not found" in extracted.lower():
                return ""
            return extracted

        except Exception as e:
            logger.warning(f"Goal info extraction failed: {e}")
            return ""

    def should_visit_link(
        self,
        goal: str,
        link: LinkCandidate,
    ) -> tuple[bool, str]:
        """
        Decide if a specific link should be visited.

        Args:
            goal: Navigation goal
            link: Link to evaluate

        Returns:
            Tuple of (should_visit, reason)
        """
        if link.visited:
            return False, "Already visited"

        if self.llm is None:
            # Simple heuristic
            goal_terms = goal.lower().split()
            link_lower = f"{link.text} {link.url}".lower()
            if any(term in link_lower for term in goal_terms):
                return True, "Contains goal keywords"
            return link.priority_score >= 0.3, "Heuristic score"

        prompt = NavigationPrompts.SHOULD_EXPLORE.format_user(
            goal=goal,
            link_text=link.text,
            link_url=link.url,
            context=link.context[:200] if link.context else "No context available",
        )

        try:
            result = self.llm.generate(
                prompt=prompt,
                config=GenerationConfig(max_new_tokens=50, temperature=0.1),
                system_prompt=NavigationPrompts.SHOULD_EXPLORE.system,
            )

            response_lower = result.text.lower().strip()
            should_visit = response_lower.startswith("yes")

            # Extract reason
            reason = result.text.split(
                ":", 1)[-1].strip() if ":" in result.text else result.text

            return should_visit, reason[:100]

        except Exception as e:
            logger.warning(f"Visit decision failed: {e}")
            return link.priority_score >= 0.4, "Fallback to score threshold"

    def clear_cache(self) -> None:
        """Clear assessment cache."""
        self._assessment_cache.clear()

    def __repr__(self) -> str:
        return f"NavigationPlanner(cache_size={len(self._assessment_cache)})"
