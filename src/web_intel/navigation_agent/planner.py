"""
Navigation planning with tiered decision system.

Uses a three-tier approach to reduce LLM usage:
1. Fast heuristics (keyword overlap, URL patterns)
2. Embedding similarity (optional, fast)
3. LLM reasoning (only when confidence is low)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from web_intel.llm import LocalLLM, GenerationConfig
from web_intel.llm.prompt_templates import PromptTemplate
from web_intel.config.settings import NavigationSettings
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Navigation Decision
# ============================================================================


@dataclass
class NavigationDecision:
    """
    Result of a navigation decision with confidence and rationale.

    Captures the selected action, how confident the system is,
    and why this decision was made.
    """

    selected_action: str  # e.g., "visit", "skip", "explore"
    confidence: float  # 0.0 to 1.0
    rationale: str  # Human-readable explanation
    tier_used: str = "heuristic"  # "heuristic", "embedding", "llm"
    scores: dict = field(default_factory=dict)  # Detailed scoring breakdown

    @property
    def is_confident(self) -> bool:
        """Check if decision meets confidence threshold."""
        return self.confidence >= 0.7

    def __repr__(self) -> str:
        return (
            f"NavigationDecision(action={self.selected_action}, "
            f"confidence={self.confidence:.2f}, tier={self.tier_used})"
        )


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
    Plans navigation paths using a tiered decision system.

    Uses three tiers to minimize expensive LLM calls:
    - Tier 1: Fast heuristics (keyword overlap, URL patterns)
    - Tier 2: Embedding similarity (optional, still fast)
    - Tier 3: LLM reasoning (only when confidence is low)

    Example:
        >>> planner = NavigationPlanner(llm, settings.navigation)
        >>> assessment = planner.assess_page(
        ...     goal="Find pricing information",
        ...     title="Products - Company",
        ...     url="https://example.com/products",
        ...     content="Our products include..."
        ... )
        >>> if assessment.should_continue:
        ...     candidates, decision = planner.prioritize_links(goal, links)
        ...     print(f"Decision: {decision.rationale}")
    """

    # Common URL patterns that indicate specific content types
    URL_PATTERNS = {
        "pricing": ["pricing", "price", "plans", "cost", "subscription", "packages"],
        "contact": ["contact", "reach", "get-in-touch", "support"],
        "about": ["about", "team", "company", "who-we-are", "our-story"],
        "products": ["products", "solutions", "services", "offerings", "features"],
        "documentation": ["docs", "documentation", "guide", "help", "faq", "how-to"],
        "blog": ["blog", "news", "articles", "posts", "updates"],
        "legal": ["privacy", "terms", "legal", "policy", "gdpr", "cookies"],
    }

    # Negative URL patterns (typically low value for navigation)
    NEGATIVE_URL_PATTERNS = [
        "login", "logout", "signin", "signout", "register", "signup",
        "cart", "checkout", "account", "profile", "settings",
        "share", "print", "download", "subscribe", "newsletter",
    ]

    def __init__(
        self,
        llm: LocalLLM | None = None,
        nav_settings: NavigationSettings | None = None,
        generation_config: GenerationConfig | None = None,
        embedder_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        """
        Initialize navigation planner.

        Args:
            llm: Local LLM instance for reasoning (Tier 3)
            nav_settings: Navigation configuration
            generation_config: LLM generation settings
            embedder_fn: Function to embed text for similarity (Tier 2)
        """
        self.llm = llm
        self.settings = nav_settings or NavigationSettings()
        self.config = generation_config or GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,
        )
        self.embedder_fn = embedder_fn

        self._assessment_cache: dict[str, PageAssessment] = {}
        self._stats = {
            "heuristic_decisions": 0,
            "embedding_decisions": 0,
            "llm_decisions": 0,
        }

        logger.info(
            f"NavigationPlanner initialized (llm_enabled={self.settings.use_llm_for_navigation})"
        )

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

        Uses tiered approach:
        1. Heuristic assessment (always)
        2. LLM assessment (only if confidence < threshold AND enabled)

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

        # Tier 1: Always start with heuristic
        assessment = self._heuristic_assessment(goal, title, url, content)
        self._stats["heuristic_decisions"] += 1

        # Tier 3: Use LLM only if enabled AND confidence is low
        if (
            self.settings.use_llm_for_navigation
            and self.llm is not None
            and assessment.confidence < self.settings.llm_confidence_threshold
        ):
            logger.debug(
                f"Low confidence ({assessment.confidence:.2f}), using LLM")
            assessment = self._llm_assessment(goal, title, url, content)
            self._stats["llm_decisions"] += 1

        if use_cache:
            self._assessment_cache[cache_key] = assessment

        return assessment

    def _llm_assessment(
        self,
        goal: str,
        title: str,
        url: str,
        content: str,
    ) -> PageAssessment:
        """Perform LLM-based page assessment."""
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
        """
        Enhanced heuristic assessment using keyword overlap and patterns.

        Scoring breakdown:
        - Exact keyword matches in title: high weight
        - Exact keyword matches in content: medium weight
        - Partial/fuzzy matches: lower weight
        - URL pattern matches: bonus
        """
        goal_lower = goal.lower()
        goal_terms = self._extract_key_terms(goal_lower)
        content_lower = content.lower()
        title_lower = title.lower()
        url_lower = url.lower()

        scores = {
            "title_exact": 0.0,
            "content_exact": 0.0,
            "url_pattern": 0.0,
            "density": 0.0,
        }

        # Title matches (high value)
        title_matches = sum(1 for term in goal_terms if term in title_lower)
        scores["title_exact"] = min(
            title_matches / max(len(goal_terms), 1), 1.0) * 0.4

        # Content matches
        content_matches = sum(
            1 for term in goal_terms if term in content_lower)
        scores["content_exact"] = min(
            content_matches / max(len(goal_terms), 1), 1.0) * 0.3

        # URL pattern bonus
        for category, patterns in self.URL_PATTERNS.items():
            if any(p in goal_lower for p in patterns):
                if any(p in url_lower for p in patterns):
                    scores["url_pattern"] = 0.2
                    break

        # Keyword density in first 500 chars (indicates prominent placement)
        first_content = content_lower[:500]
        density_matches = sum(
            1 for term in goal_terms if term in first_content)
        scores["density"] = min(
            density_matches / max(len(goal_terms), 1), 1.0) * 0.1

        # Calculate total score
        total_score = sum(scores.values())

        # Determine relevance level
        if total_score >= 0.6:
            relevance = RelevanceLevel.HIGH
            confidence = min(0.9, 0.7 + total_score * 0.2)
        elif total_score >= 0.3:
            relevance = RelevanceLevel.PARTIAL
            confidence = 0.5 + total_score * 0.3
        else:
            relevance = RelevanceLevel.LOW
            confidence = max(0.3, total_score + 0.2)

        return PageAssessment(
            url=url,
            title=title,
            relevance=relevance,
            found_info=[],
            should_continue=True,
            confidence=confidence,
        )

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract meaningful terms from text, filtering stop words."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "and", "but", "if", "or", "because", "until", "while",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "i", "me", "my", "myself", "we", "our", "ours", "you",
            "your", "yours", "he", "him", "his", "she", "her", "hers",
            "it", "its", "they", "them", "their", "find", "get", "look",
            "want", "need", "information", "about", "page", "website",
        }

        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        return [w for w in words if w not in stop_words]

    def prioritize_links(
        self,
        goal: str,
        current_page: str,
        links: list[LinkCandidate],
        max_links: int = 10,
    ) -> tuple[list[LinkCandidate], NavigationDecision]:
        """
        Prioritize links using tiered decision system.

        Tiers:
        1. Heuristic scoring (always used)
        2. Embedding similarity boost (if enabled and available)
        3. LLM re-ranking (only if confidence < threshold AND enabled)

        Args:
            goal: Navigation goal
            current_page: Current page title/URL
            links: Available links to prioritize
            max_links: Maximum links to return

        Returns:
            Tuple of (prioritized links, decision metadata)
        """
        if not links:
            return [], NavigationDecision(
                selected_action="no_links",
                confidence=1.0,
                rationale="No links available to evaluate",
                tier_used="none",
            )

        # Filter already visited
        unvisited = [l for l in links if not l.visited]
        if not unvisited:
            return [], NavigationDecision(
                selected_action="all_visited",
                confidence=1.0,
                rationale="All links have been visited",
                tier_used="none",
            )

        # Limit to max_links_to_evaluate
        candidates = unvisited[:self.settings.max_links_to_evaluate]

        # Tier 1: Heuristic scoring (always)
        scored_links, heuristic_confidence = self._heuristic_prioritize(
            goal, candidates)
        self._stats["heuristic_decisions"] += 1

        tier_used = "heuristic"
        final_confidence = heuristic_confidence

        # Tier 2: Embedding similarity boost (if enabled and available)
        if (
            self.settings.use_embeddings_for_ranking
            and self.embedder_fn is not None
        ):
            scored_links, embed_confidence = self._embedding_boost(
                goal, scored_links
            )
            self._stats["embedding_decisions"] += 1
            tier_used = "embedding"
            final_confidence = max(heuristic_confidence, embed_confidence)

        # Tier 3: LLM re-ranking (only if enabled AND confidence is low)
        if (
            self.settings.use_llm_for_navigation
            and self.llm is not None
            and final_confidence < self.settings.llm_confidence_threshold
        ):
            logger.debug(
                f"Low confidence ({final_confidence:.2f}), using LLM for ranking")
            scored_links = self._llm_prioritize(
                goal, current_page, scored_links)
            self._stats["llm_decisions"] += 1
            tier_used = "llm"
            final_confidence = 0.85  # LLM decisions are higher confidence

        # Build decision
        top_link = scored_links[0] if scored_links else None
        decision = NavigationDecision(
            selected_action="explore" if top_link else "stop",
            confidence=final_confidence,
            rationale=self._build_prioritization_rationale(
                goal, top_link, tier_used, final_confidence
            ),
            tier_used=tier_used,
            scores={
                "top_score": top_link.priority_score if top_link else 0.0,
                "candidates_evaluated": len(candidates),
                "above_threshold": sum(1 for l in scored_links if l.priority_score >= 0.5),
            },
        )

        return scored_links[:max_links], decision

    def _build_prioritization_rationale(
        self,
        goal: str,
        top_link: LinkCandidate | None,
        tier_used: str,
        confidence: float,
    ) -> str:
        """Build human-readable rationale for prioritization decision."""
        if not top_link:
            return "No promising links found for the goal"

        tier_desc = {
            "heuristic": "keyword matching and URL patterns",
            "embedding": "semantic similarity analysis",
            "llm": "LLM reasoning",
        }

        return (
            f"Selected '{top_link.text[:50]}' (score: {top_link.priority_score:.2f}) "
            f"using {tier_desc.get(tier_used, tier_used)}. "
            f"Confidence: {confidence:.0%}"
        )

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
    ) -> tuple[list[LinkCandidate], float]:
        """
        Enhanced heuristic link prioritization.

        Scoring factors:
        - Keyword overlap with goal
        - URL pattern matching
        - Negative pattern penalties
        - Link text quality indicators

        Returns:
            Tuple of (sorted links, confidence score)
        """
        goal_terms = self._extract_key_terms(goal.lower())
        goal_lower = goal.lower()

        max_score = 0.0

        for link in links:
            link_text_lower = link.text.lower()
            url_lower = link.url.lower()
            context_lower = link.context.lower() if link.context else ""

            scores = {
                "keyword": 0.0,
                "url_pattern": 0.0,
                "negative": 0.0,
                "quality": 0.0,
            }

            # Keyword overlap scoring
            link_terms = self._extract_key_terms(link_text_lower)
            term_overlap = len(set(goal_terms) & set(link_terms))
            if goal_terms:
                scores["keyword"] = (
                    term_overlap / len(goal_terms)) * self.settings.keyword_weight

            # Direct term presence in link text
            direct_matches = sum(
                1 for term in goal_terms if term in link_text_lower)
            scores["keyword"] += (direct_matches /
                                  max(len(goal_terms), 1)) * 0.2

            # URL pattern matching
            for category, patterns in self.URL_PATTERNS.items():
                if any(p in goal_lower for p in patterns):
                    url_matches = sum(1 for p in patterns if p in url_lower)
                    text_matches = sum(
                        1 for p in patterns if p in link_text_lower)
                    if url_matches or text_matches:
                        scores["url_pattern"] = self.settings.url_weight * min(
                            (url_matches + text_matches) / len(patterns), 1.0
                        )
                        break

            # Negative patterns (penalty)
            negative_hits = sum(
                1 for p in self.NEGATIVE_URL_PATTERNS if p in url_lower)
            scores["negative"] = -0.2 * min(negative_hits, 2)

            # Link text quality (prefer descriptive links)
            if len(link.text) > 3 and not link.text.startswith(("http", "www")):
                scores["quality"] = 0.1
            if link.context and any(term in context_lower for term in goal_terms):
                scores["quality"] += 0.1

            # Calculate total score
            total_score = max(0.0, min(1.0, sum(scores.values())))
            link.priority_score = total_score
            link.reason = f"Heuristic: keyword={scores['keyword']:.2f}, url={scores['url_pattern']:.2f}"

            max_score = max(max_score, total_score)

        # Sort by score descending
        links.sort(key=lambda l: l.priority_score, reverse=True)

        # Confidence based on score distribution
        high_scoring = sum(1 for l in links if l.priority_score >= 0.5)
        confidence = min(0.9, max_score +
                         (high_scoring / max(len(links), 1)) * 0.2)

        return links, confidence

    def _embedding_boost(
        self,
        goal: str,
        links: list[LinkCandidate],
    ) -> tuple[list[LinkCandidate], float]:
        """
        Apply embedding similarity boost to link scores.

        Uses cosine similarity between goal and link text embeddings.

        Returns:
            Tuple of (re-scored links, confidence score)
        """
        if not self.embedder_fn:
            return links, 0.5

        try:
            # Embed goal
            goal_embedding = self.embedder_fn(goal)

            max_similarity = 0.0

            for link in links:
                # Combine link text and context for embedding
                link_text = f"{link.text} {link.context[:100] if link.context else ''}"
                link_embedding = self.embedder_fn(link_text)

                # Cosine similarity
                similarity = self._cosine_similarity(
                    goal_embedding, link_embedding)
                max_similarity = max(max_similarity, similarity)

                # Blend embedding score with existing score
                embedding_score = similarity * self.settings.embedding_weight
                link.priority_score = (
                    link.priority_score * (1 - self.settings.embedding_weight)
                    + embedding_score
                )
                link.reason += f", embed={similarity:.2f}"

            # Re-sort after embedding boost
            links.sort(key=lambda l: l.priority_score, reverse=True)

            return links, min(0.9, max_similarity + 0.3)

        except Exception as e:
            logger.warning(f"Embedding boost failed: {e}")
            return links, 0.5

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _llm_prioritize(
        self,
        goal: str,
        current_page: str,
        links: list[LinkCandidate],
    ) -> list[LinkCandidate]:
        """
        Use LLM to re-rank links (Tier 3).

        Only called when heuristic confidence is low.
        """
        if not self.llm:
            return links

        # Format links for prompt
        link_descriptions = []
        for i, link in enumerate(links[:15], 1):  # Limit to 15 for prompt
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

            return self._parse_link_priorities(result.text, links[:15])

        except Exception as e:
            logger.warning(f"LLM prioritization failed: {e}")
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
    ) -> NavigationDecision:
        """
        Decide if a specific link should be visited using tiered system.

        Args:
            goal: Navigation goal
            link: Link to evaluate

        Returns:
            NavigationDecision with action, confidence, and rationale
        """
        if link.visited:
            return NavigationDecision(
                selected_action="skip",
                confidence=1.0,
                rationale="Link already visited",
                tier_used="none",
            )

        # Tier 1: Heuristic evaluation
        goal_terms = self._extract_key_terms(goal.lower())
        link_text = f"{link.text} {link.url}".lower()

        keyword_matches = sum(1 for term in goal_terms if term in link_text)
        keyword_score = keyword_matches / max(len(goal_terms), 1)

        # URL pattern check
        url_score = 0.0
        url_lower = link.url.lower()
        for category, patterns in self.URL_PATTERNS.items():
            if any(p in goal.lower() for p in patterns):
                if any(p in url_lower for p in patterns):
                    url_score = 0.3
                    break

        # Negative pattern check
        negative_penalty = -0.3 if any(
            p in url_lower for p in self.NEGATIVE_URL_PATTERNS
        ) else 0.0

        heuristic_score = min(1.0, max(0.0,
                                       keyword_score * 0.5 + url_score + link.priority_score * 0.2 + negative_penalty
                                       ))

        confidence = 0.5 + heuristic_score * 0.4
        self._stats["heuristic_decisions"] += 1

        # Determine action based on heuristic score
        if heuristic_score >= 0.4:
            should_visit = True
            tier_used = "heuristic"
            rationale = f"Keyword match score: {keyword_score:.0%}, URL pattern: {url_score:.0%}"
        elif heuristic_score >= 0.2:
            should_visit = link.priority_score >= 0.3
            tier_used = "heuristic"
            rationale = f"Moderate relevance (score: {heuristic_score:.2f})"
        else:
            should_visit = False
            tier_used = "heuristic"
            rationale = f"Low relevance (score: {heuristic_score:.2f})"

        # Tier 3: Use LLM only if confidence is low AND LLM is enabled
        if (
            self.settings.use_llm_for_navigation
            and self.llm is not None
            and confidence < self.settings.llm_confidence_threshold
        ):
            logger.debug(
                f"Low confidence ({confidence:.2f}), using LLM for visit decision")
            should_visit, llm_reason = self._llm_should_visit(goal, link)
            self._stats["llm_decisions"] += 1
            tier_used = "llm"
            rationale = llm_reason
            confidence = 0.85

        return NavigationDecision(
            selected_action="visit" if should_visit else "skip",
            confidence=confidence,
            rationale=rationale,
            tier_used=tier_used,
            scores={
                "keyword_score": keyword_score,
                "url_score": url_score,
                "priority_score": link.priority_score,
                "heuristic_total": heuristic_score,
            },
        )

    def _llm_should_visit(
        self,
        goal: str,
        link: LinkCandidate,
    ) -> tuple[bool, str]:
        """Use LLM to decide if link should be visited."""
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
            logger.warning(f"LLM visit decision failed: {e}")
            return link.priority_score >= 0.4, "Fallback to score threshold"

    def clear_cache(self) -> None:
        """Clear assessment cache."""
        self._assessment_cache.clear()

    def get_decision_stats(self) -> dict:
        """
        Get statistics about decision tier usage.

        Returns:
            Dictionary with counts for each tier
        """
        total = sum(self._stats.values())
        return {
            **self._stats,
            "total_decisions": total,
            "llm_usage_pct": (
                self._stats["llm_decisions"] / max(total, 1) * 100
            ),
        }

    def reset_stats(self) -> None:
        """Reset decision statistics."""
        self._stats = {
            "heuristic_decisions": 0,
            "embedding_decisions": 0,
            "llm_decisions": 0,
        }

    def __repr__(self) -> str:
        stats = self.get_decision_stats()
        return (
            f"NavigationPlanner("
            f"llm_enabled={self.settings.use_llm_for_navigation}, "
            f"cache_size={len(self._assessment_cache)}, "
            f"llm_usage={stats['llm_usage_pct']:.1f}%)"
        )
