"""
Exploration state tracking for loop prevention and coverage.

Maintains records of visited pages and states to prevent
infinite loops and ensure comprehensive coverage.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse, parse_qs
import hashlib

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class PageStateType(str, Enum):
    """Types of page states tracked."""

    INITIAL = "initial"  # First load of page
    AFTER_CLICK = "after_click"  # After clicking element
    AFTER_SCROLL = "after_scroll"  # After scrolling
    AFTER_INPUT = "after_input"  # After form input
    MODAL_OPEN = "modal_open"  # Modal/popup opened
    TAB_SWITCH = "tab_switch"  # After tab switch


@dataclass
class VisitRecord:
    """
    Record of a page visit.

    Tracks when and how a page was visited for loop prevention.
    """

    url: str
    normalized_url: str
    visit_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    state_type: PageStateType = PageStateType.INITIAL
    content_hash: str | None = None
    parent_url: str | None = None
    action_taken: str | None = None
    depth: int = 0


@dataclass
class PageState:
    """
    Represents the state of a page at a point in time.

    Used to detect if page content has changed after interaction.
    """

    url: str
    content_hash: str
    title: str
    element_count: int
    link_count: int
    text_length: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def is_similar_to(self, other: "PageState", threshold: float = 0.9) -> bool:
        """
        Check if this state is similar to another.

        Args:
            other: State to compare with
            threshold: Similarity threshold (0-1)

        Returns:
            True if states are similar
        """
        if self.content_hash == other.content_hash:
            return True

        # Compare structural similarity
        if self.url != other.url:
            return False

        # Check element counts
        if other.element_count > 0:
            element_ratio = min(self.element_count, other.element_count) / max(
                self.element_count, other.element_count
            )
        else:
            element_ratio = 1.0 if self.element_count == 0 else 0.0

        # Check text length
        if other.text_length > 0:
            text_ratio = min(self.text_length, other.text_length) / max(
                self.text_length, other.text_length
            )
        else:
            text_ratio = 1.0 if self.text_length == 0 else 0.0

        # Average similarity
        similarity = (element_ratio + text_ratio) / 2
        return similarity >= threshold


class ExplorationState:
    """
    Tracks exploration state across a crawl session.

    Maintains visited URLs, page states, and interaction history
    to prevent loops and ensure coverage.

    Example:
        >>> state = ExplorationState(max_visits_per_url=3)
        >>> if state.should_visit(url):
        ...     state.record_visit(url, content_hash)
        ...     # Visit page...
    """

    def __init__(
        self,
        max_visits_per_url: int = 3,
        max_states_per_page: int = 5,
        loop_detection_window: int = 10,
    ) -> None:
        """
        Initialize exploration state tracker.

        Args:
            max_visits_per_url: Max times to visit same URL
            max_states_per_page: Max unique states to track per page
            loop_detection_window: Recent actions to check for loops
        """
        self.max_visits_per_url = max_visits_per_url
        self.max_states_per_page = max_states_per_page
        self.loop_detection_window = loop_detection_window

        # Visit tracking
        self._visits: dict[str, list[VisitRecord]] = {}
        self._visit_count: dict[str, int] = {}

        # State tracking
        self._page_states: dict[str, list[PageState]] = {}

        # Action history for loop detection
        self._action_history: list[tuple[str, str]] = []  # (url, action)

        # Content hashes seen
        self._content_hashes: set[str] = set()

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.

        Removes fragments, sorts query params, lowercases.
        """
        try:
            parsed = urlparse(url)

            # Lowercase scheme and host
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Normalize path
            path = parsed.path or "/"
            if path != "/" and path.endswith("/"):
                path = path.rstrip("/")

            # Sort query params
            if parsed.query:
                params = parse_qs(parsed.query, keep_blank_values=True)
                sorted_params = sorted(params.items())
                query = "&".join(f"{k}={v[0]}" for k, v in sorted_params if v)
            else:
                query = ""

            from urllib.parse import urlunparse
            return urlunparse((scheme, netloc, path, "", query, ""))

        except Exception:
            return url

    def compute_content_hash(self, content: str) -> str:
        """
        Compute hash of page content.

        Args:
            content: Page content (text or HTML)

        Returns:
            SHA256 hash of normalized content
        """
        # Normalize whitespace for consistent hashing
        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def should_visit(self, url: str) -> bool:
        """
        Check if URL should be visited.

        Args:
            url: URL to check

        Returns:
            True if URL should be visited
        """
        normalized = self.normalize_url(url)
        visit_count = self._visit_count.get(normalized, 0)

        if visit_count >= self.max_visits_per_url:
            logger.debug(f"Skipping URL (max visits reached): {url}")
            return False

        return True

    def record_visit(
        self,
        url: str,
        content_hash: str | None = None,
        state_type: PageStateType = PageStateType.INITIAL,
        parent_url: str | None = None,
        action: str | None = None,
        depth: int = 0,
    ) -> VisitRecord:
        """
        Record a page visit.

        Args:
            url: Visited URL
            content_hash: Hash of page content
            state_type: Type of visit/state
            parent_url: URL that led to this page
            action: Action taken (e.g., "click_nav")
            depth: Crawl depth

        Returns:
            Created visit record
        """
        normalized = self.normalize_url(url)

        record = VisitRecord(
            url=url,
            normalized_url=normalized,
            state_type=state_type,
            content_hash=content_hash,
            parent_url=parent_url,
            action_taken=action,
            depth=depth,
        )

        # Update visit tracking
        if normalized not in self._visits:
            self._visits[normalized] = []
        self._visits[normalized].append(record)
        self._visit_count[normalized] = self._visit_count.get(
            normalized, 0) + 1

        # Track content hash
        if content_hash:
            self._content_hashes.add(content_hash)

        # Update action history
        if action:
            self._action_history.append((normalized, action))
            # Trim history
            if len(self._action_history) > self.loop_detection_window * 2:
                self._action_history = self._action_history[-self.loop_detection_window:]

        logger.debug(
            f"Recorded visit: {url} (type={state_type.value}, visits={self._visit_count[normalized]})"
        )

        return record

    def record_state(self, state: PageState) -> bool:
        """
        Record a page state.

        Args:
            state: Page state to record

        Returns:
            True if this is a new state, False if duplicate
        """
        normalized = self.normalize_url(state.url)

        if normalized not in self._page_states:
            self._page_states[normalized] = []

        # Check if similar state already exists
        for existing in self._page_states[normalized]:
            if state.is_similar_to(existing):
                return False

        # Add new state
        self._page_states[normalized].append(state)

        # Trim old states
        if len(self._page_states[normalized]) > self.max_states_per_page:
            self._page_states[normalized] = self._page_states[normalized][-self.max_states_per_page:]

        return True

    def is_new_content(self, content_hash: str) -> bool:
        """
        Check if content hash is new.

        Args:
            content_hash: Hash to check

        Returns:
            True if content hasn't been seen
        """
        return content_hash not in self._content_hashes

    def detect_loop(self, url: str, action: str) -> bool:
        """
        Detect if current action would create a loop.

        Args:
            url: Current URL
            action: Action about to take

        Returns:
            True if loop detected
        """
        normalized = self.normalize_url(url)
        current = (normalized, action)

        # Check recent history for repeated pattern
        recent = self._action_history[-self.loop_detection_window:]

        if len(recent) < 3:
            return False

        # Count occurrences of current action
        count = sum(1 for r in recent if r == current)

        if count >= 2:
            logger.debug(
                f"Loop detected: {action} on {url} repeated {count} times")
            return True

        # Check for A->B->A pattern
        if len(recent) >= 2:
            prev = recent[-1]
            if len(recent) >= 3:
                prev_prev = recent[-2]
                if prev_prev == current and prev[0] != normalized:
                    logger.debug(f"A->B->A loop detected")
                    return True

        return False

    def get_visit_count(self, url: str) -> int:
        """Get number of times URL has been visited."""
        normalized = self.normalize_url(url)
        return self._visit_count.get(normalized, 0)

    def get_visits(self, url: str) -> list[VisitRecord]:
        """Get all visit records for URL."""
        normalized = self.normalize_url(url)
        return self._visits.get(normalized, [])

    def get_states(self, url: str) -> list[PageState]:
        """Get all recorded states for URL."""
        normalized = self.normalize_url(url)
        return self._page_states.get(normalized, [])

    def get_stats(self) -> dict:
        """Get exploration statistics."""
        return {
            "urls_visited": len(self._visit_count),
            "total_visits": sum(self._visit_count.values()),
            "unique_content": len(self._content_hashes),
            "states_tracked": sum(len(s) for s in self._page_states.values()),
            "action_history_size": len(self._action_history),
        }

    def clear(self) -> None:
        """Clear all tracking state."""
        self._visits.clear()
        self._visit_count.clear()
        self._page_states.clear()
        self._action_history.clear()
        self._content_hashes.clear()
