"""
URL queue management for web crawling.

Provides priority-based URL queue with deduplication,
depth tracking, and persistence support.
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Iterator
from urllib.parse import urlparse, urlunparse

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class URLPriority(IntEnum):
    """
    Priority levels for URL queue.

    Lower values = higher priority (processed first).
    """

    CRITICAL = 0  # Must visit (e.g., sitemaps)
    HIGH = 10  # Important pages (homepage, main nav)
    NORMAL = 50  # Standard discovered links
    LOW = 80  # Less important (deep pages, pagination)
    DEFERRED = 100  # Visit only if time permits


@dataclass(order=True)
class QueuedURL:
    """
    URL entry in the crawl queue.

    Sortable by priority for heap-based queue.
    """

    priority: int
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    parent_url: str | None = field(compare=False, default=None)
    discovered_at: datetime = field(
        compare=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )
    retry_count: int = field(compare=False, default=0)
    metadata: dict = field(compare=False, default_factory=dict)

    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        parsed = urlparse(self.url)
        return parsed.netloc.lower()


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent comparison.

    - Lowercases scheme and host
    - Removes default ports
    - Removes trailing slashes (except root)
    - Removes fragments
    - Sorts query parameters

    Args:
        url: URL to normalize

    Returns:
        Normalized URL string
    """
    try:
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Sort query parameters for consistency
        query = parsed.query
        if query:
            params = sorted(query.split("&"))
            query = "&".join(params)

        # Reconstruct without fragment
        normalized = urlunparse(
            (scheme, netloc, path, parsed.params, query, ""))

        return normalized

    except Exception:
        return url


class URLQueue:
    """
    Priority queue for URLs to crawl.

    Features:
    - Priority-based ordering (heap)
    - URL deduplication via normalization
    - Depth tracking
    - Thread-safe async operations

    Example:
        >>> queue = URLQueue(max_depth=5)
        >>> await queue.put("https://example.com", priority=URLPriority.HIGH)
        >>> url_entry = await queue.get()
        >>> print(url_entry.url)
    """

    def __init__(self, max_depth: int = 10) -> None:
        """
        Initialize URL queue.

        Args:
            max_depth: Maximum crawl depth (0 = seed only)
        """
        self.max_depth = max_depth
        self._heap: list[QueuedURL] = []
        self._seen: set[str] = set()
        self._in_progress: set[str] = set()
        self._completed: set[str] = set()
        self._failed: dict[str, int] = {}  # url -> failure count
        self._lock = asyncio.Lock()

    async def put(
        self,
        url: str,
        priority: int = URLPriority.NORMAL,
        depth: int = 0,
        parent_url: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """
        Add URL to queue if not already seen.

        Args:
            url: URL to add
            priority: Processing priority
            depth: Crawl depth from seed
            parent_url: URL that linked to this one
            metadata: Additional data to attach

        Returns:
            True if URL was added, False if duplicate or too deep
        """
        normalized = normalize_url(url)

        async with self._lock:
            # Check depth limit
            if depth > self.max_depth:
                logger.debug(
                    f"Skipping URL (depth {depth} > {self.max_depth}): {url}")
                return False

            # Check if already processed or queued
            if normalized in self._seen:
                return False

            # Add to queue
            entry = QueuedURL(
                priority=priority,
                url=normalized,
                depth=depth,
                parent_url=parent_url,
                metadata=metadata or {},
            )
            heapq.heappush(self._heap, entry)
            self._seen.add(normalized)

            logger.debug(f"Queued (priority={priority}, depth={depth}): {url}")
            return True

    async def put_many(
        self,
        urls: list[str],
        priority: int = URLPriority.NORMAL,
        depth: int = 0,
        parent_url: str | None = None,
    ) -> int:
        """
        Add multiple URLs to queue.

        Args:
            urls: URLs to add
            priority: Processing priority for all
            depth: Crawl depth for all
            parent_url: Parent URL for all

        Returns:
            Number of URLs actually added
        """
        added = 0
        for url in urls:
            if await self.put(url, priority, depth, parent_url):
                added += 1
        return added

    async def get(self) -> QueuedURL | None:
        """
        Get next URL to process.

        Returns highest priority URL and marks it as in-progress.

        Returns:
            QueuedURL or None if queue is empty
        """
        async with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)

                # Skip if already completed (shouldn't happen, but safety check)
                if entry.url in self._completed:
                    continue

                self._in_progress.add(entry.url)
                return entry

            return None

    async def complete(self, url: str) -> None:
        """
        Mark URL as successfully completed.

        Args:
            url: URL that was processed
        """
        normalized = normalize_url(url)

        async with self._lock:
            self._in_progress.discard(normalized)
            self._completed.add(normalized)

    async def fail(self, url: str, max_retries: int = 3) -> bool:
        """
        Mark URL as failed, possibly requeuing for retry.

        Args:
            url: URL that failed
            max_retries: Maximum retry attempts

        Returns:
            True if requeued for retry, False if max retries exceeded
        """
        normalized = normalize_url(url)

        async with self._lock:
            self._in_progress.discard(normalized)

            # Track failures
            self._failed[normalized] = self._failed.get(normalized, 0) + 1
            failure_count = self._failed[normalized]

            if failure_count <= max_retries:
                # Find original entry to get metadata
                # Requeue with lower priority
                entry = QueuedURL(
                    priority=URLPriority.DEFERRED,
                    url=normalized,
                    retry_count=failure_count,
                )
                heapq.heappush(self._heap, entry)
                logger.debug(
                    f"Requeued for retry ({failure_count}/{max_retries}): {url}")
                return True
            else:
                logger.warning(
                    f"Max retries exceeded ({failure_count}): {url}")
                return False

    async def requeue(
        self,
        url: str,
        priority: int = URLPriority.DEFERRED,
    ) -> None:
        """
        Return URL to queue (e.g., after rate limit).

        Args:
            url: URL to requeue
            priority: New priority level
        """
        normalized = normalize_url(url)

        async with self._lock:
            self._in_progress.discard(normalized)

            entry = QueuedURL(priority=priority, url=normalized)
            heapq.heappush(self._heap, entry)

    @property
    def pending_count(self) -> int:
        """Number of URLs waiting in queue."""
        return len(self._heap)

    @property
    def in_progress_count(self) -> int:
        """Number of URLs currently being processed."""
        return len(self._in_progress)

    @property
    def completed_count(self) -> int:
        """Number of URLs successfully completed."""
        return len(self._completed)

    @property
    def seen_count(self) -> int:
        """Total unique URLs seen."""
        return len(self._seen)

    def is_empty(self) -> bool:
        """Check if queue is empty and nothing in progress."""
        return len(self._heap) == 0 and len(self._in_progress) == 0

    def has_pending(self) -> bool:
        """Check if there are URLs waiting to be processed."""
        return len(self._heap) > 0

    def is_seen(self, url: str) -> bool:
        """Check if URL has been seen."""
        return normalize_url(url) in self._seen

    def is_completed(self, url: str) -> bool:
        """Check if URL has been completed."""
        return normalize_url(url) in self._completed

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "pending": self.pending_count,
            "in_progress": self.in_progress_count,
            "completed": self.completed_count,
            "seen": self.seen_count,
            "failed": len(self._failed),
        }

    def clear(self) -> None:
        """Clear all queue state."""
        self._heap.clear()
        self._seen.clear()
        self._in_progress.clear()
        self._completed.clear()
        self._failed.clear()

    def get_completed_urls(self) -> list[str]:
        """Get list of all completed URLs."""
        return list(self._completed)

    def get_failed_urls(self) -> dict[str, int]:
        """Get dict of failed URLs and their failure counts."""
        return dict(self._failed)
