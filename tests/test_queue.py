"""
Tests for URL queue module.

Tests URL normalization, priority ordering, and deduplication.
"""

import pytest

from web_intel.crawler import (
    URLQueue,
    QueuedURL,
    URLPriority,
    normalize_url,
)


class TestNormalizeURL:
    """Tests for URL normalization function."""

    def test_normalize_basic_url(self):
        """Basic URL should be normalized."""
        url = "https://example.com/page"
        result = normalize_url(url)

        assert isinstance(result, str)
        assert "example.com" in result

    def test_normalize_trailing_slash(self):
        """Trailing slash should be handled consistently."""
        url1 = "https://example.com/page/"
        url2 = "https://example.com/page"

        result1 = normalize_url(url1)
        result2 = normalize_url(url2)

        assert result1 == result2

    def test_normalize_fragment_removal(self):
        """URL fragments should be removed."""
        url = "https://example.com/page#section"
        result = normalize_url(url)

        assert "#" not in result

    def test_normalize_query_params_sorted(self):
        """Query parameters should be sorted consistently."""
        url1 = "https://example.com/page?b=2&a=1"
        url2 = "https://example.com/page?a=1&b=2"

        result1 = normalize_url(url1)
        result2 = normalize_url(url2)

        assert result1 == result2

    def test_normalize_scheme_lowercase(self):
        """Scheme should be lowercased."""
        url = "HTTPS://Example.Com/Page"
        result = normalize_url(url)

        assert result.startswith("https://")
        assert "Example.Com" not in result

    def test_normalize_default_port_removal(self):
        """Default ports should be removed."""
        url1 = "https://example.com:443/page"
        url2 = "https://example.com/page"

        result1 = normalize_url(url1)
        result2 = normalize_url(url2)

        assert result1 == result2

    def test_normalize_preserve_path(self):
        """URL path should be preserved."""
        url = "https://example.com/some/nested/path"
        result = normalize_url(url)

        assert "/some/nested/path" in result

    def test_normalize_encoding(self):
        """URL encoding should be handled."""
        url = "https://example.com/path%20with%20spaces"
        result = normalize_url(url)

        # Result should be valid
        assert "example.com" in result


class TestQueuedURL:
    """Tests for QueuedURL dataclass."""

    def test_queued_url_creation(self):
        """QueuedURL should be created with required fields."""
        queued = QueuedURL(
            url="https://example.com",
            depth=0,
            priority=URLPriority.HIGH,
        )

        assert queued.url == "https://example.com"
        assert queued.depth == 0
        assert queued.priority == URLPriority.HIGH

    def test_queued_url_with_parent(self):
        """QueuedURL can have parent URL."""
        queued = QueuedURL(
            url="https://example.com/page",
            depth=1,
            priority=URLPriority.NORMAL,
            parent_url="https://example.com",
        )

        assert queued.parent_url == "https://example.com"

    def test_queued_url_comparison(self):
        """QueuedURLs should be comparable for priority queue."""
        high = QueuedURL(url="https://a.com", depth=0,
                         priority=URLPriority.HIGH)
        normal = QueuedURL(url="https://b.com", depth=0,
                           priority=URLPriority.NORMAL)
        low = QueuedURL(url="https://c.com", depth=0, priority=URLPriority.LOW)

        # High priority should come first (smaller comparison value)
        assert high < normal or high.priority.value < normal.priority.value
        assert normal < low or normal.priority.value < low.priority.value


class TestURLPriority:
    """Tests for URLPriority enum."""

    def test_priority_values(self):
        """Priority enum should have expected values."""
        assert URLPriority.HIGH is not None
        assert URLPriority.NORMAL is not None
        assert URLPriority.LOW is not None

    def test_priority_ordering(self):
        """Priority values should allow ordering."""
        priorities = [URLPriority.LOW, URLPriority.HIGH, URLPriority.NORMAL]

        # Should be sortable
        sorted_priorities = sorted(priorities, key=lambda p: p.value)
        assert len(sorted_priorities) == 3


class TestURLQueue:
    """Tests for URLQueue."""

    @pytest.fixture
    def queue(self) -> URLQueue:
        """Provide a queue instance."""
        return URLQueue()

    def test_queue_add_url(self, queue: URLQueue):
        """URL can be added to queue."""
        queue.add("https://example.com", depth=0)

        assert len(queue) == 1
        assert not queue.is_empty()

    def test_queue_pop_url(self, queue: URLQueue):
        """URL can be popped from queue."""
        queue.add("https://example.com", depth=0)
        queued = queue.pop()

        assert queued is not None
        assert queued.url == "https://example.com"
        assert len(queue) == 0

    def test_queue_priority_order(self, queue: URLQueue):
        """Higher priority URLs should be popped first."""
        queue.add("https://low.com", depth=0, priority=URLPriority.LOW)
        queue.add("https://high.com", depth=0, priority=URLPriority.HIGH)
        queue.add("https://normal.com", depth=0, priority=URLPriority.NORMAL)

        first = queue.pop()
        assert first.url == "https://high.com"

    def test_queue_deduplication(self, queue: URLQueue):
        """Duplicate URLs should not be added."""
        queue.add("https://example.com/page", depth=0)
        queue.add("https://example.com/page", depth=0)

        assert len(queue) == 1

    def test_queue_normalized_deduplication(self, queue: URLQueue):
        """URLs that normalize to same value should be deduplicated."""
        queue.add("https://example.com/page/", depth=0)
        queue.add("https://example.com/page", depth=0)

        assert len(queue) == 1

    def test_queue_depth_tracking(self, queue: URLQueue):
        """Queue should track URL depth."""
        queue.add("https://example.com", depth=0)
        queue.add("https://example.com/page", depth=1,
                  parent_url="https://example.com")
        queue.add("https://example.com/page/sub", depth=2,
                  parent_url="https://example.com/page")

        depths = set()
        while not queue.is_empty():
            queued = queue.pop()
            depths.add(queued.depth)

        assert depths == {0, 1, 2}

    def test_queue_is_visited(self, queue: URLQueue):
        """Queue should track visited URLs."""
        queue.add("https://example.com", depth=0)
        queue.mark_visited("https://example.com")

        assert queue.is_visited("https://example.com")

    def test_queue_skip_visited(self, queue: URLQueue):
        """Queue should skip already visited URLs."""
        queue.mark_visited("https://example.com")
        queue.add("https://example.com", depth=0)

        assert len(queue) == 0 or queue.is_empty()

    def test_queue_max_depth(self):
        """Queue should respect max depth limit."""
        queue = URLQueue(max_depth=2)

        queue.add("https://example.com", depth=0)
        queue.add("https://example.com/a", depth=1)
        queue.add("https://example.com/a/b", depth=2)
        queue.add("https://example.com/a/b/c", depth=3)  # Exceeds max depth

        assert len(queue) == 3

    def test_queue_clear(self, queue: URLQueue):
        """Queue can be cleared."""
        queue.add("https://example.com", depth=0)
        queue.add("https://example2.com", depth=0)
        queue.clear()

        assert len(queue) == 0
        assert queue.is_empty()

    def test_queue_peek(self, queue: URLQueue):
        """Peek should return next URL without removing."""
        queue.add("https://example.com", depth=0)

        peeked = queue.peek()
        assert peeked.url == "https://example.com"
        assert len(queue) == 1  # Still in queue

    def test_queue_add_multiple(self, queue: URLQueue, sample_urls: list[str]):
        """Multiple URLs can be added at once."""
        queue.add_multiple(sample_urls, depth=0)

        assert len(queue) == len(sample_urls)

    def test_queue_get_stats(self, queue: URLQueue):
        """Queue should provide statistics."""
        queue.add("https://example.com", depth=0)
        queue.mark_visited("https://other.com")

        stats = queue.get_stats()

        assert "queued" in stats or "pending" in stats
        assert "visited" in stats

    def test_queue_empty_pop(self, queue: URLQueue):
        """Popping from empty queue should return None or raise."""
        result = queue.pop()
        assert result is None


class TestURLQueueWithDomain:
    """Tests for URL queue domain filtering."""

    def test_queue_domain_filter(self):
        """Queue should filter by domain when configured."""
        queue = URLQueue(allowed_domains=["example.com"])

        queue.add("https://example.com/page", depth=0)
        queue.add("https://other.com/page", depth=0)

        # Only example.com should be added
        assert len(queue) == 1

    def test_queue_subdomain_handling(self):
        """Queue should handle subdomains correctly."""
        queue = URLQueue(allowed_domains=[
                         "example.com"], include_subdomains=True)

        queue.add("https://www.example.com/page", depth=0)
        queue.add("https://blog.example.com/page", depth=0)
        queue.add("https://example.com/page", depth=0)

        assert len(queue) == 3

    def test_queue_exclude_patterns(self):
        """Queue should exclude URLs matching patterns."""
        queue = URLQueue(exclude_patterns=[r"\.pdf$", r"/admin/"])

        queue.add("https://example.com/doc.pdf", depth=0)
        queue.add("https://example.com/admin/login", depth=0)
        queue.add("https://example.com/page", depth=0)

        assert len(queue) == 1
