"""
URL filtering for web crawling.

Provides pattern-based URL filtering, robots.txt compliance,
and domain boundary enforcement.
"""

import re
from functools import lru_cache
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from web_intel.core.exceptions import RobotsBlockedError
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class URLFilter:
    """
    Filter URLs based on patterns and rules.

    Supports:
    - Include/exclude regex patterns
    - Domain boundary enforcement
    - File extension filtering

    Example:
        >>> filter = URLFilter(
        ...     patterns_exclude=[r".*\\.pdf$", r".*/login.*"],
        ...     allowed_domains=["example.com"],
        ... )
        >>> filter.is_allowed("https://example.com/page")
        True
        >>> filter.is_allowed("https://example.com/file.pdf")
        False
    """

    def __init__(
        self,
        patterns_include: list[str] | None = None,
        patterns_exclude: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        follow_external: bool = False,
    ) -> None:
        """
        Initialize URL filter.

        Args:
            patterns_include: Regex patterns URLs must match (empty = all)
            patterns_exclude: Regex patterns to reject
            allowed_domains: Domains to allow (empty = all if follow_external)
            follow_external: Whether to follow external domain links
        """
        self.patterns_include = patterns_include or []
        self.patterns_exclude = patterns_exclude or []
        self.allowed_domains = set(d.lower() for d in (allowed_domains or []))
        self.follow_external = follow_external

        # Compile regex patterns
        self._include_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.patterns_include
        ]
        self._exclude_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.patterns_exclude
        ]

    def add_allowed_domain(self, domain: str) -> None:
        """Add domain to allowed list."""
        self.allowed_domains.add(domain.lower())

    def is_allowed(self, url: str) -> bool:
        """
        Check if URL passes all filters.

        Args:
            url: URL to check

        Returns:
            True if URL is allowed, False otherwise
        """
        try:
            parsed = urlparse(url)

            # Must be HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                return False

            # Must have a host
            if not parsed.netloc:
                return False

            domain = parsed.netloc.lower()

            # Check domain allowlist
            if not self.follow_external and self.allowed_domains:
                if not self._domain_matches(domain):
                    return False

            # Check exclude patterns first (reject if any match)
            for pattern in self._exclude_compiled:
                if pattern.search(url):
                    logger.debug(f"URL excluded by pattern: {url}")
                    return False

            # Check include patterns (must match at least one, if any defined)
            if self._include_compiled:
                if not any(p.search(url) for p in self._include_compiled):
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error checking URL filter: {e}")
            return False

    def _domain_matches(self, domain: str) -> bool:
        """Check if domain is in allowed list (including subdomains)."""
        for allowed in self.allowed_domains:
            if domain == allowed or domain.endswith("." + allowed):
                return True
        return False

    def filter_urls(self, urls: list[str]) -> list[str]:
        """
        Filter a list of URLs.

        Args:
            urls: URLs to filter

        Returns:
            List of allowed URLs
        """
        return [url for url in urls if self.is_allowed(url)]

    def get_rejection_reason(self, url: str) -> str | None:
        """
        Get reason why URL was rejected.

        Args:
            url: URL to check

        Returns:
            Rejection reason or None if allowed
        """
        try:
            parsed = urlparse(url)

            if parsed.scheme not in ("http", "https"):
                return f"Invalid scheme: {parsed.scheme}"

            if not parsed.netloc:
                return "No host in URL"

            domain = parsed.netloc.lower()

            if not self.follow_external and self.allowed_domains:
                if not self._domain_matches(domain):
                    return f"External domain: {domain}"

            for pattern in self._exclude_compiled:
                if pattern.search(url):
                    return f"Matched exclude pattern: {pattern.pattern}"

            if self._include_compiled:
                if not any(p.search(url) for p in self._include_compiled):
                    return "No include pattern matched"

            return None

        except Exception as e:
            return f"Error: {e}"


class RobotsChecker:
    """
    Checks URLs against robots.txt rules.

    Caches robots.txt per domain for efficiency.

    Example:
        >>> checker = RobotsChecker(user_agent="WebIntelBot/1.0")
        >>> await checker.is_allowed("https://example.com/page")
        True
        >>> await checker.is_allowed("https://example.com/admin")
        False
    """

    def __init__(
        self,
        user_agent: str = "WebIntelBot/1.0",
        timeout_seconds: float = 10.0,
        respect_robots: bool = True,
    ) -> None:
        """
        Initialize robots.txt checker.

        Args:
            user_agent: User agent string for robots.txt matching
            timeout_seconds: Timeout for fetching robots.txt
            respect_robots: If False, allows all URLs (disabled mode)
        """
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.respect_robots = respect_robots
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._fetch_errors: set[str] = set()

    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL for a given URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _get_domain_key(self, url: str) -> str:
        """Get domain key for caching."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _fetch_robots(self, domain_key: str) -> RobotFileParser | None:
        """Fetch and parse robots.txt for a domain."""
        robots_url = f"{domain_key}/robots.txt"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(robots_url)

                if response.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(response.text.splitlines())
                    logger.debug(f"Loaded robots.txt from {robots_url}")
                    return parser
                elif response.status_code in (404, 403):
                    # No robots.txt or forbidden = allow all
                    logger.debug(
                        f"No robots.txt at {robots_url} ({response.status_code})")
                    return None
                else:
                    logger.warning(
                        f"Unexpected status {response.status_code} for {robots_url}"
                    )
                    return None

        except Exception as e:
            logger.warning(f"Error fetching robots.txt from {robots_url}: {e}")
            self._fetch_errors.add(domain_key)
            return None

    async def is_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            url: URL to check

        Returns:
            True if allowed, False if blocked

        Raises:
            RobotsBlockedError: If URL is explicitly blocked (when raise_on_block=True)
        """
        if not self.respect_robots:
            return True

        domain_key = self._get_domain_key(url)

        # Check cache
        if domain_key not in self._parsers:
            self._parsers[domain_key] = await self._fetch_robots(domain_key)

        parser = self._parsers[domain_key]

        # No robots.txt = allow all
        if parser is None:
            return True

        # Check against rules
        allowed = parser.can_fetch(self.user_agent, url)

        if not allowed:
            logger.debug(f"Blocked by robots.txt: {url}")

        return allowed

    async def check_or_raise(self, url: str) -> None:
        """
        Check URL and raise exception if blocked.

        Args:
            url: URL to check

        Raises:
            RobotsBlockedError: If URL is blocked by robots.txt
        """
        if not await self.is_allowed(url):
            raise RobotsBlockedError(
                "URL blocked by robots.txt",
                url=url,
            )

    def get_crawl_delay(self, url: str) -> float | None:
        """
        Get crawl delay specified in robots.txt.

        Args:
            url: URL to check

        Returns:
            Crawl delay in seconds, or None if not specified
        """
        if not self.respect_robots:
            return None

        domain_key = self._get_domain_key(url)
        parser = self._parsers.get(domain_key)

        if parser is None:
            return None

        try:
            delay = parser.crawl_delay(self.user_agent)
            return float(delay) if delay is not None else None
        except Exception:
            return None

    def clear_cache(self, url: str | None = None) -> None:
        """
        Clear robots.txt cache.

        Args:
            url: Clear cache for specific domain (None = clear all)
        """
        if url is not None:
            domain_key = self._get_domain_key(url)
            self._parsers.pop(domain_key, None)
            self._fetch_errors.discard(domain_key)
        else:
            self._parsers.clear()
            self._fetch_errors.clear()


class CombinedFilter:
    """
    Combines URL pattern filtering and robots.txt checking.

    Provides a single interface for all URL filtering needs.

    Example:
        >>> filter = CombinedFilter(
        ...     patterns_exclude=[r".*\\.pdf$"],
        ...     respect_robots=True,
        ... )
        >>> await filter.is_allowed("https://example.com/page")
        True
    """

    def __init__(
        self,
        patterns_include: list[str] | None = None,
        patterns_exclude: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        follow_external: bool = False,
        respect_robots: bool = True,
        user_agent: str = "WebIntelBot/1.0",
    ) -> None:
        """
        Initialize combined filter.

        Args:
            patterns_include: URL patterns to include
            patterns_exclude: URL patterns to exclude
            allowed_domains: Allowed domains
            follow_external: Follow external links
            respect_robots: Check robots.txt
            user_agent: User agent for robots.txt
        """
        self.url_filter = URLFilter(
            patterns_include=patterns_include,
            patterns_exclude=patterns_exclude,
            allowed_domains=allowed_domains,
            follow_external=follow_external,
        )
        self.robots_checker = RobotsChecker(
            user_agent=user_agent,
            respect_robots=respect_robots,
        )

    def add_seed_domain(self, url: str) -> None:
        """Add domain from seed URL to allowed list."""
        parsed = urlparse(url)
        if parsed.netloc:
            self.url_filter.add_allowed_domain(parsed.netloc)

    async def is_allowed(self, url: str) -> bool:
        """
        Check if URL passes all filters.

        Args:
            url: URL to check

        Returns:
            True if URL is allowed
        """
        # Check pattern filter first (synchronous, fast)
        if not self.url_filter.is_allowed(url):
            return False

        # Check robots.txt (async, may fetch)
        return await self.robots_checker.is_allowed(url)

    async def filter_urls(self, urls: list[str]) -> list[str]:
        """
        Filter a list of URLs.

        Args:
            urls: URLs to filter

        Returns:
            List of allowed URLs
        """
        # First pass: pattern filtering (fast)
        pattern_allowed = self.url_filter.filter_urls(urls)

        # Second pass: robots.txt (may need fetching)
        result = []
        for url in pattern_allowed:
            if await self.robots_checker.is_allowed(url):
                result.append(url)

        return result

    def get_crawl_delay(self, url: str) -> float | None:
        """Get crawl delay from robots.txt."""
        return self.robots_checker.get_crawl_delay(url)
