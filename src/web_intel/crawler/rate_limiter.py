"""
Rate limiting for web crawling.

Provides request throttling to respect server limits and avoid being blocked.
Supports per-domain limiting, adaptive backoff, and burst handling.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlparse

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitState:
    """
    Tracks rate limit state for a single domain.

    Attributes:
        last_request_time: Timestamp of last request (monotonic clock)
        request_count: Total requests made to this domain
        error_count: Consecutive errors (resets on success)
        backoff_until: Timestamp until which requests are blocked
        current_delay: Current delay between requests (may be adjusted)
    """

    last_request_time: float = 0.0
    request_count: int = 0
    error_count: int = 0
    backoff_until: float = 0.0
    current_delay: float = 1.0


class RateLimiter:
    """
    Simple rate limiter enforcing minimum delay between requests.

    Tracks requests per domain and ensures minimum spacing.
    Thread-safe for use with asyncio.

    Example:
        >>> limiter = RateLimiter(delay_seconds=1.0)
        >>> await limiter.acquire("https://example.com/page1")
        >>> # Makes request...
        >>> await limiter.acquire("https://example.com/page2")
        >>> # Waits ~1 second before returning
    """

    def __init__(
        self,
        delay_seconds: float = 1.0,
        per_domain: bool = True,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            delay_seconds: Minimum seconds between requests
            per_domain: If True, track limits per domain; if False, global limit
        """
        self.delay_seconds = delay_seconds
        self.per_domain = per_domain
        self._states: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(current_delay=delay_seconds)
        )
        self._global_state = RateLimitState(current_delay=delay_seconds)
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limit tracking."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"

    def _get_state(self, url: str) -> RateLimitState:
        """Get rate limit state for URL."""
        if not self.per_domain:
            return self._global_state
        domain = self._get_domain(url)
        return self._states[domain]

    async def acquire(self, url: str) -> float:
        """
        Acquire permission to make a request.

        Blocks until the rate limit allows the request.

        Args:
            url: URL to be requested

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            state = self._get_state(url)
            now = time.monotonic()
            waited = 0.0

            # Check if in backoff period
            if state.backoff_until > now:
                wait_time = state.backoff_until - now
                logger.debug(f"Backoff active, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                waited += wait_time
                now = time.monotonic()

            # Check minimum delay since last request
            time_since_last = now - state.last_request_time
            if time_since_last < state.current_delay:
                wait_time = state.current_delay - time_since_last
                await asyncio.sleep(wait_time)
                waited += wait_time

            # Update state
            state.last_request_time = time.monotonic()
            state.request_count += 1

            return waited

    async def release_success(self, url: str) -> None:
        """
        Signal successful request completion.

        Resets error count for the domain.

        Args:
            url: URL that was successfully requested
        """
        async with self._lock:
            state = self._get_state(url)
            state.error_count = 0

    async def release_error(
        self,
        url: str,
        retry_after: float | None = None,
    ) -> None:
        """
        Signal request error for backoff tracking.

        Args:
            url: URL that failed
            retry_after: Server-specified retry delay (e.g., from 429 response)
        """
        async with self._lock:
            state = self._get_state(url)
            state.error_count += 1

            if retry_after is not None:
                # Use server-specified delay
                state.backoff_until = time.monotonic() + retry_after
                logger.info(
                    f"Rate limit backoff: {retry_after:.1f}s for {self._get_domain(url)}")

    def get_stats(self, url: str | None = None) -> dict:
        """
        Get rate limiter statistics.

        Args:
            url: URL to get stats for (None = all domains)

        Returns:
            Dictionary with request counts and state
        """
        if url is not None:
            state = self._get_state(url)
            return {
                "domain": self._get_domain(url) if self.per_domain else "global",
                "request_count": state.request_count,
                "error_count": state.error_count,
                "current_delay": state.current_delay,
            }

        return {
            "total_domains": len(self._states),
            "total_requests": sum(s.request_count for s in self._states.values()),
            "domains": {
                domain: {
                    "request_count": state.request_count,
                    "error_count": state.error_count,
                }
                for domain, state in self._states.items()
            },
        }

    def reset(self, url: str | None = None) -> None:
        """
        Reset rate limiter state.

        Args:
            url: URL/domain to reset (None = reset all)
        """
        if url is not None:
            domain = self._get_domain(url)
            if domain in self._states:
                del self._states[domain]
        else:
            self._states.clear()
            self._global_state = RateLimitState(
                current_delay=self.delay_seconds)


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter with adaptive delay based on server responses.

    Automatically increases delay on errors and decreases on success,
    finding the optimal request rate for each domain.

    Example:
        >>> limiter = AdaptiveRateLimiter(
        ...     base_delay=1.0,
        ...     min_delay=0.5,
        ...     max_delay=30.0,
        ... )
        >>> await limiter.acquire(url)
        >>> try:
        ...     response = await fetch(url)
        ...     await limiter.release_success(url)
        ... except RateLimitError:
        ...     await limiter.release_error(url, retry_after=10.0)
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        min_delay: float = 0.5,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        recovery_factor: float = 0.9,
        success_threshold: int = 10,
        per_domain: bool = True,
    ) -> None:
        """
        Initialize adaptive rate limiter.

        Args:
            base_delay: Starting delay between requests
            min_delay: Minimum allowed delay
            max_delay: Maximum allowed delay
            backoff_factor: Multiplier for delay on error
            recovery_factor: Multiplier for delay on sustained success
            success_threshold: Consecutive successes before reducing delay
            per_domain: Track limits per domain
        """
        super().__init__(delay_seconds=base_delay, per_domain=per_domain)
        self.base_delay = base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.success_threshold = success_threshold
        self._success_streaks: dict[str, int] = defaultdict(int)

    async def release_success(self, url: str) -> None:
        """
        Signal successful request and potentially reduce delay.

        After sustained success, gradually reduces delay toward minimum.

        Args:
            url: URL that was successfully requested
        """
        async with self._lock:
            state = self._get_state(url)
            state.error_count = 0

            # Track success streak
            key = self._get_domain(url) if self.per_domain else "global"
            self._success_streaks[key] += 1

            # Reduce delay after sustained success
            if self._success_streaks[key] >= self.success_threshold:
                old_delay = state.current_delay
                state.current_delay = max(
                    self.min_delay,
                    state.current_delay * self.recovery_factor,
                )
                if state.current_delay < old_delay:
                    logger.debug(
                        f"Reduced delay for {key}: {old_delay:.2f}s -> {state.current_delay:.2f}s"
                    )
                self._success_streaks[key] = 0

    async def release_error(
        self,
        url: str,
        retry_after: float | None = None,
    ) -> None:
        """
        Signal request error and increase delay.

        Applies exponential backoff on errors.

        Args:
            url: URL that failed
            retry_after: Server-specified retry delay
        """
        async with self._lock:
            state = self._get_state(url)
            state.error_count += 1

            # Reset success streak
            key = self._get_domain(url) if self.per_domain else "global"
            self._success_streaks[key] = 0

            # Apply backoff
            if retry_after is not None:
                # Use server-specified delay
                state.backoff_until = time.monotonic() + retry_after
                state.current_delay = max(state.current_delay, retry_after)
                logger.info(
                    f"Server-specified backoff: {retry_after:.1f}s for {key}"
                )
            else:
                # Exponential backoff
                old_delay = state.current_delay
                state.current_delay = min(
                    self.max_delay,
                    state.current_delay * self.backoff_factor,
                )
                logger.info(
                    f"Backoff for {key}: {old_delay:.2f}s -> {state.current_delay:.2f}s "
                    f"(errors: {state.error_count})"
                )

    def reset(self, url: str | None = None) -> None:
        """Reset rate limiter state including success streaks."""
        super().reset(url)
        if url is not None:
            key = self._get_domain(url) if self.per_domain else "global"
            self._success_streaks.pop(key, None)
        else:
            self._success_streaks.clear()


class TokenBucketLimiter:
    """
    Token bucket rate limiter for burst-friendly throttling.

    Allows short bursts of requests while maintaining average rate.
    Useful when server allows occasional bursts but limits sustained rate.

    Example:
        >>> limiter = TokenBucketLimiter(rate=2.0, burst=5)
        >>> # Can make 5 requests immediately, then 2 per second
        >>> for url in urls:
        ...     await limiter.acquire()
        ...     await fetch(url)
    """

    def __init__(
        self,
        rate: float = 1.0,
        burst: int = 1,
        per_domain: bool = True,
    ) -> None:
        """
        Initialize token bucket limiter.

        Args:
            rate: Tokens added per second (requests per second)
            burst: Maximum tokens (burst capacity)
            per_domain: Track buckets per domain
        """
        self.rate = rate
        self.burst = burst
        self.per_domain = per_domain
        # domain -> (tokens, last_update)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return "unknown"

    def _get_bucket_key(self, url: str) -> str:
        """Get bucket key for URL."""
        return self._get_domain(url) if self.per_domain else "global"

    async def acquire(self, url: str = "") -> float:
        """
        Acquire a token to make a request.

        Blocks until a token is available.

        Args:
            url: URL to be requested

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            key = self._get_bucket_key(url)
            now = time.monotonic()
            waited = 0.0

            # Get or initialize bucket
            if key not in self._buckets:
                self._buckets[key] = (float(self.burst), now)

            tokens, last_update = self._buckets[key]

            # Add tokens based on elapsed time
            elapsed = now - last_update
            tokens = min(self.burst, tokens + elapsed * self.rate)

            # Wait if no tokens available
            if tokens < 1.0:
                wait_time = (1.0 - tokens) / self.rate
                await asyncio.sleep(wait_time)
                waited = wait_time
                tokens = 1.0  # After waiting, we have exactly 1 token

            # Consume token
            tokens -= 1.0
            self._buckets[key] = (tokens, time.monotonic())

            return waited

    def get_available_tokens(self, url: str = "") -> float:
        """
        Get current available tokens for a domain.

        Args:
            url: URL to check

        Returns:
            Number of available tokens (may be fractional)
        """
        key = self._get_bucket_key(url)

        if key not in self._buckets:
            return float(self.burst)

        tokens, last_update = self._buckets[key]
        elapsed = time.monotonic() - last_update
        return min(self.burst, tokens + elapsed * self.rate)

    def reset(self, url: str | None = None) -> None:
        """Reset bucket state."""
        if url is not None:
            key = self._get_bucket_key(url)
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()
