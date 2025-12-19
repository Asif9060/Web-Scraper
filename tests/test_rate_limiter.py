"""
Tests for rate limiter module.

Tests delay enforcement, per-domain tracking, and adaptive backoff.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from web_intel.crawler import (
    RateLimiter,
    AdaptiveRateLimiter,
    RateLimitState,
)


class TestRateLimitState:
    """Tests for RateLimitState dataclass."""

    def test_state_creation(self):
        """RateLimitState should be created with defaults."""
        state = RateLimitState(
            domain="example.com",
            last_request_time=time.time(),
        )

        assert state.domain == "example.com"
        assert state.last_request_time > 0

    def test_state_with_delay(self):
        """RateLimitState should track current delay."""
        state = RateLimitState(
            domain="example.com",
            last_request_time=time.time(),
            current_delay=2.0,
        )

        assert state.current_delay == 2.0


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Provide a rate limiter with short delay for testing."""
        return RateLimiter(default_delay=0.1)  # 100ms for fast tests

    @pytest.mark.asyncio
    async def test_rate_limit_first_request(self, limiter: RateLimiter):
        """First request should not be delayed."""
        start = time.time()
        await limiter.wait("https://example.com/page1")
        elapsed = time.time() - start

        # First request should be immediate
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_rate_limit_subsequent_request(self, limiter: RateLimiter):
        """Subsequent requests to same domain should be delayed."""
        await limiter.wait("https://example.com/page1")

        start = time.time()
        await limiter.wait("https://example.com/page2")
        elapsed = time.time() - start

        # Should wait for delay
        assert elapsed >= 0.08  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limit_different_domains(self, limiter: RateLimiter):
        """Different domains should have independent delays."""
        await limiter.wait("https://example.com/page1")

        start = time.time()
        await limiter.wait("https://other.com/page1")  # Different domain
        elapsed = time.time() - start

        # Different domain should be immediate
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_rate_limit_domain_extraction(self, limiter: RateLimiter):
        """Limiter should correctly extract domain from URL."""
        await limiter.wait("https://www.example.com/page1")

        start = time.time()
        await limiter.wait("https://www.example.com/page2")
        elapsed = time.time() - start

        # Same domain, should be delayed
        assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_rate_limit_per_domain_config(self):
        """Limiter should support per-domain delays."""
        limiter = RateLimiter(
            default_delay=0.1,
            domain_delays={"slow.com": 0.3},
        )

        await limiter.wait("https://slow.com/page1")

        start = time.time()
        await limiter.wait("https://slow.com/page2")
        elapsed = time.time() - start

        # Should use domain-specific delay
        assert elapsed >= 0.28

    def test_get_domain_state(self, limiter: RateLimiter):
        """Limiter should provide domain state."""
        # Make a synchronous call to register domain
        asyncio.run(limiter.wait("https://example.com/page"))

        state = limiter.get_state("example.com")

        assert state is not None
        assert state.domain == "example.com"

    def test_reset_domain(self, limiter: RateLimiter):
        """Limiter should allow resetting domain state."""
        asyncio.run(limiter.wait("https://example.com/page"))

        limiter.reset("example.com")
        state = limiter.get_state("example.com")

        assert state is None or state.last_request_time == 0

    def test_clear_all_states(self, limiter: RateLimiter):
        """Limiter should allow clearing all states."""
        asyncio.run(limiter.wait("https://example.com/page"))
        asyncio.run(limiter.wait("https://other.com/page"))

        limiter.clear()

        assert limiter.get_state("example.com") is None
        assert limiter.get_state("other.com") is None


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter with backoff."""

    @pytest.fixture
    def adaptive_limiter(self) -> AdaptiveRateLimiter:
        """Provide an adaptive rate limiter."""
        return AdaptiveRateLimiter(
            base_delay=0.1,
            max_delay=1.0,
            backoff_factor=2.0,
        )

    @pytest.mark.asyncio
    async def test_adaptive_backoff_on_error(self, adaptive_limiter: AdaptiveRateLimiter):
        """Delay should increase on errors."""
        domain = "example.com"

        initial_delay = adaptive_limiter.get_delay(domain)

        adaptive_limiter.report_error(domain, status_code=429)

        new_delay = adaptive_limiter.get_delay(domain)
        assert new_delay > initial_delay

    @pytest.mark.asyncio
    async def test_adaptive_backoff_max_delay(self, adaptive_limiter: AdaptiveRateLimiter):
        """Delay should not exceed max_delay."""
        domain = "example.com"

        # Report many errors
        for _ in range(10):
            adaptive_limiter.report_error(domain, status_code=429)

        delay = adaptive_limiter.get_delay(domain)
        assert delay <= 1.0  # max_delay

    @pytest.mark.asyncio
    async def test_adaptive_success_reduces_delay(self, adaptive_limiter: AdaptiveRateLimiter):
        """Successful requests should reduce delay."""
        domain = "example.com"

        # Increase delay with error
        adaptive_limiter.report_error(domain, status_code=429)
        error_delay = adaptive_limiter.get_delay(domain)

        # Report success
        adaptive_limiter.report_success(domain)
        success_delay = adaptive_limiter.get_delay(domain)

        assert success_delay <= error_delay

    @pytest.mark.asyncio
    async def test_adaptive_429_handling(self, adaptive_limiter: AdaptiveRateLimiter):
        """429 status should trigger significant backoff."""
        domain = "example.com"

        initial_delay = adaptive_limiter.get_delay(domain)
        adaptive_limiter.report_error(domain, status_code=429)

        new_delay = adaptive_limiter.get_delay(domain)

        # 429 should at least double the delay
        assert new_delay >= initial_delay * 2

    @pytest.mark.asyncio
    async def test_adaptive_503_handling(self, adaptive_limiter: AdaptiveRateLimiter):
        """503 status should trigger backoff."""
        domain = "example.com"

        initial_delay = adaptive_limiter.get_delay(domain)
        adaptive_limiter.report_error(domain, status_code=503)

        new_delay = adaptive_limiter.get_delay(domain)
        assert new_delay > initial_delay

    @pytest.mark.asyncio
    async def test_adaptive_retry_after_header(self, adaptive_limiter: AdaptiveRateLimiter):
        """Limiter should respect Retry-After header."""
        domain = "example.com"

        adaptive_limiter.report_error(
            domain,
            status_code=429,
            retry_after=5,
        )

        delay = adaptive_limiter.get_delay(domain)
        assert delay >= 5

    def test_adaptive_get_stats(self, adaptive_limiter: AdaptiveRateLimiter):
        """Adaptive limiter should provide statistics."""
        asyncio.run(adaptive_limiter.wait("https://example.com/page"))
        adaptive_limiter.report_success("example.com")
        adaptive_limiter.report_error("other.com", status_code=500)

        stats = adaptive_limiter.get_stats()

        assert "domains" in stats or "total_requests" in stats


class TestRateLimiterConcurrency:
    """Tests for rate limiter under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_same_domain(self):
        """Concurrent requests to same domain should be serialized."""
        limiter = RateLimiter(default_delay=0.1)

        start = time.time()

        # Launch 3 concurrent requests to same domain
        await asyncio.gather(
            limiter.wait("https://example.com/page1"),
            limiter.wait("https://example.com/page2"),
            limiter.wait("https://example.com/page3"),
        )

        elapsed = time.time() - start

        # Should take at least 2 delays (first immediate, 2 more delayed)
        assert elapsed >= 0.15

    @pytest.mark.asyncio
    async def test_concurrent_different_domains(self):
        """Concurrent requests to different domains should be parallel."""
        limiter = RateLimiter(default_delay=0.1)

        start = time.time()

        # Launch concurrent requests to different domains
        await asyncio.gather(
            limiter.wait("https://domain1.com/page"),
            limiter.wait("https://domain2.com/page"),
            limiter.wait("https://domain3.com/page"),
        )

        elapsed = time.time() - start

        # Should be nearly immediate (all different domains)
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Rate limiter should be thread-safe."""
        limiter = RateLimiter(default_delay=0.01)

        async def make_requests(domain: str, count: int):
            for i in range(count):
                await limiter.wait(f"https://{domain}/page{i}")

        # Run concurrent requests from multiple "threads"
        await asyncio.gather(
            make_requests("domain1.com", 5),
            make_requests("domain2.com", 5),
            make_requests("domain1.com", 5),  # Same domain as first
        )

        # Should complete without errors
        assert True
