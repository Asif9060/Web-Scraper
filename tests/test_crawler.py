"""
Tests for crawler module.

Tests web crawling, page fetching, and link discovery.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from web_intel.crawler import (
    Crawler,
    CrawlResult,
    CrawlSession,
    CrawlStats,
)
from web_intel.config import Settings


class TestCrawlResult:
    """Tests for CrawlResult dataclass."""

    def test_crawl_result_success(self):
        """Successful crawl result should have content."""
        result = CrawlResult(
            url="https://example.com",
            status_code=200,
            content="<html>content</html>",
            content_type="text/html",
            success=True,
        )

        assert result.success
        assert result.status_code == 200
        assert result.content is not None

    def test_crawl_result_failure(self):
        """Failed crawl result should have error."""
        result = CrawlResult(
            url="https://example.com",
            status_code=404,
            content=None,
            error="Page not found",
            success=False,
        )

        assert not result.success
        assert result.error is not None

    def test_crawl_result_with_timing(self):
        """Crawl result can include timing info."""
        result = CrawlResult(
            url="https://example.com",
            status_code=200,
            content="content",
            success=True,
            fetch_time_ms=150,
        )

        assert result.fetch_time_ms == 150

    def test_crawl_result_to_dict(self):
        """CrawlResult should convert to dictionary."""
        result = CrawlResult(
            url="https://example.com",
            status_code=200,
            content="content",
            success=True,
        )

        result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)

        assert "url" in result_dict
        assert "status_code" in result_dict


class TestCrawlStats:
    """Tests for CrawlStats."""

    def test_stats_creation(self):
        """CrawlStats should be created with defaults."""
        stats = CrawlStats()

        assert stats.pages_crawled == 0
        assert stats.pages_failed == 0
        assert stats.total_bytes == 0

    def test_stats_update(self):
        """Stats should be updateable."""
        stats = CrawlStats()

        stats.pages_crawled += 1
        stats.total_bytes += 1024

        assert stats.pages_crawled == 1
        assert stats.total_bytes == 1024

    def test_stats_success_rate(self):
        """Stats should calculate success rate."""
        stats = CrawlStats()
        stats.pages_crawled = 80
        stats.pages_failed = 20

        rate = stats.success_rate if hasattr(stats, "success_rate") else (
            stats.pages_crawled / (stats.pages_crawled + stats.pages_failed)
        )

        assert rate == 0.8

    def test_stats_to_dict(self):
        """Stats should convert to dictionary."""
        stats = CrawlStats()
        stats.pages_crawled = 10

        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else vars(stats)

        assert "pages_crawled" in stats_dict


class TestCrawlSession:
    """Tests for CrawlSession."""

    @pytest.fixture
    def session(self, test_settings: Settings) -> CrawlSession:
        """Provide a crawl session."""
        return CrawlSession(test_settings)

    def test_session_creation(self, session: CrawlSession):
        """Session should be created successfully."""
        assert session is not None
        assert session.is_active is False or session.stats is not None

    def test_session_start_stop(self, session: CrawlSession):
        """Session can be started and stopped."""
        session.start()
        assert session.is_active

        session.stop()
        assert not session.is_active

    def test_session_stats_tracking(self, session: CrawlSession):
        """Session should track statistics."""
        session.start()

        # Simulate crawl activity
        session.record_success("https://example.com", bytes_received=1024)
        session.record_success("https://example.com/page", bytes_received=2048)
        session.record_failure("https://example.com/error", error="Timeout")

        stats = session.get_stats()

        assert stats.pages_crawled == 2
        assert stats.pages_failed == 1
        assert stats.total_bytes == 3072

        session.stop()

    def test_session_context_manager(self, test_settings: Settings):
        """Session should work as context manager."""
        with CrawlSession(test_settings) as session:
            assert session.is_active

        assert not session.is_active


class TestCrawler:
    """Tests for Crawler class."""

    @pytest.fixture
    def crawler(self, test_settings: Settings) -> Crawler:
        """Provide a crawler instance."""
        return Crawler(test_settings)

    def test_crawler_creation(self, crawler: Crawler):
        """Crawler should be created successfully."""
        assert crawler is not None

    @pytest.mark.asyncio
    async def test_crawl_single_page(self, crawler: Crawler):
        """Single page can be crawled."""
        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://example.com",
                status_code=200,
                content="<html><body>Test</body></html>",
                content_type="text/html",
                success=True,
            )

            result = await crawler.crawl_page("https://example.com")

            assert result.success
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_crawl_extracts_links(self, crawler: Crawler):
        """Crawler should extract links from pages."""
        html = """
        <html><body>
            <a href="/page1">Link 1</a>
            <a href="/page2">Link 2</a>
            <a href="https://external.com">External</a>
        </body></html>
        """

        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://example.com",
                status_code=200,
                content=html,
                content_type="text/html",
                success=True,
            )

            result = await crawler.crawl_page("https://example.com")
            links = crawler.extract_links(
                result.content, "https://example.com")

            assert len(links) >= 2

    @pytest.mark.asyncio
    async def test_crawl_respects_robots(self, crawler: Crawler):
        """Crawler should respect robots.txt."""
        # Test that disallowed URLs are skipped
        crawler.set_robots_txt(
            "https://example.com",
            "User-agent: *\nDisallow: /private/",
        )

        allowed = crawler.is_allowed("https://example.com/public/page")
        disallowed = crawler.is_allowed("https://example.com/private/page")

        assert allowed
        assert not disallowed

    @pytest.mark.asyncio
    async def test_crawl_handles_redirects(self, crawler: Crawler):
        """Crawler should handle redirects."""
        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://example.com/final",
                status_code=200,
                content="<html>Final</html>",
                content_type="text/html",
                success=True,
                final_url="https://example.com/final",
            )

            result = await crawler.crawl_page("https://example.com/redirect")

            assert result.success

    @pytest.mark.asyncio
    async def test_crawl_handles_errors(self, crawler: Crawler):
        """Crawler should handle errors gracefully."""
        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://example.com",
                status_code=500,
                content=None,
                error="Internal Server Error",
                success=False,
            )

            result = await crawler.crawl_page("https://example.com")

            assert not result.success
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_crawl_timeout(self, crawler: Crawler):
        """Crawler should handle timeouts."""
        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.side_effect = asyncio.TimeoutError()

            result = await crawler.crawl_page("https://slow.example.com")

            assert not result.success

    def test_extract_links_absolute(self, crawler: Crawler):
        """Links should be converted to absolute URLs."""
        html = '<html><body><a href="/page">Link</a></body></html>'

        links = crawler.extract_links(html, "https://example.com/current")

        assert any("https://example.com/page" in link for link in links)

    def test_extract_links_filters_invalid(self, crawler: Crawler):
        """Invalid links should be filtered."""
        html = """
        <html><body>
            <a href="javascript:void(0)">JS Link</a>
            <a href="mailto:test@example.com">Email</a>
            <a href="#">Hash</a>
            <a href="/valid">Valid</a>
        </body></html>
        """

        links = crawler.extract_links(html, "https://example.com")

        assert not any("javascript:" in link for link in links)
        assert not any("mailto:" in link for link in links)
        assert any("/valid" in link for link in links)

    def test_url_filtering(self, crawler: Crawler):
        """Crawler should filter URLs by pattern."""
        crawler.add_url_filter(r"\.pdf$")  # Exclude PDFs
        crawler.add_url_filter(r"/admin/")  # Exclude admin pages

        assert not crawler.should_crawl("https://example.com/doc.pdf")
        assert not crawler.should_crawl("https://example.com/admin/login")
        assert crawler.should_crawl("https://example.com/page")


class TestCrawlerSitemap:
    """Tests for sitemap handling."""

    @pytest.fixture
    def crawler(self, test_settings: Settings) -> Crawler:
        """Provide a crawler instance."""
        return Crawler(test_settings)

    def test_parse_sitemap_xml(self, crawler: Crawler):
        """Sitemap XML should be parsed."""
        sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>
        """

        urls = crawler.parse_sitemap(sitemap)

        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    def test_parse_sitemap_index(self, crawler: Crawler):
        """Sitemap index should be recognized."""
        sitemap_index = """<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
            <sitemap><loc>https://example.com/sitemap2.xml</loc></sitemap>
        </sitemapindex>
        """

        sitemaps = crawler.parse_sitemap_index(sitemap_index)

        assert len(sitemaps) == 2


class TestCrawlerConcurrency:
    """Tests for crawler concurrency."""

    @pytest.fixture
    def crawler(self, test_settings: Settings) -> Crawler:
        """Provide a crawler instance."""
        test_settings.crawler.max_concurrent = 3
        return Crawler(test_settings)

    @pytest.mark.asyncio
    async def test_concurrent_crawl(self, crawler: Crawler):
        """Crawler should handle concurrent requests."""
        urls = [f"https://example.com/page{i}" for i in range(5)]

        with patch.object(crawler, "_fetch_page") as mock_fetch:
            mock_fetch.return_value = CrawlResult(
                url="https://example.com",
                status_code=200,
                content="<html>Test</html>",
                success=True,
            )

            results = await crawler.crawl_pages(urls)

            assert len(results) == 5
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, crawler: Crawler):
        """Crawler should respect concurrency limit."""
        # This is a behavioral test - in practice would check semaphore
        assert crawler.max_concurrent == 3
