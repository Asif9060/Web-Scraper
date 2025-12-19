"""
Crawl orchestration for the Web Intelligence System.

Coordinates all crawling components: browser, queue, filters, rate limiting.
Provides the main entry point for crawling websites.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Callable, Awaitable

from web_intel.browser.manager import BrowserManager
from web_intel.browser.page_context import PageContext, PageContent
from web_intel.config.settings import BrowserSettings, CrawlerSettings
from web_intel.core.exceptions import (
    CrawlerError,
    NavigationError,
    PageLoadError,
    RateLimitError,
    RobotsBlockedError,
    is_retryable,
    get_retry_delay,
)
from web_intel.crawler.queue import URLQueue, URLPriority, normalize_url
from web_intel.crawler.filters import CombinedFilter
from web_intel.crawler.rate_limiter import AdaptiveRateLimiter
from web_intel.utils.logging import get_logger
from web_intel.utils.metrics import Metrics

logger = get_logger(__name__)


class CrawlStatus(str, Enum):
    """Status of a crawl operation."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CrawlProgress:
    """
    Current progress of a crawl operation.

    Updated in real-time during crawling.
    """

    status: CrawlStatus = CrawlStatus.PENDING
    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    pages_pending: int = 0
    current_url: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since crawl started."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def pages_per_minute(self) -> float:
        """Average pages crawled per minute."""
        elapsed = self.elapsed_seconds
        if elapsed < 1:
            return 0.0
        return (self.pages_crawled / elapsed) * 60


@dataclass
class CrawlResult:
    """
    Result of a completed crawl operation.

    Contains summary statistics and any errors encountered.
    """

    seed_url: str
    status: CrawlStatus
    pages_crawled: int
    pages_failed: int
    pages_skipped: int
    duration_seconds: float
    started_at: datetime
    completed_at: datetime
    errors: list[dict] = field(default_factory=list)


# Type alias for page callback
PageCallback = Callable[[PageContent], Awaitable[None]]


class Crawler:
    """
    Main crawler orchestrating all crawling operations.

    Coordinates browser automation, URL queue, filtering,
    and rate limiting to crawl websites efficiently.

    Example:
        >>> crawler = Crawler(browser_settings, crawler_settings)
        >>> async for content in crawler.crawl("https://example.com"):
        ...     print(f"Crawled: {content.title}")
        ...     # Process content...
    """

    def __init__(
        self,
        browser_settings: BrowserSettings,
        crawler_settings: CrawlerSettings,
    ) -> None:
        """
        Initialize crawler.

        Args:
            browser_settings: Browser configuration
            crawler_settings: Crawler configuration
        """
        self.browser_settings = browser_settings
        self.crawler_settings = crawler_settings

        # Components (initialized on crawl start)
        self._browser_manager: BrowserManager | None = None
        self._queue: URLQueue | None = None
        self._filter: CombinedFilter | None = None
        self._rate_limiter: AdaptiveRateLimiter | None = None

        # State
        self._progress = CrawlProgress()
        self._cancel_requested = False
        self._pause_requested = False

    @property
    def progress(self) -> CrawlProgress:
        """Get current crawl progress."""
        return self._progress

    def _init_components(self, seed_url: str) -> None:
        """Initialize crawl components for a new crawl."""
        self._queue = URLQueue(max_depth=self.crawler_settings.max_depth)

        self._filter = CombinedFilter(
            patterns_include=self.crawler_settings.url_patterns_include,
            patterns_exclude=self.crawler_settings.url_patterns_exclude,
            follow_external=self.crawler_settings.follow_external_links,
            respect_robots=self.crawler_settings.respect_robots_txt,
            user_agent=self.browser_settings.user_agent or "WebIntelBot/1.0",
        )
        self._filter.add_seed_domain(seed_url)

        self._rate_limiter = AdaptiveRateLimiter(
            base_delay=self.crawler_settings.delay_seconds,
            min_delay=max(0.5, self.crawler_settings.delay_seconds / 2),
            max_delay=30.0,
        )

        self._progress = CrawlProgress()
        self._cancel_requested = False
        self._pause_requested = False

    async def crawl(
        self,
        seed_url: str,
        on_page: PageCallback | None = None,
    ) -> AsyncIterator[PageContent]:
        """
        Crawl a website starting from seed URL.

        Yields page content as pages are crawled.
        Optionally calls on_page callback for each page.

        Args:
            seed_url: Starting URL for crawl
            on_page: Optional callback for each crawled page

        Yields:
            PageContent for each successfully crawled page

        Raises:
            CrawlerError: If crawl fails to start
        """
        seed_url = normalize_url(seed_url)
        self._init_components(seed_url)

        logger.info(f"Starting crawl from: {seed_url}")
        self._progress.status = CrawlStatus.RUNNING
        self._progress.started_at = datetime.now(timezone.utc)

        # Add seed URL to queue
        await self._queue.put(seed_url, priority=URLPriority.CRITICAL, depth=0)

        try:
            async with BrowserManager(self.browser_settings) as browser:
                self._browser_manager = browser
                context = await browser.new_context()

                try:
                    page = await context.new_page()
                    page_ctx = PageContext(page)

                    async for content in self._crawl_loop(page_ctx):
                        if on_page:
                            await on_page(content)
                        yield content

                finally:
                    await context.close()

        except Exception as e:
            self._progress.status = CrawlStatus.FAILED
            self._progress.error_message = str(e)
            logger.error(f"Crawl failed: {e}")
            raise CrawlerError(f"Crawl failed: {e}") from e

        finally:
            self._progress.completed_at = datetime.now(timezone.utc)
            if self._progress.status == CrawlStatus.RUNNING:
                self._progress.status = CrawlStatus.COMPLETED

            logger.info(
                f"Crawl finished: {self._progress.pages_crawled} pages, "
                f"{self._progress.pages_failed} failed, "
                f"{self._progress.elapsed_seconds:.1f}s"
            )

    async def _crawl_loop(
        self,
        page_ctx: PageContext,
    ) -> AsyncIterator[PageContent]:
        """
        Main crawl loop processing URLs from queue.

        Args:
            page_ctx: Page context for browser operations

        Yields:
            PageContent for each crawled page
        """
        max_pages = self.crawler_settings.max_pages

        while not self._cancel_requested:
            # Check pause
            while self._pause_requested:
                self._progress.status = CrawlStatus.PAUSED
                await asyncio.sleep(0.5)
            self._progress.status = CrawlStatus.RUNNING

            # Check page limit
            if self._progress.pages_crawled >= max_pages:
                logger.info(f"Reached page limit: {max_pages}")
                break

            # Get next URL
            queued = await self._queue.get()
            if queued is None:
                logger.debug("Queue empty, crawl complete")
                break

            self._progress.current_url = queued.url
            self._progress.pages_pending = self._queue.pending_count

            # Process URL
            try:
                content = await self._process_url(page_ctx, queued)
                if content:
                    self._progress.pages_crawled += 1
                    Metrics.get().increment("pages_crawled")
                    yield content
                else:
                    self._progress.pages_skipped += 1

            except Exception as e:
                self._progress.pages_failed += 1
                logger.error(f"Failed to process {queued.url}: {e}")

        if self._cancel_requested:
            self._progress.status = CrawlStatus.CANCELLED
            logger.info("Crawl cancelled by user")

    async def _process_url(
        self,
        page_ctx: PageContext,
        queued,
    ) -> PageContent | None:
        """
        Process a single URL from the queue.

        Args:
            page_ctx: Page context
            queued: Queued URL entry

        Returns:
            PageContent if successful, None if skipped
        """
        url = queued.url

        # Check filters
        if not await self._filter.is_allowed(url):
            logger.debug(f"Filtered out: {url}")
            await self._queue.complete(url)
            return None

        # Apply rate limiting
        await self._rate_limiter.acquire(url)

        try:
            # Navigate and extract
            logger.debug(f"Crawling: {url}")
            content = await page_ctx.navigate_and_extract(url)

            # Signal success to rate limiter
            await self._rate_limiter.release_success(url)

            # Mark complete
            await self._queue.complete(url)

            # Discover and queue new links
            await self._discover_links(content, queued.depth)

            logger.info(
                f"[{self._progress.pages_crawled + 1}/{self.crawler_settings.max_pages}] "
                f"Crawled: {content.title[:50] if content.title else url}"
            )

            return content

        except RobotsBlockedError:
            logger.debug(f"Blocked by robots.txt: {url}")
            await self._queue.complete(url)
            return None

        except (NavigationError, PageLoadError) as e:
            await self._rate_limiter.release_error(url, retry_after=get_retry_delay(e))

            if is_retryable(e) and queued.retry_count < self.crawler_settings.max_retries:
                await self._queue.fail(url, max_retries=self.crawler_settings.max_retries)
                logger.warning(f"Will retry: {url} ({e})")
            else:
                await self._queue.complete(url)
                logger.warning(f"Giving up on: {url} ({e})")

            return None

        except Exception as e:
            await self._rate_limiter.release_error(url)
            await self._queue.complete(url)
            logger.error(f"Unexpected error for {url}: {e}")
            return None

    async def _discover_links(self, content: PageContent, current_depth: int) -> None:
        """
        Discover and queue links from page content.

        Args:
            content: Extracted page content
            current_depth: Current crawl depth
        """
        if current_depth >= self.crawler_settings.max_depth:
            return

        new_depth = current_depth + 1

        # Filter links before queueing
        allowed_links = await self._filter.filter_urls(content.links)

        # Queue discovered links
        added = await self._queue.put_many(
            allowed_links,
            priority=URLPriority.NORMAL,
            depth=new_depth,
            parent_url=content.url,
        )

        if added > 0:
            logger.debug(f"Discovered {added} new links from {content.url}")

    def cancel(self) -> None:
        """Request crawl cancellation."""
        logger.info("Cancellation requested")
        self._cancel_requested = True

    def pause(self) -> None:
        """Pause the crawl."""
        logger.info("Pause requested")
        self._pause_requested = True

    def resume(self) -> None:
        """Resume a paused crawl."""
        logger.info("Resume requested")
        self._pause_requested = False

    def get_result(self) -> CrawlResult:
        """
        Get crawl result after completion.

        Returns:
            CrawlResult with summary statistics
        """
        return CrawlResult(
            seed_url=self._progress.current_url or "",
            status=self._progress.status,
            pages_crawled=self._progress.pages_crawled,
            pages_failed=self._progress.pages_failed,
            pages_skipped=self._progress.pages_skipped,
            duration_seconds=self._progress.elapsed_seconds,
            started_at=self._progress.started_at or datetime.now(timezone.utc),
            completed_at=self._progress.completed_at or datetime.now(
                timezone.utc),
            errors=[],
        )


async def crawl_website(
    seed_url: str,
    browser_settings: BrowserSettings,
    crawler_settings: CrawlerSettings,
    on_page: PageCallback | None = None,
) -> CrawlResult:
    """
    Convenience function to crawl a website.

    Collects all pages and returns result.

    Args:
        seed_url: Starting URL
        browser_settings: Browser configuration
        crawler_settings: Crawler configuration
        on_page: Optional callback for each page

    Returns:
        CrawlResult with statistics
    """
    crawler = Crawler(browser_settings, crawler_settings)

    async for _ in crawler.crawl(seed_url, on_page):
        pass

    return crawler.get_result()
