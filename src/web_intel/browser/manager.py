"""
Browser lifecycle management using Playwright.

Handles browser instance creation, configuration, and cleanup.
Supports chromium, firefox, and webkit engines.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Playwright,
)

from web_intel.config.settings import BrowserSettings
from web_intel.core.exceptions import BrowserError
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class BrowserManager:
    """
    Manages Playwright browser lifecycle.

    Provides async context manager for safe browser creation and cleanup.
    Handles browser type selection, viewport configuration, and context setup.

    Example:
        >>> async with BrowserManager(settings) as manager:
        ...     context = await manager.new_context()
        ...     page = await context.new_page()
        ...     await page.goto("https://example.com")
    """

    def __init__(self, settings: BrowserSettings) -> None:
        """
        Initialize browser manager with configuration.

        Args:
            settings: Browser configuration from app settings
        """
        self.settings = settings
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def start(self) -> None:
        """
        Start Playwright and launch browser.

        Raises:
            BrowserError: If browser fails to launch
        """
        if self._browser is not None:
            logger.warning("Browser already started, skipping launch")
            return

        try:
            logger.info(
                f"Starting {self.settings.browser_type} browser "
                f"(headless={self.settings.headless})"
            )

            self._playwright = await async_playwright().start()

            # Select browser type
            browser_type = getattr(
                self._playwright, self.settings.browser_type)

            # Launch browser with configuration
            self._browser = await browser_type.launch(
                headless=self.settings.headless,
            )

            logger.info("Browser started successfully")

        except Exception as e:
            await self._cleanup()
            raise BrowserError(
                f"Failed to launch browser: {e}",
                details={"browser_type": self.settings.browser_type},
            ) from e

    async def stop(self) -> None:
        """
        Stop browser and cleanup Playwright resources.

        Safe to call multiple times.
        """
        await self._cleanup()
        logger.info("Browser stopped")

    async def _cleanup(self) -> None:
        """Internal cleanup of browser resources."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping Playwright: {e}")
            self._playwright = None

    async def new_context(self) -> BrowserContext:
        """
        Create a new browser context with configured settings.

        Each context has isolated cookies, cache, and storage.
        Use separate contexts for parallel crawling.

        Returns:
            Configured BrowserContext

        Raises:
            BrowserError: If browser not started or context creation fails
        """
        if self._browser is None:
            raise BrowserError("Browser not started. Call start() first.")

        try:
            context_options: dict = {
                "viewport": {
                    "width": self.settings.viewport_width,
                    "height": self.settings.viewport_height,
                },
                "ignore_https_errors": self.settings.ignore_https_errors,
            }

            if self.settings.user_agent:
                context_options["user_agent"] = self.settings.user_agent

            context = await self._browser.new_context(**context_options)

            # Set default timeouts
            context.set_default_timeout(self.settings.timeout_ms)
            context.set_default_navigation_timeout(
                self.settings.navigation_timeout_ms)

            logger.debug("Created new browser context")
            return context

        except Exception as e:
            raise BrowserError(
                f"Failed to create browser context: {e}",
            ) from e

    @property
    def is_running(self) -> bool:
        """Check if browser is currently running."""
        return self._browser is not None and self._browser.is_connected()

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.stop()


@asynccontextmanager
async def create_browser(
    settings: BrowserSettings,
) -> AsyncGenerator[BrowserManager, None]:
    """
    Convenience context manager for browser creation.

    Args:
        settings: Browser configuration

    Yields:
        Configured and started BrowserManager

    Example:
        >>> async with create_browser(settings) as browser:
        ...     context = await browser.new_context()
    """
    manager = BrowserManager(settings)
    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()
