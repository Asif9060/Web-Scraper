"""
Browser module for Web Intelligence System.

Provides Playwright-based browser automation with:
- Browser lifecycle management
- Page context wrapper with utilities
- Common navigation and interaction actions
"""

from web_intel.browser.manager import BrowserManager
from web_intel.browser.page_context import PageContext, PageContent
from web_intel.browser.actions import (
    extract_links,
    extract_text_content,
    wait_for_page_ready,
    safe_click,
    take_screenshot,
)

__all__ = [
    "BrowserManager",
    "PageContext",
    "PageContent",
    "extract_links",
    "extract_text_content",
    "wait_for_page_ready",
    "safe_click",
    "take_screenshot",
]
