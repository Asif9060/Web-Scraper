"""
Web Intelligence System - A website crawling and question-answering system.

This package provides tools to crawl websites, extract and understand content,
and answer questions about the crawled data using a hybrid LLM approach.
"""

from web_intel.config import Settings, load_config
from web_intel.utils.logging import setup_logging, get_logger
from web_intel.core.exceptions import WebIntelError
from web_intel.extraction import PageParser, ContentExtractor, MetadataExtractor

__version__ = "0.1.0"
__author__ = "Web Intel Team"

__all__ = [
    "Settings",
    "load_config",
    "setup_logging",
    "get_logger",
    "WebIntelError",
    "PageParser",
    "ContentExtractor",
    "MetadataExtractor",
]
