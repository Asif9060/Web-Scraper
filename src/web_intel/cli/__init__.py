"""
CLI module for Web Intelligence System.

Provides command-line interface using Typer:
- crawl: Crawl and index websites
- query: Ask questions about crawled content
- status: View system and crawl status
- config: Configuration management
"""

from web_intel.cli.main import app

__all__ = ["app"]
