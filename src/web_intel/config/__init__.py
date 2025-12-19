"""
Configuration module for Web Intelligence System.

Provides Pydantic-based settings management with YAML file support
and environment variable overrides.
"""

from web_intel.config.settings import (
    Settings,
    BrowserSettings,
    CrawlerSettings,
    StorageSettings,
    LocalLLMSettings,
    APILLMSettings,
    EmbeddingSettings,
    NavigationSettings,
    LoggingSettings,
)
from web_intel.config.loader import load_config, get_settings

__all__ = [
    "Settings",
    "BrowserSettings",
    "CrawlerSettings",
    "StorageSettings",
    "LocalLLMSettings",
    "APILLMSettings",
    "EmbeddingSettings",
    "NavigationSettings",
    "LoggingSettings",
    "load_config",
    "get_settings",
]
