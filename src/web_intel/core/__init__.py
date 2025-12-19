"""
Core module for Web Intelligence System.

Contains foundational types, protocols, and exceptions used throughout the application.
"""

from web_intel.core.exceptions import (
    WebIntelError,
    ConfigurationError,
    BrowserError,
    NavigationError,
    PageLoadError,
    CrawlerError,
    RateLimitError,
    RobotsBlockedError,
    StorageError,
    DatabaseError,
    VectorStorageError,
    ExtractionError,
    ContentExtractionError,
    LLMError,
    LocalLLMError,
    ModelLoadError,
    InferenceError,
    APILLMError,
    APIConnectionError,
    APIRateLimitError,
    APIAuthenticationError,
    EmbeddingError,
    QueryError,
    QueryInterpretationError,
    RetrievalError,
    RetryableError,
)

__all__ = [
    # Base
    "WebIntelError",
    "ConfigurationError",
    # Browser
    "BrowserError",
    "NavigationError",
    "PageLoadError",
    # Crawler
    "CrawlerError",
    "RateLimitError",
    "RobotsBlockedError",
    # Storage
    "StorageError",
    "DatabaseError",
    "VectorStorageError",
    # Extraction
    "ExtractionError",
    "ContentExtractionError",
    # LLM
    "LLMError",
    "LocalLLMError",
    "ModelLoadError",
    "InferenceError",
    "APILLMError",
    "APIConnectionError",
    "APIRateLimitError",
    "APIAuthenticationError",
    # Embedding
    "EmbeddingError",
    # Query
    "QueryError",
    "QueryInterpretationError",
    "RetrievalError",
    # Retry
    "RetryableError",
]
