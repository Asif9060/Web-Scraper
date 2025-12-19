"""
Custom exceptions for the Web Intelligence System.

Provides a hierarchy of exceptions for precise error handling across
all subsystems. All exceptions inherit from WebIntelError.

Exception Hierarchy:
    WebIntelError (base)
    ├── ConfigurationError
    ├── BrowserError
    │   ├── NavigationError
    │   └── PageLoadError
    ├── CrawlerError
    │   ├── RateLimitError
    │   └── RobotsBlockedError
    ├── StorageError
    │   ├── DatabaseError
    │   └── VectorStorageError
    ├── ExtractionError
    │   └── ContentExtractionError
    ├── LLMError
    │   ├── LocalLLMError
    │   │   ├── ModelLoadError
    │   │   └── InferenceError
    │   └── APILLMError
    │       ├── APIConnectionError
    │       ├── APIRateLimitError
    │       └── APIAuthenticationError
    ├── EmbeddingError
    └── QueryError
        ├── QueryInterpretationError
        └── RetrievalError
"""

from typing import Any


class WebIntelError(Exception):
    """
    Base exception for all Web Intelligence System errors.

    All custom exceptions inherit from this class, allowing for
    catch-all handling when needed.

    Attributes:
        message: Human-readable error description
        details: Optional dictionary with additional context
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(
                f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


class RetryableError(WebIntelError):
    """
    Mixin/marker class for errors that can be retried.

    Errors inheriting from this class indicate that the operation
    may succeed if attempted again after a delay.

    Attributes:
        retry_after: Suggested delay in seconds before retry (optional)
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(WebIntelError):
    """
    Error in configuration loading or validation.

    Raised when:
    - Configuration file is missing or malformed
    - Required settings are missing
    - Setting values fail validation
    """

    pass


# =============================================================================
# Browser Errors
# =============================================================================


class BrowserError(WebIntelError):
    """
    Base error for browser/Playwright operations.

    Raised for general browser-related failures not covered by
    more specific subclasses.
    """

    pass


class NavigationError(BrowserError, RetryableError):
    """
    Error during page navigation.

    Raised when:
    - URL is unreachable
    - Navigation times out
    - Redirect loop detected

    This is retryable as network issues may be transient.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        details = details or {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details, retry_after)
        self.url = url
        self.status_code = status_code


class PageLoadError(BrowserError, RetryableError):
    """
    Error loading page content after navigation.

    Raised when:
    - Page content fails to render
    - Required elements don't appear
    - JavaScript execution fails
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        details = details or {}
        if url:
            details["url"] = url
        super().__init__(message, details, retry_after)
        self.url = url


# =============================================================================
# Crawler Errors
# =============================================================================


class CrawlerError(WebIntelError):
    """
    Base error for crawling operations.

    Raised for general crawl-related failures not covered by
    more specific subclasses.
    """

    pass


class RateLimitError(CrawlerError, RetryableError):
    """
    Error when rate limit is exceeded.

    Raised when:
    - Server returns 429 Too Many Requests
    - Crawl rate exceeds configured limits
    - External rate limiting detected

    The retry_after attribute indicates when to resume.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if url:
            details["url"] = url
        super().__init__(message, details, retry_after)
        self.url = url


class RobotsBlockedError(CrawlerError):
    """
    Error when URL is blocked by robots.txt.

    This is NOT retryable as robots.txt rules are persistent.
    Crawling should skip this URL.
    """

    def __init__(
        self,
        message: str,
        url: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["url"] = url
        super().__init__(message, details)
        self.url = url


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(WebIntelError):
    """
    Base error for storage operations.

    Raised for general storage-related failures not covered by
    more specific subclasses.
    """

    pass


class DatabaseError(StorageError):
    """
    Error in SQLite database operations.

    Raised when:
    - Database connection fails
    - Query execution fails
    - Schema migration fails
    - Constraint violations occur
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if query:
            # Truncate long queries for readability
            details["query"] = query[:200] + \
                "..." if len(query) > 200 else query
        super().__init__(message, details)
        self.query = query


class VectorStorageError(StorageError):
    """
    Error in vector storage operations.

    Raised when:
    - Vector dimension mismatch
    - Index creation fails
    - Similarity search fails
    """

    pass


# =============================================================================
# Extraction Errors
# =============================================================================


class ExtractionError(WebIntelError):
    """
    Base error for content extraction operations.

    Raised for general extraction failures not covered by
    more specific subclasses.
    """

    pass


class ContentExtractionError(ExtractionError):
    """
    Error extracting structured content from page.

    Raised when:
    - HTML parsing fails
    - Required content elements not found
    - Content structure is unexpected
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        selector: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if url:
            details["url"] = url
        if selector:
            details["selector"] = selector
        super().__init__(message, details)
        self.url = url
        self.selector = selector


# =============================================================================
# LLM Errors
# =============================================================================


class LLMError(WebIntelError):
    """
    Base error for LLM operations (both local and API).

    Raised for general LLM-related failures not covered by
    more specific subclasses.
    """

    pass


class LocalLLMError(LLMError):
    """
    Base error for local LLM (HuggingFace Transformers) operations.
    """

    pass


class ModelLoadError(LocalLLMError):
    """
    Error loading local LLM model.

    Raised when:
    - Model files not found
    - Insufficient memory to load model
    - Model configuration is invalid
    - Tokenizer loading fails
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, details)
        self.model_name = model_name


class InferenceError(LocalLLMError, RetryableError):
    """
    Error during local LLM inference.

    Raised when:
    - Generation fails
    - Output parsing fails
    - Memory exhausted during inference

    May be retryable with smaller input or after memory cleanup.
    """

    pass


class APILLMError(LLMError):
    """
    Base error for API LLM (Anthropic Claude) operations.
    """

    pass


class APIConnectionError(APILLMError, RetryableError):
    """
    Error connecting to LLM API.

    Raised when:
    - Network connection fails
    - API endpoint unreachable
    - Request times out

    This is retryable as network issues may be transient.
    """

    pass


class APIRateLimitError(APILLMError, RetryableError):
    """
    Error when API rate limit is exceeded.

    Raised when API returns rate limit error (429).
    The retry_after attribute indicates when to resume.
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details, retry_after)


class APIAuthenticationError(APILLMError):
    """
    Error authenticating with LLM API.

    Raised when:
    - API key is missing
    - API key is invalid
    - API key lacks required permissions

    This is NOT retryable without fixing credentials.
    """

    pass


# =============================================================================
# Embedding Errors
# =============================================================================


class EmbeddingError(WebIntelError):
    """
    Error in embedding generation.

    Raised when:
    - Embedding model fails to load
    - Text encoding fails
    - Batch processing fails
    """

    def __init__(
        self,
        message: str,
        text_length: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if text_length:
            details["text_length"] = text_length
        super().__init__(message, details)
        self.text_length = text_length


# =============================================================================
# Query Errors
# =============================================================================


class QueryError(WebIntelError):
    """
    Base error for query processing operations.

    Raised for general query-related failures not covered by
    more specific subclasses.
    """

    pass


class QueryInterpretationError(QueryError):
    """
    Error interpreting user query.

    Raised when:
    - Query cannot be parsed
    - Query intent is ambiguous
    - Required clarification not provided
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query[:100] + \
                "..." if len(query) > 100 else query
        super().__init__(message, details)
        self.query = query


class RetrievalError(QueryError):
    """
    Error retrieving data to answer query.

    Raised when:
    - No relevant documents found
    - Retrieval operation fails
    - Ranking fails
    """

    pass


# =============================================================================
# Utility Functions
# =============================================================================


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error indicates a retryable condition
    """
    return isinstance(error, RetryableError)


def get_retry_delay(error: Exception, default: float = 5.0) -> float:
    """
    Get the recommended retry delay for an error.

    Args:
        error: The exception to check
        default: Default delay if not specified by error

    Returns:
        Recommended delay in seconds before retry
    """
    if isinstance(error, RetryableError) and error.retry_after is not None:
        return error.retry_after
    return default
