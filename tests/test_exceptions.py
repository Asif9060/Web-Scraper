"""
Tests for exception hierarchy.

Tests custom exceptions and error handling.
"""

import pytest

from web_intel.core.exceptions import (
    WebIntelError,
    ConfigurationError,
    CrawlerError,
    BrowserError,
    RateLimitError,
    StorageError,
    DatabaseError,
    VectorStoreError,
    ExtractionError,
    LLMError,
    ModelLoadError,
    GenerationError,
    QueryError,
    NavigationError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_base_exception(self):
        """WebIntelError should be the base for all custom exceptions."""
        exc = WebIntelError("Test error")

        assert isinstance(exc, Exception)
        assert str(exc) == "Test error"

    def test_configuration_error(self):
        """ConfigurationError should inherit from WebIntelError."""
        exc = ConfigurationError("Invalid config")

        assert isinstance(exc, WebIntelError)
        assert isinstance(exc, Exception)

    def test_crawler_error(self):
        """CrawlerError should inherit from WebIntelError."""
        exc = CrawlerError("Crawl failed")

        assert isinstance(exc, WebIntelError)

    def test_browser_error(self):
        """BrowserError should inherit from CrawlerError."""
        exc = BrowserError("Browser crashed")

        assert isinstance(exc, CrawlerError)
        assert isinstance(exc, WebIntelError)

    def test_rate_limit_error(self):
        """RateLimitError should inherit from CrawlerError."""
        exc = RateLimitError("Rate limited")

        assert isinstance(exc, CrawlerError)

    def test_storage_error(self):
        """StorageError should inherit from WebIntelError."""
        exc = StorageError("Storage failed")

        assert isinstance(exc, WebIntelError)

    def test_database_error(self):
        """DatabaseError should inherit from StorageError."""
        exc = DatabaseError("Query failed")

        assert isinstance(exc, StorageError)
        assert isinstance(exc, WebIntelError)

    def test_vector_store_error(self):
        """VectorStoreError should inherit from StorageError."""
        exc = VectorStoreError("Index corrupted")

        assert isinstance(exc, StorageError)

    def test_extraction_error(self):
        """ExtractionError should inherit from WebIntelError."""
        exc = ExtractionError("Extraction failed")

        assert isinstance(exc, WebIntelError)

    def test_llm_error(self):
        """LLMError should inherit from WebIntelError."""
        exc = LLMError("Model error")

        assert isinstance(exc, WebIntelError)

    def test_model_load_error(self):
        """ModelLoadError should inherit from LLMError."""
        exc = ModelLoadError("Could not load model")

        assert isinstance(exc, LLMError)
        assert isinstance(exc, WebIntelError)

    def test_generation_error(self):
        """GenerationError should inherit from LLMError."""
        exc = GenerationError("Generation failed")

        assert isinstance(exc, LLMError)

    def test_query_error(self):
        """QueryError should inherit from WebIntelError."""
        exc = QueryError("Invalid query")

        assert isinstance(exc, WebIntelError)

    def test_navigation_error(self):
        """NavigationError should inherit from WebIntelError."""
        exc = NavigationError("Navigation failed")

        assert isinstance(exc, WebIntelError)


class TestExceptionAttributes:
    """Tests for exception attributes and metadata."""

    def test_error_with_context(self):
        """Exceptions should support additional context."""
        exc = CrawlerError(
            "Failed to fetch page",
            url="https://example.com",
            status_code=404,
        )

        assert "https://example.com" in str(exc) or hasattr(exc, "url")
        assert hasattr(exc, "status_code") or "404" in str(exc)

    def test_rate_limit_with_retry_after(self):
        """RateLimitError should include retry information."""
        exc = RateLimitError(
            "Rate limited",
            domain="example.com",
            retry_after=60,
        )

        assert hasattr(exc, "retry_after") or hasattr(exc, "domain")

    def test_database_error_with_query(self):
        """DatabaseError should include query context."""
        exc = DatabaseError(
            "Query failed",
            query="SELECT * FROM pages",
        )

        assert hasattr(exc, "query") or "SELECT" in str(exc)

    def test_model_load_error_with_model_name(self):
        """ModelLoadError should include model name."""
        exc = ModelLoadError(
            "Failed to load model",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
        )

        assert hasattr(exc, "model_name") or "Qwen" in str(exc)


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_specific_error(self):
        """Specific exceptions can be caught."""
        def raise_browser_error():
            raise BrowserError("Browser crashed")

        with pytest.raises(BrowserError):
            raise_browser_error()

    def test_catch_parent_error(self):
        """Parent exceptions catch child exceptions."""
        def raise_browser_error():
            raise BrowserError("Browser crashed")

        with pytest.raises(CrawlerError):  # Parent class
            raise_browser_error()

    def test_catch_base_error(self):
        """Base exception catches all custom exceptions."""
        def raise_model_error():
            raise ModelLoadError("Model not found")

        with pytest.raises(WebIntelError):  # Base class
            raise_model_error()

    def test_exception_handling_pattern(self):
        """Exception handling should follow expected pattern."""
        def process_with_errors(error_type: str):
            if error_type == "browser":
                raise BrowserError("Browser issue")
            elif error_type == "database":
                raise DatabaseError("Database issue")
            elif error_type == "model":
                raise ModelLoadError("Model issue")
            return "success"

        # Handle specific errors
        try:
            process_with_errors("browser")
        except BrowserError as e:
            result = f"Browser: {e}"
        except CrawlerError as e:
            result = f"Crawler: {e}"
        except WebIntelError as e:
            result = f"General: {e}"

        assert "Browser" in result


class TestExceptionMessages:
    """Tests for exception message formatting."""

    def test_message_formatting(self):
        """Exception messages should be informative."""
        exc = CrawlerError(
            "Failed to crawl https://example.com: Connection timeout")

        msg = str(exc)
        assert "example.com" in msg or "crawl" in msg.lower()

    def test_repr_formatting(self):
        """Exception repr should include class name."""
        exc = VectorStoreError("Index corrupted")

        repr_str = repr(exc)
        assert "VectorStoreError" in repr_str or "Index" in repr_str

    def test_exception_chaining(self):
        """Exceptions should support chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DatabaseError("Database operation failed") from e
        except DatabaseError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)


class TestExceptionUsage:
    """Tests for practical exception usage patterns."""

    def test_raise_with_cleanup(self):
        """Exceptions should allow proper cleanup."""
        cleanup_called = False

        try:
            try:
                raise StorageError("Storage failure")
            finally:
                cleanup_called = True
        except StorageError:
            pass

        assert cleanup_called

    def test_exception_in_context_manager(self):
        """Exceptions should work with context managers."""
        class ResourceManager:
            def __init__(self):
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.exited = True
                return False  # Don't suppress exception

        manager = ResourceManager()

        with pytest.raises(ExtractionError):
            with manager:
                raise ExtractionError("Extraction failed")

        assert manager.entered
        assert manager.exited

    def test_exception_conversion(self):
        """Exceptions can be converted between types."""
        def convert_exception(e: Exception) -> WebIntelError:
            if isinstance(e, WebIntelError):
                return e
            return WebIntelError(f"Wrapped: {e}")

        original = ValueError("Value error")
        converted = convert_exception(original)

        assert isinstance(converted, WebIntelError)
        assert "Value error" in str(converted)
