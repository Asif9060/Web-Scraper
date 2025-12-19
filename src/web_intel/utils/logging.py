"""
Logging configuration and utilities for the Web Intelligence System.

Provides centralized logging setup with support for:
- Console and file output
- Log rotation
- Configurable formatting
- Per-module loggers
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from web_intel.config.settings import LoggingSettings


# Root logger name for the application
ROOT_LOGGER_NAME = "web_intel"

# Track whether logging has been configured
_logging_configured = False


def setup_logging(settings: "LoggingSettings | None" = None) -> logging.Logger:
    """
    Configure the application logging system.

    Sets up handlers for console and/or file output based on settings.
    Should be called once at application startup.

    Args:
        settings: Logging configuration. If None, uses sensible defaults.

    Returns:
        The configured root logger for the application.
    """
    global _logging_configured

    # Get or create the root logger for our application
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Avoid duplicate handlers if called multiple times
    if _logging_configured:
        return logger

    # Clear any existing handlers
    logger.handlers.clear()

    # Use defaults if no settings provided
    if settings is None:
        level = logging.INFO
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        log_to_console = True
        file_path = None
        max_file_size_mb = 10
        backup_count = 3
    else:
        level = getattr(logging, settings.level)
        log_format = settings.format
        date_format = settings.date_format
        log_to_console = settings.log_to_console
        file_path = settings.file_path
        max_file_size_mb = settings.max_file_size_mb
        backup_count = settings.backup_count

    # Set the logging level
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if path is specified
    if file_path is not None:
        file_handler = _create_file_handler(
            file_path=file_path,
            max_bytes=max_file_size_mb * 1024 * 1024,
            backup_count=backup_count,
            level=level,
            formatter=formatter,
        )
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    _logging_configured = True

    return logger


def _create_file_handler(
    file_path: Path,
    max_bytes: int,
    backup_count: int,
    level: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    """
    Create a rotating file handler for logging.

    Args:
        file_path: Path to the log file
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        level: Logging level for the handler
        formatter: Formatter for log messages

    Returns:
        Configured RotatingFileHandler
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)

    return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    All loggers are children of the application root logger, ensuring
    consistent configuration across the application.

    Args:
        name: Module name for the logger. If None, returns the root logger.
              Typically pass __name__ to get a module-specific logger.

    Returns:
        Logger instance configured as child of application root.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if name is None:
        return logging.getLogger(ROOT_LOGGER_NAME)

    # If name already starts with our root, use it directly
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)

    # Otherwise, make it a child of our root logger
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def reset_logging() -> None:
    """
    Reset the logging configuration.

    Removes all handlers and resets the configured flag.
    Useful for testing or reconfiguration.
    """
    global _logging_configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Close and remove all handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    _logging_configured = False


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to log messages.

    Useful for adding request IDs, URLs, or other context to all
    log messages within a specific scope.

    Example:
        >>> base_logger = get_logger(__name__)
        >>> logger = LoggerAdapter(base_logger, {"url": "https://example.com"})
        >>> logger.info("Page loaded")  # Outputs: "Page loaded [url=https://example.com]"
    """

    def process(
        self, msg: str, kwargs: dict
    ) -> tuple[str, dict]:
        """
        Process the logging message to add extra context.

        Args:
            msg: Original log message
            kwargs: Keyword arguments passed to the logging call

        Returns:
            Tuple of (modified message, kwargs)
        """
        if self.extra:
            context_str = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
            msg = f"{msg} {context_str}"
        return msg, kwargs


def get_logger_with_context(
    name: str | None = None,
    **context: str,
) -> LoggerAdapter:
    """
    Get a logger with additional context that appears in all messages.

    Args:
        name: Module name for the logger
        **context: Key-value pairs to include in all log messages

    Returns:
        LoggerAdapter with context attached

    Example:
        >>> logger = get_logger_with_context(__name__, task="crawl", domain="example.com")
        >>> logger.info("Started")  # Outputs with [task=crawl] [domain=example.com]
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
