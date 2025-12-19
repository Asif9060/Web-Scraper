"""
Utilities module for Web Intelligence System.

Provides common utilities including logging setup and helpers.
"""

from web_intel.utils.logging import setup_logging, get_logger
from web_intel.utils.metrics import (
    Metrics,
    TimingStats,
    increment_pages_crawled,
    increment_llm_calls,
    increment_embeddings_generated,
    observe_llm_latency,
    observe_query_latency,
    time_llm_call,
    time_query,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Metrics
    "Metrics",
    "TimingStats",
    "increment_pages_crawled",
    "increment_llm_calls",
    "increment_embeddings_generated",
    "observe_llm_latency",
    "observe_query_latency",
    "time_llm_call",
    "time_query",
]
