"""
Lightweight in-memory metrics for observability.

Provides simple counters and timing metrics without
external dependencies like Prometheus or OpenTelemetry.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, duration_ms: float) -> None:
        """Record a timing measurement."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count > 0 else 0.0,
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class Metrics:
    """
    In-memory metrics collector.

    Thread-safe counters and timing metrics for observability.
    Uses a singleton pattern for easy access across the application.

    Example:
        >>> metrics = Metrics.get()
        >>> metrics.increment("pages_crawled")
        >>> with metrics.timer("query_latency_ms"):
        ...     result = execute_query()
        >>> print(metrics.snapshot())
    """

    # Counters
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Timing stats
    _timings: dict[str, TimingStats] = field(
        default_factory=lambda: defaultdict(TimingStats)
    )

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Singleton instance
    _instance: "Metrics | None" = field(default=None, repr=False, init=False)

    @classmethod
    def get(cls) -> "Metrics":
        """
        Get the global metrics instance.

        Creates one if it doesn't exist.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset all metrics (useful for testing)."""
        if cls._instance is not None:
            with cls._instance._lock:
                cls._instance._counters.clear()
                cls._instance._timings.clear()

    def increment(self, name: str, value: int = 1) -> int:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Amount to increment (default 1)

        Returns:
            New counter value
        """
        with self._lock:
            self._counters[name] += value
            return self._counters[name]

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def observe(self, name: str, duration_ms: float) -> None:
        """
        Record a timing observation.

        Args:
            name: Timing metric name
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            self._timings[name].record(duration_ms)

    def get_timing(self, name: str) -> TimingStats | None:
        """Get timing statistics for a metric."""
        with self._lock:
            if name in self._timings:
                # Return a copy to avoid mutation
                stats = self._timings[name]
                return TimingStats(
                    count=stats.count,
                    total_ms=stats.total_ms,
                    min_ms=stats.min_ms,
                    max_ms=stats.max_ms,
                )
            return None

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """
        Context manager to time a block of code.

        Args:
            name: Name for this timing metric

        Example:
            >>> with metrics.timer("query_latency_ms"):
            ...     result = slow_operation()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.observe(name, duration_ms)

    def snapshot(self) -> dict:
        """
        Get a snapshot of all metrics.

        Returns:
            Dictionary with all counters and timings
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timings": {
                    name: stats.to_dict()
                    for name, stats in self._timings.items()
                },
            }

    def summary(self) -> str:
        """
        Get a human-readable summary of metrics.

        Returns:
            Formatted string with key metrics
        """
        snap = self.snapshot()
        lines = ["=== Metrics Summary ==="]

        # Counters
        if snap["counters"]:
            lines.append("\nCounters:")
            for name, value in sorted(snap["counters"].items()):
                lines.append(f"  {name}: {value:,}")

        # Timings
        if snap["timings"]:
            lines.append("\nTimings:")
            for name, stats in sorted(snap["timings"].items()):
                lines.append(
                    f"  {name}: {stats['count']} calls, "
                    f"avg={stats['avg_ms']:.1f}ms, "
                    f"min={stats['min_ms']:.1f}ms, "
                    f"max={stats['max_ms']:.1f}ms"
                )

        return "\n".join(lines)


# Convenience functions for common metrics
def increment_pages_crawled(count: int = 1) -> None:
    """Increment pages crawled counter."""
    Metrics.get().increment("pages_crawled", count)


def increment_llm_calls(count: int = 1) -> None:
    """Increment LLM calls counter."""
    Metrics.get().increment("llm_calls", count)


def increment_embeddings_generated(count: int = 1) -> None:
    """Increment embeddings generated counter."""
    Metrics.get().increment("embeddings_generated", count)


def observe_llm_latency(duration_ms: float) -> None:
    """Record LLM call latency."""
    Metrics.get().observe("llm_latency_ms", duration_ms)


def observe_query_latency(duration_ms: float) -> None:
    """Record query execution latency."""
    Metrics.get().observe("query_latency_ms", duration_ms)


@contextmanager
def time_llm_call() -> Iterator[None]:
    """Time an LLM call (increments counter and records latency)."""
    Metrics.get().increment("llm_calls")
    with Metrics.get().timer("llm_latency_ms"):
        yield


@contextmanager
def time_query() -> Iterator[None]:
    """Time a query execution."""
    with Metrics.get().timer("query_latency_ms"):
        yield
