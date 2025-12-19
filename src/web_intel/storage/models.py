"""
Data models for storage layer.

Defines dataclasses representing database records with
type-safe access and serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class PageStatus(str, Enum):
    """Status of a crawled page."""

    PENDING = "pending"
    CRAWLED = "crawled"
    PROCESSED = "processed"  # Content extracted and understood
    INDEXED = "indexed"  # Embeddings generated and stored
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by robots.txt


class CrawlStatus(str, Enum):
    """Status of a crawl session."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PageRecord:
    """
    Record of a crawled page.

    Stores page content, metadata, and processing status.
    """

    id: int | None = None
    crawl_id: int | None = None
    url: str = ""
    canonical_url: str | None = None
    title: str = ""
    content_text: str = ""
    content_html: str = ""
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    word_count: int = 0
    language: str | None = None
    status: PageStatus = PageStatus.PENDING
    depth: int = 0
    parent_url: str | None = None
    content_hash: str | None = None
    quality_score: float = 0.0
    crawled_at: datetime | None = None
    processed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "crawl_id": self.crawl_id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "content_text": self.content_text,
            "content_html": self.content_html,
            "summary": self.summary,
            "topics": ",".join(self.topics) if self.topics else "",
            "word_count": self.word_count,
            "language": self.language,
            "status": self.status.value,
            "depth": self.depth,
            "parent_url": self.parent_url,
            "content_hash": self.content_hash,
            "quality_score": self.quality_score,
            "crawled_at": self.crawled_at.isoformat() if self.crawled_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "metadata": str(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_row(cls, row: dict) -> "PageRecord":
        """Create from database row."""
        topics = row.get("topics", "")
        topics_list = [t.strip() for t in topics.split(",")
                       if t.strip()] if topics else []

        return cls(
            id=row.get("id"),
            crawl_id=row.get("crawl_id"),
            url=row.get("url", ""),
            canonical_url=row.get("canonical_url"),
            title=row.get("title", ""),
            content_text=row.get("content_text", ""),
            content_html=row.get("content_html", ""),
            summary=row.get("summary", ""),
            topics=topics_list,
            word_count=row.get("word_count", 0),
            language=row.get("language"),
            status=PageStatus(row.get("status", "pending")),
            depth=row.get("depth", 0),
            parent_url=row.get("parent_url"),
            content_hash=row.get("content_hash"),
            quality_score=row.get("quality_score", 0.0),
            crawled_at=_parse_datetime(row.get("crawled_at")),
            processed_at=_parse_datetime(row.get("processed_at")),
            created_at=_parse_datetime(row.get("created_at")),
            updated_at=_parse_datetime(row.get("updated_at")),
        )


@dataclass
class EntityRecord:
    """
    Record of an extracted entity.

    Stores named entities with their types and relationships to pages.
    """

    id: int | None = None
    page_id: int | None = None
    name: str = ""
    entity_type: str = ""  # person, organization, location, etc.
    normalized_name: str = ""  # Lowercase, trimmed for deduplication
    mentions: int = 1
    context: str = ""  # Surrounding text
    confidence: float = 1.0
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "page_id": self.page_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "normalized_name": self.normalized_name or self.name.lower().strip(),
            "mentions": self.mentions,
            "context": self.context,
            "confidence": self.confidence,
        }

    @classmethod
    def from_row(cls, row: dict) -> "EntityRecord":
        """Create from database row."""
        return cls(
            id=row.get("id"),
            page_id=row.get("page_id"),
            name=row.get("name", ""),
            entity_type=row.get("entity_type", ""),
            normalized_name=row.get("normalized_name", ""),
            mentions=row.get("mentions", 1),
            context=row.get("context", ""),
            confidence=row.get("confidence", 1.0),
            created_at=_parse_datetime(row.get("created_at")),
        )


@dataclass
class ChunkRecord:
    """
    Record of a text chunk with embedding.

    Stores chunks for vector similarity search.
    """

    id: int | None = None
    page_id: int | None = None
    chunk_index: int = 0
    text: str = ""
    summary: str = ""
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    embedding: np.ndarray | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "page_id": self.page_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "summary": self.summary,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
        }

    @classmethod
    def from_row(cls, row: dict, embedding: np.ndarray | None = None) -> "ChunkRecord":
        """Create from database row."""
        return cls(
            id=row.get("id"),
            page_id=row.get("page_id"),
            chunk_index=row.get("chunk_index", 0),
            text=row.get("text", ""),
            summary=row.get("summary", ""),
            start_char=row.get("start_char", 0),
            end_char=row.get("end_char", 0),
            token_count=row.get("token_count", 0),
            embedding=embedding,
            created_at=_parse_datetime(row.get("created_at")),
        )


@dataclass
class CrawlRecord:
    """
    Record of a crawl session.

    Tracks crawl progress and statistics.
    """

    id: int | None = None
    start_url: str = ""
    domain: str = ""
    status: CrawlStatus = CrawlStatus.RUNNING
    pages_crawled: int = 0
    pages_processed: int = 0
    pages_failed: int = 0
    max_pages: int = 0
    max_depth: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "start_url": self.start_url,
            "domain": self.domain,
            "status": self.status.value,
            "pages_crawled": self.pages_crawled,
            "pages_processed": self.pages_processed,
            "pages_failed": self.pages_failed,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "config": str(self.config) if self.config else None,
        }

    @classmethod
    def from_row(cls, row: dict) -> "CrawlRecord":
        """Create from database row."""
        return cls(
            id=row.get("id"),
            start_url=row.get("start_url", ""),
            domain=row.get("domain", ""),
            status=CrawlStatus(row.get("status", "running")),
            pages_crawled=row.get("pages_crawled", 0),
            pages_processed=row.get("pages_processed", 0),
            pages_failed=row.get("pages_failed", 0),
            max_pages=row.get("max_pages", 0),
            max_depth=row.get("max_depth", 0),
            started_at=_parse_datetime(row.get("started_at")),
            completed_at=_parse_datetime(row.get("completed_at")),
            error_message=row.get("error_message"),
        )


@dataclass
class LinkRecord:
    """Record of a link between pages."""

    id: int | None = None
    source_page_id: int | None = None
    target_url: str = ""
    target_page_id: int | None = None
    anchor_text: str = ""
    link_type: str = "internal"  # internal, external
    created_at: datetime | None = None


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime from database value."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
