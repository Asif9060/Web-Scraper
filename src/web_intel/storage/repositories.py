"""
Repository classes for data access.

Provides typed interfaces for database operations on each entity type.
"""

from datetime import datetime
from typing import Sequence

import numpy as np

from web_intel.core.exceptions import DatabaseError
from web_intel.storage.database import Database
from web_intel.storage.models import (
    PageRecord,
    PageStatus,
    EntityRecord,
    ChunkRecord,
    CrawlRecord,
    CrawlStatus,
)
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class PageRepository:
    """
    Repository for page records.

    Provides CRUD operations and queries for crawled pages.

    Example:
        >>> repo = PageRepository(database)
        >>> page = PageRecord(url="https://example.com", title="Example")
        >>> page_id = repo.insert(page)
        >>> retrieved = repo.get_by_id(page_id)
    """

    def __init__(self, db: Database) -> None:
        """Initialize repository with database."""
        self.db = db

    def insert(self, page: PageRecord) -> int:
        """
        Insert a new page.

        Args:
            page: PageRecord to insert

        Returns:
            Inserted page ID
        """
        data = {
            "crawl_id": page.crawl_id,
            "url": page.url,
            "canonical_url": page.canonical_url,
            "title": page.title,
            "content_text": page.content_text,
            "content_html": page.content_html,
            "summary": page.summary,
            "topics": ",".join(page.topics) if page.topics else "",
            "word_count": page.word_count,
            "language": page.language,
            "status": page.status.value,
            "depth": page.depth,
            "parent_url": page.parent_url,
            "content_hash": page.content_hash,
            "quality_score": page.quality_score,
            "crawled_at": page.crawled_at.isoformat() if page.crawled_at else None,
            "processed_at": page.processed_at.isoformat() if page.processed_at else None,
        }

        return self.db.insert("pages", data)

    def update(self, page: PageRecord) -> bool:
        """
        Update an existing page.

        Args:
            page: PageRecord with updated values

        Returns:
            True if updated, False if not found
        """
        if page.id is None:
            raise ValueError("Cannot update page without ID")

        data = {
            "title": page.title,
            "content_text": page.content_text,
            "content_html": page.content_html,
            "summary": page.summary,
            "topics": ",".join(page.topics) if page.topics else "",
            "word_count": page.word_count,
            "language": page.language,
            "status": page.status.value,
            "content_hash": page.content_hash,
            "quality_score": page.quality_score,
            "crawled_at": page.crawled_at.isoformat() if page.crawled_at else None,
            "processed_at": page.processed_at.isoformat() if page.processed_at else None,
            "updated_at": datetime.now().isoformat(),
        }

        affected = self.db.update("pages", data, "id = ?", (page.id,))
        return affected > 0

    def get_by_id(self, page_id: int) -> PageRecord | None:
        """Get page by ID."""
        row = self.db.fetch_one("SELECT * FROM pages WHERE id = ?", (page_id,))
        return PageRecord.from_row(row) if row else None

    def get_by_url(self, url: str, crawl_id: int | None = None) -> PageRecord | None:
        """Get page by URL."""
        if crawl_id:
            row = self.db.fetch_one(
                "SELECT * FROM pages WHERE url = ? AND crawl_id = ?",
                (url, crawl_id),
            )
        else:
            row = self.db.fetch_one(
                "SELECT * FROM pages WHERE url = ? ORDER BY id DESC LIMIT 1",
                (url,),
            )
        return PageRecord.from_row(row) if row else None

    def exists(self, url: str, crawl_id: int | None = None) -> bool:
        """Check if page exists."""
        if crawl_id:
            result = self.db.fetch_one(
                "SELECT 1 FROM pages WHERE url = ? AND crawl_id = ?",
                (url, crawl_id),
            )
        else:
            result = self.db.fetch_one(
                "SELECT 1 FROM pages WHERE url = ?",
                (url,),
            )
        return result is not None

    def get_by_crawl(
        self,
        crawl_id: int,
        status: PageStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PageRecord]:
        """Get pages for a crawl session."""
        if status:
            rows = self.db.fetch_all(
                "SELECT * FROM pages WHERE crawl_id = ? AND status = ? "
                "ORDER BY id LIMIT ? OFFSET ?",
                (crawl_id, status.value, limit, offset),
            )
        else:
            rows = self.db.fetch_all(
                "SELECT * FROM pages WHERE crawl_id = ? ORDER BY id LIMIT ? OFFSET ?",
                (crawl_id, limit, offset),
            )
        return [PageRecord.from_row(row) for row in rows]

    def get_pending(self, crawl_id: int, limit: int = 10) -> list[PageRecord]:
        """Get pending pages to process."""
        rows = self.db.fetch_all(
            "SELECT * FROM pages WHERE crawl_id = ? AND status = ? "
            "ORDER BY depth, id LIMIT ?",
            (crawl_id, PageStatus.PENDING.value, limit),
        )
        return [PageRecord.from_row(row) for row in rows]

    def update_status(self, page_id: int, status: PageStatus) -> bool:
        """Update page status."""
        affected = self.db.update(
            "pages",
            {"status": status.value, "updated_at": datetime.now().isoformat()},
            "id = ?",
            (page_id,),
        )
        return affected > 0

    def count_by_crawl(self, crawl_id: int, status: PageStatus | None = None) -> int:
        """Count pages in a crawl."""
        if status:
            result = self.db.fetch_one(
                "SELECT COUNT(*) as count FROM pages WHERE crawl_id = ? AND status = ?",
                (crawl_id, status.value),
            )
        else:
            result = self.db.fetch_one(
                "SELECT COUNT(*) as count FROM pages WHERE crawl_id = ?",
                (crawl_id,),
            )
        return result["count"] if result else 0

    def search_fts(self, query: str, limit: int = 20) -> list[PageRecord]:
        """Full-text search across pages."""
        rows = self.db.fetch_all(
            """
            SELECT pages.* FROM pages
            JOIN pages_fts ON pages.id = pages_fts.rowid
            WHERE pages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        )
        return [PageRecord.from_row(row) for row in rows]

    def get_by_content_hash(self, content_hash: str) -> PageRecord | None:
        """Get page by content hash for deduplication."""
        row = self.db.fetch_one(
            "SELECT * FROM pages WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return PageRecord.from_row(row) if row else None

    def delete(self, page_id: int) -> bool:
        """Delete page by ID."""
        affected = self.db.delete("pages", "id = ?", (page_id,))
        return affected > 0


class EntityRepository:
    """
    Repository for entity records.

    Manages named entities extracted from pages.
    """

    def __init__(self, db: Database) -> None:
        """Initialize repository with database."""
        self.db = db

    def insert(self, entity: EntityRecord) -> int:
        """Insert a new entity."""
        data = {
            "page_id": entity.page_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "normalized_name": entity.normalized_name or entity.name.lower().strip(),
            "mentions": entity.mentions,
            "context": entity.context,
            "confidence": entity.confidence,
        }
        return self.db.insert("entities", data)

    def insert_many(self, entities: Sequence[EntityRecord]) -> int:
        """Insert multiple entities."""
        if not entities:
            return 0

        sql = """
            INSERT INTO entities 
            (page_id, name, entity_type, normalized_name, mentions, context, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                e.page_id,
                e.name,
                e.entity_type,
                e.normalized_name or e.name.lower().strip(),
                e.mentions,
                e.context,
                e.confidence,
            )
            for e in entities
        ]

        cursor = self.db.executemany(sql, params)
        self.db._get_connection().commit()
        return cursor.rowcount

    def get_by_page(self, page_id: int) -> list[EntityRecord]:
        """Get all entities for a page."""
        rows = self.db.fetch_all(
            "SELECT * FROM entities WHERE page_id = ? ORDER BY mentions DESC",
            (page_id,),
        )
        return [EntityRecord.from_row(row) for row in rows]

    def get_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> list[EntityRecord]:
        """Get entities by type."""
        rows = self.db.fetch_all(
            "SELECT * FROM entities WHERE entity_type = ? ORDER BY mentions DESC LIMIT ?",
            (entity_type, limit),
        )
        return [EntityRecord.from_row(row) for row in rows]

    def search(self, name: str, limit: int = 20) -> list[EntityRecord]:
        """Search entities by name."""
        normalized = name.lower().strip()
        rows = self.db.fetch_all(
            "SELECT * FROM entities WHERE normalized_name LIKE ? ORDER BY mentions DESC LIMIT ?",
            (f"%{normalized}%", limit),
        )
        return [EntityRecord.from_row(row) for row in rows]

    def get_unique_entities(
        self,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get unique entities with total mention counts."""
        if entity_type:
            rows = self.db.fetch_all(
                """
                SELECT normalized_name, entity_type, SUM(mentions) as total_mentions,
                       COUNT(DISTINCT page_id) as page_count
                FROM entities
                WHERE entity_type = ?
                GROUP BY normalized_name, entity_type
                ORDER BY total_mentions DESC
                LIMIT ?
                """,
                (entity_type, limit),
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT normalized_name, entity_type, SUM(mentions) as total_mentions,
                       COUNT(DISTINCT page_id) as page_count
                FROM entities
                GROUP BY normalized_name, entity_type
                ORDER BY total_mentions DESC
                LIMIT ?
                """,
                (limit,),
            )
        return rows

    def delete_by_page(self, page_id: int) -> int:
        """Delete all entities for a page."""
        return self.db.delete("entities", "page_id = ?", (page_id,))


class ChunkRepository:
    """
    Repository for chunk records.

    Manages text chunks and their embeddings for vector search.
    """

    def __init__(self, db: Database) -> None:
        """Initialize repository with database."""
        self.db = db
        self._vector_table_created = False

    def insert(self, chunk: ChunkRecord) -> int:
        """Insert a new chunk."""
        data = {
            "page_id": chunk.page_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "summary": chunk.summary,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "token_count": chunk.token_count,
        }
        chunk_id = self.db.insert("chunks", data)

        # Store embedding if provided
        if chunk.embedding is not None:
            self._store_embedding(chunk_id, chunk.embedding)

        return chunk_id

    def insert_many(self, chunks: Sequence[ChunkRecord]) -> list[int]:
        """Insert multiple chunks."""
        ids = []
        for chunk in chunks:
            chunk_id = self.insert(chunk)
            ids.append(chunk_id)
        return ids

    def _store_embedding(self, chunk_id: int, embedding: np.ndarray) -> None:
        """Store embedding for chunk."""
        # Store as blob
        embedding_bytes = embedding.astype(np.float32).tobytes()
        self.db.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
            """,
            (chunk_id, embedding_bytes),
        )
        self.db._get_connection().commit()

    def _ensure_embedding_table(self) -> None:
        """Ensure embedding storage table exists."""
        if self._vector_table_created:
            return

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )
        """)
        self.db._get_connection().commit()
        self._vector_table_created = True

    def get_by_id(self, chunk_id: int) -> ChunkRecord | None:
        """Get chunk by ID."""
        row = self.db.fetch_one(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        if not row:
            return None

        embedding = self._get_embedding(chunk_id)
        return ChunkRecord.from_row(row, embedding)

    def _get_embedding(self, chunk_id: int) -> np.ndarray | None:
        """Get embedding for chunk."""
        row = self.db.fetch_one(
            "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        )
        if row and row["embedding"]:
            return np.frombuffer(row["embedding"], dtype=np.float32)
        return None

    def get_by_page(self, page_id: int) -> list[ChunkRecord]:
        """Get all chunks for a page."""
        rows = self.db.fetch_all(
            "SELECT * FROM chunks WHERE page_id = ? ORDER BY chunk_index",
            (page_id,),
        )
        chunks = []
        for row in rows:
            embedding = self._get_embedding(row["id"])
            chunks.append(ChunkRecord.from_row(row, embedding))
        return chunks

    def get_all_embeddings(self) -> tuple[list[int], np.ndarray]:
        """
        Get all chunk IDs and embeddings.

        Returns:
            Tuple of (chunk_ids, embeddings_matrix)
        """
        rows = self.db.fetch_all(
            "SELECT chunk_id, embedding FROM chunk_embeddings ORDER BY chunk_id"
        )

        if not rows:
            return [], np.array([])

        chunk_ids = [row["chunk_id"] for row in rows]
        embeddings = [
            np.frombuffer(row["embedding"], dtype=np.float32)
            for row in rows
        ]

        return chunk_ids, np.vstack(embeddings)

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[ChunkRecord, float]]:
        """
        Find similar chunks by embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of (chunk, similarity_score) tuples
        """
        chunk_ids, embeddings = self.get_all_embeddings()

        if len(chunk_ids) == 0:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / \
            (np.linalg.norm(query_embedding) + 1e-10)
        emb_norms = embeddings / \
            (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(emb_norms, query_norm)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.get_by_id(chunk_ids[idx])
            if chunk:
                results.append((chunk, float(similarities[idx])))

        return results

    def delete_by_page(self, page_id: int) -> int:
        """Delete all chunks for a page."""
        # Get chunk IDs first for embedding cleanup
        rows = self.db.fetch_all(
            "SELECT id FROM chunks WHERE page_id = ?",
            (page_id,),
        )
        chunk_ids = [row["id"] for row in rows]

        # Delete embeddings
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            self.db.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            )

        # Delete chunks
        return self.db.delete("chunks", "page_id = ?", (page_id,))

    def count(self) -> int:
        """Get total chunk count."""
        result = self.db.fetch_one("SELECT COUNT(*) as count FROM chunks")
        return result["count"] if result else 0


class CrawlRepository:
    """
    Repository for crawl session records.

    Manages crawl session lifecycle and statistics.
    """

    def __init__(self, db: Database) -> None:
        """Initialize repository with database."""
        self.db = db

    def create(self, crawl: CrawlRecord) -> int:
        """Create a new crawl session."""
        data = {
            "start_url": crawl.start_url,
            "domain": crawl.domain,
            "status": crawl.status.value,
            "max_pages": crawl.max_pages,
            "max_depth": crawl.max_depth,
            "started_at": datetime.now().isoformat(),
        }
        return self.db.insert("crawls", data)

    def update(self, crawl: CrawlRecord) -> bool:
        """Update crawl session."""
        if crawl.id is None:
            raise ValueError("Cannot update crawl without ID")

        data = {
            "status": crawl.status.value,
            "pages_crawled": crawl.pages_crawled,
            "pages_processed": crawl.pages_processed,
            "pages_failed": crawl.pages_failed,
            "completed_at": crawl.completed_at.isoformat() if crawl.completed_at else None,
            "error_message": crawl.error_message,
        }

        affected = self.db.update("crawls", data, "id = ?", (crawl.id,))
        return affected > 0

    def get_by_id(self, crawl_id: int) -> CrawlRecord | None:
        """Get crawl by ID."""
        row = self.db.fetch_one(
            "SELECT * FROM crawls WHERE id = ?", (crawl_id,))
        return CrawlRecord.from_row(row) if row else None

    def get_latest(self, domain: str | None = None) -> CrawlRecord | None:
        """Get most recent crawl session."""
        if domain:
            row = self.db.fetch_one(
                "SELECT * FROM crawls WHERE domain = ? ORDER BY id DESC LIMIT 1",
                (domain,),
            )
        else:
            row = self.db.fetch_one(
                "SELECT * FROM crawls ORDER BY id DESC LIMIT 1"
            )
        return CrawlRecord.from_row(row) if row else None

    def get_active(self) -> list[CrawlRecord]:
        """Get all active crawl sessions."""
        rows = self.db.fetch_all(
            "SELECT * FROM crawls WHERE status IN (?, ?) ORDER BY started_at DESC",
            (CrawlStatus.RUNNING.value, CrawlStatus.PAUSED.value),
        )
        return [CrawlRecord.from_row(row) for row in rows]

    def get_all(self, limit: int = 50) -> list[CrawlRecord]:
        """Get all crawl sessions."""
        rows = self.db.fetch_all(
            "SELECT * FROM crawls ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [CrawlRecord.from_row(row) for row in rows]

    def update_stats(
        self,
        crawl_id: int,
        pages_crawled: int | None = None,
        pages_processed: int | None = None,
        pages_failed: int | None = None,
    ) -> bool:
        """Update crawl statistics."""
        data = {}
        if pages_crawled is not None:
            data["pages_crawled"] = pages_crawled
        if pages_processed is not None:
            data["pages_processed"] = pages_processed
        if pages_failed is not None:
            data["pages_failed"] = pages_failed

        if not data:
            return False

        affected = self.db.update("crawls", data, "id = ?", (crawl_id,))
        return affected > 0

    def complete(
        self,
        crawl_id: int,
        status: CrawlStatus = CrawlStatus.COMPLETED,
        error_message: str | None = None,
    ) -> bool:
        """Mark crawl as complete."""
        data = {
            "status": status.value,
            "completed_at": datetime.now().isoformat(),
        }
        if error_message:
            data["error_message"] = error_message

        affected = self.db.update("crawls", data, "id = ?", (crawl_id,))
        return affected > 0

    def delete(self, crawl_id: int) -> bool:
        """Delete crawl and all associated data."""
        # Delete pages (cascades to entities, chunks)
        self.db.delete("pages", "crawl_id = ?", (crawl_id,))
        # Delete crawl
        affected = self.db.delete("crawls", "id = ?", (crawl_id,))
        return affected > 0
