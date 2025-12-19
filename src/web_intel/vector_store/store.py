"""
Vector store for semantic search.

Combines persistent storage with in-memory indexing
for efficient similarity search. Supports memory limits
with LRU eviction for large datasets.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from web_intel.config import Settings
from web_intel.core.exceptions import VectorStorageError
from web_intel.embeddings import Embedder
from web_intel.storage import Database, ChunkRepository
from web_intel.storage.models import ChunkRecord
from web_intel.vector_store.index import VectorIndex, IndexType
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchFilter:
    """
    Filters for vector search.

    Allows filtering results by metadata criteria.
    """

    page_ids: list[int] | None = None  # Limit to specific pages
    crawl_id: int | None = None  # Limit to specific crawl
    min_score: float = 0.0  # Minimum similarity score
    exclude_ids: set[int] = field(default_factory=set)  # Chunk IDs to exclude


@dataclass
class VectorSearchResult:
    """
    Result of a vector similarity search.

    Contains the matching chunk with its similarity score.
    """

    chunk_id: int
    page_id: int
    text: str
    score: float
    chunk_index: int = 0
    summary: str = ""

    @classmethod
    def from_chunk(cls, chunk: ChunkRecord, score: float) -> "VectorSearchResult":
        """Create from ChunkRecord."""
        return cls(
            chunk_id=chunk.id,
            page_id=chunk.page_id,
            text=chunk.text,
            score=score,
            chunk_index=chunk.chunk_index,
            summary=chunk.summary,
        )


class VectorStore:
    """
    Vector store for semantic similarity search.

    Combines SQLite persistence with in-memory indexing for
    efficient hybrid search. Supports both exact and approximate
    nearest neighbor search.

    Memory Management:
        The store enforces a maximum number of vectors in memory.
        When the limit is reached, least-recently-used vectors are
        evicted. Evicted vectors remain in the database and can be
        reloaded on demand.

    Example:
        >>> store = VectorStore.from_settings(settings)
        >>> store.load_index()  # Load from database (up to max_vectors)
        >>>
        >>> # Add new chunks with embeddings
        >>> store.add_chunks(chunks, embeddings)
        >>>
        >>> # Search by text
        >>> results = store.search_text("How to configure?", top_k=5)
        >>> for result in results:
        ...     print(f"Score: {result.score:.3f} - {result.text[:50]}")
        >>>
        >>> # Load specific domain/pages only
        >>> store.load_by_page_ids([1, 2, 3])
    """

    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        dimensions: int = 384,
        use_ivf: bool = False,
        ivf_clusters: int = 100,
        max_vectors_in_memory: int = 50000,
        eviction_batch_size: int = 5000,
    ) -> None:
        """
        Initialize vector store.

        Args:
            database: Database instance for persistence
            embedder: Embedder for text-to-vector conversion
            dimensions: Vector dimensionality
            use_ivf: Whether to use approximate search
            ivf_clusters: Number of IVF clusters
            max_vectors_in_memory: Maximum vectors to keep in memory
            eviction_batch_size: How many vectors to evict at once
        """
        self.db = database
        self.embedder = embedder
        self.dimensions = dimensions
        self.use_ivf = use_ivf
        self.ivf_clusters = ivf_clusters
        self.max_vectors = max_vectors_in_memory
        self.eviction_batch_size = eviction_batch_size

        self._index = VectorIndex(
            dimensions=dimensions,
            index_type=IndexType.IVF if use_ivf else IndexType.FLAT,
        )
        self._chunk_repo = ChunkRepository(database)
        self._loaded = False

        # LRU tracking: OrderedDict maintains insertion order
        # Key: chunk_id, Value: access timestamp (or just presence for LRU)
        self._access_order: OrderedDict[int, None] = OrderedDict()

        # Track which vectors are currently in memory
        self._loaded_chunk_ids: set[int] = set()

        # Ensure embedding table exists
        self._ensure_tables()

        logger.info(
            f"VectorStore initialized (dims={dimensions}, ivf={use_ivf}, "
            f"max_vectors={max_vectors_in_memory})"
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Database | None = None,
        embedder: Embedder | None = None,
    ) -> "VectorStore":
        """
        Create VectorStore from settings.

        Args:
            settings: Application settings
            database: Optional pre-configured database
            embedder: Optional pre-configured embedder

        Returns:
            Configured VectorStore instance
        """
        if database is None:
            database = Database.from_settings(settings)

        if embedder is None:
            embedder = Embedder.from_settings(settings)

        return cls(
            database=database,
            embedder=embedder,
            dimensions=settings.storage.vector_dimensions,
            max_vectors_in_memory=settings.storage.max_vectors_in_memory,
            eviction_batch_size=settings.storage.vector_eviction_batch_size,
        )

    def _ensure_tables(self) -> None:
        """Ensure vector storage tables exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL
            )
        """)
        self.db._get_connection().commit()

    def load_index(self, limit: int | None = None) -> int:
        """
        Load embeddings from database into memory index.

        Respects max_vectors_in_memory limit. If more vectors exist
        in the database, only the most recent ones are loaded.

        Args:
            limit: Optional override for max vectors to load

        Returns:
            Number of vectors loaded
        """
        max_to_load = limit if limit is not None else self.max_vectors

        logger.info(f"Loading vector index from database (max: {max_to_load})")

        # Load most recent embeddings up to limit
        rows = self.db.fetch_all(
            """
            SELECT chunk_id, embedding 
            FROM chunk_embeddings 
            ORDER BY chunk_id DESC 
            LIMIT ?
            """,
            (max_to_load,),
        )

        if not rows:
            logger.info("No embeddings found in database")
            self._loaded = True
            return 0

        return self._load_rows_into_index(rows)

    def load_by_page_ids(self, page_ids: list[int]) -> int:
        """
        Load embeddings for specific pages only.

        Useful for domain-specific or scoped searches.
        Evicts other vectors if needed to stay within memory limit.

        Args:
            page_ids: List of page IDs to load vectors for

        Returns:
            Number of vectors loaded
        """
        if not page_ids:
            return 0

        placeholders = ",".join("?" * len(page_ids))
        rows = self.db.fetch_all(
            f"""
            SELECT ce.chunk_id, ce.embedding
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE c.page_id IN ({placeholders})
            ORDER BY ce.chunk_id DESC
            LIMIT ?
            """,
            (*page_ids, self.max_vectors),
        )

        if not rows:
            logger.info(f"No embeddings found for pages: {page_ids}")
            return 0

        logger.info(f"Loading {len(rows)} vectors for {len(page_ids)} pages")
        return self._load_rows_into_index(rows, clear_existing=False)

    def load_by_crawl_id(self, crawl_id: int) -> int:
        """
        Load embeddings for a specific crawl only.

        Args:
            crawl_id: Crawl ID to load vectors for

        Returns:
            Number of vectors loaded
        """
        rows = self.db.fetch_all(
            """
            SELECT ce.chunk_id, ce.embedding
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            JOIN pages p ON p.id = c.page_id
            WHERE p.crawl_id = ?
            ORDER BY ce.chunk_id DESC
            LIMIT ?
            """,
            (crawl_id, self.max_vectors),
        )

        if not rows:
            logger.info(f"No embeddings found for crawl: {crawl_id}")
            return 0

        logger.info(f"Loading {len(rows)} vectors for crawl {crawl_id}")
        return self._load_rows_into_index(rows, clear_existing=False)

    def load_by_domain(self, domain: str) -> int:
        """
        Load embeddings for a specific domain.

        Args:
            domain: Domain to filter by (e.g., "example.com")

        Returns:
            Number of vectors loaded
        """
        rows = self.db.fetch_all(
            """
            SELECT ce.chunk_id, ce.embedding
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            JOIN pages p ON p.id = c.page_id
            JOIN crawls cr ON cr.id = p.crawl_id
            WHERE cr.domain = ? OR p.url LIKE ?
            ORDER BY ce.chunk_id DESC
            LIMIT ?
            """,
            (domain, f"%{domain}%", self.max_vectors),
        )

        if not rows:
            logger.info(f"No embeddings found for domain: {domain}")
            return 0

        logger.info(f"Loading {len(rows)} vectors for domain {domain}")
        return self._load_rows_into_index(rows, clear_existing=False)

    def _load_rows_into_index(
        self,
        rows: list[dict],
        clear_existing: bool = True,
    ) -> int:
        """
        Load database rows into the in-memory index.

        Args:
            rows: List of rows with chunk_id and embedding
            clear_existing: Whether to clear existing index first

        Returns:
            Number of vectors loaded
        """
        if clear_existing:
            self._index.clear()
            self._access_order.clear()
            self._loaded_chunk_ids.clear()

        ids = []
        embeddings = []

        for row in rows:
            chunk_id = row["chunk_id"]

            # Skip if already loaded
            if chunk_id in self._loaded_chunk_ids:
                continue

            ids.append(chunk_id)
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(embedding)

        if not ids:
            self._loaded = True
            return 0

        # Check if we need to evict before loading
        space_needed = len(ids)
        current_size = self._index.size
        if current_size + space_needed > self.max_vectors:
            to_evict = (current_size + space_needed) - self.max_vectors
            # Evict at least eviction_batch_size for efficiency
            to_evict = max(to_evict, min(
                self.eviction_batch_size, current_size))
            self._evict_lru(to_evict)

        embeddings_matrix = np.vstack(embeddings)

        # Add to index
        self._index.add(ids, embeddings_matrix)

        # Update tracking
        for chunk_id in ids:
            self._loaded_chunk_ids.add(chunk_id)
            self._access_order[chunk_id] = None  # Add to end (most recent)

        # Build IVF if configured and enough vectors
        if self.use_ivf and self._index.size >= self.ivf_clusters * 10:
            self._index.build_ivf(self.ivf_clusters)

        self._loaded = True
        logger.info(
            f"Loaded {len(ids)} vectors into index (total: {self._index.size})")
        return len(ids)

    def _evict_lru(self, count: int) -> int:
        """
        Evict least-recently-used vectors from memory.

        Vectors are removed from the in-memory index but remain
        in the database for future reloading.

        Args:
            count: Number of vectors to evict

        Returns:
            Number of vectors actually evicted
        """
        if count <= 0 or not self._access_order:
            return 0

        count = min(count, len(self._access_order))
        evicted = 0

        # Get oldest entries (first items in OrderedDict)
        ids_to_evict = []
        for chunk_id in self._access_order:
            if evicted >= count:
                break
            ids_to_evict.append(chunk_id)
            evicted += 1

        # Remove from tracking
        for chunk_id in ids_to_evict:
            del self._access_order[chunk_id]
            self._loaded_chunk_ids.discard(chunk_id)

        # Remove from index
        self._index.remove(ids_to_evict)

        logger.debug(
            f"Evicted {evicted} vectors from memory (remaining: {self._index.size})")
        return evicted

    def _mark_accessed(self, chunk_ids: list[int]) -> None:
        """
        Mark vectors as recently accessed (for LRU tracking).

        Moves the chunk IDs to the end of the access order,
        making them less likely to be evicted.

        Args:
            chunk_ids: List of chunk IDs that were accessed
        """
        for chunk_id in chunk_ids:
            if chunk_id in self._access_order:
                # Move to end by re-adding
                self._access_order.move_to_end(chunk_id)

    def _ensure_capacity(self, needed: int) -> None:
        """
        Ensure there is capacity for new vectors.

        Evicts LRU vectors if necessary to make room.

        Args:
            needed: Number of new vectors to accommodate
        """
        current_size = self._index.size
        available = self.max_vectors - current_size

        if available >= needed:
            return

        to_evict = needed - available
        # Evict at least eviction_batch_size for efficiency
        to_evict = max(to_evict, min(self.eviction_batch_size, current_size))
        self._evict_lru(to_evict)

    @property
    def memory_stats(self) -> dict:
        """
        Get memory usage statistics.

        Returns:
            Dict with memory stats
        """
        return {
            "vectors_in_memory": self._index.size,
            "max_vectors": self.max_vectors,
            "usage_percent": (self._index.size / self.max_vectors * 100) if self.max_vectors > 0 else 0,
            "loaded_chunk_ids": len(self._loaded_chunk_ids),
        }

    def add_chunk(
        self,
        chunk: ChunkRecord,
        embedding: np.ndarray | None = None,
    ) -> int:
        """
        Add a single chunk with embedding.

        Evicts LRU vectors if memory limit is reached.

        Args:
            chunk: Chunk record to add
            embedding: Pre-computed embedding (computed if None)

        Returns:
            Chunk ID
        """
        # Compute embedding if not provided
        if embedding is None:
            result = self.embedder.embed(chunk.text)
            embedding = result.embedding

        # Store chunk in database
        chunk.embedding = embedding
        chunk_id = self._chunk_repo.insert(chunk)

        # Store embedding in database (persists even if evicted from memory)
        self._store_embedding(chunk_id, embedding)

        # Ensure we have capacity
        self._ensure_capacity(1)

        # Add to index and track
        self._index.add([chunk_id], embedding.reshape(1, -1))
        self._loaded_chunk_ids.add(chunk_id)
        self._access_order[chunk_id] = None

        return chunk_id

    def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: np.ndarray | None = None,
    ) -> list[int]:
        """
        Add multiple chunks with embeddings.

        Evicts LRU vectors if memory limit is reached.

        Args:
            chunks: Chunk records to add
            embeddings: Pre-computed embeddings (computed if None)

        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

        # Store chunks and get IDs
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            chunk_id = self._chunk_repo.insert(chunk)
            chunk_ids.append(chunk_id)

            # Store embedding in database
            self._store_embedding(chunk_id, embeddings[i])

        # Ensure we have capacity for all new vectors
        self._ensure_capacity(len(chunk_ids))

        # Add to index and track
        self._index.add(chunk_ids, embeddings)
        for chunk_id in chunk_ids:
            self._loaded_chunk_ids.add(chunk_id)
            self._access_order[chunk_id] = None

        logger.debug(f"Added {len(chunks)} chunks to vector store")
        return chunk_ids

    def _store_embedding(self, chunk_id: int, embedding: np.ndarray) -> None:
        """Store embedding in database."""
        embedding_bytes = embedding.astype(np.float32).tobytes()
        self.db.execute(
            """
            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
            """,
            (chunk_id, embedding_bytes),
        )
        self.db._get_connection().commit()

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search by text query.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results sorted by relevance
        """
        # Embed query
        result = self.embedder.embed(query)
        return self.search_embedding(result.embedding, top_k, filters)

    def search_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search by embedding vector.

        Updates LRU tracking for returned results.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results sorted by relevance
        """
        if not self._loaded:
            self.load_index()

        if self._index.is_empty:
            return []

        filters = filters or SearchFilter()

        # Get more results than needed to allow for filtering
        search_k = top_k * 3 if filters.page_ids or filters.exclude_ids else top_k

        # Search index
        min_score = filters.min_score if filters.min_score > 0 else None
        candidates = self._index.search(query_embedding, search_k, min_score)

        # Apply filters and build results
        results = []
        accessed_ids = []
        for chunk_id, score in candidates:
            if chunk_id in filters.exclude_ids:
                continue

            # Get chunk details
            chunk = self._chunk_repo.get_by_id(chunk_id)
            if chunk is None:
                continue

            # Apply page filter
            if filters.page_ids and chunk.page_id not in filters.page_ids:
                continue

            # Apply crawl filter (need to join with pages)
            if filters.crawl_id is not None:
                page_row = self.db.fetch_one(
                    "SELECT crawl_id FROM pages WHERE id = ?",
                    (chunk.page_id,),
                )
                if not page_row or page_row["crawl_id"] != filters.crawl_id:
                    continue

            results.append(VectorSearchResult.from_chunk(chunk, score))
            accessed_ids.append(chunk_id)

            if len(results) >= top_k:
                break

        # Update LRU tracking for accessed chunks
        if accessed_ids:
            self._mark_accessed(accessed_ids)

        return results

    def search_similar_to_chunk(
        self,
        chunk_id: int,
        top_k: int = 10,
        exclude_same_page: bool = True,
    ) -> list[VectorSearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of results
            exclude_same_page: Exclude chunks from same page

        Returns:
            List of similar chunks
        """
        # Get the chunk's embedding
        embedding = self._index.get_vector(chunk_id)
        if embedding is None:
            return []

        # Build filters
        filters = SearchFilter(exclude_ids={chunk_id})

        if exclude_same_page:
            chunk = self._chunk_repo.get_by_id(chunk_id)
            if chunk:
                filters.page_ids = None  # We'll filter manually

        results = self.search_embedding(embedding, top_k * 2, filters)

        # Filter same page if needed
        if exclude_same_page and chunk:
            results = [r for r in results if r.page_id != chunk.page_id]

        return results[:top_k]

    def delete_by_page(self, page_id: int) -> int:
        """
        Delete all vectors for a page.

        Args:
            page_id: Page ID

        Returns:
            Number of vectors deleted
        """
        # Get chunk IDs for this page
        rows = self.db.fetch_all(
            "SELECT id FROM chunks WHERE page_id = ?",
            (page_id,),
        )
        chunk_ids = [row["id"] for row in rows]

        if not chunk_ids:
            return 0

        # Remove from index
        removed = self._index.remove(chunk_ids)

        # Remove from database
        placeholders = ",".join("?" * len(chunk_ids))
        self.db.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        )
        self._chunk_repo.delete_by_page(page_id)

        logger.debug(f"Deleted {removed} vectors for page {page_id}")
        return removed

    def delete_by_crawl(self, crawl_id: int) -> int:
        """
        Delete all vectors for a crawl.

        Args:
            crawl_id: Crawl ID

        Returns:
            Number of vectors deleted
        """
        # Get all page IDs for this crawl
        rows = self.db.fetch_all(
            "SELECT id FROM pages WHERE crawl_id = ?",
            (crawl_id,),
        )

        total = 0
        for row in rows:
            total += self.delete_by_page(row["id"])

        return total

    def rebuild_index(self) -> int:
        """
        Rebuild the in-memory index from database.

        Useful after bulk operations or to switch index type.

        Returns:
            Number of vectors in rebuilt index
        """
        self._index.clear()
        return self.load_index()

    def optimize(self) -> None:
        """
        Optimize the index for search performance.

        Builds IVF index if not already built and enough vectors.
        """
        if self._index.size < 1000:
            logger.info("Not enough vectors for IVF optimization")
            return

        if self._index.index_type != IndexType.IVF:
            n_clusters = min(self.ivf_clusters, self._index.size // 10)
            self._index.build_ivf(n_clusters)
            logger.info(f"Built IVF index with {n_clusters} clusters")

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        index_stats = self._index.get_stats()

        # Get database counts
        db_count = self.db.fetch_one(
            "SELECT COUNT(*) as count FROM chunk_embeddings"
        )

        return {
            "index_vectors": index_stats.vector_count,
            "database_vectors": db_count["count"] if db_count else 0,
            "dimensions": index_stats.dimensions,
            "index_type": index_stats.index_type.value,
            "memory_mb": index_stats.memory_bytes / (1024 * 1024),
            "is_loaded": self._loaded,
        }

    def save_index(self, path: str | Path) -> None:
        """
        Save index to file for fast loading.

        Args:
            path: File path to save to
        """
        self._index.save(str(path))

    def load_index_file(self, path: str | Path) -> int:
        """
        Load index from file instead of database.

        Args:
            path: File path to load from

        Returns:
            Number of vectors loaded
        """
        self._index = VectorIndex.load(str(path))
        self._loaded = True
        return self._index.size

    # =========================================================================
    # Async Methods
    # These methods wrap synchronous database operations for use in async code.
    # Use these when calling from async contexts to avoid blocking the event loop.
    # =========================================================================

    async def load_index_async(self) -> int:
        """
        Load all embeddings from database into memory index asynchronously.

        Use this method when calling from async code to avoid blocking.

        Returns:
            Number of vectors loaded
        """
        import asyncio
        return await asyncio.to_thread(self.load_index)

    async def add_chunk_async(
        self,
        chunk: ChunkRecord,
        embedding: np.ndarray | None = None,
    ) -> int:
        """
        Add a single chunk with embedding asynchronously.

        Args:
            chunk: Chunk record to add
            embedding: Pre-computed embedding (computed if None)

        Returns:
            Chunk ID
        """
        import asyncio
        return await asyncio.to_thread(self.add_chunk, chunk, embedding)

    async def add_chunks_async(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: np.ndarray | None = None,
    ) -> list[int]:
        """
        Add multiple chunks with embeddings asynchronously.

        This method batches database writes for better performance.

        Args:
            chunks: Chunk records to add
            embeddings: Pre-computed embeddings (computed if None)

        Returns:
            List of chunk IDs
        """
        import asyncio
        return await asyncio.to_thread(self._add_chunks_batched, chunks, embeddings)

    def _add_chunks_batched(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: np.ndarray | None = None,
    ) -> list[int]:
        """
        Add multiple chunks with batched database writes.

        Internal method that performs batched inserts for better performance.

        Args:
            chunks: Chunk records to add
            embeddings: Pre-computed embeddings (computed if None)

        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

        # Batch insert chunks and embeddings
        chunk_ids = []
        embedding_data = []

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            chunk_id = self._chunk_repo.insert(chunk)
            chunk_ids.append(chunk_id)
            embedding_data.append(
                (chunk_id, embeddings[i].astype(np.float32).tobytes()))

        # Batch insert embeddings
        self.db.executemany(
            """
            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
            """,
            embedding_data,
        )
        self.db._get_connection().commit()

        # Add to index in one batch
        self._index.add(chunk_ids, embeddings)

        logger.debug(f"Added {len(chunks)} chunks to vector store (batched)")
        return chunk_ids

    async def search_text_async(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search by text query asynchronously.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results sorted by relevance
        """
        import asyncio
        return await asyncio.to_thread(self.search_text, query, top_k, filters)

    async def search_embedding_async(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search by embedding vector asynchronously.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results sorted by relevance
        """
        import asyncio
        return await asyncio.to_thread(self.search_embedding, query_embedding, top_k, filters)

    async def delete_by_page_async(self, page_id: int) -> int:
        """
        Delete all vectors for a page asynchronously.

        Args:
            page_id: Page ID

        Returns:
            Number of vectors deleted
        """
        import asyncio
        return await asyncio.to_thread(self.delete_by_page, page_id)

    async def delete_by_crawl_async(self, crawl_id: int) -> int:
        """
        Delete all vectors for a crawl asynchronously.

        Args:
            crawl_id: Crawl ID

        Returns:
            Number of vectors deleted
        """
        import asyncio
        return await asyncio.to_thread(self.delete_by_crawl, crawl_id)

    def __repr__(self) -> str:
        return (
            f"VectorStore(vectors={self._index.size}, "
            f"dims={self.dimensions}, loaded={self._loaded})"
        )
