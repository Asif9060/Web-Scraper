"""
In-memory vector index for fast similarity search.

Provides efficient nearest neighbor search with support
for both exact and approximate methods.
"""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class IndexType(str, Enum):
    """Type of vector index."""

    FLAT = "flat"  # Exact brute-force search
    IVF = "ivf"  # Inverted file index (approximate)


class SimilarityMetric(str, Enum):
    """Similarity metric for vector comparison."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass
class IndexStats:
    """Statistics about the vector index."""

    vector_count: int
    dimensions: int
    index_type: IndexType
    memory_bytes: int
    is_trained: bool = True


class VectorIndex:
    """
    In-memory vector index for similarity search.

    Supports both exact (brute-force) and approximate search methods.
    Optimized for CPU-only environments.

    Example:
        >>> index = VectorIndex(dimensions=384)
        >>> index.add(ids=[1, 2, 3], vectors=embeddings)
        >>> results = index.search(query_vector, top_k=5)
        >>> for idx, score in results:
        ...     print(f"ID: {idx}, Score: {score:.3f}")
    """

    def __init__(
        self,
        dimensions: int,
        index_type: IndexType = IndexType.FLAT,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        normalize: bool = True,
    ) -> None:
        """
        Initialize vector index.

        Args:
            dimensions: Vector dimensionality
            index_type: Type of index (flat or ivf)
            metric: Similarity metric to use
            normalize: Whether to normalize vectors
        """
        self.dimensions = dimensions
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize

        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None
        self._id_to_position: dict[int, int] = {}
        self._lock = threading.Lock()

        # IVF-specific
        self._centroids: np.ndarray | None = None
        self._n_clusters: int = 0
        self._inverted_lists: dict[int, list[int]] = {}

        logger.debug(
            f"VectorIndex created (dims={dimensions}, type={index_type.value}, "
            f"metric={metric.value})"
        )

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return len(self._id_to_position)

    @property
    def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.size == 0

    def add(
        self,
        ids: Sequence[int],
        vectors: np.ndarray,
    ) -> None:
        """
        Add vectors to the index.

        Args:
            ids: Unique identifiers for vectors
            vectors: Vector matrix (n_vectors, dimensions)

        Raises:
            ValueError: If dimensions don't match or IDs already exist
        """
        if len(ids) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {vectors.shape[1]} don't match "
                f"index dimensions {self.dimensions}"
            )

        # Normalize if needed
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-10)

        with self._lock:
            # Check for duplicate IDs
            for id_ in ids:
                if id_ in self._id_to_position:
                    raise ValueError(f"ID {id_} already exists in index")

            # Initialize or append
            if self._vectors is None:
                self._vectors = vectors
                self._ids = np.array(ids, dtype=np.int64)
            else:
                self._vectors = np.vstack([self._vectors, vectors])
                self._ids = np.concatenate(
                    [self._ids, np.array(ids, dtype=np.int64)])

            # Update position mapping
            start_pos = len(self._id_to_position)
            for i, id_ in enumerate(ids):
                self._id_to_position[id_] = start_pos + i

            # Update IVF if needed
            if self.index_type == IndexType.IVF and self._centroids is not None:
                self._update_inverted_lists(vectors, list(
                    range(start_pos, start_pos + len(ids))))

        logger.debug(f"Added {len(ids)} vectors (total: {self.size})")

    def remove(self, ids: Sequence[int]) -> int:
        """
        Remove vectors from the index.

        Args:
            ids: IDs to remove

        Returns:
            Number of vectors removed
        """
        with self._lock:
            positions_to_remove = []
            for id_ in ids:
                if id_ in self._id_to_position:
                    positions_to_remove.append(self._id_to_position[id_])

            if not positions_to_remove:
                return 0

            # Create mask for keeping vectors
            mask = np.ones(len(self._ids), dtype=bool)
            mask[positions_to_remove] = False

            # Filter vectors and IDs
            self._vectors = self._vectors[mask]
            self._ids = self._ids[mask]

            # Rebuild position mapping
            self._id_to_position = {
                int(id_): i for i, id_ in enumerate(self._ids)
            }

            # Rebuild IVF lists if needed
            if self.index_type == IndexType.IVF and self._centroids is not None:
                self._rebuild_inverted_lists()

            return len(positions_to_remove)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[tuple[int, float]]:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of (id, score) tuples sorted by score descending
        """
        if self.is_empty:
            return []

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query if needed
        if self.normalize:
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        with self._lock:
            if self.index_type == IndexType.FLAT:
                results = self._search_flat(query, top_k)
            else:
                results = self._search_ivf(query, top_k)

        # Apply minimum score filter
        if min_score is not None:
            results = [(id_, score)
                       for id_, score in results if score >= min_score]

        return results

    def _search_flat(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Exact brute-force search."""
        scores = self._compute_similarity(query, self._vectors)
        scores = scores.flatten()

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(self._ids[i]), float(scores[i])) for i in top_indices]

    def _search_ivf(
        self,
        query: np.ndarray,
        top_k: int,
        n_probe: int = 10,
    ) -> list[tuple[int, float]]:
        """Approximate search using inverted file index."""
        if self._centroids is None:
            return self._search_flat(query, top_k)

        # Find nearest centroids
        centroid_scores = self._compute_similarity(
            query, self._centroids).flatten()
        n_probe = min(n_probe, len(centroid_scores))
        nearest_centroids = np.argpartition(
            centroid_scores, -n_probe)[-n_probe:]

        # Collect candidate positions from inverted lists
        candidates = []
        for c in nearest_centroids:
            candidates.extend(self._inverted_lists.get(int(c), []))

        if not candidates:
            return []

        # Score candidates
        candidate_vectors = self._vectors[candidates]
        scores = self._compute_similarity(query, candidate_vectors).flatten()

        # Get top-k
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (int(self._ids[candidates[i]]), float(scores[i]))
            for i in top_indices
        ]

    def _compute_similarity(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity scores."""
        if self.metric == SimilarityMetric.COSINE:
            # For normalized vectors, dot product = cosine similarity
            return np.dot(query, vectors.T)

        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(query, vectors.T)

        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity (1 / (1 + distance))
            distances = np.linalg.norm(vectors - query, axis=1)
            return 1.0 / (1.0 + distances)

        else:
            return np.dot(query, vectors.T)

    def build_ivf(self, n_clusters: int = 100) -> None:
        """
        Build IVF index for approximate search.

        Args:
            n_clusters: Number of clusters/centroids
        """
        if self.is_empty:
            logger.warning("Cannot build IVF on empty index")
            return

        if self.size < n_clusters:
            n_clusters = max(1, self.size // 10)

        logger.info(f"Building IVF index with {n_clusters} clusters")

        with self._lock:
            from sklearn.cluster import MiniBatchKMeans

            # Train k-means
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(1000, self.size),
                n_init=3,
            )
            kmeans.fit(self._vectors)

            self._centroids = kmeans.cluster_centers_.astype(np.float32)
            self._n_clusters = n_clusters

            # Build inverted lists
            self._rebuild_inverted_lists()

            self.index_type = IndexType.IVF

        logger.info("IVF index built successfully")

    def _rebuild_inverted_lists(self) -> None:
        """Rebuild inverted lists from current vectors."""
        self._inverted_lists = {i: [] for i in range(self._n_clusters)}

        if self._vectors is None or len(self._vectors) == 0:
            return

        # Assign vectors to nearest centroids
        similarities = np.dot(self._vectors, self._centroids.T)
        assignments = np.argmax(similarities, axis=1)

        for pos, cluster in enumerate(assignments):
            self._inverted_lists[int(cluster)].append(pos)

    def _update_inverted_lists(
        self,
        new_vectors: np.ndarray,
        positions: list[int],
    ) -> None:
        """Update inverted lists with new vectors."""
        similarities = np.dot(new_vectors, self._centroids.T)
        assignments = np.argmax(similarities, axis=1)

        for pos, cluster in zip(positions, assignments):
            self._inverted_lists[int(cluster)].append(pos)

    def get_vector(self, id_: int) -> np.ndarray | None:
        """Get vector by ID."""
        with self._lock:
            if id_ not in self._id_to_position:
                return None
            pos = self._id_to_position[id_]
            return self._vectors[pos].copy()

    def get_vectors(self, ids: Sequence[int]) -> dict[int, np.ndarray]:
        """Get multiple vectors by IDs."""
        result = {}
        with self._lock:
            for id_ in ids:
                if id_ in self._id_to_position:
                    pos = self._id_to_position[id_]
                    result[id_] = self._vectors[pos].copy()
        return result

    def clear(self) -> None:
        """Clear all vectors from index."""
        with self._lock:
            self._vectors = None
            self._ids = None
            self._id_to_position.clear()
            self._inverted_lists.clear()
            self._centroids = None

        logger.debug("Index cleared")

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        memory = 0
        if self._vectors is not None:
            memory += self._vectors.nbytes
        if self._ids is not None:
            memory += self._ids.nbytes
        if self._centroids is not None:
            memory += self._centroids.nbytes

        return IndexStats(
            vector_count=self.size,
            dimensions=self.dimensions,
            index_type=self.index_type,
            memory_bytes=memory,
            is_trained=self._centroids is not None if self.index_type == IndexType.IVF else True,
        )

    def save(self, path: str) -> None:
        """Save index to file."""
        with self._lock:
            data = {
                "dimensions": self.dimensions,
                "index_type": self.index_type.value,
                "metric": self.metric.value,
                "normalize": self.normalize,
                "vectors": self._vectors,
                "ids": self._ids,
                "centroids": self._centroids,
                "n_clusters": self._n_clusters,
            }
            np.savez_compressed(
                path, **{k: v for k, v in data.items() if v is not None})

        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load index from file."""
        data = np.load(path, allow_pickle=True)

        index = cls(
            dimensions=int(data["dimensions"]),
            index_type=IndexType(str(data["index_type"])),
            metric=SimilarityMetric(str(data["metric"])),
            normalize=bool(data["normalize"]),
        )

        if "vectors" in data and data["vectors"] is not None:
            index._vectors = data["vectors"]
            index._ids = data["ids"]
            index._id_to_position = {
                int(id_): i for i, id_ in enumerate(index._ids)
            }

        if "centroids" in data and data["centroids"] is not None:
            index._centroids = data["centroids"]
            index._n_clusters = int(data["n_clusters"])
            index._rebuild_inverted_lists()

        logger.info(f"Index loaded from {path} ({index.size} vectors)")
        return index

    def __repr__(self) -> str:
        return (
            f"VectorIndex(dimensions={self.dimensions}, size={self.size}, "
            f"type={self.index_type.value})"
        )
