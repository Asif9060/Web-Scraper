"""
Text embedding generation using sentence-transformers.

Provides efficient embedding generation optimized for CPU-only,
8GB RAM environments using all-MiniLM-L6-v2 (384 dimensions).
"""

import gc
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from web_intel.config import Settings
from web_intel.core.exceptions import EmbeddingError
from web_intel.utils.logging import get_logger
from web_intel.utils.metrics import Metrics

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of embedding generation.

    Contains the embedding vector and metadata about the operation.
    """

    embedding: np.ndarray  # Shape: (dimensions,)
    text_length: int  # Character count of source text
    token_count: int  # Approximate token count
    model_name: str = ""

    @property
    def dimensions(self) -> int:
        """Number of dimensions in embedding."""
        return self.embedding.shape[0]

    def to_list(self) -> list[float]:
        """Convert embedding to list for serialization."""
        return self.embedding.tolist()


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""

    embeddings: np.ndarray  # Shape: (n_texts, dimensions)
    text_lengths: list[int]
    model_name: str = ""

    @property
    def count(self) -> int:
        """Number of embeddings."""
        return self.embeddings.shape[0]

    @property
    def dimensions(self) -> int:
        """Number of dimensions per embedding."""
        return self.embeddings.shape[1]


@dataclass
class SimilarityResult:
    """Result of similarity computation."""

    score: float  # Cosine similarity [-1, 1], typically [0, 1] for normalized
    text1_length: int
    text2_length: int

    @property
    def is_similar(self) -> bool:
        """Check if texts are similar (threshold 0.7)."""
        return self.score >= 0.7


class Embedder:
    """
    Text embedding generator using sentence-transformers.

    Optimized for memory efficiency with:
    - Lazy model loading
    - Batch processing
    - Optional normalization for cosine similarity

    Example:
        >>> embedder = Embedder.from_settings(settings)
        >>> result = embedder.embed("Hello, world!")
        >>> print(f"Embedding shape: {result.embedding.shape}")

        >>> # Batch embedding
        >>> texts = ["First text", "Second text", "Third text"]
        >>> embeddings = embedder.embed_batch(texts)
        >>> print(f"Batch shape: {embeddings.shape}")

        >>> # Similarity
        >>> sim = embedder.similarity("Hello", "Hi there")
        >>> print(f"Similarity: {sim.score:.3f}")
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialize embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cpu, cuda, mps)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar during encoding
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self.cache_dir = str(cache_dir) if cache_dir else None

        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._loaded = False
        self._dimensions: int | None = None

        logger.info(
            f"Embedder initialized (model={model_name}, device={device})"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Embedder":
        """
        Create Embedder from application settings.

        Args:
            settings: Application settings

        Returns:
            Configured Embedder instance
        """
        embed_settings = settings.embedding

        return cls(
            model_name=embed_settings.model_name,
            device=embed_settings.device,
            batch_size=embed_settings.batch_size,
            normalize=embed_settings.normalize_embeddings,
            show_progress=embed_settings.show_progress_bar,
            cache_dir=embed_settings.cache_dir,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions (loads model if needed)."""
        if self._dimensions is None:
            self.load()
        return self._dimensions

    def load(self) -> None:
        """
        Load the embedding model.

        Thread-safe lazy loading on first use.

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            logger.info(f"Loading embedding model: {self.model_name}")

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir,
                )

                # Get dimensions from model
                self._dimensions = self._model.get_sentence_embedding_dimension()

                self._loaded = True
                logger.info(
                    f"Embedding model loaded (dimensions={self._dimensions})"
                )

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise EmbeddingError(
                    f"Failed to load embedding model {self.model_name}",
                    details={"error": str(e), "model": self.model_name},
                ) from e

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._lock:
            if not self._loaded:
                return

            logger.info("Unloading embedding model")

            del self._model
            self._model = None
            self._loaded = False

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Embedding model unloaded")

    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        self.load()

        try:
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Record metrics
            Metrics.get().increment("embeddings_generated")

            return EmbeddingResult(
                embedding=embedding,
                text_length=len(text),
                token_count=len(text.split()),
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(
                "Embedding generation failed",
                details={"error": str(e), "text_length": len(text)},
            ) from e

    def embed_batch(
        self,
        texts: Sequence[str],
        show_progress: bool | None = None,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: Sequence of texts to embed
            show_progress: Override default progress bar setting

        Returns:
            NumPy array of shape (n_texts, dimensions)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return np.array([])

        self.load()

        try:
            show = show_progress if show_progress is not None else self.show_progress

            embeddings = self._model.encode(
                list(texts),
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
                show_progress_bar=show,
                convert_to_numpy=True,
            )

            # Record metrics for batch
            Metrics.get().increment("embeddings_generated", len(texts))

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(
                "Batch embedding generation failed",
                details={"error": str(e), "batch_size": len(texts)},
            ) from e

    def embed_batch_with_metadata(
        self,
        texts: Sequence[str],
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings with metadata.

        Args:
            texts: Sequence of texts to embed

        Returns:
            BatchEmbeddingResult with embeddings and metadata
        """
        embeddings = self.embed_batch(texts)

        return BatchEmbeddingResult(
            embeddings=embeddings,
            text_lengths=[len(t) for t in texts],
            model_name=self.model_name,
        )

    def similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            SimilarityResult with score and metadata
        """
        embeddings = self.embed_batch([text1, text2])

        # Compute cosine similarity
        if self.normalize:
            # Already normalized, dot product = cosine similarity
            score = float(np.dot(embeddings[0], embeddings[1]))
        else:
            # Compute cosine similarity manually
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            if norm1 > 0 and norm2 > 0:
                score = float(
                    np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2))
            else:
                score = 0.0

        return SimilarityResult(
            score=score,
            text1_length=len(text1),
            text2_length=len(text2),
        )

    def similarity_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for texts.

        Args:
            texts: Sequence of texts

        Returns:
            NumPy array of shape (n, n) with similarity scores
        """
        embeddings = self.embed_batch(texts)

        if self.normalize:
            # Normalized embeddings: dot product = cosine similarity
            return np.dot(embeddings, embeddings.T)
        else:
            # Compute cosine similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.maximum(norms, 1e-10)
            return np.dot(normalized, normalized.T)

    def find_most_similar(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: Candidate texts to search
            top_k: Number of results to return

        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        if not candidates:
            return []

        # Embed query and candidates together
        all_texts = [query] + list(candidates)
        embeddings = self.embed_batch(all_texts)

        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        # Compute similarities
        if self.normalize:
            scores = np.dot(candidate_embeddings, query_embedding)
        else:
            query_norm = np.linalg.norm(query_embedding)
            cand_norms = np.linalg.norm(candidate_embeddings, axis=1)
            scores = np.dot(candidate_embeddings, query_embedding) / (
                cand_norms * query_norm + 1e-10
            )

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def find_similar_to_embedding(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to a query embedding.

        Args:
            query_embedding: Query vector (dimensions,)
            candidate_embeddings: Candidate matrix (n, dimensions)
            top_k: Number of results

        Returns:
            List of (index, score) tuples
        """
        if len(candidate_embeddings) == 0:
            return []

        # Compute similarities
        if self.normalize:
            scores = np.dot(candidate_embeddings, query_embedding)
        else:
            query_norm = np.linalg.norm(query_embedding)
            cand_norms = np.linalg.norm(candidate_embeddings, axis=1)
            scores = np.dot(candidate_embeddings, query_embedding) / (
                cand_norms * query_norm + 1e-10
            )

        # Get top-k
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def cluster_texts(
        self,
        texts: Sequence[str],
        n_clusters: int,
    ) -> list[int]:
        """
        Cluster texts using K-means on embeddings.

        Args:
            texts: Texts to cluster
            n_clusters: Number of clusters

        Returns:
            List of cluster assignments (one per text)
        """
        from sklearn.cluster import KMeans

        if len(texts) < n_clusters:
            return list(range(len(texts)))

        embeddings = self.embed_batch(texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        return labels.tolist()

    def deduplicate(
        self,
        texts: Sequence[str],
        threshold: float = 0.95,
    ) -> list[int]:
        """
        Find duplicate/near-duplicate texts.

        Args:
            texts: Texts to deduplicate
            threshold: Similarity threshold for duplicates

        Returns:
            List of indices to keep (unique texts)
        """
        if len(texts) <= 1:
            return list(range(len(texts)))

        embeddings = self.embed_batch(texts)
        similarity_matrix = self.similarity_matrix(texts)

        # Greedy selection of unique texts
        keep_indices = []
        seen = set()

        for i in range(len(texts)):
            if i in seen:
                continue

            keep_indices.append(i)

            # Mark similar texts as seen
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i, j] >= threshold:
                    seen.add(j)

        return keep_indices

    def __enter__(self) -> "Embedder":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        dims = f", dims={self._dimensions}" if self._dimensions else ""
        return f"Embedder(model={self.model_name!r}, {status}{dims})"
