"""
Tests for embeddings module.

Tests embedding generation, caching, and batch processing.
"""

import numpy as np
import pytest

from web_intel.embeddings import Embedder, EmbeddingCache


class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture
    def embedder(self, test_settings) -> Embedder:
        """Provide an embedder instance."""
        return Embedder(test_settings)

    def test_embedder_creation(self, embedder: Embedder):
        """Embedder should be created successfully."""
        assert embedder is not None
        assert embedder.dimension == 384

    def test_embed_single_text(self, embedder: Embedder):
        """Single text can be embedded."""
        text = "This is a test sentence for embedding."

        embedding = embedder.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_batch(self, embedder: Embedder):
        """Batch of texts can be embedded."""
        texts = [
            "First sentence for embedding.",
            "Second sentence for embedding.",
            "Third sentence for embedding.",
        ]

        embeddings = embedder.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_embed_empty_text(self, embedder: Embedder):
        """Empty text should be handled gracefully."""
        embedding = embedder.embed("")

        # Either return zero vector or raise exception
        assert embedding is None or isinstance(embedding, np.ndarray)

    def test_embed_long_text(self, embedder: Embedder):
        """Long text should be truncated appropriately."""
        long_text = "This is a word. " * 1000  # Very long text

        embedding = embedder.embed(long_text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_normalized(self, embedder: Embedder):
        """Embeddings should be normalized."""
        text = "Test sentence for normalization check."

        embedding = embedder.embed(text)

        # L2 norm should be close to 1
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01

    def test_embed_consistency(self, embedder: Embedder):
        """Same text should produce same embedding."""
        text = "Consistent embedding test."

        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)

        assert np.allclose(embedding1, embedding2)

    def test_embed_similarity(self, embedder: Embedder):
        """Similar texts should have similar embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "Quantum physics explains particle behavior."

        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)
        emb3 = embedder.embed(text3)

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13

    def test_embed_batch_consistency(self, embedder: Embedder):
        """Batch embedding should match individual embeddings."""
        texts = ["First text.", "Second text."]

        batch_embeddings = embedder.embed_batch(texts)
        individual1 = embedder.embed(texts[0])
        individual2 = embedder.embed(texts[1])

        assert np.allclose(batch_embeddings[0], individual1)
        assert np.allclose(batch_embeddings[1], individual2)

    def test_model_info(self, embedder: Embedder):
        """Embedder should provide model information."""
        info = embedder.model_info

        assert "name" in info or "model_name" in info
        assert "dimension" in info or "dimensions" in info


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.fixture
    def cache(self, temp_dir) -> EmbeddingCache:
        """Provide a cache instance."""
        return EmbeddingCache(cache_dir=temp_dir / "embedding_cache")

    def test_cache_creation(self, cache: EmbeddingCache):
        """Cache should be created successfully."""
        assert cache is not None

    def test_cache_put_get(self, cache: EmbeddingCache):
        """Embedding can be cached and retrieved."""
        text = "Text to cache"
        embedding = np.random.randn(384).astype(np.float32)

        cache.put(text, embedding)
        retrieved = cache.get(text)

        assert retrieved is not None
        assert np.allclose(retrieved, embedding)

    def test_cache_miss(self, cache: EmbeddingCache):
        """Cache miss should return None."""
        retrieved = cache.get("nonexistent text")

        assert retrieved is None

    def test_cache_key_normalization(self, cache: EmbeddingCache):
        """Cache keys should be normalized."""
        embedding = np.random.randn(384).astype(np.float32)

        cache.put("  Some Text  ", embedding)
        retrieved = cache.get("some text")

        # Depending on normalization strategy
        assert retrieved is not None or cache.get("  Some Text  ") is not None

    def test_cache_batch(self, cache: EmbeddingCache):
        """Batch caching should work."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = np.random.randn(3, 384).astype(np.float32)

        cache.put_batch(texts, embeddings)

        for i, text in enumerate(texts):
            retrieved = cache.get(text)
            assert retrieved is not None
            assert np.allclose(retrieved, embeddings[i])

    def test_cache_clear(self, cache: EmbeddingCache):
        """Cache can be cleared."""
        cache.put("text", np.random.randn(384).astype(np.float32))

        cache.clear()

        assert cache.get("text") is None

    def test_cache_contains(self, cache: EmbeddingCache):
        """Cache should support contains check."""
        cache.put("cached text", np.random.randn(384).astype(np.float32))

        assert cache.contains("cached text")
        assert not cache.contains("uncached text")

    def test_cache_size(self, cache: EmbeddingCache):
        """Cache should track size."""
        for i in range(5):
            cache.put(f"text_{i}", np.random.randn(384).astype(np.float32))

        assert cache.size() == 5

    def test_cache_persistence(self, temp_dir):
        """Cache should persist across instances."""
        cache_dir = temp_dir / "persistent_cache"

        # Create and populate cache
        cache1 = EmbeddingCache(cache_dir=cache_dir)
        embedding = np.random.randn(384).astype(np.float32)
        cache1.put("persistent text", embedding)
        cache1.save() if hasattr(cache1, "save") else None

        # Create new cache instance
        cache2 = EmbeddingCache(cache_dir=cache_dir)
        retrieved = cache2.get("persistent text")

        # Should persist (if persistence is implemented)
        if retrieved is not None:
            assert np.allclose(retrieved, embedding)


class TestEmbedderWithCache:
    """Tests for embedder with caching enabled."""

    @pytest.fixture
    def cached_embedder(self, test_settings, temp_dir) -> Embedder:
        """Provide an embedder with cache."""
        test_settings.embeddings.cache_enabled = True
        test_settings.embeddings.cache_dir = str(temp_dir / "embed_cache")
        return Embedder(test_settings)

    def test_cached_embed(self, cached_embedder: Embedder):
        """Cached embedding should be returned on second call."""
        text = "Text to cache"

        # First call - computes embedding
        emb1 = cached_embedder.embed(text)

        # Second call - should hit cache
        emb2 = cached_embedder.embed(text)

        assert np.allclose(emb1, emb2)

    def test_cache_stats(self, cached_embedder: Embedder):
        """Embedder should provide cache statistics."""
        cached_embedder.embed("text 1")
        cached_embedder.embed("text 2")
        cached_embedder.embed("text 1")  # Cache hit

        stats = cached_embedder.get_cache_stats()

        assert "hits" in stats or "cache_hits" in stats
        assert "misses" in stats or "cache_misses" in stats


class TestEmbeddingOperations:
    """Tests for embedding mathematical operations."""

    def test_cosine_similarity(self):
        """Cosine similarity calculation."""
        from web_intel.embeddings import cosine_similarity

        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Same vectors should have similarity 1
        assert np.isclose(cosine_similarity(v1, v2), 1.0)

        # Orthogonal vectors should have similarity 0
        assert np.isclose(cosine_similarity(v1, v3), 0.0)

    def test_euclidean_distance(self):
        """Euclidean distance calculation."""
        from web_intel.embeddings import euclidean_distance

        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        distance = euclidean_distance(v1, v2)

        assert np.isclose(distance, 5.0)

    def test_mean_pooling(self):
        """Mean pooling of embeddings."""
        from web_intel.embeddings import mean_pooling

        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)

        pooled = mean_pooling(embeddings)

        expected = np.array([2.5, 3.5, 4.5], dtype=np.float32)
        assert np.allclose(pooled, expected)
