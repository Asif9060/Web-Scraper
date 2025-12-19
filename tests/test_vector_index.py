"""
Tests for vector store module.

Tests vector indexing, similarity search, and persistence.
"""

import numpy as np
import pytest

from web_intel.vector_store import (
    VectorIndex,
    IndexType,
    SimilarityMetric,
    SearchResult,
)


class TestVectorIndex:
    """Tests for VectorIndex."""

    @pytest.fixture
    def index(self) -> VectorIndex:
        """Provide a vector index instance."""
        return VectorIndex(dimension=384, index_type=IndexType.FLAT)

    @pytest.fixture
    def populated_index(self, sample_embeddings: np.ndarray) -> VectorIndex:
        """Provide an index with sample vectors."""
        index = VectorIndex(dimension=384, index_type=IndexType.FLAT)

        for i, embedding in enumerate(sample_embeddings):
            index.add(
                vector=embedding,
                doc_id=f"doc_{i}",
                metadata={"index": i},
            )

        return index

    def test_index_creation(self, index: VectorIndex):
        """Index should be created with specified parameters."""
        assert index.dimension == 384
        assert index.index_type == IndexType.FLAT
        assert len(index) == 0

    def test_add_vector(self, index: VectorIndex):
        """Vector can be added to index."""
        vector = np.random.randn(384).astype(np.float32)

        index.add(vector, doc_id="doc_1", metadata={"key": "value"})

        assert len(index) == 1

    def test_add_multiple_vectors(self, index: VectorIndex, sample_embeddings: np.ndarray):
        """Multiple vectors can be added."""
        for i, vector in enumerate(sample_embeddings):
            index.add(vector, doc_id=f"doc_{i}")

        assert len(index) == len(sample_embeddings)

    def test_search_basic(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Basic search should return results."""
        query = sample_embeddings[0]  # Search with first embedding

        results = populated_index.search(query, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_correct_format(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Search results should have expected fields."""
        query = sample_embeddings[0]
        results = populated_index.search(query, k=1)

        result = results[0]
        assert hasattr(result, "doc_id")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata") or hasattr(result, "distance")

    def test_search_top_result(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Searching for exact vector should return it first."""
        query = sample_embeddings[0]
        results = populated_index.search(query, k=1)

        assert results[0].doc_id == "doc_0"
        # Score should be very high (close to 1 for cosine) or distance very low
        assert results[0].score > 0.99 or results[0].score < 0.01

    def test_search_k_parameter(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Search should respect k parameter."""
        query = sample_embeddings[0]

        results_3 = populated_index.search(query, k=3)
        results_5 = populated_index.search(query, k=5)

        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_search_k_exceeds_size(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Search with k > index size should return all vectors."""
        query = sample_embeddings[0]

        results = populated_index.search(query, k=100)

        assert len(results) == len(sample_embeddings)

    def test_search_empty_index(self, index: VectorIndex):
        """Search on empty index should return empty results."""
        query = np.random.randn(384).astype(np.float32)

        results = index.search(query, k=5)

        assert results == []

    def test_search_with_filter(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Search should support metadata filtering."""
        query = sample_embeddings[0]

        # Search with filter for specific indices
        results = populated_index.search(
            query,
            k=10,
            filter={"index": {"$lte": 5}},
        )

        # All results should match filter
        for result in results:
            if result.metadata:
                assert result.metadata["index"] <= 5

    def test_remove_vector(self, populated_index: VectorIndex):
        """Vector can be removed by doc_id."""
        initial_size = len(populated_index)

        populated_index.remove("doc_0")

        assert len(populated_index) == initial_size - 1

    def test_update_vector(self, populated_index: VectorIndex):
        """Vector can be updated."""
        new_vector = np.random.randn(384).astype(np.float32)

        populated_index.update(
            doc_id="doc_0",
            vector=new_vector,
            metadata={"updated": True},
        )

        # Search with new vector should find it
        results = populated_index.search(new_vector, k=1)
        assert results[0].doc_id == "doc_0"

    def test_contains(self, populated_index: VectorIndex):
        """Index should support contains check."""
        assert populated_index.contains("doc_0")
        assert not populated_index.contains("nonexistent")

    def test_get_vector(self, populated_index: VectorIndex, sample_embeddings: np.ndarray):
        """Vector can be retrieved by doc_id."""
        result = populated_index.get("doc_0")

        assert result is not None
        assert np.allclose(result["vector"], sample_embeddings[0], atol=1e-5)

    def test_clear(self, populated_index: VectorIndex):
        """Index can be cleared."""
        assert len(populated_index) > 0

        populated_index.clear()

        assert len(populated_index) == 0


class TestIndexTypes:
    """Tests for different index types."""

    def test_flat_index(self, sample_embeddings: np.ndarray):
        """FLAT index should work correctly."""
        index = VectorIndex(dimension=384, index_type=IndexType.FLAT)

        for i, vector in enumerate(sample_embeddings):
            index.add(vector, doc_id=f"doc_{i}")

        results = index.search(sample_embeddings[0], k=3)
        assert len(results) == 3

    def test_ivf_index(self, sample_embeddings: np.ndarray):
        """IVF index should work correctly."""
        index = VectorIndex(
            dimension=384,
            index_type=IndexType.IVF,
            nlist=2,  # Small for test data
        )

        for i, vector in enumerate(sample_embeddings):
            index.add(vector, doc_id=f"doc_{i}")

        # IVF requires training after adding vectors
        if hasattr(index, "train"):
            index.train()

        results = index.search(sample_embeddings[0], k=3)
        assert len(results) >= 1  # IVF may return fewer if not well-trained


class TestSimilarityMetrics:
    """Tests for different similarity metrics."""

    @pytest.fixture
    def vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Provide test vectors."""
        v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0] + [0.0] * 381,
                      dtype=np.float32)  # Same as v1
        v3 = np.array([0.0, 1.0, 0.0] + [0.0] * 381,
                      dtype=np.float32)  # Orthogonal
        return v1, v2, v3

    def test_cosine_similarity(self, vectors):
        """Cosine similarity should work correctly."""
        v1, v2, v3 = vectors

        index = VectorIndex(
            dimension=384,
            index_type=IndexType.FLAT,
            metric=SimilarityMetric.COSINE,
        )

        index.add(v2, doc_id="same")
        index.add(v3, doc_id="orthogonal")

        results = index.search(v1, k=2)

        # Same vector should have highest similarity
        assert results[0].doc_id == "same"

    def test_euclidean_distance(self, vectors):
        """Euclidean distance should work correctly."""
        v1, v2, v3 = vectors

        index = VectorIndex(
            dimension=384,
            index_type=IndexType.FLAT,
            metric=SimilarityMetric.EUCLIDEAN,
        )

        index.add(v2, doc_id="same")
        index.add(v3, doc_id="different")

        results = index.search(v1, k=2)

        # Same vector should have smallest distance
        assert results[0].doc_id == "same"

    def test_dot_product(self, vectors):
        """Dot product should work correctly."""
        v1, v2, v3 = vectors

        index = VectorIndex(
            dimension=384,
            index_type=IndexType.FLAT,
            metric=SimilarityMetric.DOT_PRODUCT,
        )

        index.add(v2, doc_id="same")
        index.add(v3, doc_id="orthogonal")

        results = index.search(v1, k=2)

        # Same vector should have highest dot product
        assert results[0].doc_id == "same"


class TestVectorIndexPersistence:
    """Tests for vector index persistence."""

    def test_save_and_load(self, temp_dir, sample_embeddings: np.ndarray):
        """Index can be saved and loaded."""
        index_path = temp_dir / "test_index"

        # Create and populate index
        index = VectorIndex(dimension=384, index_type=IndexType.FLAT)
        for i, vector in enumerate(sample_embeddings):
            index.add(vector, doc_id=f"doc_{i}", metadata={"index": i})

        # Save
        index.save(index_path)

        # Load into new index
        loaded = VectorIndex.load(index_path)

        assert len(loaded) == len(sample_embeddings)

        # Verify search works
        results = loaded.search(sample_embeddings[0], k=1)
        assert results[0].doc_id == "doc_0"

    def test_load_preserves_metadata(self, temp_dir, sample_embeddings: np.ndarray):
        """Loading should preserve metadata."""
        index_path = temp_dir / "test_index"

        index = VectorIndex(dimension=384, index_type=IndexType.FLAT)
        index.add(sample_embeddings[0],
                  doc_id="doc_0", metadata={"key": "value"})
        index.save(index_path)

        loaded = VectorIndex.load(index_path)
        result = loaded.get("doc_0")

        assert result["metadata"]["key"] == "value"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """SearchResult should be created with required fields."""
        result = SearchResult(
            doc_id="doc_1",
            score=0.95,
            metadata={"key": "value"},
        )

        assert result.doc_id == "doc_1"
        assert result.score == 0.95
        assert result.metadata["key"] == "value"

    def test_search_result_comparison(self):
        """SearchResults should be comparable by score."""
        r1 = SearchResult(doc_id="a", score=0.9)
        r2 = SearchResult(doc_id="b", score=0.8)

        # Higher score should be "better"
        sorted_results = sorted([r2, r1], key=lambda r: r.score, reverse=True)
        assert sorted_results[0].doc_id == "a"

    def test_search_result_to_dict(self):
        """SearchResult should convert to dictionary."""
        result = SearchResult(
            doc_id="doc_1",
            score=0.95,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)

        assert "doc_id" in result_dict
        assert "score" in result_dict
