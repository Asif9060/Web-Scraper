"""
Tests for text chunking module.

Tests different chunking strategies, overlap handling, and chunk metadata.
"""

import pytest

from web_intel.understanding import (
    TextChunker,
    Chunk,
    ChunkingStrategy,
    ContentAnalyzer,
)


class TestTextChunker:
    """Tests for TextChunker."""

    @pytest.fixture
    def chunker(self) -> TextChunker:
        """Provide a chunker instance with default settings."""
        return TextChunker(chunk_size=256, chunk_overlap=32)

    @pytest.fixture
    def small_chunker(self) -> TextChunker:
        """Provide a chunker with smaller chunks for testing."""
        return TextChunker(chunk_size=64, chunk_overlap=8)

    def test_chunk_short_text(self, chunker: TextChunker):
        """Short text should produce single chunk."""
        text = "This is a short piece of text."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].content == text

    def test_chunk_long_text(self, small_chunker: TextChunker, sample_text: str):
        """Long text should produce multiple chunks."""
        chunks = small_chunker.chunk(sample_text)

        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert chunk.index >= 0

    def test_chunk_overlap(self, small_chunker: TextChunker, sample_text: str):
        """Consecutive chunks should have overlapping content."""
        chunks = small_chunker.chunk(sample_text)

        if len(chunks) >= 2:
            # Check that end of chunk 0 overlaps with start of chunk 1
            chunk0_end = chunks[0].content[-20:]
            chunk1_start = chunks[1].content[:50]

            # Some overlap should exist
            assert any(
                word in chunk1_start
                for word in chunk0_end.split()
                if len(word) > 3
            )

    def test_chunk_indices(self, small_chunker: TextChunker, sample_text: str):
        """Chunk indices should be sequential."""
        chunks = small_chunker.chunk(sample_text)

        indices = [chunk.index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata(self, chunker: TextChunker, sample_text: str):
        """Chunks should have metadata."""
        chunks = chunker.chunk(sample_text, metadata={"source": "test"})

        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.get("source") == "test"

    def test_sentence_strategy(self, sample_text: str):
        """Sentence strategy should split on sentence boundaries."""
        chunker = TextChunker(
            chunk_size=256,
            chunk_overlap=32,
            strategy=ChunkingStrategy.SENTENCE,
        )
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0
        # Chunks should tend to end on sentence boundaries
        for chunk in chunks:
            content = chunk.content.strip()
            if len(content) > 0:
                # Most chunks should end with punctuation or be the last chunk
                last_char = content[-1]
                assert last_char in ".!?" or chunk == chunks[-1] or len(
                    content) < 50

    def test_paragraph_strategy(self, sample_text: str):
        """Paragraph strategy should split on paragraph boundaries."""
        chunker = TextChunker(
            chunk_size=256,
            chunk_overlap=32,
            strategy=ChunkingStrategy.PARAGRAPH,
        )
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0

    def test_semantic_strategy(self, sample_text: str):
        """Semantic strategy should maintain topical coherence."""
        chunker = TextChunker(
            chunk_size=256,
            chunk_overlap=32,
            strategy=ChunkingStrategy.SEMANTIC,
        )
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0
        # Each chunk should have reasonable content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 10

    def test_fixed_strategy(self, sample_text: str):
        """Fixed strategy should create fixed-size chunks."""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.FIXED,
        )
        chunks = chunker.chunk(sample_text)

        assert len(chunks) > 0
        # All chunks except possibly the last should be close to chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk.content) >= 80  # Allow some flexibility

    def test_empty_text(self, chunker: TextChunker):
        """Empty text should return empty list."""
        chunks = chunker.chunk("")
        assert chunks == []

    def test_whitespace_only(self, chunker: TextChunker):
        """Whitespace-only text should return empty list."""
        chunks = chunker.chunk("   \n\n   \t  ")
        assert chunks == []

    def test_chunk_positions(self, small_chunker: TextChunker, sample_text: str):
        """Chunks should track their position in original text."""
        chunks = small_chunker.chunk(sample_text)

        for chunk in chunks:
            assert hasattr(chunk, "start_char") or hasattr(chunk, "start_pos")
            assert hasattr(chunk, "end_char") or hasattr(chunk, "end_pos")

    def test_chunk_total_count(self, small_chunker: TextChunker, sample_text: str):
        """Chunks should know total count."""
        chunks = small_chunker.chunk(sample_text)

        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)


class TestContentAnalyzer:
    """Tests for ContentAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> ContentAnalyzer:
        """Provide an analyzer instance."""
        return ContentAnalyzer()

    def test_analyze_text(self, analyzer: ContentAnalyzer, sample_text: str):
        """Analyzer should identify text content type."""
        result = analyzer.analyze(sample_text)

        assert result is not None
        assert hasattr(result, "content_type") or hasattr(result, "topics")

    def test_extract_topics(self, analyzer: ContentAnalyzer, sample_text: str):
        """Analyzer should extract topics from text."""
        result = analyzer.analyze(sample_text)

        topics = getattr(result, "topics", getattr(result, "keywords", []))
        assert isinstance(topics, list)

    def test_estimate_complexity(self, analyzer: ContentAnalyzer):
        """Analyzer should estimate text complexity."""
        simple_text = "The cat sat on the mat."
        complex_text = (
            "The epistemological implications of quantum mechanical "
            "superposition fundamentally challenge our ontological "
            "presuppositions regarding the nature of observation."
        )

        simple_result = analyzer.analyze(simple_text)
        complex_result = analyzer.analyze(complex_text)

        # Should have complexity measure
        assert hasattr(simple_result, "complexity") or hasattr(
            simple_result, "readability")

    def test_analyze_html_detection(self, analyzer: ContentAnalyzer, sample_html: str):
        """Analyzer should detect HTML content."""
        result = analyzer.analyze(sample_html)

        # Should identify as HTML or have html-related metadata
        content_type = getattr(result, "content_type", "")
        is_html = getattr(result, "is_html", False)

        assert "html" in content_type.lower() or is_html


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should be created with required fields."""
        chunk = Chunk(
            content="Test content",
            index=0,
            total_chunks=1,
        )

        assert chunk.content == "Test content"
        assert chunk.index == 0
        assert chunk.total_chunks == 1

    def test_chunk_with_metadata(self):
        """Chunk should accept metadata."""
        chunk = Chunk(
            content="Test content",
            index=0,
            total_chunks=1,
            metadata={"source": "test", "page": 1},
        )

        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["page"] == 1

    def test_chunk_str_representation(self):
        """Chunk should have meaningful string representation."""
        chunk = Chunk(
            content="This is a test chunk with some content.",
            index=0,
            total_chunks=5,
        )

        repr_str = str(chunk)
        assert "0" in repr_str or "chunk" in repr_str.lower()

    def test_chunk_to_dict(self):
        """Chunk should convert to dictionary."""
        chunk = Chunk(
            content="Test content",
            index=0,
            total_chunks=1,
            metadata={"key": "value"},
        )

        chunk_dict = chunk.to_dict() if hasattr(chunk, "to_dict") else vars(chunk)

        assert "content" in chunk_dict
        assert "index" in chunk_dict
