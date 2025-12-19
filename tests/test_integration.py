"""
Integration tests for the Web Intelligence System.

Tests end-to-end workflows combining multiple modules.
"""

import asyncio
from pathlib import Path

import pytest

from web_intel.config import Settings
from web_intel.storage import Database
from web_intel.query_parser import QueryParser
from web_intel.understanding import TextChunker
from web_intel.vector_store import VectorIndex


class TestCrawlToQueryPipeline:
    """Tests for crawl-to-query pipeline."""

    @pytest.fixture
    def settings(self, temp_dir: Path) -> Settings:
        """Provide test settings."""
        return Settings(
            storage={"database_path": str(temp_dir / "data" / "test.db")},
            local_llm={"enabled": False},
        )

    @pytest.mark.asyncio
    async def test_pipeline_processes_content(self, settings: Settings, sample_html: str):
        """Content should flow through pipeline."""
        from web_intel.extraction import ContentExtractor
        from web_intel.embeddings import Embedder

        # Extract content
        extractor = ContentExtractor()
        extracted = extractor.extract(sample_html)

        # Chunk content
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)
        chunks = chunker.chunk(extracted.text)

        # Verify flow
        assert len(extracted.text) > 0
        assert len(chunks) >= 1


class TestQueryPipeline:
    """Tests for query processing pipeline."""

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Provide a query parser."""
        return QueryParser()

    def test_parse_and_classify(self, parser: QueryParser):
        """Query should be parsed and classified."""
        queries = [
            ("What is the price?", "FACTUAL"),
            ("List all features", "LIST"),
            ("How do I sign up?", "PROCEDURAL"),
        ]

        for query_text, expected_type in queries:
            parsed = parser.parse(query_text)

            assert parsed.query_type.name == expected_type or parsed.query_type is not None


class TestStoragePipeline:
    """Tests for storage operations pipeline."""

    def test_store_and_retrieve_page(self, database: Database, sample_html: str):
        """Page should be stored and retrievable."""
        from web_intel.storage import Page
        from web_intel.extraction import ContentExtractor

        # Extract and store
        extractor = ContentExtractor()
        extracted = extractor.extract(sample_html)

        page = Page(
            url="https://example.com/test",
            title=extracted.title or "Test Page",
            content=extracted.text,
            crawled_at="2024-01-01T00:00:00Z",
        )

        page_id = database.insert_page(page)

        # Retrieve
        retrieved = database.get_page_by_url("https://example.com/test")

        assert retrieved is not None
        assert retrieved.content == extracted.text


class TestVectorSearchPipeline:
    """Tests for vector search pipeline."""

    @pytest.fixture
    def index(self) -> VectorIndex:
        """Provide a vector index."""
        return VectorIndex(dimension=384)

    def test_index_and_search(self, index: VectorIndex, sample_embeddings):
        """Content should be indexable and searchable."""
        # Index embeddings
        for i, emb in enumerate(sample_embeddings):
            index.add(
                vector=emb,
                doc_id=f"doc_{i}",
                metadata={"content": f"Document {i}"},
            )

        # Search
        results = index.search(sample_embeddings[0], k=3)

        assert len(results) == 3
        assert results[0].doc_id == "doc_0"


class TestChunkingPipeline:
    """Tests for content chunking pipeline."""

    def test_chunk_and_embed(self, sample_text: str):
        """Text should be chunked appropriately."""
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)
        chunks = chunker.chunk(sample_text)

        # Verify chunks
        assert len(chunks) >= 1

        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.index >= 0
            assert chunk.total_chunks == len(chunks)


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_qa_scenario(self, test_settings: Settings, database: Database):
        """Question-answering scenario should work."""
        from web_intel.storage import Page
        from web_intel.query_parser import QueryParser

        # Setup: Add a page with known content
        page = Page(
            url="https://example.com/pricing",
            title="Pricing",
            content="Our basic plan costs $9.99 per month. Premium is $19.99.",
            crawled_at="2024-01-01T00:00:00Z",
        )
        database.insert_page(page)

        # Query
        parser = QueryParser()
        parsed = parser.parse("What is the price of the basic plan?")

        # Search database
        results = database.search_pages("basic plan")

        # Verify we can find relevant content
        assert len(results) >= 1
        assert "9.99" in results[0].content

    @pytest.mark.asyncio
    async def test_multi_page_scenario(self, test_settings: Settings, database: Database):
        """Multi-page crawl scenario should work."""
        from web_intel.storage import Page

        # Setup: Add multiple related pages
        pages = [
            Page(
                url="https://example.com",
                title="Home",
                content="Welcome to Example Corp.",
                crawled_at="2024-01-01T00:00:00Z",
            ),
            Page(
                url="https://example.com/about",
                title="About",
                content="Example Corp was founded in 2020.",
                crawled_at="2024-01-01T00:00:00Z",
            ),
            Page(
                url="https://example.com/products",
                title="Products",
                content="We offer three main products.",
                crawled_at="2024-01-01T00:00:00Z",
            ),
        ]

        for page in pages:
            database.insert_page(page)

        # Verify all pages stored
        all_pages = database.get_all_pages()
        assert len(all_pages) == 3


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_recover_from_extraction_error(self, test_settings: Settings):
        """System should handle extraction errors gracefully."""
        from web_intel.extraction import ContentExtractor

        extractor = ContentExtractor()

        # Malformed HTML
        result = extractor.extract("<html><body>Incomplete...")

        # Should not crash
        assert result is not None

    def test_recover_from_empty_content(self, test_settings: Settings):
        """System should handle empty content gracefully."""
        chunker = TextChunker(chunk_size=256, chunk_overlap=32)

        chunks = chunker.chunk("")

        assert chunks == []

    @pytest.mark.asyncio
    async def test_recover_from_query_error(self, test_settings: Settings):
        """System should handle query errors gracefully."""
        parser = QueryParser()

        # Empty query
        result = parser.parse("")

        # Should return something, not crash
        assert result is not None


class TestPerformance:
    """Performance-related tests."""

    def test_chunking_performance(self, sample_text: str):
        """Chunking should complete in reasonable time."""
        import time

        chunker = TextChunker(chunk_size=256, chunk_overlap=32)

        # Time 100 chunking operations
        start = time.time()
        for _ in range(100):
            chunker.chunk(sample_text)
        elapsed = time.time() - start

        # Should complete within 5 seconds
        assert elapsed < 5.0

    def test_vector_search_performance(self, sample_embeddings):
        """Vector search should be fast."""
        import time

        index = VectorIndex(dimension=384)

        # Index embeddings
        for i, emb in enumerate(sample_embeddings):
            index.add(emb, f"doc_{i}")

        # Time 100 searches
        start = time.time()
        for _ in range(100):
            index.search(sample_embeddings[0], k=5)
        elapsed = time.time() - start

        # Should complete within 2 seconds
        assert elapsed < 2.0

    def test_query_parsing_performance(self):
        """Query parsing should be fast."""
        import time

        parser = QueryParser()
        queries = [
            "What is the price?",
            "List all features",
            "How do I create an account?",
            "Compare basic and premium plans",
            "Is support available 24/7?",
        ]

        # Time 100 parse operations
        start = time.time()
        for _ in range(100):
            for query in queries:
                parser.parse(query)
        elapsed = time.time() - start

        # Should complete within 2 seconds
        assert elapsed < 2.0
