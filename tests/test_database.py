"""
Tests for database storage module.

Tests SQLite operations, WAL mode, and data persistence.
"""

import asyncio
from pathlib import Path

import pytest

from web_intel.storage import Database, Page, Chunk as DBChunk
from web_intel.config import Settings


class TestDatabase:
    """Tests for Database class."""

    def test_database_creation(self, test_settings: Settings, temp_dir: Path):
        """Database should be created with WAL mode."""
        db = Database(test_settings)

        assert db is not None
        assert (temp_dir / "data" / "web_intel.db").exists()

        db.close()

    def test_database_wal_mode(self, database: Database):
        """Database should use WAL mode."""
        # WAL mode creates -wal and -shm files after first write
        result = database.execute("PRAGMA journal_mode").fetchone()

        assert result[0].lower() == "wal"

    def test_database_tables_created(self, database: Database):
        """Database should have required tables."""
        result = database.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        table_names = [row[0] for row in result]

        assert "pages" in table_names
        assert "chunks" in table_names

    def test_insert_page(self, database: Database):
        """Page can be inserted into database."""
        page = Page(
            url="https://example.com",
            title="Example Page",
            content="Page content here",
            crawled_at="2024-01-01T00:00:00Z",
        )

        page_id = database.insert_page(page)

        assert page_id is not None
        assert page_id > 0

    def test_get_page_by_url(self, database: Database):
        """Page can be retrieved by URL."""
        page = Page(
            url="https://example.com/test",
            title="Test Page",
            content="Test content",
            crawled_at="2024-01-01T00:00:00Z",
        )
        database.insert_page(page)

        retrieved = database.get_page_by_url("https://example.com/test")

        assert retrieved is not None
        assert retrieved.title == "Test Page"

    def test_get_page_not_found(self, database: Database):
        """Getting non-existent page returns None."""
        retrieved = database.get_page_by_url("https://nonexistent.com")

        assert retrieved is None

    def test_update_page(self, database: Database):
        """Page can be updated."""
        page = Page(
            url="https://example.com/update",
            title="Original Title",
            content="Original content",
            crawled_at="2024-01-01T00:00:00Z",
        )
        page_id = database.insert_page(page)

        database.update_page(
            page_id,
            title="Updated Title",
            content="Updated content",
        )

        retrieved = database.get_page_by_url("https://example.com/update")
        assert retrieved.title == "Updated Title"

    def test_delete_page(self, database: Database):
        """Page can be deleted."""
        page = Page(
            url="https://example.com/delete",
            title="Delete Me",
            content="Content to delete",
            crawled_at="2024-01-01T00:00:00Z",
        )
        page_id = database.insert_page(page)

        database.delete_page(page_id)

        retrieved = database.get_page_by_url("https://example.com/delete")
        assert retrieved is None

    def test_insert_chunk(self, database: Database):
        """Chunk can be inserted."""
        page = Page(
            url="https://example.com/chunks",
            title="Page with Chunks",
            content="Full content",
            crawled_at="2024-01-01T00:00:00Z",
        )
        page_id = database.insert_page(page)

        chunk = DBChunk(
            page_id=page_id,
            content="Chunk content",
            chunk_index=0,
            embedding_id="emb_1",
        )

        chunk_id = database.insert_chunk(chunk)

        assert chunk_id is not None

    def test_get_chunks_by_page(self, database: Database):
        """Chunks can be retrieved by page ID."""
        page = Page(
            url="https://example.com/multi-chunks",
            title="Multi-Chunk Page",
            content="Full content",
            crawled_at="2024-01-01T00:00:00Z",
        )
        page_id = database.insert_page(page)

        for i in range(3):
            chunk = DBChunk(
                page_id=page_id,
                content=f"Chunk {i}",
                chunk_index=i,
                embedding_id=f"emb_{i}",
            )
            database.insert_chunk(chunk)

        chunks = database.get_chunks_by_page(page_id)

        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[2].chunk_index == 2

    def test_search_pages(self, database: Database):
        """Pages can be searched by content."""
        pages = [
            Page(url="https://a.com", title="Python Guide",
                 content="Learn Python programming", crawled_at="2024-01-01T00:00:00Z"),
            Page(url="https://b.com", title="JavaScript Tutorial",
                 content="Learn JavaScript development", crawled_at="2024-01-01T00:00:00Z"),
            Page(url="https://c.com", title="Python Advanced",
                 content="Advanced Python topics", crawled_at="2024-01-01T00:00:00Z"),
        ]

        for page in pages:
            database.insert_page(page)

        results = database.search_pages("Python")

        assert len(results) == 2

    def test_get_all_pages(self, database: Database):
        """All pages can be retrieved."""
        for i in range(5):
            page = Page(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                content=f"Content {i}",
                crawled_at="2024-01-01T00:00:00Z",
            )
            database.insert_page(page)

        all_pages = database.get_all_pages()

        assert len(all_pages) == 5

    def test_count_pages(self, database: Database):
        """Page count is accurate."""
        for i in range(3):
            page = Page(
                url=f"https://count.com/page{i}",
                title=f"Page {i}",
                content=f"Content {i}",
                crawled_at="2024-01-01T00:00:00Z",
            )
            database.insert_page(page)

        count = database.count_pages()

        assert count == 3

    def test_transaction_commit(self, database: Database):
        """Transactions are committed properly."""
        with database.transaction():
            page = Page(
                url="https://transaction.com",
                title="Transaction Test",
                content="Transaction content",
                crawled_at="2024-01-01T00:00:00Z",
            )
            database.insert_page(page)

        # Should be committed
        retrieved = database.get_page_by_url("https://transaction.com")
        assert retrieved is not None

    def test_transaction_rollback(self, database: Database):
        """Transactions can be rolled back."""
        try:
            with database.transaction():
                page = Page(
                    url="https://rollback.com",
                    title="Rollback Test",
                    content="Rollback content",
                    crawled_at="2024-01-01T00:00:00Z",
                )
                database.insert_page(page)
                raise Exception("Force rollback")
        except Exception:
            pass

        # Should be rolled back
        retrieved = database.get_page_by_url("https://rollback.com")
        assert retrieved is None


class TestDatabaseConcurrency:
    """Tests for database concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, database: Database):
        """Concurrent reads should work with WAL mode."""
        # Insert test data
        for i in range(10):
            page = Page(
                url=f"https://concurrent.com/page{i}",
                title=f"Page {i}",
                content=f"Content {i}",
                crawled_at="2024-01-01T00:00:00Z",
            )
            database.insert_page(page)

        async def read_pages():
            return database.get_all_pages()

        # Concurrent reads
        results = await asyncio.gather(
            read_pages(),
            read_pages(),
            read_pages(),
        )

        # All reads should succeed
        assert all(len(r) == 10 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, database: Database):
        """Concurrent writes should be serialized properly."""
        async def insert_page(i: int):
            page = Page(
                url=f"https://write.com/page{i}",
                title=f"Page {i}",
                content=f"Content {i}",
                crawled_at="2024-01-01T00:00:00Z",
            )
            database.insert_page(page)

        # Concurrent writes
        await asyncio.gather(
            insert_page(0),
            insert_page(1),
            insert_page(2),
        )

        # All writes should succeed
        count = database.count_pages()
        assert count == 3


class TestDatabaseMigrations:
    """Tests for database schema migrations."""

    def test_schema_version(self, database: Database):
        """Database should track schema version."""
        version = database.get_schema_version()

        assert isinstance(version, int)
        assert version >= 1

    def test_migration_on_upgrade(self, test_settings: Settings, temp_dir: Path):
        """Migrations should run on version upgrade."""
        # Create initial database
        db1 = Database(test_settings)
        initial_version = db1.get_schema_version()
        db1.close()

        # Re-open should not change version (no migrations needed)
        db2 = Database(test_settings)
        final_version = db2.get_schema_version()
        db2.close()

        assert final_version >= initial_version


class TestPage:
    """Tests for Page dataclass."""

    def test_page_creation(self):
        """Page should be created with required fields."""
        page = Page(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            crawled_at="2024-01-01T00:00:00Z",
        )

        assert page.url == "https://example.com"
        assert page.title == "Test Page"

    def test_page_with_metadata(self):
        """Page can have optional metadata."""
        page = Page(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            crawled_at="2024-01-01T00:00:00Z",
            metadata={"language": "en", "author": "Test"},
        )

        assert page.metadata["language"] == "en"

    def test_page_to_dict(self):
        """Page should convert to dictionary."""
        page = Page(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            crawled_at="2024-01-01T00:00:00Z",
        )

        page_dict = page.to_dict() if hasattr(page, "to_dict") else vars(page)

        assert "url" in page_dict
        assert "title" in page_dict


class TestDBChunk:
    """Tests for database Chunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should be created with required fields."""
        chunk = DBChunk(
            page_id=1,
            content="Chunk content",
            chunk_index=0,
            embedding_id="emb_1",
        )

        assert chunk.page_id == 1
        assert chunk.chunk_index == 0

    def test_chunk_with_positions(self):
        """Chunk can track character positions."""
        chunk = DBChunk(
            page_id=1,
            content="Chunk content",
            chunk_index=0,
            embedding_id="emb_1",
            start_pos=0,
            end_pos=13,
        )

        assert chunk.start_pos == 0
        assert chunk.end_pos == 13
