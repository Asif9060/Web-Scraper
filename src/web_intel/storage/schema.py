"""
Database schema definition and migration.

Manages SQLite schema creation and versioning.
"""

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)

# Current schema version
SCHEMA_VERSION = 1


class SchemaManager:
    """
    Manages database schema creation and migrations.

    Handles initial schema setup and version upgrades.

    Example:
        >>> manager = SchemaManager(connection)
        >>> manager.initialize()
        >>> if manager.needs_migration():
        ...     manager.migrate()
    """

    def __init__(self, connection) -> None:
        """
        Initialize schema manager.

        Args:
            connection: SQLite database connection
        """
        self.conn = connection

    def initialize(self) -> None:
        """
        Initialize database schema.

        Creates all tables if they don't exist.
        """
        logger.info("Initializing database schema")

        # Create schema version table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Check if we need to create schema
        cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
        current_version = cursor.fetchone()[0]

        if current_version is None:
            self._create_schema_v1()
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
            self.conn.commit()
            logger.info(f"Created schema version {SCHEMA_VERSION}")
        else:
            logger.info(f"Schema version {current_version} already exists")

    def _create_schema_v1(self) -> None:
        """Create version 1 of the database schema."""

        # Crawl sessions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS crawls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_url TEXT NOT NULL,
                domain TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                pages_crawled INTEGER DEFAULT 0,
                pages_processed INTEGER DEFAULT 0,
                pages_failed INTEGER DEFAULT 0,
                max_pages INTEGER DEFAULT 0,
                max_depth INTEGER DEFAULT 0,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                error_message TEXT,
                config TEXT
            )
        """)

        # Pages table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crawl_id INTEGER REFERENCES crawls(id),
                url TEXT NOT NULL,
                canonical_url TEXT,
                title TEXT,
                content_text TEXT,
                content_html TEXT,
                summary TEXT,
                topics TEXT,
                word_count INTEGER DEFAULT 0,
                language TEXT,
                status TEXT DEFAULT 'pending',
                depth INTEGER DEFAULT 0,
                parent_url TEXT,
                content_hash TEXT,
                quality_score REAL DEFAULT 0.0,
                crawled_at TEXT,
                processed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                UNIQUE(crawl_id, url)
            )
        """)

        # Create indexes for pages
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_crawl_id ON pages(crawl_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_content_hash ON pages(content_hash)"
        )

        # Entities table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER REFERENCES pages(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                mentions INTEGER DEFAULT 1,
                context TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for entities
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_page_id ON entities(page_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)"
        )

        # Chunks table (for vector search)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER REFERENCES pages(id) ON DELETE CASCADE,
                chunk_index INTEGER DEFAULT 0,
                text TEXT NOT NULL,
                summary TEXT,
                start_char INTEGER DEFAULT 0,
                end_char INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for chunks
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)"
        )

        # Links table (page graph)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_page_id INTEGER REFERENCES pages(id) ON DELETE CASCADE,
                target_url TEXT NOT NULL,
                target_page_id INTEGER REFERENCES pages(id) ON DELETE SET NULL,
                anchor_text TEXT,
                link_type TEXT DEFAULT 'internal',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for links
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_page_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_page_id)"
        )

        # Facts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER REFERENCES pages(id) ON DELETE CASCADE,
                statement TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_chunk_id INTEGER REFERENCES chunks(id),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_page_id ON facts(page_id)"
        )

        # Full-text search virtual table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                title,
                content_text,
                summary,
                content='pages',
                content_rowid='id'
            )
        """)

        # Triggers to keep FTS in sync
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_fts_insert AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, title, content_text, summary)
                VALUES (new.id, new.title, new.content_text, new.summary);
            END
        """)

        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_fts_delete AFTER DELETE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, title, content_text, summary)
                VALUES ('delete', old.id, old.title, old.content_text, old.summary);
            END
        """)

        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS pages_fts_update AFTER UPDATE ON pages BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, title, content_text, summary)
                VALUES ('delete', old.id, old.title, old.content_text, old.summary);
                INSERT INTO pages_fts(rowid, title, content_text, summary)
                VALUES (new.id, new.title, new.content_text, new.summary);
            END
        """)

        logger.info("Schema v1 created successfully")

    def get_version(self) -> int:
        """Get current schema version."""
        cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
        version = cursor.fetchone()[0]
        return version or 0

    def needs_migration(self) -> bool:
        """Check if schema needs migration."""
        return self.get_version() < SCHEMA_VERSION

    def migrate(self) -> None:
        """Run pending migrations."""
        current = self.get_version()

        if current < SCHEMA_VERSION:
            logger.info(
                f"Migrating from version {current} to {SCHEMA_VERSION}")
            # Add migration logic here as schema evolves
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
            self.conn.commit()
