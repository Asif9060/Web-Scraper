"""
Database module for Web Intelligence System.

Provides SQLite-based storage with:
- Connection management with WAL mode
- Schema initialization and migrations
- Repository pattern for data access
- Vector storage using sqlite-vec
"""

from web_intel.storage.database import (
    Database,
    get_database,
)
from web_intel.storage.schema import (
    SchemaManager,
    SCHEMA_VERSION,
)
from web_intel.storage.repositories import (
    PageRepository,
    EntityRepository,
    ChunkRepository,
    CrawlRepository,
)
from web_intel.storage.models import (
    PageRecord,
    EntityRecord,
    ChunkRecord,
    CrawlRecord,
    PageStatus,
)

__all__ = [
    # Database
    "Database",
    "get_database",
    # Schema
    "SchemaManager",
    "SCHEMA_VERSION",
    # Repositories
    "PageRepository",
    "EntityRepository",
    "ChunkRepository",
    "CrawlRepository",
    # Models
    "PageRecord",
    "EntityRecord",
    "ChunkRecord",
    "CrawlRecord",
    "PageStatus",
]
