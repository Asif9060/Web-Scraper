"""
SQLite database connection management.

Provides thread-safe database access with WAL mode
and connection pooling.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from web_intel.config import Settings
from web_intel.core.exceptions import DatabaseError
from web_intel.storage.schema import SchemaManager
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)

# Thread-local storage for connections
_local = threading.local()

# Global database instance
_database: "Database | None" = None
_database_lock = threading.Lock()


def get_database() -> "Database":
    """
    Get the global database instance.

    Returns:
        Database singleton instance

    Raises:
        DatabaseError: If database not initialized
    """
    global _database
    if _database is None:
        raise DatabaseError(
            "Database not initialized. Call Database.initialize() first.")
    return _database


class Database:
    """
    SQLite database manager with connection pooling.

    Provides thread-safe access to SQLite database with:
    - WAL mode for concurrent reads
    - Connection per thread
    - Automatic schema initialization

    Example:
        >>> db = Database.initialize(settings)
        >>> with db.connection() as conn:
        ...     cursor = conn.execute("SELECT * FROM pages")
        ...     rows = cursor.fetchall()

        >>> # Or use get_database() anywhere
        >>> db = get_database()
    """

    def __init__(
        self,
        database_path: Path,
        wal_mode: bool = True,
        cache_size_mb: int = 64,
        vector_dimensions: int = 384,
    ) -> None:
        """
        Initialize database manager.

        Args:
            database_path: Path to SQLite database file
            wal_mode: Enable WAL mode for better concurrency
            cache_size_mb: SQLite cache size in megabytes
            vector_dimensions: Dimensions for vector storage
        """
        self.database_path = Path(database_path)
        self.wal_mode = wal_mode
        self.cache_size_mb = cache_size_mb
        self.vector_dimensions = vector_dimensions

        self._connections: dict[int, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        self._initialized = False

        logger.info(f"Database manager created (path={database_path})")

    @classmethod
    def initialize(cls, settings: Settings) -> "Database":
        """
        Initialize and return global database instance.

        Args:
            settings: Application settings

        Returns:
            Database instance
        """
        global _database

        with _database_lock:
            if _database is not None:
                return _database

            storage = settings.storage
            _database = cls(
                database_path=storage.database_path,
                wal_mode=storage.wal_mode,
                cache_size_mb=storage.cache_size_mb,
                vector_dimensions=storage.vector_dimensions,
            )
            _database._setup()
            return _database

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        """Create database from settings (alias for initialize)."""
        return cls.initialize(settings)

    def _setup(self) -> None:
        """Setup database: create directory, initialize schema."""
        # Ensure directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        conn = self._get_connection()
        schema = SchemaManager(conn)
        schema.initialize()

        self._initialized = True
        logger.info("Database setup complete")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get connection for current thread.

        Creates new connection if none exists for this thread.
        """
        thread_id = threading.get_ident()

        if thread_id not in self._connections:
            with self._lock:
                if thread_id not in self._connections:
                    conn = self._create_connection()
                    self._connections[thread_id] = conn

        return self._connections[thread_id]

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new connection."""
        try:
            conn = sqlite3.connect(
                str(self.database_path),
                check_same_thread=False,
                timeout=30.0,
            )

            # Enable row factory for dict-like access
            conn.row_factory = sqlite3.Row

            # Configure pragmas
            cache_pages = (self.cache_size_mb * 1024 * 1024) // 4096
            conn.execute(f"PRAGMA cache_size = -{cache_pages}")
            conn.execute(
                "PRAGMA journal_mode = WAL" if self.wal_mode else "PRAGMA journal_mode = DELETE")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA foreign_keys = ON")

            logger.debug(
                f"Created new connection for thread {threading.get_ident()}")
            return conn

        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to create database connection: {e}",
                details={"path": str(self.database_path)},
            ) from e

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Get database connection as context manager.

        Automatically handles transactions.

        Yields:
            SQLite connection
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Execute operations in a transaction.

        Commits on success, rolls back on error.

        Yields:
            SQLite connection
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute SQL statement.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Cursor with results
        """
        conn = self._get_connection()
        try:
            return conn.execute(sql, params)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Query execution failed: {e}", query=sql) from e

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """
        Execute SQL statement for multiple parameter sets.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Cursor
        """
        conn = self._get_connection()
        try:
            return conn.executemany(sql, params_list)
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Batch execution failed: {e}", query=sql) from e

    def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """
        Fetch single row as dictionary.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Row as dict or None
        """
        cursor = self.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Fetch all rows as dictionaries.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of rows as dicts
        """
        cursor = self.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def insert(self, table: str, data: dict) -> int:
        """
        Insert row and return ID.

        Args:
            table: Table name
            data: Column-value pairs

        Returns:
            Inserted row ID
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        cursor = self.execute(sql, tuple(data.values()))
        self._get_connection().commit()
        return cursor.lastrowid

    def update(
        self,
        table: str,
        data: dict,
        where: str,
        params: tuple = (),
    ) -> int:
        """
        Update rows.

        Args:
            table: Table name
            data: Column-value pairs to update
            where: WHERE clause
            params: Parameters for WHERE clause

        Returns:
            Number of affected rows
        """
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

        all_params = tuple(data.values()) + params
        cursor = self.execute(sql, all_params)
        self._get_connection().commit()
        return cursor.rowcount

    def delete(self, table: str, where: str, params: tuple = ()) -> int:
        """
        Delete rows.

        Args:
            table: Table name
            where: WHERE clause
            params: Parameters for WHERE clause

        Returns:
            Number of deleted rows
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        cursor = self.execute(sql, params)
        self._get_connection().commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close all connections."""
        with self._lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()

        logger.info("All database connections closed")

    def vacuum(self) -> None:
        """Run VACUUM to optimize database."""
        logger.info("Running VACUUM")
        conn = self._get_connection()
        conn.execute("VACUUM")

    def checkpoint(self) -> None:
        """Force WAL checkpoint."""
        if self.wal_mode:
            logger.info("Running WAL checkpoint")
            conn = self._get_connection()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @property
    def size_bytes(self) -> int:
        """Get database file size in bytes."""
        if self.database_path.exists():
            return self.database_path.stat().st_size
        return 0

    @property
    def size_mb(self) -> float:
        """Get database file size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"Database(path={self.database_path!r}, {status})"
