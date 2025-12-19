"""
Conversation memory storage.

Manages conversation history with persistence and
efficient retrieval for context-aware QA.
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence
from uuid import uuid4

from web_intel.config import Settings
from web_intel.storage import Database
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory entries."""

    USER_QUERY = "user_query"
    ASSISTANT_RESPONSE = "assistant_response"
    SYSTEM_MESSAGE = "system"
    RETRIEVED_CONTEXT = "context"
    SUMMARY = "summary"


@dataclass
class MemoryEntry:
    """
    A single entry in conversation memory.

    Represents one turn in a conversation with metadata.
    """

    id: str = ""
    session_id: str = ""
    memory_type: MemoryType = MemoryType.USER_QUERY
    content: str = ""
    role: str = "user"  # user, assistant, system
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid4())

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }

    @classmethod
    def from_row(cls, row: dict) -> "MemoryEntry":
        """Create from database row."""
        timestamp = datetime.now(timezone.utc)
        if row.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
            except (ValueError, TypeError):
                pass

        metadata = {}
        if row.get("metadata"):
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return cls(
            id=row.get("id", ""),
            session_id=row.get("session_id", ""),
            memory_type=MemoryType(row.get("memory_type", "user_query")),
            content=row.get("content", ""),
            role=row.get("role", "user"),
            timestamp=timestamp,
            token_count=row.get("token_count", 0),
            metadata=metadata,
        )

    def to_message_dict(self) -> dict:
        """Convert to LLM message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationMemory:
    """
    Memory for a single conversation session.

    Contains all entries and provides access methods.
    """

    session_id: str
    entries: list[MemoryEntry] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    total_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add an entry to the conversation."""
        entry.session_id = self.session_id
        self.entries.append(entry)
        self.total_tokens += entry.token_count

    def get_messages(
        self,
        include_context: bool = False,
        include_summaries: bool = True,
    ) -> list[dict]:
        """
        Get conversation as message list for LLM.

        Args:
            include_context: Include retrieved context entries
            include_summaries: Include summary entries

        Returns:
            List of message dictionaries
        """
        messages = []

        for entry in self.entries:
            if entry.memory_type == MemoryType.RETRIEVED_CONTEXT and not include_context:
                continue
            if entry.memory_type == MemoryType.SUMMARY and not include_summaries:
                continue

            messages.append(entry.to_message_dict())

        return messages

    def get_last_n_turns(self, n: int) -> list[MemoryEntry]:
        """Get the last n conversation turns (user + assistant pairs)."""
        # Filter to user queries and assistant responses
        turns = [
            e for e in self.entries
            if e.memory_type in (MemoryType.USER_QUERY, MemoryType.ASSISTANT_RESPONSE)
        ]
        return turns[-(n * 2):]

    def get_user_queries(self) -> list[str]:
        """Get all user queries in the conversation."""
        return [
            e.content for e in self.entries
            if e.memory_type == MemoryType.USER_QUERY
        ]

    @property
    def turn_count(self) -> int:
        """Count of user-assistant turn pairs."""
        user_count = sum(
            1 for e in self.entries
            if e.memory_type == MemoryType.USER_QUERY
        )
        return user_count

    @property
    def is_empty(self) -> bool:
        """Check if memory has no entries."""
        return len(self.entries) == 0


class MemoryStore:
    """
    Persistent storage for conversation memories.

    Stores and retrieves conversation history with support
    for multiple sessions and efficient querying.

    Example:
        >>> store = MemoryStore.from_settings(settings)
        >>>
        >>> # Create new session
        >>> session_id = store.create_session()
        >>>
        >>> # Add entries
        >>> store.add_entry(session_id, MemoryEntry(
        ...     memory_type=MemoryType.USER_QUERY,
        ...     content="What is this website about?",
        ...     role="user"
        ... ))
        >>>
        >>> # Get conversation
        >>> memory = store.get_conversation(session_id)
        >>> print(memory.turn_count)
    """

    def __init__(self, database: Database) -> None:
        """
        Initialize memory store.

        Args:
            database: Database instance for persistence
        """
        self.db = database
        self._lock = threading.Lock()
        self._ensure_tables()

        # In-memory cache for active sessions
        self._cache: dict[str, ConversationMemory] = {}
        self._cache_max_size = 10

        logger.info("MemoryStore initialized")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Database | None = None,
    ) -> "MemoryStore":
        """
        Create MemoryStore from settings.

        Args:
            settings: Application settings
            database: Optional pre-configured database

        Returns:
            Configured MemoryStore instance
        """
        if database is None:
            database = Database.from_settings(settings)

        return cls(database=database)

    def _ensure_tables(self) -> None:
        """Create memory tables if they don't exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                summary TEXT,
                total_tokens INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)"
        )

        self.db._get_connection().commit()

    def create_session(self, metadata: dict | None = None) -> str:
        """
        Create a new conversation session.

        Args:
            metadata: Optional session metadata

        Returns:
            Session ID
        """
        session_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db.execute(
            """
            INSERT INTO conversation_sessions (id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, now, now, json.dumps(metadata) if metadata else None),
        )
        self.db._get_connection().commit()

        # Initialize cache
        with self._lock:
            self._cache[session_id] = ConversationMemory(session_id=session_id)
            self._evict_cache_if_needed()

        logger.debug(f"Created conversation session: {session_id}")
        return session_id

    def add_entry(
        self,
        session_id: str,
        entry: MemoryEntry,
    ) -> str:
        """
        Add an entry to a conversation.

        Args:
            session_id: Session to add to
            entry: Memory entry to add

        Returns:
            Entry ID
        """
        entry.session_id = session_id
        if not entry.id:
            entry.id = str(uuid4())

        # Insert into database
        self.db.execute(
            """
            INSERT INTO memory_entries 
            (id, session_id, memory_type, content, role, timestamp, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                session_id,
                entry.memory_type.value,
                entry.content,
                entry.role,
                entry.timestamp.isoformat(),
                entry.token_count,
                json.dumps(entry.metadata) if entry.metadata else None,
            ),
        )

        # Update session
        self.db.execute(
            """
            UPDATE conversation_sessions 
            SET updated_at = ?, total_tokens = total_tokens + ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), entry.token_count, session_id),
        )
        self.db._get_connection().commit()

        # Update cache
        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].add_entry(entry)

        return entry.id

    def add_user_query(
        self,
        session_id: str,
        query: str,
        token_count: int = 0,
    ) -> str:
        """Add a user query to the conversation."""
        entry = MemoryEntry(
            memory_type=MemoryType.USER_QUERY,
            content=query,
            role="user",
            token_count=token_count,
        )
        return self.add_entry(session_id, entry)

    def add_assistant_response(
        self,
        session_id: str,
        response: str,
        token_count: int = 0,
        metadata: dict | None = None,
    ) -> str:
        """Add an assistant response to the conversation."""
        entry = MemoryEntry(
            memory_type=MemoryType.ASSISTANT_RESPONSE,
            content=response,
            role="assistant",
            token_count=token_count,
            metadata=metadata or {},
        )
        return self.add_entry(session_id, entry)

    def add_context(
        self,
        session_id: str,
        context: str,
        source: str = "",
    ) -> str:
        """Add retrieved context to the conversation."""
        entry = MemoryEntry(
            memory_type=MemoryType.RETRIEVED_CONTEXT,
            content=context,
            role="system",
            metadata={"source": source} if source else {},
        )
        return self.add_entry(session_id, entry)

    def get_conversation(self, session_id: str) -> ConversationMemory | None:
        """
        Get full conversation memory.

        Args:
            session_id: Session to retrieve

        Returns:
            ConversationMemory or None if not found
        """
        # Check cache first
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]

        # Load from database
        session_row = self.db.fetch_one(
            "SELECT * FROM conversation_sessions WHERE id = ?",
            (session_id,),
        )

        if not session_row:
            return None

        entries_rows = self.db.fetch_all(
            "SELECT * FROM memory_entries WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )

        # Parse metadata
        metadata = {}
        if session_row.get("metadata"):
            try:
                metadata = json.loads(session_row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Build memory object
        memory = ConversationMemory(
            session_id=session_id,
            entries=[MemoryEntry.from_row(r) for r in entries_rows],
            summary=session_row.get("summary", ""),
            total_tokens=session_row.get("total_tokens", 0),
            metadata=metadata,
        )

        # Update cache
        with self._lock:
            self._cache[session_id] = memory
            self._evict_cache_if_needed()

        return memory

    def get_recent_entries(
        self,
        session_id: str,
        limit: int = 10,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryEntry]:
        """
        Get recent entries from a conversation.

        Args:
            session_id: Session to query
            limit: Maximum entries to return
            memory_types: Filter by memory types

        Returns:
            List of recent entries
        """
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            type_values = [t.value for t in memory_types]
            rows = self.db.fetch_all(
                f"""
                SELECT * FROM memory_entries 
                WHERE session_id = ? AND memory_type IN ({placeholders})
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, *type_values, limit),
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT * FROM memory_entries 
                WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (session_id, limit),
            )

        # Return in chronological order
        return [MemoryEntry.from_row(r) for r in reversed(rows)]

    def set_summary(self, session_id: str, summary: str) -> None:
        """
        Set conversation summary.

        Args:
            session_id: Session to update
            summary: Summary text
        """
        self.db.execute(
            "UPDATE conversation_sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )
        self.db._get_connection().commit()

        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].summary = summary

    def get_all_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Get all conversation sessions."""
        rows = self.db.fetch_all(
            """
            SELECT id, created_at, updated_at, summary, total_tokens
            FROM conversation_sessions
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session and all its entries.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
        """
        # Entries are deleted by CASCADE
        affected = self.db.delete(
            "conversation_sessions",
            "id = ?",
            (session_id,),
        )

        with self._lock:
            self._cache.pop(session_id, None)

        return affected > 0

    def clear_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of sessions deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=days)).isoformat()

        # Get sessions to delete
        rows = self.db.fetch_all(
            "SELECT id FROM conversation_sessions WHERE updated_at < ?",
            (cutoff,),
        )

        count = 0
        for row in rows:
            if self.delete_session(row["id"]):
                count += 1

        logger.info(f"Cleared {count} old conversation sessions")
        return count

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cached conversations if over limit."""
        if len(self._cache) <= self._cache_max_size:
            return

        # Sort by last entry timestamp
        sorted_sessions = sorted(
            self._cache.items(),
            key=lambda x: x[1].entries[-1].timestamp if x[1].entries else x[1].created_at,
        )

        # Remove oldest
        while len(self._cache) > self._cache_max_size:
            session_id, _ = sorted_sessions.pop(0)
            del self._cache[session_id]

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        session_count = self.db.fetch_one(
            "SELECT COUNT(*) as count FROM conversation_sessions"
        )
        entry_count = self.db.fetch_one(
            "SELECT COUNT(*) as count FROM memory_entries"
        )
        total_tokens = self.db.fetch_one(
            "SELECT SUM(total_tokens) as total FROM conversation_sessions"
        )

        return {
            "session_count": session_count["count"] if session_count else 0,
            "entry_count": entry_count["count"] if entry_count else 0,
            "total_tokens": total_tokens["total"] if total_tokens and total_tokens["total"] else 0,
            "cached_sessions": len(self._cache),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"MemoryStore(sessions={stats['session_count']}, entries={stats['entry_count']})"
