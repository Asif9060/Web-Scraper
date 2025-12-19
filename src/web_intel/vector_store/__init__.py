"""
Vector store module for Web Intelligence System.

Provides efficient vector similarity search with:
- In-memory index for fast retrieval
- Persistent storage in SQLite
- Multiple similarity metrics
- Approximate nearest neighbor search
"""

from web_intel.vector_store.store import (
    VectorStore,
    VectorSearchResult,
    SearchFilter,
)
from web_intel.vector_store.index import (
    VectorIndex,
    IndexType,
)

__all__ = [
    "VectorStore",
    "VectorSearchResult",
    "SearchFilter",
    "VectorIndex",
    "IndexType",
]
