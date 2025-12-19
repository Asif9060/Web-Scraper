"""
Page understanding module for Web Intelligence System.

Provides LLM-powered content understanding including:
- Summarization
- Topic extraction
- Entity extraction
- Key fact extraction
- Content classification
- Relationship extraction
"""

from web_intel.understanding.page_understanding import (
    PageUnderstanding,
    UnderstandingResult,
    PageSummary,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    EntityType,
)
from web_intel.understanding.chunker import (
    TextChunker,
    Chunk,
    ChunkingStrategy,
)

__all__ = [
    # Page understanding
    "PageUnderstanding",
    "UnderstandingResult",
    "PageSummary",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "EntityType",
    # Chunking
    "TextChunker",
    "Chunk",
    "ChunkingStrategy",
]
