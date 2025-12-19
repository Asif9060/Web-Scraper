"""
Embeddings module for Web Intelligence System.

Provides text embedding generation using sentence-transformers:
- Text-to-vector conversion
- Batch processing for efficiency
- Similarity computation
- Memory-efficient loading
"""

from web_intel.embeddings.embedder import (
    Embedder,
    EmbeddingResult,
    SimilarityResult,
)

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "SimilarityResult",
]
