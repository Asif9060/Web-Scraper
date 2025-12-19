"""
Extraction module for Web Intelligence System.

Provides content extraction and parsing including:
- HTML/DOM parsing
- Main content detection
- Metadata extraction
- Text cleaning and normalization
"""

from web_intel.extraction.page_parser import (
    PageParser,
    ParsedPage,
    ContentBlock,
    BlockType,
)
from web_intel.extraction.content_extractor import (
    ContentExtractor,
    ExtractedContent,
    MainContentDetector,
)
from web_intel.extraction.metadata_extractor import (
    MetadataExtractor,
    PageMetadata,
    StructuredData,
)

__all__ = [
    # Page parsing
    "PageParser",
    "ParsedPage",
    "ContentBlock",
    "BlockType",
    # Content extraction
    "ContentExtractor",
    "ExtractedContent",
    "MainContentDetector",
    # Metadata
    "MetadataExtractor",
    "PageMetadata",
    "StructuredData",
]
