"""
Query parser module for Web Intelligence System.

Provides query understanding and processing:
- Intent classification
- Entity extraction from queries
- Query expansion and reformulation
- Search query generation
"""

from web_intel.query_parser.parser import (
    QueryParser,
    ParsedQuery,
    QueryIntent,
    QueryType,
)
from web_intel.query_parser.expander import (
    QueryExpander,
    ExpandedQuery,
)

__all__ = [
    "QueryParser",
    "ParsedQuery",
    "QueryIntent",
    "QueryType",
    "QueryExpander",
    "ExpandedQuery",
]
