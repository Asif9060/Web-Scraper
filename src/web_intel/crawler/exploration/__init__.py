"""
Exploration agent module for Web Intelligence System.

Provides intelligent page discovery and navigation decisions including:
- Interactive element detection
- Click decision making
- Navigation pattern recognition
- Loop prevention
"""

from web_intel.crawler.exploration.agent import ExplorationAgent
from web_intel.crawler.exploration.element_analyzer import (
    ElementAnalyzer,
    InteractiveElement,
    ElementType,
)
from web_intel.crawler.exploration.navigation_patterns import (
    NavigationPatternDetector,
    NavigationPattern,
    PaginationInfo,
)
from web_intel.crawler.exploration.state_tracker import (
    ExplorationState,
    PageState,
    VisitRecord,
)

__all__ = [
    "ExplorationAgent",
    "ElementAnalyzer",
    "InteractiveElement",
    "ElementType",
    "NavigationPatternDetector",
    "NavigationPattern",
    "PaginationInfo",
    "ExplorationState",
    "PageState",
    "VisitRecord",
]
