"""
Navigation agent module for Web Intelligence System.

Provides LLM-powered intelligent navigation:
- Goal-directed browsing
- Link prioritization based on objectives
- Adaptive exploration strategies
- Multi-step navigation planning
"""

from web_intel.navigation_agent.agent import (
    NavigationAgent,
    NavigationGoal,
    NavigationPlan,
    NavigationStep,
    NavigationResult,
    StepOutcome,
)
from web_intel.navigation_agent.planner import (
    NavigationPlanner,
    LinkCandidate,
    PageAssessment,
)

__all__ = [
    "NavigationAgent",
    "NavigationGoal",
    "NavigationPlan",
    "NavigationStep",
    "NavigationResult",
    "StepOutcome",
    "NavigationPlanner",
    "LinkCandidate",
    "PageAssessment",
]
