"""
Answer generator module for Web Intelligence System.

Provides LLM-based answer generation:
- Context-aware answer generation
- Multi-step refinement
- Source citation
- Conversation continuity
"""

from web_intel.answer_generator.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    AnswerConfig,
    GenerationStrategy,
)

__all__ = [
    "AnswerGenerator",
    "GeneratedAnswer",
    "AnswerConfig",
    "GenerationStrategy",
]
