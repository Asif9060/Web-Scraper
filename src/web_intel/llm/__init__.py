"""
LLM module for Web Intelligence System.

Provides local and API-based language model interfaces:
- Local LLM using Hugging Face Transformers (Qwen2.5-1.5B)
- API LLM using Anthropic Claude
"""

from web_intel.llm.local_llm import (
    LocalLLM,
    GenerationConfig,
    GenerationResult,
    ConversationMessage,
)
from web_intel.llm.prompt_templates import (
    PromptTemplate,
    ExtractionPrompts,
    QueryPrompts,
)

__all__ = [
    # Local LLM
    "LocalLLM",
    "GenerationConfig",
    "GenerationResult",
    "ConversationMessage",
    # Prompts
    "PromptTemplate",
    "ExtractionPrompts",
    "QueryPrompts",
]
