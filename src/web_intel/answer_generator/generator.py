"""
Answer generation using LLMs.

Generates answers from retrieved context using
local or API-based language models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from web_intel.config import Settings
from web_intel.llm import LocalLLM, GenerationConfig, ConversationMessage
from web_intel.llm.local_llm import Role
from web_intel.llm.prompt_templates import QueryPrompts
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationStrategy(str, Enum):
    """Strategy for generating answers."""

    DIRECT = "direct"  # Single-shot generation
    REFINE = "refine"  # Iterative refinement
    MAP_REDUCE = "map_reduce"  # Process chunks then combine


@dataclass
class AnswerConfig:
    """Configuration for answer generation."""

    max_tokens: int = 512
    temperature: float = 0.1
    include_sources: bool = True
    strategy: GenerationStrategy = GenerationStrategy.DIRECT
    max_context_chunks: int = 5
    confidence_threshold: float = 0.3


@dataclass
class GeneratedAnswer:
    """
    Result of answer generation.

    Contains the answer text and generation metadata.
    """

    answer: str
    strategy_used: GenerationStrategy
    tokens_used: int = 0
    chunks_used: int = 0
    refinement_steps: int = 0
    confidence: float = 1.0
    sources_cited: list[str] = field(default_factory=list)

    @property
    def is_uncertain(self) -> bool:
        """Check if the answer indicates uncertainty."""
        uncertain_phrases = [
            "i don't have enough information",
            "the context doesn't",
            "cannot find",
            "not mentioned",
            "no information",
            "unclear from",
            "i'm not sure",
        ]
        answer_lower = self.answer.lower()
        return any(phrase in answer_lower for phrase in uncertain_phrases)


class LLMInterface(Protocol):
    """Protocol for LLM implementations."""

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> "GenerationResult": ...

    def chat(
        self,
        messages: list[ConversationMessage],
        config: GenerationConfig | None = None,
    ) -> "GenerationResult": ...


class AnswerGenerator:
    """
    Generates answers using LLMs with retrieved context.

    Supports multiple generation strategies and can use
    either local or API-based LLMs.

    Example:
        >>> generator = AnswerGenerator.from_settings(settings)
        >>>
        >>> # Simple generation
        >>> answer = generator.generate(
        ...     question="What is the return policy?",
        ...     context="Our return policy allows returns within 30 days..."
        ... )
        >>> print(answer.answer)
        >>>
        >>> # With sources
        >>> answer = generator.generate_with_sources(
        ...     question="What products do you offer?",
        ...     sources=[
        ...         {"title": "Products Page", "content": "We offer..."},
        ...         {"title": "Catalog", "content": "Our catalog includes..."}
        ...     ]
        ... )
    """

    def __init__(
        self,
        llm: LocalLLM | None = None,
        api_llm: Callable[[str, str | None], str] | None = None,
        config: AnswerConfig | None = None,
        use_local: bool = True,
    ) -> None:
        """
        Initialize answer generator.

        Args:
            llm: Local LLM instance
            api_llm: Optional API LLM callable (prompt, system) -> response
            config: Generation configuration
            use_local: Whether to prefer local LLM
        """
        self.llm = llm
        self.api_llm = api_llm
        self.config = config or AnswerConfig()
        self.use_local = use_local

        self._gen_config = GenerationConfig(
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        logger.info(
            f"AnswerGenerator initialized (strategy={self.config.strategy.value}, "
            f"use_local={use_local})"
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        llm: LocalLLM | None = None,
        api_llm: Callable[[str, str | None], str] | None = None,
    ) -> "AnswerGenerator":
        """
        Create AnswerGenerator from settings.

        Args:
            settings: Application settings
            llm: Optional pre-configured local LLM
            api_llm: Optional API LLM callable

        Returns:
            Configured AnswerGenerator instance
        """
        if llm is None and settings.local_llm.enabled:
            llm = LocalLLM.from_settings(settings)

        config = AnswerConfig(
            max_tokens=settings.local_llm.max_new_tokens,
            temperature=settings.local_llm.temperature,
        )

        return cls(
            llm=llm,
            api_llm=api_llm,
            config=config,
            use_local=settings.local_llm.enabled,
        )

    def generate(
        self,
        question: str,
        context: str,
        strategy: GenerationStrategy | None = None,
    ) -> GeneratedAnswer:
        """
        Generate an answer for a question using provided context.

        Args:
            question: User's question
            context: Retrieved context to answer from
            strategy: Override default strategy

        Returns:
            GeneratedAnswer with response
        """
        strategy = strategy or self.config.strategy

        if strategy == GenerationStrategy.DIRECT:
            return self._generate_direct(question, context)
        elif strategy == GenerationStrategy.REFINE:
            return self._generate_with_refinement(question, context)
        elif strategy == GenerationStrategy.MAP_REDUCE:
            return self._generate_map_reduce(question, context)
        else:
            return self._generate_direct(question, context)

    def _generate_direct(
        self,
        question: str,
        context: str,
    ) -> GeneratedAnswer:
        """Single-shot answer generation."""
        prompt = QueryPrompts.ANSWER_QUESTION.format_user(
            question=question,
            context=context,
        )
        system = QueryPrompts.ANSWER_QUESTION.system

        response = self._call_llm(prompt, system)

        return GeneratedAnswer(
            answer=response,
            strategy_used=GenerationStrategy.DIRECT,
            chunks_used=1,
        )

    def _generate_with_refinement(
        self,
        question: str,
        context: str,
        max_refinements: int = 2,
    ) -> GeneratedAnswer:
        """
        Generate with iterative refinement.

        First generates an initial answer, then refines it
        with additional context passes.
        """
        # Split context into chunks for refinement
        chunks = self._split_context(context)

        if len(chunks) <= 1:
            return self._generate_direct(question, context)

        # Initial answer from first chunk
        initial_context = chunks[0]
        prompt = QueryPrompts.ANSWER_QUESTION.format_user(
            question=question,
            context=initial_context,
        )
        system = QueryPrompts.ANSWER_QUESTION.system

        current_answer = self._call_llm(prompt, system)
        refinement_count = 0

        # Refine with remaining chunks
        for chunk in chunks[1:max_refinements + 1]:
            prompt = QueryPrompts.REFINE_ANSWER.format_user(
                question=question,
                previous_answer=current_answer,
                context=chunk,
            )
            system = QueryPrompts.REFINE_ANSWER.system

            current_answer = self._call_llm(prompt, system)
            refinement_count += 1

        return GeneratedAnswer(
            answer=current_answer,
            strategy_used=GenerationStrategy.REFINE,
            chunks_used=min(len(chunks), max_refinements + 1),
            refinement_steps=refinement_count,
        )

    def _generate_map_reduce(
        self,
        question: str,
        context: str,
    ) -> GeneratedAnswer:
        """
        Map-reduce style generation.

        Generates answers from each chunk independently,
        then synthesizes a final answer.
        """
        chunks = self._split_context(context)

        if len(chunks) <= 1:
            return self._generate_direct(question, context)

        # Map: generate answer for each chunk
        chunk_answers = []
        for chunk in chunks[:self.config.max_context_chunks]:
            prompt = QueryPrompts.ANSWER_QUESTION.format_user(
                question=question,
                context=chunk,
            )
            system = QueryPrompts.ANSWER_QUESTION.system

            answer = self._call_llm(prompt, system)
            if answer and not self._is_no_info_answer(answer):
                chunk_answers.append(answer)

        if not chunk_answers:
            return GeneratedAnswer(
                answer="I couldn't find relevant information to answer this question.",
                strategy_used=GenerationStrategy.MAP_REDUCE,
                chunks_used=len(chunks),
                confidence=0.2,
            )

        if len(chunk_answers) == 1:
            return GeneratedAnswer(
                answer=chunk_answers[0],
                strategy_used=GenerationStrategy.MAP_REDUCE,
                chunks_used=len(chunks),
            )

        # Reduce: synthesize answers
        combined_sources = "\n\n".join(
            f"Source {i+1}:\n{ans}" for i, ans in enumerate(chunk_answers)
        )

        prompt = QueryPrompts.SYNTHESIZE_INFO.format_user(
            topic=question,
            sources=combined_sources,
        )
        system = QueryPrompts.SYNTHESIZE_INFO.system

        final_answer = self._call_llm(prompt, system)

        return GeneratedAnswer(
            answer=final_answer,
            strategy_used=GenerationStrategy.MAP_REDUCE,
            chunks_used=len(chunks),
        )

    def generate_with_sources(
        self,
        question: str,
        sources: list[dict],
    ) -> GeneratedAnswer:
        """
        Generate answer with explicit source citations.

        Args:
            question: User's question
            sources: List of source dicts with 'title' and 'content'

        Returns:
            GeneratedAnswer with source citations
        """
        # Format sources
        formatted_sources = "\n\n".join(
            f"[{i+1}] {src.get('title', 'Source')}:\n{src.get('content', '')}"
            for i, src in enumerate(sources)
        )

        prompt = QueryPrompts.ANSWER_WITH_SOURCES.format_user(
            question=question,
            sources=formatted_sources,
        )
        system = QueryPrompts.ANSWER_WITH_SOURCES.system

        response = self._call_llm(prompt, system)

        # Extract cited sources
        cited = []
        for i, src in enumerate(sources, 1):
            if f"[{i}]" in response or src.get("title", "").lower() in response.lower():
                cited.append(src.get("title", f"Source {i}"))

        return GeneratedAnswer(
            answer=response,
            strategy_used=GenerationStrategy.DIRECT,
            chunks_used=len(sources),
            sources_cited=cited,
        )

    def generate_follow_up(
        self,
        question: str,
        context: str,
        conversation_history: str,
    ) -> GeneratedAnswer:
        """
        Generate answer for a follow-up question.

        Uses conversation history for context continuity.

        Args:
            question: Follow-up question
            context: New retrieved context
            conversation_history: Previous conversation

        Returns:
            GeneratedAnswer
        """
        prompt = QueryPrompts.CONVERSATIONAL_FOLLOW_UP.format_user(
            history=conversation_history,
            context=context,
            question=question,
        )
        system = QueryPrompts.CONVERSATIONAL_FOLLOW_UP.system

        response = self._call_llm(prompt, system)

        return GeneratedAnswer(
            answer=response,
            strategy_used=GenerationStrategy.DIRECT,
            chunks_used=1,
        )

    def generate_comparison(
        self,
        page1_title: str,
        page1_content: str,
        page2_title: str,
        page2_content: str,
    ) -> GeneratedAnswer:
        """
        Generate a comparison between two pages.

        Args:
            page1_title: First page title
            page1_content: First page content
            page2_title: Second page title
            page2_content: Second page content

        Returns:
            GeneratedAnswer with comparison
        """
        prompt = QueryPrompts.COMPARE_PAGES.format_user(
            page1_title=page1_title,
            page1_content=page1_content,
            page2_title=page2_title,
            page2_content=page2_content,
        )
        system = QueryPrompts.COMPARE_PAGES.system

        response = self._call_llm(prompt, system)

        return GeneratedAnswer(
            answer=response,
            strategy_used=GenerationStrategy.DIRECT,
            chunks_used=2,
            sources_cited=[page1_title, page2_title],
        )

    def _call_llm(self, prompt: str, system: str | None = None) -> str:
        """Call the configured LLM."""
        # Try local LLM first if configured
        if self.use_local and self.llm is not None:
            try:
                result = self.llm.generate(
                    prompt=prompt,
                    config=self._gen_config,
                    system_prompt=system,
                )
                return result.text.strip()
            except Exception as e:
                logger.warning(f"Local LLM generation failed: {e}")
                if self.api_llm is None:
                    raise

        # Fall back to API LLM
        if self.api_llm is not None:
            try:
                response = self.api_llm(prompt, system)
                return response.strip()
            except Exception as e:
                logger.error(f"API LLM generation failed: {e}")
                raise

        raise ValueError("No LLM configured for generation")

    def _split_context(self, context: str, max_chunk_size: int = 1500) -> list[str]:
        """Split context into chunks for processing."""
        # Try to split on double newlines (paragraph boundaries)
        paragraphs = context.split("\n\n")

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _is_no_info_answer(self, answer: str) -> bool:
        """Check if answer indicates no information found."""
        no_info_phrases = [
            "doesn't contain",
            "does not contain",
            "no information",
            "not mentioned",
            "cannot find",
            "unable to find",
            "not enough information",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in no_info_phrases)

    def as_callable(self) -> Callable[[str, str], str]:
        """
        Return a simple callable for use with QueryExecutor.

        Returns:
            Function(question, context) -> answer
        """
        def generate_fn(question: str, context: str) -> str:
            result = self.generate(question, context)
            return result.answer

        return generate_fn

    def __repr__(self) -> str:
        return (
            f"AnswerGenerator(strategy={self.config.strategy.value}, "
            f"use_local={self.use_local})"
        )
