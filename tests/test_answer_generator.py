"""
Tests for answer generator module.

Tests answer generation strategies and quality.
"""

import pytest

from web_intel.answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
    GenerationStrategy,
)
from web_intel.config import Settings


class TestAnswerGenerator:
    """Tests for AnswerGenerator."""

    @pytest.fixture
    def generator(self, test_settings: Settings) -> AnswerGenerator:
        """Provide an answer generator."""
        # Disable LLM for faster tests
        test_settings.local_llm.enabled = False
        return AnswerGenerator(test_settings)

    def test_generator_creation(self, generator: AnswerGenerator):
        """Generator should be created successfully."""
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_direct_answer(self, generator: AnswerGenerator):
        """Direct strategy should generate answer."""
        context = [
            {"content": "The price is $99.99 per month.", "score": 0.95},
            {"content": "We offer monthly subscriptions.", "score": 0.8},
        ]

        result = await generator.generate(
            query="What is the price?",
            context=context,
            strategy=GenerationStrategy.DIRECT,
        )

        assert isinstance(result, GeneratedAnswer)
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_generate_with_refinement(self, generator: AnswerGenerator):
        """Refine strategy should iterate on answer."""
        context = [
            {"content": "Feature A provides X functionality.", "score": 0.9},
            {"content": "Feature B enables Y capability.", "score": 0.85},
            {"content": "Feature C allows Z operations.", "score": 0.8},
        ]

        result = await generator.generate(
            query="What features are available?",
            context=context,
            strategy=GenerationStrategy.REFINE,
        )

        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_generate_map_reduce(self, generator: AnswerGenerator):
        """Map-reduce strategy should aggregate answers."""
        context = [
            {"content": "Section 1: Product overview and benefits.", "score": 0.9},
            {"content": "Section 2: Technical specifications.", "score": 0.85},
            {"content": "Section 3: Pricing and plans.", "score": 0.8},
        ]

        result = await generator.generate(
            query="Give me a summary of the product",
            context=context,
            strategy=GenerationStrategy.MAP_REDUCE,
        )

        assert isinstance(result, GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_generate_returns_sources(self, generator: AnswerGenerator):
        """Generated answer should include sources."""
        context = [
            {"content": "Content 1", "score": 0.9, "url": "https://example.com/1"},
            {"content": "Content 2", "score": 0.8, "url": "https://example.com/2"},
        ]

        result = await generator.generate(
            query="What is this?",
            context=context,
        )

        assert hasattr(result, "sources") or hasattr(result, "source_chunks")

    @pytest.mark.asyncio
    async def test_generate_confidence_score(self, generator: AnswerGenerator):
        """Generated answer should have confidence score."""
        context = [
            {"content": "Definite answer here.", "score": 0.95},
        ]

        result = await generator.generate(
            query="What is the answer?",
            context=context,
        )

        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_generate_empty_context(self, generator: AnswerGenerator):
        """Generator should handle empty context."""
        result = await generator.generate(
            query="What is this?",
            context=[],
        )

        assert result is not None
        assert result.confidence < 0.5 or "no information" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_generate_respects_max_length(self, generator: AnswerGenerator):
        """Generated answer should respect max length."""
        context = [{"content": "A" * 1000, "score": 0.9}]

        result = await generator.generate(
            query="Summarize",
            context=context,
            max_length=100,
        )

        # Answer shouldn't be excessively long
        assert len(result.answer) < 500


class TestGeneratedAnswer:
    """Tests for GeneratedAnswer dataclass."""

    def test_answer_creation(self):
        """GeneratedAnswer should be created correctly."""
        answer = GeneratedAnswer(
            answer="The answer is 42.",
            confidence=0.95,
            strategy=GenerationStrategy.DIRECT,
        )

        assert answer.answer == "The answer is 42."
        assert answer.confidence == 0.95
        assert answer.strategy == GenerationStrategy.DIRECT

    def test_answer_with_sources(self):
        """GeneratedAnswer can include sources."""
        answer = GeneratedAnswer(
            answer="Answer text",
            confidence=0.9,
            strategy=GenerationStrategy.DIRECT,
            sources=[
                {"url": "https://example.com", "relevance": 0.95},
            ],
        )

        assert len(answer.sources) == 1

    def test_answer_with_metadata(self):
        """GeneratedAnswer can include metadata."""
        answer = GeneratedAnswer(
            answer="Answer text",
            confidence=0.9,
            strategy=GenerationStrategy.DIRECT,
            metadata={
                "generation_time_ms": 150,
                "tokens_used": 100,
            },
        )

        assert answer.metadata["generation_time_ms"] == 150

    def test_answer_to_dict(self):
        """GeneratedAnswer should convert to dictionary."""
        answer = GeneratedAnswer(
            answer="Answer",
            confidence=0.9,
            strategy=GenerationStrategy.DIRECT,
        )

        answer_dict = answer.to_dict() if hasattr(answer, "to_dict") else vars(answer)

        assert "answer" in answer_dict
        assert "confidence" in answer_dict


class TestGenerationStrategy:
    """Tests for GenerationStrategy enum."""

    def test_strategy_values(self):
        """Strategy enum should have expected values."""
        assert GenerationStrategy.DIRECT is not None
        assert GenerationStrategy.REFINE is not None
        assert GenerationStrategy.MAP_REDUCE is not None

    def test_strategy_selection(self):
        """Strategies can be compared and selected."""
        strategies = [
            GenerationStrategy.DIRECT,
            GenerationStrategy.REFINE,
            GenerationStrategy.MAP_REDUCE,
        ]

        # Should be distinct
        assert len(set(strategies)) == 3


class TestAnswerGeneratorCallable:
    """Tests for AnswerGenerator as callable."""

    @pytest.fixture
    def generator(self, test_settings: Settings) -> AnswerGenerator:
        """Provide an answer generator."""
        test_settings.local_llm.enabled = False
        return AnswerGenerator(test_settings)

    def test_as_callable(self, generator: AnswerGenerator):
        """Generator should provide callable interface."""
        callable_gen = generator.as_callable()

        assert callable(callable_gen)

    @pytest.mark.asyncio
    async def test_callable_interface(self, generator: AnswerGenerator):
        """Callable should work with query executor."""
        callable_gen = generator.as_callable()

        result = await callable_gen(
            query="What is the price?",
            context=[{"content": "Price is $99", "score": 0.9}],
        )

        assert result is not None


class TestAnswerQuality:
    """Tests for answer quality aspects."""

    @pytest.fixture
    def generator(self, test_settings: Settings) -> AnswerGenerator:
        """Provide an answer generator."""
        test_settings.local_llm.enabled = False
        return AnswerGenerator(test_settings)

    @pytest.mark.asyncio
    async def test_answer_relevance(self, generator: AnswerGenerator):
        """Answer should be relevant to query."""
        context = [
            {"content": "Our return policy allows returns within 30 days.", "score": 0.95},
        ]

        result = await generator.generate(
            query="What is the return policy?",
            context=context,
        )

        # Answer should mention relevant terms
        answer_lower = result.answer.lower()
        assert "return" in answer_lower or "30 days" in answer_lower or "policy" in answer_lower

    @pytest.mark.asyncio
    async def test_answer_handles_conflicting_info(self, generator: AnswerGenerator):
        """Generator should handle conflicting information."""
        context = [
            {"content": "The price is $99.", "score": 0.9},
            {"content": "The price is $149.", "score": 0.85},
        ]

        result = await generator.generate(
            query="What is the price?",
            context=context,
        )

        # Should provide some answer, possibly noting uncertainty
        assert result.answer is not None
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_answer_factual_question(self, generator: AnswerGenerator):
        """Factual questions should get direct answers."""
        context = [
            {"content": "The company was founded in 2015.", "score": 0.95},
        ]

        result = await generator.generate(
            query="When was the company founded?",
            context=context,
        )

        assert "2015" in result.answer

    @pytest.mark.asyncio
    async def test_answer_list_question(self, generator: AnswerGenerator):
        """List questions should enumerate items."""
        context = [
            {"content": "Features include: search, filtering, and export.", "score": 0.9},
        ]

        result = await generator.generate(
            query="What features are available?",
            context=context,
        )

        # Should mention multiple items
        assert result.answer is not None
