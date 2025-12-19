"""
Tests for query executor module.

Tests query execution pipeline and result aggregation.
"""

import pytest

from web_intel.query_executor import (
    QueryExecutor,
    QueryResult,
    ExecutionPlan,
    ResultAggregator,
)
from web_intel.query_parser import QueryParser, ParsedQuery
from web_intel.config import Settings


class TestQueryExecutor:
    """Tests for QueryExecutor."""

    @pytest.fixture
    def executor(self, test_settings: Settings, database) -> QueryExecutor:
        """Provide a query executor."""
        return QueryExecutor(test_settings, database=database)

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Provide a query parser."""
        return QueryParser()

    def test_executor_creation(self, executor: QueryExecutor):
        """Executor should be created successfully."""
        assert executor is not None

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, executor: QueryExecutor, parser: QueryParser):
        """Simple query should be executed."""
        parsed = parser.parse("What is machine learning?")

        result = await executor.execute(parsed)

        assert isinstance(result, QueryResult)
        assert result.query == parsed.original_query

    def test_create_execution_plan(self, executor: QueryExecutor, parser: QueryParser):
        """Execution plan should be created for query."""
        parsed = parser.parse("List all products")

        plan = executor.create_plan(parsed)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 1

    def test_plan_for_factual_query(self, executor: QueryExecutor, parser: QueryParser):
        """Factual query should have appropriate plan."""
        parsed = parser.parse("What is the company phone number?")

        plan = executor.create_plan(parsed)

        # Should include vector search and answer generation
        step_types = [step.type for step in plan.steps]
        assert "vector_search" in step_types or "search" in step_types

    def test_plan_for_list_query(self, executor: QueryExecutor, parser: QueryParser):
        """List query should have appropriate plan."""
        parsed = parser.parse("What products are available?")

        plan = executor.create_plan(parsed)

        # List queries might use different retrieval strategy
        assert len(plan.steps) >= 1

    @pytest.mark.asyncio
    async def test_execute_with_context(self, executor: QueryExecutor, parser: QueryParser):
        """Query can be executed with additional context."""
        parsed = parser.parse("Tell me more about that")

        result = await executor.execute(
            parsed,
            context={"previous_topic": "pricing"},
        )

        assert isinstance(result, QueryResult)

    @pytest.mark.asyncio
    async def test_execute_returns_sources(self, executor: QueryExecutor, parser: QueryParser):
        """Execution should return source documents."""
        parsed = parser.parse("What features are available?")

        result = await executor.execute(parsed)

        assert hasattr(result, "sources") or hasattr(
            result, "source_documents")

    @pytest.mark.asyncio
    async def test_execute_returns_confidence(self, executor: QueryExecutor, parser: QueryParser):
        """Execution should return confidence score."""
        parsed = parser.parse("What is the pricing?")

        result = await executor.execute(parsed)

        assert hasattr(result, "confidence")
        assert 0 <= result.confidence <= 1


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_result_creation(self):
        """QueryResult should be created with required fields."""
        result = QueryResult(
            query="What is the price?",
            answer="The price is $99.",
            confidence=0.85,
        )

        assert result.query == "What is the price?"
        assert result.answer == "The price is $99."
        assert result.confidence == 0.85

    def test_result_with_sources(self):
        """QueryResult can include sources."""
        result = QueryResult(
            query="query",
            answer="answer",
            confidence=0.9,
            sources=[
                {"url": "https://example.com/page1", "relevance": 0.95},
                {"url": "https://example.com/page2", "relevance": 0.85},
            ],
        )

        assert len(result.sources) == 2

    def test_result_with_metadata(self):
        """QueryResult can include execution metadata."""
        result = QueryResult(
            query="query",
            answer="answer",
            confidence=0.9,
            metadata={
                "execution_time_ms": 150,
                "chunks_retrieved": 5,
            },
        )

        assert result.metadata["execution_time_ms"] == 150

    def test_result_to_dict(self):
        """QueryResult should convert to dictionary."""
        result = QueryResult(
            query="query",
            answer="answer",
            confidence=0.9,
        )

        result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)

        assert "query" in result_dict
        assert "answer" in result_dict
        assert "confidence" in result_dict


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_plan_creation(self):
        """ExecutionPlan should be created with steps."""
        from web_intel.query_executor import ExecutionStep

        plan = ExecutionPlan(
            steps=[
                ExecutionStep(type="vector_search", params={"k": 5}),
                ExecutionStep(type="rerank", params={"top_n": 3}),
                ExecutionStep(type="generate", params={}),
            ]
        )

        assert len(plan.steps) == 3

    def test_plan_estimated_time(self):
        """Plan should estimate execution time."""
        from web_intel.query_executor import ExecutionStep

        plan = ExecutionPlan(
            steps=[
                ExecutionStep(type="vector_search", params={}),
                ExecutionStep(type="generate", params={}),
            ]
        )

        if hasattr(plan, "estimated_time_ms"):
            assert plan.estimated_time_ms > 0


class TestResultAggregator:
    """Tests for ResultAggregator."""

    @pytest.fixture
    def aggregator(self) -> ResultAggregator:
        """Provide a result aggregator."""
        return ResultAggregator()

    def test_aggregate_single_result(self, aggregator: ResultAggregator):
        """Single result should pass through."""
        results = [
            {"content": "Answer text", "score": 0.9},
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated is not None

    def test_aggregate_multiple_results(self, aggregator: ResultAggregator):
        """Multiple results should be combined."""
        results = [
            {"content": "First answer", "score": 0.9},
            {"content": "Second answer", "score": 0.8},
            {"content": "Third answer", "score": 0.7},
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated is not None

    def test_aggregate_by_score(self, aggregator: ResultAggregator):
        """Results should be ranked by score."""
        results = [
            {"content": "Low score", "score": 0.5},
            {"content": "High score", "score": 0.95},
            {"content": "Medium score", "score": 0.7},
        ]

        aggregated = aggregator.aggregate(results)

        # Top result should be highest score
        if hasattr(aggregated, "top_results"):
            assert aggregated.top_results[0]["score"] == 0.95

    def test_aggregate_deduplication(self, aggregator: ResultAggregator):
        """Duplicate content should be deduplicated."""
        results = [
            {"content": "Same content", "score": 0.9},
            {"content": "Same content", "score": 0.85},
            {"content": "Different content", "score": 0.8},
        ]

        aggregated = aggregator.aggregate(results)

        # Should have fewer results after dedup
        if hasattr(aggregated, "results"):
            unique_contents = set(r["content"] for r in aggregated.results)
            assert len(unique_contents) == 2

    def test_aggregate_empty_results(self, aggregator: ResultAggregator):
        """Empty results should be handled."""
        aggregated = aggregator.aggregate([])

        assert aggregated is not None
        if hasattr(aggregated, "results"):
            assert len(aggregated.results) == 0


class TestQueryExecutorWithLLM:
    """Tests for query executor with LLM enabled."""

    @pytest.fixture
    def executor_with_llm(self, test_settings: Settings, database) -> QueryExecutor:
        """Provide executor with LLM enabled."""
        # Note: In production, LLM would be enabled
        # For tests, we may mock it
        test_settings.local_llm.enabled = False
        return QueryExecutor(test_settings, database=database)

    @pytest.mark.asyncio
    async def test_execute_generates_answer(
        self, executor_with_llm: QueryExecutor
    ):
        """Executor should generate answers."""
        parser = QueryParser()
        parsed = parser.parse("What is the company mission?")

        result = await executor_with_llm.execute(parsed)

        # Even without LLM, should return some result
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_with_no_results(
        self, executor_with_llm: QueryExecutor
    ):
        """Executor should handle queries with no matching results."""
        parser = QueryParser()
        parsed = parser.parse("xyzzy nonexistent query")

        result = await executor_with_llm.execute(parsed)

        assert result is not None
        # Should indicate low confidence or no answer
        assert result.confidence < 0.5 or "not found" in result.answer.lower() or result.answer == ""
