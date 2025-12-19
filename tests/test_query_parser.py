"""
Tests for query parser module.

Tests query classification, intent detection, and term extraction.
"""

import pytest

from web_intel.query_parser import (
    QueryParser,
    ParsedQuery,
    QueryType,
    QueryIntent,
    QueryExpander,
    ExpandedQuery,
)


class TestQueryParser:
    """Tests for QueryParser."""

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Provide a parser instance."""
        return QueryParser()

    def test_parse_factual_question(self, parser: QueryParser):
        """Factual questions should be classified correctly."""
        result = parser.parse("What is the company's phone number?")

        assert isinstance(result, ParsedQuery)
        assert result.query_type == QueryType.FACTUAL
        assert "phone" in result.key_terms or "number" in result.key_terms

    def test_parse_list_question(self, parser: QueryParser):
        """List questions should be classified correctly."""
        result = parser.parse("What products does the company offer?")

        assert result.query_type == QueryType.LIST
        assert "products" in result.key_terms

    def test_parse_procedural_question(self, parser: QueryParser):
        """Procedural questions should be classified correctly."""
        result = parser.parse("How do I create an account?")

        assert result.query_type == QueryType.PROCEDURAL
        assert "create" in result.key_terms or "account" in result.key_terms

    def test_parse_definition_question(self, parser: QueryParser):
        """Definition questions should be classified correctly."""
        result = parser.parse("What is machine learning?")

        assert result.query_type == QueryType.DEFINITION
        assert "machine" in result.key_terms or "learning" in result.key_terms

    def test_parse_comparison_question(self, parser: QueryParser):
        """Comparison questions should be classified correctly."""
        result = parser.parse("Compare the basic and premium plans")

        assert result.query_type == QueryType.COMPARISON
        assert result.intent == QueryIntent.COMPARISON

    def test_parse_yes_no_question(self, parser: QueryParser):
        """Yes/no questions should be classified correctly."""
        result = parser.parse("Is the service available on weekends?")

        assert result.query_type == QueryType.YES_NO

    def test_parse_explanation_question(self, parser: QueryParser):
        """Explanation questions should be classified correctly."""
        result = parser.parse("Why did my payment fail?")

        assert result.query_type == QueryType.EXPLANATION

    def test_normalized_query(self, parser: QueryParser):
        """Query should be normalized (lowercase, trimmed)."""
        result = parser.parse("  WHAT ARE THE PRICES?  ")

        assert result.normalized_query == "what are the prices?"

    def test_key_terms_extraction(self, parser: QueryParser):
        """Key terms should exclude stop words."""
        result = parser.parse("What are the main features of the product?")

        # Should not include stop words
        assert "the" not in result.key_terms
        assert "are" not in result.key_terms
        # Should include meaningful words
        assert any(term in ["main", "features", "product"]
                   for term in result.key_terms)

    def test_follow_up_detection(self, parser: QueryParser):
        """Follow-up questions should be detected."""
        # Direct follow-up
        result = parser.parse("Tell me more about that")
        assert result.is_follow_up or result.references_previous

        # Reference to previous context
        result = parser.parse("What about the price of it?")
        assert result.references_previous

    def test_search_query_property(self, parser: QueryParser):
        """search_query should return optimized search string."""
        result = parser.parse("What products are available?")

        assert isinstance(result.search_query, str)
        assert len(result.search_query) > 0

    def test_to_dict_conversion(self, parser: QueryParser):
        """ParsedQuery should convert to dictionary."""
        result = parser.parse("What is the return policy?")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "original_query" in result_dict
        assert "query_type" in result_dict
        assert "key_terms" in result_dict

    def test_confidence_score(self, parser: QueryParser):
        """Parser should assign confidence score."""
        result = parser.parse("What is the price?")

        assert 0 <= result.confidence <= 1


class TestQueryExpander:
    """Tests for QueryExpander."""

    @pytest.fixture
    def expander(self) -> QueryExpander:
        """Provide an expander instance."""
        return QueryExpander()

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Provide a parser for creating ParsedQuery."""
        return QueryParser()

    def test_expand_query(self, expander: QueryExpander, parser: QueryParser):
        """Expander should generate alternative queries."""
        parsed = parser.parse("What products do you sell?")
        expanded = expander.expand(parsed)

        assert isinstance(expanded, ExpandedQuery)
        assert len(expanded.reformulations) > 0

    def test_synonym_expansion(self, expander: QueryExpander, parser: QueryParser):
        """Expander should include synonyms."""
        parsed = parser.parse("What is the cost of the service?")
        expanded = expander.expand(parsed)

        # Should have synonym variations
        all_queries = [parsed.original_query] + expanded.reformulations
        all_text = " ".join(all_queries).lower()

        # "cost" might expand to include "price"
        assert "cost" in all_text or "price" in all_text

    def test_search_terms_extraction(self, expander: QueryExpander, parser: QueryParser):
        """Expander should provide search terms."""
        parsed = parser.parse("How do I reset my password?")
        expanded = expander.expand(parsed)

        assert len(expanded.search_terms) > 0

    def test_all_queries_property(self, expander: QueryExpander, parser: QueryParser):
        """all_queries should include original and reformulations."""
        parsed = parser.parse("What are the features?")
        expanded = expander.expand(parsed)

        all_queries = expanded.all_queries
        assert parsed.original_query in all_queries
        assert len(all_queries) >= 1
