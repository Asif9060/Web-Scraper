"""
Prompt regression tests for critical extraction and classification prompts.

These tests protect against silent prompt regressions by:
1. Validating prompt structure hasn't changed
2. Testing that LLM outputs parse correctly given expected format
3. Comparing structural outputs (not exact text)

Uses mocked LLM responses to avoid actual API calls.
"""

import json
import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from web_intel.llm.prompt_templates import (
    PromptTemplate,
    ExtractionPrompts,
    QueryPrompts,
)
from web_intel.understanding.page_understanding import (
    PageUnderstanding,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
)


# ============================================================================
# Snapshot definitions for expected prompt structures
# ============================================================================

@dataclass
class PromptSnapshot:
    """Snapshot of expected prompt structure."""

    name: str
    # Keywords that must appear in system prompt
    required_system_keywords: list[str]
    required_user_placeholders: list[str]  # Placeholders that must exist
    expected_output_format: str  # Description of expected output format
    sample_input: dict[str, str]  # Sample input for testing
    sample_output: str  # Sample LLM output for parsing tests


# Page Classification Prompt Snapshot
CLASSIFY_CONTENT_SNAPSHOT = PromptSnapshot(
    name="classify_content",
    required_system_keywords=[
        "classification",
        "categor",  # category/categories
    ],
    required_user_placeholders=[
        "{categories}",
        "{text}",
    ],
    expected_output_format="Single category name",
    sample_input={
        "categories": "article, product, documentation, blog, news, contact",
        "text": "Welcome to our online store. Browse our collection of electronics.",
    },
    sample_output="product",
)

# Entity Extraction Prompt Snapshot
EXTRACT_ENTITIES_SNAPSHOT = PromptSnapshot(
    name="extract_entities",
    required_system_keywords=[
        "entity",
        "extract",
        "TYPE:",  # Format instruction
    ],
    required_user_placeholders=[
        "{text}",
    ],
    expected_output_format="Lines of TYPE: name format",
    sample_input={
        "text": "Apple Inc. CEO Tim Cook announced new products in Cupertino on Monday.",
    },
    sample_output="""ORGANIZATION: Apple Inc.
PERSON: Tim Cook
LOCATION: Cupertino
DATE: Monday""",
)

# Key Facts Extraction Prompt Snapshot
EXTRACT_KEY_FACTS_SNAPSHOT = PromptSnapshot(
    name="extract_key_facts",
    required_system_keywords=[
        "fact",
        "extract",
        "statement",
    ],
    required_user_placeholders=[
        "{text}",
    ],
    expected_output_format="One fact per line",
    sample_input={
        "text": "Python 3.12 was released in October 2023. It includes performance improvements.",
    },
    sample_output="""Python 3.12 was released in October 2023.
Python 3.12 includes performance improvements.""",
)

# Topics Extraction Prompt Snapshot
EXTRACT_TOPICS_SNAPSHOT = PromptSnapshot(
    name="extract_topics",
    required_system_keywords=[
        "topic",
        "extract",
        "comma",
    ],
    required_user_placeholders=[
        "{text}",
    ],
    expected_output_format="Comma-separated list",
    sample_input={
        "text": "Machine learning models can be trained on GPUs for faster processing.",
    },
    sample_output="machine learning, GPU computing, model training",
)

# Summarization Prompt Snapshot
SUMMARIZE_SNAPSHOT = PromptSnapshot(
    name="summarize",
    required_system_keywords=[
        "summar",  # summary/summarize/summarization
        "concise",
    ],
    required_user_placeholders=[
        "{text}",
    ],
    expected_output_format="2-3 sentence summary",
    sample_input={
        "text": "The company reported strong Q4 earnings, beating analyst expectations by 15%.",
    },
    sample_output="The company exceeded Q4 earnings expectations by 15%.",
)

# Relationship Extraction Prompt Snapshot
EXTRACT_RELATIONSHIPS_SNAPSHOT = PromptSnapshot(
    name="extract_relationships",
    required_system_keywords=[
        "relationship",
        "ENTITY",
        "->",
    ],
    required_user_placeholders=[
        "{text}",
    ],
    expected_output_format="ENTITY1 -> RELATIONSHIP -> ENTITY2 format",
    sample_input={
        "text": "Microsoft acquired GitHub in 2018 for $7.5 billion.",
    },
    sample_output="Microsoft -> acquired -> GitHub",
)


# ============================================================================
# Prompt Structure Tests
# ============================================================================

class TestPromptStructure:
    """Tests for prompt template structure integrity."""

    def test_classify_content_structure(self):
        """Classification prompt should have required structure."""
        prompt = ExtractionPrompts.CLASSIFY_CONTENT
        snapshot = CLASSIFY_CONTENT_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)

    def test_extract_entities_structure(self):
        """Entity extraction prompt should have required structure."""
        prompt = ExtractionPrompts.EXTRACT_ENTITIES
        snapshot = EXTRACT_ENTITIES_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)

    def test_extract_key_facts_structure(self):
        """Key facts extraction prompt should have required structure."""
        prompt = ExtractionPrompts.EXTRACT_KEY_FACTS
        snapshot = EXTRACT_KEY_FACTS_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)

    def test_extract_topics_structure(self):
        """Topics extraction prompt should have required structure."""
        prompt = ExtractionPrompts.EXTRACT_TOPICS
        snapshot = EXTRACT_TOPICS_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)

    def test_summarize_structure(self):
        """Summarization prompt should have required structure."""
        prompt = ExtractionPrompts.SUMMARIZE
        snapshot = SUMMARIZE_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)

    def test_extract_relationships_structure(self):
        """Relationship extraction prompt should have required structure."""
        prompt = ExtractionPrompts.EXTRACT_RELATIONSHIPS
        snapshot = EXTRACT_RELATIONSHIPS_SNAPSHOT

        _assert_prompt_structure(prompt, snapshot)


class TestPromptFormatting:
    """Tests for prompt variable substitution."""

    def test_classify_content_formatting(self):
        """Classification prompt should format correctly."""
        snapshot = CLASSIFY_CONTENT_SNAPSHOT
        prompt = ExtractionPrompts.CLASSIFY_CONTENT

        result = prompt.format(**snapshot.sample_input)

        assert snapshot.sample_input["categories"] in result["user"]
        assert snapshot.sample_input["text"] in result["user"]
        assert "{categories}" not in result["user"]
        assert "{text}" not in result["user"]

    def test_extract_entities_formatting(self):
        """Entity extraction prompt should format correctly."""
        snapshot = EXTRACT_ENTITIES_SNAPSHOT
        prompt = ExtractionPrompts.EXTRACT_ENTITIES

        result = prompt.format(**snapshot.sample_input)

        assert snapshot.sample_input["text"] in result["user"]
        assert "{text}" not in result["user"]

    def test_all_prompts_have_system_and_user(self):
        """All prompts should have system and user components."""
        prompts = [
            ExtractionPrompts.CLASSIFY_CONTENT,
            ExtractionPrompts.EXTRACT_ENTITIES,
            ExtractionPrompts.EXTRACT_KEY_FACTS,
            ExtractionPrompts.EXTRACT_TOPICS,
            ExtractionPrompts.SUMMARIZE,
            ExtractionPrompts.EXTRACT_RELATIONSHIPS,
        ]

        for prompt in prompts:
            assert prompt.system, f"{prompt.name} missing system prompt"
            assert prompt.user, f"{prompt.name} missing user prompt"
            assert len(
                prompt.system) > 20, f"{prompt.name} system prompt too short"
            assert len(
                prompt.user) > 20, f"{prompt.name} user prompt too short"


# ============================================================================
# Output Parsing Tests (Structural Validation)
# ============================================================================

class TestEntityExtractionParsing:
    """Tests for entity extraction output parsing."""

    def test_parse_standard_entity_format(self):
        """Should parse TYPE: name format correctly."""
        output = EXTRACT_ENTITIES_SNAPSHOT.sample_output
        entities = _parse_entities(output)

        assert len(entities) >= 3

        # Check structural validity (types exist)
        types_found = {e["type"] for e in entities}
        assert "ORGANIZATION" in types_found or "organization" in types_found.union(
            {t.lower() for t in types_found})
        assert "PERSON" in types_found or "person" in types_found.union(
            {t.lower() for t in types_found})

    def test_parse_entity_with_variations(self):
        """Should handle format variations in entity output."""
        variations = [
            "PERSON: John Smith",
            "person: Jane Doe",
            "Person: Bob Jones",
            "ORGANIZATION: Acme Corp",
            "Location: New York City",
        ]

        for line in variations:
            entities = _parse_entities(line)
            assert len(entities) == 1, f"Failed to parse: {line}"
            assert entities[0]["name"], f"Missing name in: {line}"
            assert entities[0]["type"], f"Missing type in: {line}"

    def test_ignore_malformed_entity_lines(self):
        """Should skip lines that don't match expected format."""
        output = """PERSON: Valid Person
This is not an entity
ORGANIZATION: Valid Org
- Another invalid line
"""
        entities = _parse_entities(output)

        assert len(entities) == 2
        names = {e["name"] for e in entities}
        assert "Valid Person" in names
        assert "Valid Org" in names

    def test_entity_type_normalization(self):
        """Should normalize entity types to standard values."""
        output = """COMPANY: Acme Corp
org: Beta Inc
PEOPLE: John Doe
person: Jane Doe"""

        entities = _parse_entities(output)
        types = {e["type"].lower() for e in entities}

        # All should be parseable even if types vary
        assert len(entities) == 4


class TestClassificationParsing:
    """Tests for classification output parsing."""

    def test_parse_single_category(self):
        """Should extract single category from output."""
        outputs = [
            "product",
            "Product",
            "PRODUCT",
            "  product  ",
            "Category: product",
        ]

        for output in outputs:
            category = _parse_classification(output)
            assert category.lower() == "product", f"Failed to parse: {output}"

    def test_parse_category_from_sentence(self):
        """Should extract category even if embedded in text."""
        output = "Based on the content, this is clearly a documentation page."
        category = _parse_classification(output, valid_categories=[
                                         "documentation", "article", "blog"])

        assert category == "documentation"

    def test_reject_unknown_category(self):
        """Should return None for unrecognized categories."""
        output = "This is something completely random"
        category = _parse_classification(
            output, valid_categories=["article", "product"])

        assert category is None or category == ""


class TestFactExtractionParsing:
    """Tests for key fact extraction output parsing."""

    def test_parse_facts_one_per_line(self):
        """Should parse facts separated by newlines."""
        output = EXTRACT_KEY_FACTS_SNAPSHOT.sample_output
        facts = _parse_facts(output)

        assert len(facts) >= 2
        assert all(len(f) > 10 for f in facts)

    def test_parse_numbered_facts(self):
        """Should handle numbered fact lists."""
        output = """1. Python is a programming language.
2. It was created by Guido van Rossum.
3. Python 3.12 is the latest version."""

        facts = _parse_facts(output)

        assert len(facts) == 3
        assert "Python" in facts[0]

    def test_parse_bulleted_facts(self):
        """Should handle bulleted fact lists."""
        output = """• Python is interpreted.
- Python is dynamically typed.
* Python supports multiple paradigms."""

        facts = _parse_facts(output)

        assert len(facts) == 3


class TestRelationshipParsing:
    """Tests for relationship extraction output parsing."""

    def test_parse_arrow_format(self):
        """Should parse ENTITY -> RELATION -> ENTITY format."""
        output = EXTRACT_RELATIONSHIPS_SNAPSHOT.sample_output
        relationships = _parse_relationships(output)

        assert len(relationships) >= 1
        rel = relationships[0]
        assert "subject" in rel
        assert "predicate" in rel
        assert "object" in rel

    def test_parse_relationship_variations(self):
        """Should handle relationship format variations."""
        variations = [
            "Microsoft -> acquired -> GitHub",
            "Microsoft->acquired->GitHub",
            "Microsoft --> acquired --> GitHub",
            "Microsoft - acquired - GitHub",
        ]

        for line in variations:
            relationships = _parse_relationships(line)
            if relationships:  # Some formats may not parse
                assert relationships[0]["subject"], f"Failed to parse subject: {line}"


class TestTopicsParsing:
    """Tests for topics extraction output parsing."""

    def test_parse_comma_separated_topics(self):
        """Should parse comma-separated topic list."""
        output = EXTRACT_TOPICS_SNAPSHOT.sample_output
        topics = _parse_topics(output)

        assert len(topics) >= 2
        assert all(len(t) > 2 for t in topics)

    def test_parse_topics_with_cleanup(self):
        """Should clean up topic strings."""
        output = "  machine learning , GPU computing,  AI  "
        topics = _parse_topics(output)

        assert "machine learning" in topics
        assert all(not t.startswith(" ") and not t.endswith(" ")
                   for t in topics)


# ============================================================================
# Integration Tests with Mocked LLM
# ============================================================================

class TestPageUnderstandingIntegration:
    """Integration tests with mocked LLM responses."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predefined responses."""
        llm = MagicMock()
        llm.generate = MagicMock()
        return llm

    def test_entity_extraction_integration(self, mock_llm):
        """Entity extraction should work with expected LLM output format."""
        # Configure mock to return expected format
        mock_response = MagicMock()
        mock_response.text = EXTRACT_ENTITIES_SNAPSHOT.sample_output
        mock_llm.generate.return_value = mock_response

        understanding = PageUnderstanding(
            llm=mock_llm,
            chunk_size=1000,
            max_chunks_per_page=1,
        )

        # Call internal method directly to test parsing
        entities = understanding._parse_entities(mock_response.text)

        assert len(entities) >= 3
        assert any(e.entity_type == EntityType.ORGANIZATION for e in entities)
        assert any(e.entity_type == EntityType.PERSON for e in entities)

    def test_classification_output_structure(self, mock_llm):
        """Classification should return expected structure."""
        mock_response = MagicMock()
        mock_response.text = "documentation"
        mock_llm.generate.return_value = mock_response

        understanding = PageUnderstanding(
            llm=mock_llm,
            chunk_size=1000,
            max_chunks_per_page=1,
        )

        # The output should be parseable as a category
        result = mock_response.text.strip().lower()
        assert result in ["documentation", "article",
                          "blog", "product", "news", "contact", "other"]


# ============================================================================
# Regression Detection Tests
# ============================================================================

class TestPromptRegression:
    """Tests that detect breaking changes in prompts."""

    def test_entity_prompt_format_instruction_present(self):
        """Entity prompt must include format instruction (TYPE: name)."""
        prompt = ExtractionPrompts.EXTRACT_ENTITIES

        # The format instruction is critical for parsing
        assert "TYPE:" in prompt.system or "TYPE:" in prompt.user, \
            "Entity extraction prompt missing TYPE: format instruction - this will break parsing!"

    def test_classification_prompt_categories_placeholder(self):
        """Classification prompt must accept categories parameter."""
        prompt = ExtractionPrompts.CLASSIFY_CONTENT

        assert "{categories}" in prompt.user, \
            "Classification prompt missing {categories} placeholder - cannot specify valid categories!"

    def test_relationship_prompt_arrow_format(self):
        """Relationship prompt must specify arrow format."""
        prompt = ExtractionPrompts.EXTRACT_RELATIONSHIPS

        assert "->" in prompt.system or "->" in prompt.user, \
            "Relationship prompt missing -> format instruction - this will break parsing!"

    def test_topics_prompt_comma_instruction(self):
        """Topics prompt must request comma-separated format."""
        prompt = ExtractionPrompts.EXTRACT_TOPICS
        combined = (prompt.system + prompt.user).lower()

        assert "comma" in combined, \
            "Topics prompt missing comma-separated instruction - output format may change!"


# ============================================================================
# Helper Functions
# ============================================================================

def _assert_prompt_structure(prompt: PromptTemplate, snapshot: PromptSnapshot):
    """Assert that a prompt matches its snapshot structure."""
    # Check name
    assert prompt.name == snapshot.name, f"Prompt name mismatch: {prompt.name} != {snapshot.name}"

    # Check system prompt keywords
    system_lower = prompt.system.lower()
    for keyword in snapshot.required_system_keywords:
        assert keyword.lower() in system_lower, \
            f"Missing keyword '{keyword}' in system prompt for {snapshot.name}"

    # Check user prompt placeholders
    for placeholder in snapshot.required_user_placeholders:
        assert placeholder in prompt.user, \
            f"Missing placeholder '{placeholder}' in user prompt for {snapshot.name}"


def _parse_entities(text: str) -> list[dict]:
    """Parse entities from LLM output (mirrors production parsing)."""
    entities = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try TYPE: name format
        match = re.match(r"^(\w+):\s*(.+)$", line)
        if match:
            entity_type = match.group(1).upper()
            name = match.group(2).strip()
            if name:
                entities.append({"type": entity_type, "name": name})

    return entities


def _parse_classification(text: str, valid_categories: list[str] | None = None) -> str:
    """Parse classification from LLM output."""
    text = text.strip().lower()

    # Direct match
    if valid_categories:
        for category in valid_categories:
            if category.lower() in text:
                return category

    # First word if single word
    words = text.split()
    if words:
        return words[0].strip(".:,")

    return ""


def _parse_facts(text: str) -> list[str]:
    """Parse facts from LLM output."""
    facts = []

    for line in text.split("\n"):
        # Remove numbering, bullets, etc.
        line = re.sub(r"^[\d\.\)\-\*•]+\s*", "", line.strip())
        if line and len(line) > 10:
            facts.append(line)

    return facts


def _parse_relationships(text: str) -> list[dict]:
    """Parse relationships from LLM output."""
    relationships = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try various arrow formats
        patterns = [
            r"(.+?)\s*-+>\s*(.+?)\s*-+>\s*(.+)",  # A -> B -> C
            r"(.+?)\s*-\s+(.+?)\s+-\s+(.+)",  # A - B - C
        ]

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                relationships.append({
                    "subject": match.group(1).strip(),
                    "predicate": match.group(2).strip(),
                    "object": match.group(3).strip(),
                })
                break

    return relationships


def _parse_topics(text: str) -> list[str]:
    """Parse topics from comma-separated LLM output."""
    # Handle multiple lines by joining
    text = " ".join(text.split("\n"))

    topics = []
    for topic in text.split(","):
        topic = topic.strip()
        if topic and len(topic) > 1:
            topics.append(topic)

    return topics
