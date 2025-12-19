"""
Query parsing and intent classification.

Analyzes user queries to extract intent, entities,
and structure for effective retrieval.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Type of query based on expected answer format."""

    FACTUAL = "factual"  # Single fact answer
    EXPLANATION = "explanation"  # How/why explanation
    LIST = "list"  # List of items
    COMPARISON = "comparison"  # Compare multiple things
    DEFINITION = "definition"  # What is X?
    PROCEDURAL = "procedural"  # How to do X?
    NAVIGATIONAL = "navigational"  # Where to find X?
    YES_NO = "yes_no"  # Boolean question
    OPEN_ENDED = "open_ended"  # General discussion


class QueryIntent(str, Enum):
    """User's underlying intent."""

    INFORMATION_SEEKING = "information_seeking"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    SUMMARIZATION = "summarization"


@dataclass
class ParsedQuery:
    """
    Result of query parsing.

    Contains structured information extracted from the query.
    """

    original_query: str
    normalized_query: str
    query_type: QueryType
    intent: QueryIntent
    key_terms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    is_follow_up: bool = False
    references_previous: bool = False
    confidence: float = 1.0

    @property
    def search_query(self) -> str:
        """Get optimized query for search."""
        if self.key_terms:
            return " ".join(self.key_terms)
        return self.normalized_query

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type.value,
            "intent": self.intent.value,
            "key_terms": self.key_terms,
            "entities": self.entities,
            "constraints": self.constraints,
            "is_follow_up": self.is_follow_up,
            "references_previous": self.references_previous,
            "confidence": self.confidence,
        }


class QueryParser:
    """
    Parses and analyzes user queries.

    Extracts structured information from natural language queries
    to improve retrieval and answer generation.

    Example:
        >>> parser = QueryParser()
        >>> parsed = parser.parse("What products does this company sell?")
        >>> print(parsed.query_type)  # QueryType.LIST
        >>> print(parsed.key_terms)   # ['products', 'company', 'sell']
    """

    # Question word patterns for type classification
    FACTUAL_PATTERNS = [
        r"^(what|which|who|when|where)\b.*\?*$",
        r"^(how much|how many|how long|how far)\b",
    ]

    EXPLANATION_PATTERNS = [
        r"^(why|how come)\b",
        r"^how\b(?!.*\b(much|many|long|far|to)\b)",
        r"\bexplain\b",
        r"\bdescribe\b",
    ]

    DEFINITION_PATTERNS = [
        r"^what (is|are|was|were)\b",
        r"^define\b",
        r"\bdefinition\b",
        r"\bmean(s|ing)?\b",
    ]

    PROCEDURAL_PATTERNS = [
        r"^how (to|do|can|should)\b",
        r"\bsteps?\b.*\b(to|for)\b",
        r"\bprocess\b.*\b(to|for|of)\b",
        r"\bway to\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bdifference\s+(between|of)\b",
        r"\bbetter\b.*\bor\b",
        r"\bsimilar(ity|ities)?\b",
    ]

    LIST_PATTERNS = [
        r"\blist\b",
        r"\ball\b.*\b(the|of)\b",
        r"\bwhat (are|were)\b.*\b(types|kinds|categories|options)\b",
        r"\benumerate\b",
    ]

    YES_NO_PATTERNS = [
        r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
    ]

    # Follow-up indicators
    FOLLOW_UP_PATTERNS = [
        r"^(and|also|what about|how about)\b",
        r"^(tell me more|more about|elaborate)\b",
        r"\bprevious(ly)?\b",
        r"\babove\b",
        r"\bearlier\b",
    ]

    # Reference patterns (pronouns referring to previous context)
    REFERENCE_PATTERNS = [
        r"\b(it|they|them|this|that|these|those|its|their)\b",
        r"\b(the same|such|said)\b",
    ]

    # Stop words for term extraction
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "as", "until", "while", "about", "against", "between",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "any", "both",
    }

    def __init__(
        self,
        custom_entities: list[str] | None = None,
        entity_matcher: Callable[[str], list[str]] | None = None,
    ) -> None:
        """
        Initialize query parser.

        Args:
            custom_entities: Domain-specific entities to recognize
            entity_matcher: Custom function to extract entities
        """
        self.custom_entities = set(e.lower() for e in (custom_entities or []))
        self.entity_matcher = entity_matcher

        # Compile patterns
        self._factual_re = [re.compile(p, re.IGNORECASE)
                            for p in self.FACTUAL_PATTERNS]
        self._explanation_re = [re.compile(
            p, re.IGNORECASE) for p in self.EXPLANATION_PATTERNS]
        self._definition_re = [re.compile(
            p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS]
        self._procedural_re = [re.compile(
            p, re.IGNORECASE) for p in self.PROCEDURAL_PATTERNS]
        self._comparison_re = [re.compile(
            p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._list_re = [re.compile(p, re.IGNORECASE)
                         for p in self.LIST_PATTERNS]
        self._yes_no_re = [re.compile(p, re.IGNORECASE)
                           for p in self.YES_NO_PATTERNS]
        self._follow_up_re = [re.compile(p, re.IGNORECASE)
                              for p in self.FOLLOW_UP_PATTERNS]
        self._reference_re = [re.compile(p, re.IGNORECASE)
                              for p in self.REFERENCE_PATTERNS]

        logger.debug("QueryParser initialized")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a user query.

        Args:
            query: Raw user query text

        Returns:
            ParsedQuery with extracted information
        """
        # Normalize
        normalized = self._normalize_query(query)

        # Classify query type
        query_type = self._classify_query_type(normalized)

        # Determine intent
        intent = self._determine_intent(normalized, query_type)

        # Extract key terms
        key_terms = self._extract_key_terms(normalized)

        # Extract entities
        entities = self._extract_entities(normalized)

        # Check for constraints
        constraints = self._extract_constraints(normalized)

        # Check if follow-up
        is_follow_up = self._is_follow_up(normalized)
        references_previous = self._references_previous(normalized)

        # Calculate confidence
        confidence = self._calculate_confidence(query_type, key_terms)

        return ParsedQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            intent=intent,
            key_terms=key_terms,
            entities=entities,
            constraints=constraints,
            is_follow_up=is_follow_up,
            references_previous=references_previous,
            confidence=confidence,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query text."""
        # Strip and lowercase
        normalized = query.strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Ensure ends with question mark if it's a question
        if not normalized.endswith("?") and self._looks_like_question(normalized):
            normalized += "?"

        return normalized

    def _looks_like_question(self, text: str) -> bool:
        """Check if text appears to be a question."""
        text_lower = text.lower()
        question_starters = [
            "what", "where", "when", "why", "how", "who", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "will", "would", "should", "has", "have",
        ]
        return any(text_lower.startswith(w + " ") for w in question_starters)

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the query type."""
        query_lower = query.lower()

        # Check patterns in order of specificity
        if any(p.search(query_lower) for p in self._comparison_re):
            return QueryType.COMPARISON

        if any(p.search(query_lower) for p in self._procedural_re):
            return QueryType.PROCEDURAL

        if any(p.search(query_lower) for p in self._list_re):
            return QueryType.LIST

        if any(p.search(query_lower) for p in self._definition_re):
            return QueryType.DEFINITION

        if any(p.search(query_lower) for p in self._explanation_re):
            return QueryType.EXPLANATION

        if any(p.search(query_lower) for p in self._yes_no_re):
            return QueryType.YES_NO

        if any(p.search(query_lower) for p in self._factual_re):
            return QueryType.FACTUAL

        return QueryType.OPEN_ENDED

    def _determine_intent(self, query: str, query_type: QueryType) -> QueryIntent:
        """Determine user intent."""
        query_lower = query.lower()

        # Check for specific intent patterns
        if any(w in query_lower for w in ["fix", "error", "problem", "issue", "not working", "broken"]):
            return QueryIntent.TROUBLESHOOTING

        if any(w in query_lower for w in ["compare", "versus", "vs", "difference", "better"]):
            return QueryIntent.COMPARISON

        if any(w in query_lower for w in ["summarize", "summary", "overview", "brief"]):
            return QueryIntent.SUMMARIZATION

        if any(w in query_lower for w in ["true", "false", "correct", "accurate", "really"]):
            return QueryIntent.VERIFICATION

        if any(w in query_lower for w in ["explore", "browse", "discover", "find out"]):
            return QueryIntent.EXPLORATION

        # Default based on query type
        return QueryIntent.INFORMATION_SEEKING

    def _extract_key_terms(self, query: str) -> list[str]:
        """Extract key search terms from query."""
        # Remove punctuation except hyphens in compounds
        text = re.sub(r"[^\w\s-]", " ", query.lower())

        # Tokenize
        words = text.split()

        # Filter stop words and short words
        key_terms = [
            w for w in words
            if w not in self.STOP_WORDS
            and len(w) > 2
            and not w.isdigit()
        ]

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities from query."""
        entities = []

        # Use custom matcher if provided
        if self.entity_matcher:
            entities.extend(self.entity_matcher(query))

        # Check for custom entities
        query_lower = query.lower()
        for entity in self.custom_entities:
            if entity in query_lower:
                entities.append(entity)

        # Extract quoted phrases as entities
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        quoted_single = re.findall(r"'([^']+)'", query)
        entities.extend(quoted_single)

        # Extract capitalized phrases (potential proper nouns)
        # Skip first word as it's usually capitalized anyway
        words = query.split()
        if len(words) > 1:
            for i, word in enumerate(words[1:], 1):
                if word[0].isupper() and word.lower() not in self.STOP_WORDS:
                    # Check if part of a multi-word entity
                    entity_words = [word]
                    j = i + 1
                    while j < len(words) and words[j][0].isupper():
                        entity_words.append(words[j])
                        j += 1
                    entity = " ".join(entity_words)
                    if entity not in entities:
                        entities.append(entity)

        return entities

    def _extract_constraints(self, query: str) -> dict:
        """Extract constraints from query."""
        constraints = {}
        query_lower = query.lower()

        # Time constraints
        time_patterns = {
            "recent": r"\b(recent|latest|new|current)\b",
            "past": r"\b(old|previous|past|earlier|before)\b",
            "specific_time": r"\b(in|during|since|after|before)\s+(\d{4}|\w+\s+\d{4})\b",
        }

        for constraint_type, pattern in time_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                constraints["time"] = constraint_type
                if constraint_type == "specific_time" and match.group(2):
                    constraints["time_value"] = match.group(2)
                break

        # Quantity constraints
        quantity_match = re.search(
            r"\b(top|first|last)\s+(\d+)\b", query_lower)
        if quantity_match:
            constraints["limit"] = int(quantity_match.group(2))

        # Scope constraints
        if any(w in query_lower for w in ["all", "every", "complete", "full"]):
            constraints["scope"] = "comprehensive"
        elif any(w in query_lower for w in ["brief", "quick", "short", "simple"]):
            constraints["scope"] = "brief"

        return constraints

    def _is_follow_up(self, query: str) -> bool:
        """Check if query is a follow-up question."""
        return any(p.search(query) for p in self._follow_up_re)

    def _references_previous(self, query: str) -> bool:
        """Check if query references previous context."""
        # More likely a reference if short query with pronouns
        has_reference = any(p.search(query) for p in self._reference_re)

        # Short queries with pronouns are likely referencing previous context
        word_count = len(query.split())
        return has_reference and word_count < 10

    def _calculate_confidence(
        self,
        query_type: QueryType,
        key_terms: list[str],
    ) -> float:
        """Calculate confidence in parsing."""
        confidence = 1.0

        # Lower confidence for open-ended queries
        if query_type == QueryType.OPEN_ENDED:
            confidence *= 0.7

        # Lower confidence if few key terms
        if len(key_terms) < 2:
            confidence *= 0.8

        # Higher confidence for clear question types
        if query_type in (QueryType.DEFINITION, QueryType.YES_NO, QueryType.PROCEDURAL):
            confidence *= 1.1

        return min(1.0, confidence)

    def add_entity(self, entity: str) -> None:
        """Add a custom entity to recognize."""
        self.custom_entities.add(entity.lower())

    def add_entities(self, entities: list[str]) -> None:
        """Add multiple custom entities."""
        for entity in entities:
            self.add_entity(entity)
