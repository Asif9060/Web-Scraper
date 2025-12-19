"""
Query expansion for improved retrieval.

Generates alternative queries and search terms
to improve retrieval coverage.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from web_intel.query_parser.parser import ParsedQuery, QueryType
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpandedQuery:
    """
    Result of query expansion.

    Contains the original query plus generated alternatives.
    """

    original_query: str
    parsed: ParsedQuery
    search_queries: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    reformulations: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        """Get all query variations."""
        queries = [self.original_query]
        queries.extend(self.search_queries)
        queries.extend(self.reformulations)
        return list(dict.fromkeys(queries))  # Dedupe preserving order

    def get_weighted_queries(self) -> list[tuple[str, float]]:
        """Get queries with their weights."""
        result = [(self.original_query, 1.0)]

        for query in self.search_queries:
            weight = self.weights.get(query, 0.8)
            result.append((query, weight))

        for query in self.reformulations:
            weight = self.weights.get(query, 0.6)
            result.append((query, weight))

        return result


class QueryExpander:
    """
    Expands queries for improved retrieval.

    Generates synonyms, reformulations, and alternative
    search queries to improve recall.

    Example:
        >>> expander = QueryExpander()
        >>> expanded = expander.expand(parsed_query)
        >>> for query in expanded.all_queries:
        ...     results = search(query)
    """

    # Common synonyms for query expansion
    SYNONYM_MAP: dict[str, list[str]] = {
        "price": ["cost", "pricing", "fee", "charge", "rate"],
        "buy": ["purchase", "order", "get", "acquire"],
        "sell": ["offer", "provide", "supply"],
        "product": ["item", "goods", "merchandise", "offering"],
        "service": ["solution", "offering", "support"],
        "help": ["support", "assistance", "guide", "documentation"],
        "contact": ["reach", "email", "call", "phone"],
        "location": ["address", "place", "where", "office"],
        "job": ["career", "position", "employment", "opening", "vacancy"],
        "about": ["information", "info", "details", "overview"],
        "start": ["begin", "get started", "setup", "initialize"],
        "stop": ["end", "cancel", "terminate", "disable"],
        "change": ["modify", "update", "edit", "alter"],
        "delete": ["remove", "erase", "clear"],
        "create": ["make", "add", "new", "generate"],
        "find": ["search", "locate", "discover", "look for"],
        "show": ["display", "view", "see", "list"],
        "error": ["problem", "issue", "bug", "failure"],
        "fix": ["solve", "resolve", "repair", "correct"],
        "fast": ["quick", "rapid", "speedy"],
        "slow": ["sluggish", "delayed", "lagging"],
        "feature": ["capability", "function", "functionality"],
        "login": ["sign in", "log in", "authenticate"],
        "logout": ["sign out", "log out"],
        "signup": ["register", "sign up", "create account"],
        "download": ["get", "obtain", "fetch"],
        "upload": ["send", "submit", "transfer"],
        "free": ["no cost", "complimentary", "gratis"],
        "paid": ["premium", "pro", "subscription"],
    }

    # Query reformulation templates
    REFORMULATION_TEMPLATES: dict[QueryType, list[str]] = {
        QueryType.FACTUAL: [
            "{terms}",
            "what is {terms}",
            "{terms} information",
        ],
        QueryType.DEFINITION: [
            "definition of {terms}",
            "{terms} meaning",
            "what is {terms}",
        ],
        QueryType.PROCEDURAL: [
            "how to {terms}",
            "{terms} guide",
            "{terms} tutorial",
            "{terms} steps",
            "{terms} instructions",
        ],
        QueryType.LIST: [
            "list of {terms}",
            "{terms} options",
            "all {terms}",
            "{terms} types",
        ],
        QueryType.COMPARISON: [
            "{terms} comparison",
            "{terms} vs",
            "{terms} differences",
        ],
        QueryType.EXPLANATION: [
            "why {terms}",
            "{terms} explanation",
            "how {terms} works",
            "{terms} reason",
        ],
        QueryType.YES_NO: [
            "{terms}",
            "does {terms}",
            "is {terms}",
        ],
        QueryType.NAVIGATIONAL: [
            "{terms} page",
            "where {terms}",
            "find {terms}",
        ],
        QueryType.OPEN_ENDED: [
            "{terms}",
            "{terms} information",
            "about {terms}",
        ],
    }

    def __init__(
        self,
        llm_expander: Callable[[str], list[str]] | None = None,
        max_synonyms_per_term: int = 3,
        max_reformulations: int = 5,
    ) -> None:
        """
        Initialize query expander.

        Args:
            llm_expander: Optional LLM-based expansion function
            max_synonyms_per_term: Maximum synonyms to include per term
            max_reformulations: Maximum reformulations to generate
        """
        self.llm_expander = llm_expander
        self.max_synonyms_per_term = max_synonyms_per_term
        self.max_reformulations = max_reformulations

        # Build reverse synonym map for lookup
        self._synonym_lookup: dict[str, list[str]] = {}
        for key, synonyms in self.SYNONYM_MAP.items():
            self._synonym_lookup[key] = synonyms
            for syn in synonyms:
                if syn not in self._synonym_lookup:
                    self._synonym_lookup[syn] = []
                if key not in self._synonym_lookup[syn]:
                    self._synonym_lookup[syn].append(key)

        logger.debug("QueryExpander initialized")

    def expand(
        self,
        parsed: ParsedQuery,
        use_llm: bool = True,
    ) -> ExpandedQuery:
        """
        Expand a parsed query.

        Args:
            parsed: Parsed query to expand
            use_llm: Whether to use LLM expansion if available

        Returns:
            ExpandedQuery with alternatives
        """
        expanded = ExpandedQuery(
            original_query=parsed.original_query,
            parsed=parsed,
        )

        # Generate synonyms for key terms
        expanded.synonyms = self._generate_synonyms(parsed.key_terms)

        # Generate search queries
        expanded.search_queries = self._generate_search_queries(parsed)

        # Generate reformulations
        expanded.reformulations = self._generate_reformulations(parsed)

        # Assign weights
        expanded.weights = self._assign_weights(expanded)

        # Use LLM expansion if available
        if use_llm and self.llm_expander:
            try:
                llm_queries = self.llm_expander(parsed.original_query)
                for q in llm_queries:
                    if q not in expanded.search_queries:
                        expanded.search_queries.append(q)
                        expanded.weights[q] = 0.85
            except Exception as e:
                logger.warning(f"LLM expansion failed: {e}")

        return expanded

    def _generate_synonyms(self, terms: list[str]) -> dict[str, list[str]]:
        """Generate synonyms for terms."""
        synonyms = {}

        for term in terms:
            term_lower = term.lower()
            if term_lower in self._synonym_lookup:
                syns = self._synonym_lookup[term_lower][: self.max_synonyms_per_term]
                if syns:
                    synonyms[term] = syns

        return synonyms

    def _generate_search_queries(self, parsed: ParsedQuery) -> list[str]:
        """Generate search query variations."""
        queries = []

        # Base search query from key terms
        if parsed.key_terms:
            base_query = " ".join(parsed.key_terms)
            queries.append(base_query)

            # Add entity-focused queries
            for entity in parsed.entities:
                entity_query = f"{entity} {base_query}"
                if entity_query not in queries:
                    queries.append(entity_query)

            # Add synonym-expanded queries
            expanded_terms = self._expand_with_synonyms(parsed.key_terms)
            for terms in expanded_terms[:3]:
                query = " ".join(terms)
                if query not in queries:
                    queries.append(query)

        # Add constraint-based queries
        if parsed.constraints.get("time") == "recent":
            queries.append(f"latest {' '.join(parsed.key_terms)}")
        elif parsed.constraints.get("time") == "past":
            queries.append(f"previous {' '.join(parsed.key_terms)}")

        return queries[: self.max_reformulations * 2]

    def _expand_with_synonyms(self, terms: list[str]) -> list[list[str]]:
        """Generate term combinations with synonyms."""
        if not terms:
            return []

        # Find terms with synonyms
        term_options = []
        for term in terms:
            options = [term]
            if term.lower() in self._synonym_lookup:
                options.extend(self._synonym_lookup[term.lower()][:2])
            term_options.append(options)

        # Generate combinations (limited to avoid explosion)
        combinations = []
        if len(term_options) == 1:
            for opt in term_options[0]:
                combinations.append([opt])
        elif len(term_options) == 2:
            for opt1 in term_options[0][:2]:
                for opt2 in term_options[1][:2]:
                    combinations.append([opt1, opt2])
        else:
            # For more terms, just do single substitutions
            base = [opts[0] for opts in term_options]
            combinations.append(base)
            for i, opts in enumerate(term_options):
                for opt in opts[1:2]:  # Just first synonym
                    new_combo = base.copy()
                    new_combo[i] = opt
                    combinations.append(new_combo)

        return combinations

    def _generate_reformulations(self, parsed: ParsedQuery) -> list[str]:
        """Generate query reformulations."""
        reformulations = []
        terms_str = " ".join(parsed.key_terms) if parsed.key_terms else ""

        if not terms_str:
            return []

        # Get templates for this query type
        templates = self.REFORMULATION_TEMPLATES.get(
            parsed.query_type,
            self.REFORMULATION_TEMPLATES[QueryType.OPEN_ENDED],
        )

        for template in templates[: self.max_reformulations]:
            try:
                reformulation = template.format(terms=terms_str)
                if reformulation != parsed.original_query.lower():
                    reformulations.append(reformulation)
            except (KeyError, ValueError):
                continue

        return reformulations

    def _assign_weights(self, expanded: ExpandedQuery) -> dict[str, float]:
        """Assign weights to expanded queries."""
        weights = {}

        # Search queries get higher weight
        for i, query in enumerate(expanded.search_queries):
            # Decay weight for later queries
            weights[query] = max(0.5, 0.9 - (i * 0.1))

        # Reformulations get lower weight
        for i, query in enumerate(expanded.reformulations):
            weights[query] = max(0.3, 0.7 - (i * 0.1))

        return weights

    def expand_for_semantic_search(
        self,
        parsed: ParsedQuery,
    ) -> list[str]:
        """
        Generate queries optimized for semantic/embedding search.

        Semantic search works best with natural language queries,
        so we generate full sentence variations.

        Args:
            parsed: Parsed query

        Returns:
            List of queries for semantic search
        """
        queries = [parsed.original_query]

        # Add normalized version
        if parsed.normalized_query != parsed.original_query:
            queries.append(parsed.normalized_query)

        # Generate natural language variations
        terms = " ".join(
            parsed.key_terms) if parsed.key_terms else parsed.normalized_query

        variations = [
            f"Information about {terms}",
            f"Tell me about {terms}",
            f"What is {terms}",
        ]

        # Add type-specific variations
        if parsed.query_type == QueryType.PROCEDURAL:
            variations.extend([
                f"How to {terms}",
                f"Steps to {terms}",
                f"Guide for {terms}",
            ])
        elif parsed.query_type == QueryType.EXPLANATION:
            variations.extend([
                f"Why {terms}",
                f"Explanation of {terms}",
            ])
        elif parsed.query_type == QueryType.COMPARISON:
            variations.extend([
                f"Comparing {terms}",
                f"Differences in {terms}",
            ])

        queries.extend(variations[:3])
        return list(dict.fromkeys(queries))  # Dedupe

    def expand_for_keyword_search(
        self,
        parsed: ParsedQuery,
    ) -> list[str]:
        """
        Generate queries optimized for keyword/BM25 search.

        Keyword search works best with specific terms,
        so we generate focused term combinations.

        Args:
            parsed: Parsed query

        Returns:
            List of queries for keyword search
        """
        queries = []

        # Start with key terms
        if parsed.key_terms:
            queries.append(" ".join(parsed.key_terms))

            # Add entity + terms combinations
            for entity in parsed.entities[:2]:
                queries.append(f"{entity} {' '.join(parsed.key_terms[:3])}")

            # Add synonym variations
            for term in parsed.key_terms[:2]:
                if term.lower() in self._synonym_lookup:
                    for syn in self._synonym_lookup[term.lower()][:2]:
                        new_terms = [syn if t ==
                                     term else t for t in parsed.key_terms]
                        queries.append(" ".join(new_terms))

        return list(dict.fromkeys(queries))[:5]  # Dedupe and limit

    def add_synonym(self, word: str, synonyms: list[str]) -> None:
        """Add custom synonyms."""
        word_lower = word.lower()
        self.SYNONYM_MAP[word_lower] = synonyms
        self._synonym_lookup[word_lower] = synonyms
        for syn in synonyms:
            if syn not in self._synonym_lookup:
                self._synonym_lookup[syn] = []
            if word_lower not in self._synonym_lookup[syn]:
                self._synonym_lookup[syn].append(word_lower)
